from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch

from .common import git_metadata, resolve_checkpoint, sha256_file, write_json


PACKAGES = (
    "torch",
    "torchvision",
    "lightning",
    "transformers",
    "diffusers",
    "timm",
    "xformers",
    "flash-attn",
)


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": repr(exc)}


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a reproducible Phase 0 environment manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--settings-json", help="Optional JSON object containing solver/CFG/precision settings.")
    args = parser.parse_args()

    checkpoints = []
    for requested in args.checkpoint:
        resolved = resolve_checkpoint(requested)
        checkpoints.append(
            {
                "requested": requested,
                "path": resolved,
                "bytes": os.path.getsize(resolved),
                "sha256": sha256_file(resolved),
            }
        )

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]

    settings = json.loads(args.settings_json) if args.settings_json else {}
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "checkpoints": checkpoints,
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": package_versions(),
        "cuda": cuda,
        "nvidia_smi_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,"
                "clocks.current.graphics,clocks.max.graphics,power.limit,power.max_limit",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_full": command_output(["nvidia-smi", "-q"]),
        "settings": settings,
    }
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

