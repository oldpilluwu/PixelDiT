from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pixdit_core.pixeldit_c2i import PixDiT
from pixdit_core.pixeldit_t2i import PixDiT_T2I


DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_core(config_path: str | os.PathLike[str]) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """Build the unmodified C2I or T2I core described by a repository config."""
    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    if "denoiser" in model_cfg:
        model = PixDiT(**model_cfg["denoiser"].get("init_args", {}))
        return model, "c2i", cfg
    if "extra" in model_cfg:
        extra = model_cfg["extra"]
        text_cfg = cfg.get("text_encoder", {})
        model = PixDiT_T2I(
            in_channels=3,
            num_groups=int(extra.get("num_groups", 24)),
            hidden_size=int(extra.get("hidden_size", 1536)),
            pixel_hidden_size=int(extra.get("pixel_hidden_size", 16)),
            pixel_attn_hidden_size=extra.get("pixel_attn_hidden_size"),
            pixel_num_groups=extra.get("pixel_num_groups"),
            patch_depth=int(extra.get("patch_depth", 14)),
            pixel_depth=int(extra.get("pixel_depth", 2)),
            num_text_blocks=int(extra.get("num_text_blocks", 4)),
            patch_size=int(extra.get("patch_size", 16)),
            txt_embed_dim=int(extra.get("txt_embed_dim", text_cfg.get("caption_channels", 2304))),
            txt_max_length=int(extra.get("txt_max_length", text_cfg.get("model_max_length", 300))),
            use_text_rope=bool(extra.get("use_text_rope", True)),
            text_rope_theta=float(extra.get("text_rope_theta", 10000.0)),
            repa_encoder_index=int(extra.get("repa_encoder_index", -1)),
            use_pixel_abs_pos=bool(extra.get("use_pixel_abs_pos", True)),
            pit_adaln_post_modulation=bool(extra.get("pit_adaln_post_modulation", False)),
        )
        return model, "t2i", cfg
    raise ValueError(f"Cannot detect PixelDiT model type from {config_path}")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    from tools.download import resolve_checkpoint as repository_resolver

    resolved = repository_resolver(path)
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint was not found or downloaded: {path}")
    return os.path.abspath(resolved)


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    for key in ("state_dict_ema", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {str(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}


def _key_candidates(target_key: str, mode: str) -> Iterable[str]:
    if mode == "c2i":
        prefixes = (
            "ema_denoiser._orig_mod.",
            "ema_denoiser.",
            "denoiser._orig_mod.",
            "denoiser.",
            "_orig_mod.",
            "module.",
            "",
        )
    else:
        prefixes = (
            "module.core.",
            "_orig_mod.core.",
            "core.",
            "module.",
            "_orig_mod.",
            "",
        )
    for prefix in prefixes:
        yield prefix + target_key


def load_weights(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    mode: str,
    minimum_coverage: float = 0.90,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"loaded": False, "checkpoint": None, "coverage": 0.0}
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _state_dict_from_checkpoint(raw)
    target = model.state_dict()
    selected: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for target_key, target_value in target.items():
        for candidate in _key_candidates(target_key, mode):
            value = source.get(candidate)
            if value is None:
                continue
            if value.shape != target_value.shape:
                shape_mismatches.append(
                    f"{candidate}: checkpoint {tuple(value.shape)} != model {tuple(target_value.shape)}"
                )
                continue
            selected[target_key] = value
            break
    coverage = len(selected) / max(len(target), 1)
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"Only {coverage:.1%} of model tensors matched {checkpoint_path}; "
            f"refusing an untrustworthy baseline load"
        )
    missing, unexpected = model.load_state_dict(selected, strict=False)
    return {
        "loaded": True,
        "checkpoint": checkpoint_path,
        "sha256": sha256_file(checkpoint_path),
        "coverage": coverage,
        "matched_tensors": len(selected),
        "model_tensors": len(target),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "shape_mismatches": shape_mismatches,
    }


def make_inputs(
    model: torch.nn.Module,
    mode: str,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    text_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch_size, model.in_channels, height, width, device=device, dtype=dtype, generator=generator)
    t = torch.linspace(0.2, 0.8, batch_size, device=device, dtype=dtype)
    if mode == "c2i":
        y = torch.arange(batch_size, device=device, dtype=torch.long) % model.num_classes
    else:
        y = torch.randn(
            batch_size,
            text_length,
            model.txt_embed_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return x, t, y


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q).item())


def summarize(values: list[float]) -> dict[str, float]:
    median = percentile(values, 0.5)
    mean = sum(values) / max(len(values), 1)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
    return {
        "median": median,
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "mean": mean,
        "std": variance**0.5,
        "cv_percent": 100.0 * variance**0.5 / median if median else float("inf"),
    }


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

