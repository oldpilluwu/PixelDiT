from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from .common import git_metadata, sha256_file, write_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".npz"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze baseline outputs as a content-addressed manifest.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-metadata", help="Environment or invocation manifest associated with this run.")
    args = parser.parse_args()

    root = Path(args.input_dir).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": os.fspath(root),
        "file_count": len(files),
        "git": git_metadata(),
        "run_metadata": os.path.abspath(args.run_metadata) if args.run_metadata else None,
        "files": files,
    }
    write_json(args.output, payload)
    print(f"Frozen {len(files)} outputs in {args.output}")


if __name__ == "__main__":
    main()

