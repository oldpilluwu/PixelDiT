from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from .common import DTYPES, ROOT, build_core, load_weights, write_json

if str(ROOT / "c2i") not in sys.path:
    sys.path.insert(0, str(ROOT / "c2i"))


def rows_from_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seeded_noise(rows: list[dict], shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    values = []
    for row in rows:
        generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
        values.append(torch.randn(shape, generator=generator, device=device, dtype=torch.float32))
    return torch.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic C2I Phase 0 regression samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--timeshift", type=float, required=True)
    parser.add_argument("--guidance-min", type=float, default=0.1)
    parser.add_argument("--guidance-max", type=float, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from torchvision.utils import save_image
    from src.diffusion import FlowDPMSolverSampler, LinearScheduler, ode_step_fn, simple_guidance_fn

    if not torch.cuda.is_available():
        raise RuntimeError("C2I regression sampling requires CUDA")
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    model, mode, _ = build_core(args.config)
    if mode != "c2i":
        raise ValueError(f"Expected a C2I config, got {mode}")
    weights = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    sampler = FlowDPMSolverSampler(
        scheduler=LinearScheduler(),
        w_scheduler=LinearScheduler(),
        num_steps=args.num_steps,
        guidance=args.cfg_scale,
        timeshift=args.timeshift,
        guidance_interval_min=args.guidance_min,
        guidance_interval_max=args.guidance_max,
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    ).to(device)

    rows = rows_from_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            noise = seeded_noise(batch_rows, (model.in_channels, args.height, args.width), device)
            condition = torch.tensor([row["class_id"] for row in batch_rows], device=device, dtype=torch.long)
            uncondition = torch.full_like(condition, model.num_classes)
            samples = sampler(model, noise, condition, uncondition)
            for row, sample in zip(batch_rows, samples):
                name = f"{int(row['index']):05d}_class{int(row['class_id']):04d}_seed{int(row['seed'])}.png"
                save_image(sample.float(), output / name, normalize=True, value_range=(-1, 1))
            print(f"{min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    write_json(
        output / "run.json",
        {
            "config": os.path.abspath(args.config),
            "checkpoint": weights,
            "manifest": os.path.abspath(args.manifest),
            "resolution": [args.height, args.width],
            "batch_size": args.batch_size,
            "precision": args.dtype,
            "solver": "FlowDPMSolverSampler",
            "num_steps": args.num_steps,
            "nfe": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "timeshift": args.timeshift,
            "guidance_interval": [args.guidance_min, args.guidance_max],
            "sample_count": len(rows),
        },
    )
    print(f"Wrote {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
