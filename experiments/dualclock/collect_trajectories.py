from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from .common import (
    DTYPES,
    build_core,
    git_metadata,
    load_weights,
    sha256_file,
    write_json,
)
from .parity import compare, extract_and_reinject
from .phase1 import (
    CollectionOptions,
    autocast_context,
    collect_c2i_batch,
    default_patch_layers,
    parse_indices,
)
from .sample_c2i import rows_from_jsonl, seeded_noise


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect exact C2I trajectories and Phase 1 semantic-substitution probes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="experiments/dualclock/traces")
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=2.75)
    parser.add_argument("--timeshift", type=float, default=1.0)
    parser.add_argument("--guidance-min", type=float, default=0.1)
    parser.add_argument("--guidance-max", type=float, default=0.9)
    parser.add_argument("--patch-layers", help="Comma-separated; default is early,middle,late.")
    parser.add_argument("--stale-horizons", default="1,2,3")
    parser.add_argument("--generic-pit-layer", type=int, default=-1)
    parser.add_argument("--activation-storage", choices=("sketch", "full"), default="sketch")
    parser.add_argument("--sketch-dim", type=int, default=1024)
    parser.add_argument(
        "--trace-dtype",
        choices=("same", "bfloat16", "float16", "float32"),
        default="same",
        help="Defaults to the model dtype so exact states are archived losslessly.",
    )
    parser.add_argument(
        "--sensitivity-steps",
        default="0,25,50,75,99",
        help="Steps at which grouped token/channel influence is evaluated.",
    )
    parser.add_argument("--token-groups", type=int, default=16)
    parser.add_argument("--channel-groups", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    trace_dtype = {
        "same": dtype,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.trace_dtype]
    model, mode, _ = build_core(args.config)
    if mode != "c2i":
        raise ValueError("Phase 1 collection currently starts with the official C2I track")
    weights = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    baseline_path = Path(args.baseline_report).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Phase 0 baseline report was not found: {baseline_path}")
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    if baseline_payload.get("mode") != "c2i":
        raise ValueError("The supplied Phase 0 baseline report is not a C2I report")
    if baseline_payload.get("resolution") != [args.height, args.width]:
        raise ValueError(
            f"Baseline resolution {baseline_payload.get('resolution')} does not match "
            f"collection resolution {[args.height, args.width]}"
        )
    baseline_checkpoint = baseline_payload.get("checkpoint", {}).get("sha256")
    if baseline_checkpoint and baseline_checkpoint != weights.get("sha256"):
        raise ValueError("Phase 0 baseline and Phase 1 checkpoint hashes do not match")
    baseline_dtype = baseline_payload.get("dtype")
    if baseline_dtype and baseline_dtype != str(dtype):
        raise ValueError(
            f"Baseline dtype {baseline_dtype} does not match requested Phase 1 dtype {dtype}"
        )
    patch_layers = (
        parse_indices(args.patch_layers, len(model.patch_blocks))
        if args.patch_layers
        else default_patch_layers(len(model.patch_blocks))
    )
    if patch_layers[-1] != len(model.patch_blocks) - 1:
        patch_layers.append(len(model.patch_blocks) - 1)

    rows = rows_from_jsonl(args.manifest)[: args.limit]
    if not rows:
        raise ValueError("The regression manifest did not contain any rows")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (Path(args.output_dir) / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = CollectionOptions(
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        timeshift=args.timeshift,
        guidance_min=args.guidance_min,
        guidance_max=args.guidance_max,
        stale_horizons=_parse_csv_ints(args.stale_horizons),
        generic_pit_layer=args.generic_pit_layer,
        activation_storage=args.activation_storage,
        sketch_dim=args.sketch_dim,
        output_dtype=trace_dtype,
        sensitivity_steps=_parse_csv_ints(args.sensitivity_steps),
        token_groups=args.token_groups,
        channel_groups=args.channel_groups,
    )
    run = {
        "phase": 1,
        "created_utc": stamp,
        "git": git_metadata(),
        "config": os.path.abspath(args.config),
        "checkpoint": weights,
        "baseline_report": {
            "path": os.fspath(baseline_path),
            "sha256": sha256_file(baseline_path),
            "checkpoint_sha256": baseline_checkpoint,
        },
        "regression_manifest": os.path.abspath(args.manifest),
        "resolution": [args.height, args.width],
        "sample_count": len(rows),
        "batch_size": args.batch_size,
        "precision": args.dtype,
        "trace_dtype": str(trace_dtype),
        "solver": "FlowDPMSolverSampler-compatible AB2",
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "timeshift": args.timeshift,
        "guidance_interval": [args.guidance_min, args.guidance_max],
        "patch_layers": patch_layers,
        "stale_horizons": list(options.stale_horizons),
        "generic_pit_layer": args.generic_pit_layer,
        "activation_storage": args.activation_storage,
        "sketch_dim": args.sketch_dim,
        "sensitivity_steps": list(options.sensitivity_steps),
        "token_groups": args.token_groups,
        "channel_groups": args.channel_groups,
        "measurement_contract": {
            "exact": ["x_t", "final semantic tokens", "velocity branches", "guided velocity"],
            "semantic_and_velocity_storage_lossless": (
                trace_dtype == dtype or trace_dtype == torch.float32
            ),
            "solver_state_storage": "native FP32, matching the released C2I sampler",
            "activation_mode": (
                "exact full tensors"
                if args.activation_storage == "full"
                else "deterministic pooled temporal sketches with exact L2 norms"
            ),
            "substitutions": "same-state decoder evaluations; scalar errors retained",
        },
    }
    write_json(output / "run.json", run)

    first = rows[: min(args.batch_size, len(rows))]
    parity_noise = seeded_noise(
        first, (model.in_channels, args.height, args.width), device
    )
    parity_condition = torch.tensor(
        [row["class_id"] for row in first], device=device, dtype=torch.long
    )
    parity_t = torch.full((len(first),), 0.5, device=device, dtype=torch.float32)
    with autocast_context(device, dtype), torch.inference_mode():
        exact, reinjected, _ = extract_and_reinject(model, parity_noise, parity_t, parity_condition)
    parity = compare(exact, reinjected, atol=0.0, rtol=0.0)
    write_json(output / "semantic_parity.json", parity)
    if not parity["allclose"]:
        raise RuntimeError("Semantic parity failed; refusing to collect Phase 1 traces")

    manifest_rows: list[dict] = []
    with torch.inference_mode(), autocast_context(device, dtype):
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            noise = seeded_noise(
                batch_rows, (model.in_channels, args.height, args.width), device
            )
            condition = torch.tensor(
                [row["class_id"] for row in batch_rows], device=device, dtype=torch.long
            )
            uncondition = torch.full_like(condition, model.num_classes)
            trace = collect_c2i_batch(
                model, noise, condition, uncondition, options, patch_layers=patch_layers
            )
            trace["samples"] = batch_rows
            name = f"trajectory_{start:05d}_{start + len(batch_rows) - 1:05d}.pt"
            torch.save(trace, output / name)
            manifest_rows.append(
                {
                    "file": name,
                    "start": start,
                    "count": len(batch_rows),
                    "indices": [int(row["index"]) for row in batch_rows],
                    "classes": [int(row["class_id"]) for row in batch_rows],
                    "seeds": [int(row["seed"]) for row in batch_rows],
                }
            )
            write_json(output / "manifest.json", {"schema_version": 1, "shards": manifest_rows})
            print(f"{start + len(batch_rows)}/{len(rows)}", flush=True)
    print(f"Phase 1 traces: {output}")


if __name__ == "__main__":
    main()
