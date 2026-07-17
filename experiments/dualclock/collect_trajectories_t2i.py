from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .common import (
    DTYPES,
    ROOT,
    build_core,
    git_metadata,
    load_weights,
    load_yaml,
    sha256_file,
    write_json,
)
from .parity import compare, extract_and_reinject
from .phase1 import (
    CollectionOptions,
    default_patch_layers,
    parse_indices,
)
from .phase1_t2i import collect_t2i_batch
from .sample_c2i import rows_from_jsonl


if str(ROOT / "t2i") not in sys.path:
    sys.path.insert(0, str(ROOT / "t2i"))


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _encode_prompts(
    prompts: list[str],
    tokenizer: Any,
    text_encoder: torch.nn.Module,
    text_config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_max_length = int(text_config.get("model_max_length", 300))
    chi_lines = text_config.get("chi_prompt") or []
    if chi_lines:
        chi_prompt = "\n".join(chi_lines)
        prompts_all = [chi_prompt + prompt for prompt in prompts]
        num_chi_tokens = len(tokenizer.encode(chi_prompt))
        max_length_all = num_chi_tokens + model_max_length - 2
    else:
        prompts_all = prompts
        max_length_all = model_max_length
    tokens = tokenizer(
        prompts_all,
        max_length=max_length_all,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    select_index = [0] + list(range(-model_max_length + 1, 0))
    embeddings = text_encoder(
        tokens.input_ids, tokens.attention_mask
    )[0][:, select_index]
    masks = tokens.attention_mask[:, select_index]
    return embeddings, masks


def _encode_negative_prompt(
    prompt: str,
    tokenizer: Any,
    text_encoder: torch.nn.Module,
    model_max_length: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        prompt,
        max_length=model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    return text_encoder(tokens.input_ids, tokens.attention_mask)[0]


def _validate_baseline(
    path: Path,
    weights: dict[str, Any],
    height: int,
    width: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 0 baseline report was not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("mode") != "t2i":
        raise ValueError("The supplied Phase 0 baseline report is not T2I")
    if payload.get("resolution") != [height, width]:
        raise ValueError(
            f"Baseline resolution {payload.get('resolution')} does not match "
            f"{[height, width]}"
        )
    baseline_checkpoint = payload.get("checkpoint", {}).get("sha256")
    if baseline_checkpoint and baseline_checkpoint != weights.get("sha256"):
        raise ValueError("Phase 0 and Phase 1 checkpoint hashes do not match")
    if payload.get("dtype") and payload["dtype"] != str(dtype):
        raise ValueError(
            f"Baseline dtype {payload['dtype']} does not match {dtype}"
        )
    return {
        "path": os.fspath(path),
        "sha256": sha256_file(path),
        "checkpoint_sha256": baseline_checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official T2I-1024 flow-DPM trajectories and Phase 1 "
            "semantic-substitution probes."
        )
    )
    parser.add_argument(
        "--config",
        default="t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml",
    )
    parser.add_argument("--checkpoint", default="pixeldit_t2i_v1.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument(
        "--output-dir",
        default="experiments/dualclock/traces/phase1_t2i1024",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.75)
    parser.add_argument("--flow-shift", type=float, default=4.0)
    parser.add_argument("--guidance-min", type=float, default=0.0)
    parser.add_argument("--guidance-max", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=(
        "low quality, worst quality, over-saturated, blurry, deformed, watermark"
    ))
    parser.add_argument("--patch-layers")
    parser.add_argument("--stale-horizons", default="1,2,3")
    parser.add_argument("--generic-pit-layer", type=int, default=-1)
    parser.add_argument(
        "--activation-storage", choices=("sketch", "full"), default="sketch"
    )
    parser.add_argument("--sketch-dim", type=int, default=1024)
    parser.add_argument(
        "--trace-dtype",
        choices=("same", "bfloat16", "float16", "float32"),
        default="same",
    )
    parser.add_argument("--sensitivity-steps", default="10,25,40,49")
    parser.add_argument("--token-groups", type=int, default=8)
    parser.add_argument("--channel-groups", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("The accepted released T2I path requires batch size 1")
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

    config = load_yaml(args.config)
    model, mode, _ = build_core(args.config)
    if mode != "t2i":
        raise ValueError("Expected a T2I config")
    weights = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    baseline = _validate_baseline(
        Path(args.baseline_report).resolve(),
        weights,
        args.height,
        args.width,
        dtype,
    )
    patch_layers = (
        parse_indices(args.patch_layers, len(model.patch_blocks))
        if args.patch_layers
        else default_patch_layers(len(model.patch_blocks))
    )
    final_layer = len(model.patch_blocks) - 1
    if final_layer not in patch_layers:
        patch_layers.append(final_layer)

    rows = rows_from_jsonl(args.manifest)[: args.limit]
    if not rows or any("prompt" not in row for row in rows):
        raise ValueError("T2I manifest must contain JSONL rows with prompts")

    from diffusion.model.builder import get_tokenizer_and_text_encoder

    text_config = config.get("text_encoder", {})
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=text_config.get("text_encoder_name", "gemma-2-2b-it"),
        device=str(device),
    )
    text_encoder.eval()
    null_embedding = _encode_negative_prompt(
        args.negative_prompt,
        tokenizer,
        text_encoder,
        int(text_config.get("model_max_length", 300)),
        device,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (Path(args.output_dir) / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = CollectionOptions(
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
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
    write_json(
        output / "run.json",
        {
            "phase": 1,
            "mode": "t2i",
            "track": "t2i1024_official",
            "created_utc": stamp,
            "git": git_metadata(),
            "config": os.path.abspath(args.config),
            "checkpoint": weights,
            "baseline_report": baseline,
            "regression_manifest": os.path.abspath(args.manifest),
            "resolution": [args.height, args.width],
            "sample_count": len(rows),
            "batch_size": 1,
            "precision": args.dtype,
            "trace_dtype": str(trace_dtype),
            "solver": "official flow DPM-Solver++ multistep order 2",
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "flow_shift": args.flow_shift,
            "guidance_interval": [
                args.guidance_min,
                args.guidance_max,
            ],
            "negative_prompt": args.negative_prompt,
            "patch_layers": patch_layers,
            "stale_horizons": list(options.stale_horizons),
            "activation_storage": args.activation_storage,
            "sketch_dim": args.sketch_dim,
            "sensitivity_steps": list(options.sensitivity_steps),
            "token_groups": args.token_groups,
            "channel_groups": args.channel_groups,
            "released_mask_behavior": (
                "text attention mask recorded but not forwarded, matching "
                "PixDiTTrainer.forward"
            ),
            "storage_warning": (
                "Exact T2I-1024 x, raw patch, fused semantic, and velocity "
                "states require approximately 4 GiB per prompt."
            ),
        },
    )

    first_embedding, first_mask = _encode_prompts(
        [rows[0]["prompt"]],
        tokenizer,
        text_encoder,
        text_config,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(int(rows[0]["seed"]))
    parity_x = torch.randn(
        1,
        model.in_channels,
        args.height,
        args.width,
        device=device,
        generator=generator,
    ).to(dtype=dtype)
    parity_t = torch.tensor([500.0], device=device, dtype=dtype)
    with torch.inference_mode():
        exact, reinjected, _ = extract_and_reinject(
            model, parity_x, parity_t, first_embedding.to(dtype=dtype)
        )
    parity = compare(exact, reinjected, atol=0.0, rtol=0.0)
    write_json(output / "semantic_parity.json", parity)
    if not parity["allclose"]:
        raise RuntimeError("T2I semantic parity failed")

    manifest_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = str(row["prompt"])
            condition, text_mask = _encode_prompts(
                [prompt],
                tokenizer,
                text_encoder,
                text_config,
                device,
            )
            seed = int(row["seed"])
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                1,
                model.in_channels,
                args.height,
                args.width,
                device=device,
                generator=generator,
            )
            trace = collect_t2i_batch(
                model,
                noise,
                condition,
                null_embedding,
                options,
                flow_shift=args.flow_shift,
                patch_layers=patch_layers,
                text_mask=text_mask,
            )
            trace["samples"] = [row]
            trace["prompts"] = [prompt]
            name = f"trajectory_{index:05d}_{index:05d}.pt"
            torch.save(trace, output / name)
            manifest_rows.append(
                {
                    "file": name,
                    "start": index,
                    "count": 1,
                    "indices": [int(row["index"])],
                    "seeds": [seed],
                    "prompt": prompt,
                }
            )
            write_json(
                output / "manifest.json",
                {"schema_version": 1, "mode": "t2i", "shards": manifest_rows},
            )
            print(f"{index + 1}/{len(rows)}", flush=True)
            del trace, condition, text_mask, noise
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"T2I-1024 Phase 1 traces: {output}")


if __name__ == "__main__":
    main()
