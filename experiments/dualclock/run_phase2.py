from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import (
    DTYPES,
    build_core,
    git_metadata,
    load_weights,
    summarize,
    write_json,
)
from .phase2 import (
    CausalErrorPredictor,
    RefreshPolicy,
    TerminalPiTCacheModel,
    paired_quality,
    sample_c2i_cached,
)
from .sample_c2i import rows_from_jsonl, seeded_noise


DEFAULT_VARIANTS = (
    "exact",
    "fixed_k2",
    "adaptive_q50",
    "adaptive_q75",
    "adaptive_q75_late",
    "timestep_only_q75",
)


def late_guard_steps(num_steps: int) -> frozenset[int]:
    """Map the Phase 1 hard regions (87-90 and 97-99 of 100) to an NFE."""

    fractions = (0.87, 0.88, 0.89, 0.90, 0.97, 0.98, 0.99)
    return frozenset(
        min(num_steps - 1, max(0, round(value * num_steps)))
        for value in fractions
    )


def make_policy(
    variant: str,
    predictor_path: str | None,
    num_steps: int,
) -> RefreshPolicy:
    if variant == "exact":
        return RefreshPolicy("exact")
    if variant.startswith("fixed_k"):
        interval = int(variant.removeprefix("fixed_k"))
        return RefreshPolicy("fixed", fixed_interval=interval)
    if predictor_path is None:
        raise ValueError(f"{variant} requires --predictor")
    model_name = (
        "timestep_only"
        if variant.startswith("timestep_only")
        else "full"
    )
    quantile = 0.50 if "q50" in variant else 0.75
    predictor = CausalErrorPredictor.from_file(
        predictor_path, model_name=model_name, quantile=quantile
    )
    guardrail = (
        late_guard_steps(num_steps)
        if variant.endswith("_late")
        else frozenset()
    )
    return RefreshPolicy(
        "adaptive",
        predictor=predictor,
        forced_refresh_steps=guardrail,
    )


def _save_npz(path: Path, samples: torch.Tensor) -> None:
    array = samples.permute(0, 2, 3, 1).contiguous().numpy()
    np.savez(path, arr_0=array)


def _lpips_report(
    exact: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import lpips  # type: ignore[import-not-found]
    except ImportError:
        return {
            "available": False,
            "reason": (
                "The optional `lpips` package is not installed. Pixel and "
                "frequency diagnostics were still computed."
            ),
        }
    metric = lpips.LPIPS(net="alex").eval().to(device)
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, exact.shape[0], batch_size):
            lhs = exact[start : start + batch_size].to(device)
            rhs = candidate[start : start + batch_size].to(device)
            values.append(metric(lhs, rhs).flatten().cpu())
    scores = torch.cat(values).float()
    return {
        "available": True,
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "p90": float(torch.quantile(scores, 0.9)),
        "max": float(scores.max()),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DualClock Phase 2 terminal-PiT rollout",
        "",
        f"Samples: {report['sample_count']}.",
        "",
        "| Variant | Median batch latency | Speedup | Skip rate | Policy overhead |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in report["variants"].items():
        lines.append(
            f"| `{name}` | {value['latency_ms']['median']:.3f} ms | "
            f"{value['speedup_vs_exact']:.3f}x | "
            f"{value['skip_rate']:.3f} | "
            f"{value['policy_overhead_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Paired metrics compare complete rolled-out samples against the "
            "exact wrapper using identical seeds and class labels. Distributional "
            "FID/sFID/IS/precision/recall must be run on the emitted `samples.npz` "
            "files with the repository's external ImageNet evaluation suite.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 synchronized performance runs require CUDA")
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    model, mode, _ = build_core(args.config)
    if mode != "c2i":
        raise ValueError("Phase 2 terminal-PiT rollout currently supports C2I")
    weights = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    wrapper = TerminalPiTCacheModel(model).eval()

    rows = rows_from_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("The regression manifest selected no samples")
    variants = tuple(
        value.strip()
        for value in args.variants.split(",")
        if value.strip()
    )
    unknown = set(variants) - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown Phase 2 variants: {sorted(unknown)}")
    if "exact" not in variants:
        variants = ("exact", *variants)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exact_samples: torch.Tensor | None = None
    variant_reports: dict[str, Any] = {}
    autocast_enabled = dtype in {torch.float16, torch.bfloat16}

    parity_rows = rows[: min(args.batch_size, len(rows))]
    parity_noise = seeded_noise(
        parity_rows,
        (model.in_channels, args.height, args.width),
        device,
    )
    parity_condition = torch.tensor(
        [row["class_id"] for row in parity_rows],
        device=device,
        dtype=torch.long,
    )
    parity_uncondition = torch.full_like(parity_condition, model.num_classes)
    parity_x = torch.cat((parity_noise, parity_noise), dim=0)
    parity_t = torch.zeros(
        parity_x.shape[0], device=device, dtype=parity_noise.dtype
    )
    parity_y = torch.cat((parity_uncondition, parity_condition), dim=0)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        released_output = model(parity_x, parity_t, parity_y)
        wrapper_output, parity_state = wrapper.forward_exact(
            parity_x, parity_t, parity_y
        )
        cached_output = wrapper.forward_cached(parity_state)
    parity_delta = (wrapper_output - released_output).detach().float()
    reinjection_delta = (cached_output - wrapper_output).detach().float()
    parity = {
        "allclose": bool(
            torch.allclose(
                wrapper_output, released_output, atol=1e-5, rtol=1e-5
            )
        ),
        "max_absolute_error": float(parity_delta.abs().max()),
        "cached_reinjection_allclose": bool(
            torch.equal(cached_output, wrapper_output)
        ),
        "cached_reinjection_max_absolute_error": float(
            reinjection_delta.abs().max()
        ),
    }
    if not parity["allclose"] or not parity["cached_reinjection_allclose"]:
        raise RuntimeError(f"Phase 2 exact-wrapper parity failed: {parity}")
    del (
        released_output,
        wrapper_output,
        cached_output,
        parity_state,
        parity_delta,
        reinjection_delta,
        parity_noise,
        parity_condition,
        parity_uncondition,
        parity_x,
        parity_t,
        parity_y,
    )
    torch.cuda.empty_cache()

    for variant in variants:
        policy = make_policy(variant, args.predictor, args.num_steps)
        sample_batches: list[torch.Tensor] = []
        latency_values: list[float] = []
        policy_values: list[float] = []
        refresh_count = 0
        cached_count = 0
        peak_memory = 0
        decision_rows: list[dict[str, Any]] = []

        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            noise = seeded_noise(
                batch_rows,
                (model.in_channels, args.height, args.width),
                device,
            )
            condition = torch.tensor(
                [row["class_id"] for row in batch_rows],
                device=device,
                dtype=torch.long,
            )
            uncondition = torch.full_like(condition, model.num_classes)
            if start == 0:
                for _ in range(args.warmup):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype,
                        enabled=autocast_enabled,
                    ):
                        sample_c2i_cached(
                            wrapper,
                            noise,
                            condition,
                            uncondition,
                            policy,
                            num_steps=args.num_steps,
                            cfg_scale=args.cfg_scale,
                            timeshift=args.timeshift,
                            guidance_min=args.guidance_min,
                            guidance_max=args.guidance_max,
                        )
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                result = sample_c2i_cached(
                    wrapper,
                    noise,
                    condition,
                    uncondition,
                    policy,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    timeshift=args.timeshift,
                    guidance_min=args.guidance_min,
                    guidance_max=args.guidance_max,
                )
            peak_memory = max(
                peak_memory, torch.cuda.max_memory_allocated(device)
            )
            sample_batches.append(result.sample.float().cpu())
            latency_values.append(result.elapsed_ms)
            policy_values.append(result.policy_ms)
            refresh_count += result.refresh_count
            cached_count += result.cached_count
            for row in result.decisions:
                row["batch_start"] = start
                decision_rows.append(row)
            print(
                f"{variant}: {min(start + len(batch_rows), len(rows))}/"
                f"{len(rows)}",
                flush=True,
            )

        samples = torch.cat(sample_batches)
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        _save_npz(variant_dir / "samples.npz", samples)
        with open(
            variant_dir / "decisions.jsonl", "w", encoding="utf-8"
        ) as handle:
            for row in decision_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if args.save_png:
            from torchvision.utils import save_image

            for row, sample in zip(rows, samples):
                name = (
                    f"{int(row['index']):05d}_"
                    f"class{int(row['class_id']):04d}_"
                    f"seed{int(row['seed'])}.png"
                )
                save_image(
                    sample,
                    variant_dir / name,
                    normalize=True,
                    value_range=(-1, 1),
                )

        total_steps = refresh_count + cached_count
        variant_reports[variant] = {
            "latency_ms": summarize(latency_values),
            "policy_ms": summarize(policy_values),
            "policy_overhead_percent": (
                100.0 * sum(policy_values) / sum(latency_values)
            ),
            "refresh_count": refresh_count,
            "cached_count": cached_count,
            "skip_rate": cached_count / total_steps,
            "peak_allocated_bytes": peak_memory,
            "sample_npz": str((variant_dir / "samples.npz").resolve()),
            "decision_log": str(
                (variant_dir / "decisions.jsonl").resolve()
            ),
        }
        if variant == "exact":
            exact_samples = samples
            variant_reports[variant]["paired_quality"] = paired_quality(
                samples, samples
            )
        else:
            assert exact_samples is not None
            variant_reports[variant]["paired_quality"] = paired_quality(
                exact_samples, samples
            )
            if args.lpips:
                variant_reports[variant]["lpips"] = _lpips_report(
                    exact_samples,
                    samples,
                    device,
                    args.batch_size,
                )

    exact_latency = variant_reports["exact"]["latency_ms"]["median"]
    for value in variant_reports.values():
        value["speedup_vs_exact"] = (
            exact_latency / value["latency_ms"]["median"]
        )
    candidate_gates = {
        name: {
            "speedup_at_least_1_4": value["speedup_vs_exact"] >= 1.4,
            "policy_overhead_below_5_percent": (
                value["policy_overhead_percent"] < 5.0
            ),
            "distributional_quality_complete": False,
        }
        for name, value in variant_reports.items()
        if name != "exact"
    }

    report = {
        "schema_version": 1,
        "phase": 2,
        "experiment": "terminal_pit_cache_rollout",
        "git": git_metadata(),
        "config": os.path.abspath(args.config),
        "checkpoint": weights,
        "manifest": os.path.abspath(args.manifest),
        "predictor": (
            os.path.abspath(args.predictor) if args.predictor else None
        ),
        "sample_count": len(rows),
        "batch_size": args.batch_size,
        "resolution": [args.height, args.width],
        "precision": args.dtype,
        "solver": "FlowDPMSolverSampler-compatible AB2",
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "timeshift": args.timeshift,
        "guidance_interval": [args.guidance_min, args.guidance_max],
        "exact_wrapper_parity": parity,
        "variants": variant_reports,
        "distributional_evaluation": {
            "status": "pending_external_adm_evaluation",
            "required": ["FID", "sFID", "IS", "precision", "recall"],
            "artifact": "Each variant directory contains samples.npz.",
        },
        "gate": {
            "status": "incomplete_pending_distributional_quality",
            "criteria": {
                "speedup": "at least 1.4x",
                "policy_overhead": "below 5 percent",
                "preliminary_fid_degradation": "no worse than approximately 0.15",
                "matched_latency": (
                    "adaptive quality better than fixed generic caching"
                ),
            },
            "candidates": candidate_gates,
        },
    }
    write_json(output / "phase2_report.json", report)
    with open(
        output / "phase2_report.md", "w", encoding="utf-8"
    ) as handle:
        handle.write(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact, fixed-k2, and causal adaptive terminal-PiT cache "
            "rollouts with synchronized end-to-end timing."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictor")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=2.75)
    parser.add_argument("--timeshift", type=float, default=1.0)
    parser.add_argument("--guidance-min", type=float, default=0.1)
    parser.add_argument("--guidance-max", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--variants", default=",".join(DEFAULT_VARIANTS)
    )
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        f"Phase 2 report: "
        f"{(Path(args.output_dir) / 'phase2_report.json').resolve()}"
    )
    for name, value in report["variants"].items():
        print(
            f"{name}: {value['speedup_vs_exact']:.3f}x, "
            f"skip={value['skip_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
