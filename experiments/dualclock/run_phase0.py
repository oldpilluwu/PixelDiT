from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, resolve_checkpoint, write_json


CASES = {
    "c2i256": {
        "config": "c2i/configs/pix256_xl.yaml",
        "checkpoint": "imagenet256_pixeldit_xl_epoch320.ckpt",
        "height": 256,
        "width": 256,
        "sampling": {
            "solver": "FlowDPMSolverSampler",
            "nfe": 100,
            "cfg_scale": 2.75,
            "cfg_interval": [0.1, 0.9],
            "timeshift": 1.0,
        },
    },
    "c2i512": {
        "config": "c2i/configs/pix512_xl.yaml",
        "checkpoint": "imagenet512_pixeldit_xl.ckpt",
        "height": 512,
        "width": 512,
        "sampling": {
            "solver": "FlowDPMSolverSampler",
            "nfe": 100,
            "cfg_scale": 3.5,
            "cfg_interval": [0.1, 1.0],
            "timeshift": 2.0,
        },
    },
    "t2i512": {
        "config": "t2i/configs/PixelDiT_512px_pixel_diffusion_stage1.yaml",
        "checkpoint": "pixeldit_t2i_v1.pth",
        "height": 512,
        "width": 512,
        "sampling": {
            "solver": "flow_dpm-solver",
            "nfe": 50,
            "cfg_scale": 3.5,
            "cfg_interval": [0.0, 1.0],
            "flow_shift": 3.0,
        },
    },
    "t2i1024": {
        "config": "t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml",
        "checkpoint": "pixeldit_t2i_v1.pth",
        "height": 1024,
        "width": 1024,
        "sampling": {
            "solver": "flow_dpm-solver",
            "nfe": 50,
            "cfg_scale": 2.75,
            "cfg_interval": [0.0, 1.0],
            "flow_shift": 4.0,
        },
    },
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 parity and profiling on selected baselines.")
    parser.add_argument("--cases", default="c2i256", help=f"Comma-separated: {','.join(CASES)} or all")
    parser.add_argument("--output-dir", default="experiments/dualclock/reports")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-sizes", default="1,auto")
    parser.add_argument("--max-auto-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--component-repeats", type=int, default=10)
    parser.add_argument("--compiled", action="store_true", help="Also record total-only torch.compile baselines.")
    parser.add_argument("--compile-mode", default="default")
    args = parser.parse_args()

    requested = list(CASES) if args.cases == "all" else [item.strip() for item in args.cases.split(",")]
    unknown = sorted(set(requested) - set(CASES))
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (ROOT / args.output_dir / stamp).resolve()
    report_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        report_dir / "baseline_settings.json",
        {
            name: {
                **CASES[name]["sampling"],
                "resolution": [CASES[name]["height"], CASES[name]["width"]],
                "precision": args.dtype,
                "benchmark_batch_sizes": args.batch_sizes,
                "compile_requested": args.compiled,
                "compile_mode": args.compile_mode if args.compiled else None,
            }
            for name in requested
        },
    )

    resolved: dict[str, str] = {}
    for name in requested:
        resolved[name] = resolve_checkpoint(CASES[name]["checkpoint"]) or ""
    unique_checkpoints = list(dict.fromkeys(resolved.values()))
    environment_command = [
        sys.executable,
        "-m",
        "experiments.dualclock.capture_environment",
        "--output",
        os.fspath(report_dir / "environment.json"),
    ]
    for checkpoint in unique_checkpoints:
        environment_command.extend(["--checkpoint", checkpoint])
    run(environment_command)
    run(
        [
            sys.executable,
            "-m",
            "experiments.dualclock.make_regression_set",
            "--output-dir",
            os.fspath(report_dir / "regression"),
        ]
    )

    for name in requested:
        case = CASES[name]
        common = [
            "--config",
            os.fspath(ROOT / case["config"]),
            "--checkpoint",
            resolved[name],
            "--height",
            str(case["height"]),
            "--width",
            str(case["width"]),
            "--dtype",
            args.dtype,
        ]
        parity_command = [
            sys.executable,
            "-m",
            "experiments.dualclock.parity",
            *common,
            "--output",
            os.fspath(report_dir / f"{name}_parity.json"),
        ]
        if name.startswith("t2i"):
            parity_command.append("--check-mask")
        run(parity_command)

        benchmark_command = [
            sys.executable,
            "-m",
            "experiments.dualclock.benchmark",
            *common,
            "--batch-sizes",
            args.batch_sizes,
            "--max-auto-batch",
            str(args.max_auto_batch),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--trials",
            str(args.trials),
            "--component-repeats",
            str(args.component_repeats),
            "--output",
            os.fspath(report_dir / f"{name}_eager.json"),
        ]
        run(benchmark_command)
        if args.compiled:
            run(
                [
                    *benchmark_command[:-2],
                    "--compile",
                    "--compile-mode",
                    args.compile_mode,
                    "--total-only",
                    "--output",
                    os.fspath(report_dir / f"{name}_compiled.json"),
                ]
            )
    try:
        run(
            [
                sys.executable,
                "-m",
                "experiments.dualclock.validate",
                "--report-dir",
                os.fspath(report_dir),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Phase 0 instrumentation gate failed (exit {exc.returncode}). "
            f"Reports were preserved at {report_dir}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
    print(f"Phase 0 reports: {report_dir}")


if __name__ == "__main__":
    main()
