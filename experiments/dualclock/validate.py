from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


REQUIRED_COMPONENTS = {
    "patch_embedding_text_projection",
    "patch_attention",
    "patch_mlp",
    "patch_adaln",
    "pit_adaln",
    "pit_compaction_expansion",
    "pit_attention",
    "pit_mlp",
    "final_projection_reconstruction",
}


def read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def trial_median_cv_percent(latency: dict) -> float:
    recorded = latency.get("trial_median_cv_percent")
    if recorded is not None:
        return float(recorded)
    values = [float(value) for value in latency.get("trial_medians_ms", [])]
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean == 0:
        return float("inf")
    sample_variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return 100.0 * sample_variance**0.5 / mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 instrumentation exit criteria.")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--max-variation-percent", type=float, default=3.0)
    parser.add_argument("--min-component-coverage-percent", type=float, default=95.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    directory = Path(args.report_dir)
    checks = []
    for path in sorted(directory.glob("*_parity.json")):
        payload = read(path)
        checks.extend(
            [
                {
                    "name": f"{path.stem}: checkpoint coverage",
                    "pass": bool(payload["checkpoint"].get("loaded"))
                    and payload["checkpoint"].get("coverage", 0.0) >= 0.90,
                    "value": payload["checkpoint"].get("coverage"),
                },
                {
                    "name": f"{path.stem}: semantic reinjection",
                    "pass": payload["semantic_reinjection"]["allclose"],
                    "value": payload["semantic_reinjection"]["max_abs"],
                },
                {
                    "name": f"{path.stem}: repeatability",
                    "pass": payload["repeatability"]["allclose"],
                    "value": payload["repeatability"]["max_abs"],
                },
            ]
        )
    for path in sorted(directory.glob("*_eager.json")):
        payload = read(path)
        for result in payload["results"]:
            batch = result["batch_size"]
            latency = result["latency_ms"]
            components = set(result["components_ms"])
            variation_cv = trial_median_cv_percent(latency)
            checks.extend(
                [
                    {
                        "name": f"{path.stem}: batch {batch} variation (trial-median CV)",
                        "pass": variation_cv < args.max_variation_percent,
                        "value": variation_cv,
                        "threshold": args.max_variation_percent,
                        "range_percent_diagnostic": latency.get(
                            "trial_median_range_percent",
                            latency.get("trial_median_variation_percent"),
                        ),
                    },
                    {
                        "name": f"{path.stem}: batch {batch} component coverage",
                        "pass": latency["component_coverage_percent"]
                        >= args.min_component_coverage_percent,
                        "value": latency["component_coverage_percent"],
                        "threshold": args.min_component_coverage_percent,
                    },
                    {
                        "name": f"{path.stem}: batch {batch} required components",
                        "pass": REQUIRED_COMPONENTS.issubset(components),
                        "missing": sorted(REQUIRED_COMPONENTS - components),
                    },
                ]
            )
    if not checks:
        raise RuntimeError(f"No parity/eager reports found in {directory}")
    summary = {
        "pass": all(check["pass"] for check in checks),
        "criteria": {
            "max_variation_percent": args.max_variation_percent,
            "min_component_coverage_percent": args.min_component_coverage_percent,
        },
        "checks": checks,
        "note": "Quality reproduction/output hashes remain a separate sampling gate.",
    }
    output = Path(args.output) if args.output else directory / "instrumentation_gate.json"
    write_json(output, summary)
    for check in checks:
        print(f"{'PASS' if check['pass'] else 'FAIL'} {check['name']}: {check.get('value', check.get('missing'))}")
    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
