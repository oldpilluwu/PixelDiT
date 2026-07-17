from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .common import write_json
from .phase1 import deterministic_signed_pool


EPS = 1e-12
DEFAULT_ANALYSIS_DIM = 4096


def _finite(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().flatten()
    return value[torch.isfinite(value)]


def _summary(value: torch.Tensor | list[float]) -> dict[str, float | int | None]:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {"count": 0, "median": None, "mean": None, "p10": None, "p90": None}
    return {
        "count": int(tensor.numel()),
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
    }


def _flatten_streams(
    values: torch.Tensor, max_features: int = DEFAULT_ANALYSIS_DIM
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError(f"Expected [time, stream, ...], got {tuple(values.shape)}")
    flattened = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    if max_features > 0 and flattened.shape[-1] > max_features:
        original_shape = flattened.shape
        flattened = deterministic_signed_pool(
            flattened.reshape(-1, original_shape[-1]), max_features
        ).reshape(original_shape[0], original_shape[1], max_features)
    return flattened


def temporal_measurements(
    values: torch.Tensor,
    max_features: int = DEFAULT_ANALYSIS_DIM,
    exact_norms: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    z = _flatten_streams(values, max_features=max_features)
    norms = (
        exact_norms.detach().float()
        if exact_norms is not None
        else torch.linalg.vector_norm(z, dim=-1)
    ).clamp_min(EPS)
    first = torch.linalg.vector_norm(z[1:] - z[:-1], dim=-1) / norms[1:]
    second = (
        torch.linalg.vector_norm(z[2:] - 2.0 * z[1:-1] + z[:-2], dim=-1)
        / norms[2:]
    )
    result: dict[str, torch.Tensor] = {
        "normalized_first_difference": first,
        "normalized_curvature": second,
    }
    for lag in range(1, min(5, z.shape[0] - 1) + 1):
        result[f"cosine_lag_{lag}"] = F.cosine_similarity(
            z[lag:], z[:-lag], dim=-1, eps=EPS
        )

    ranks: list[float] = []
    ranks_95: list[float] = []
    high_frequency: list[float] = []
    spectral_centroids: list[float] = []
    for stream in range(z.shape[1]):
        matrix = z[:, stream]
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        gram = centered @ centered.T
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
        total = eigenvalues.sum()
        if total <= EPS:
            ranks.append(0.0)
            ranks_95.append(0.0)
        else:
            ranks.append(float(total.square() / eigenvalues.square().sum().clamp_min(EPS)))
            cumulative = eigenvalues.cumsum(0) / total
            ranks_95.append(float((cumulative < 0.95).sum() + 1))
        spectrum = torch.fft.rfft(matrix, dim=0)
        energy = spectrum.abs().square().sum(dim=-1)
        non_dc = energy[1:]
        if non_dc.numel() == 0 or non_dc.sum() <= EPS:
            high_frequency.append(0.0)
            spectral_centroids.append(0.0)
        else:
            cutoff = max(1, math.ceil(non_dc.numel() / 2))
            high_frequency.append(float(non_dc[cutoff:].sum() / non_dc.sum()))
            frequencies = torch.linspace(
                0.0, 1.0, non_dc.numel(), dtype=non_dc.dtype
            )
            spectral_centroids.append(
                float((frequencies * non_dc).sum() / non_dc.sum())
            )
    result["effective_temporal_rank"] = torch.tensor(ranks)
    result["temporal_rank_95"] = torch.tensor(ranks_95)
    result["high_frequency_energy_fraction"] = torch.tensor(high_frequency)
    result["low_frequency_energy_fraction"] = 1.0 - result["high_frequency_energy_fraction"]
    result["normalized_spectral_centroid"] = torch.tensor(spectral_centroids)
    return result


def temporal_report(
    values: torch.Tensor, max_features: int = DEFAULT_ANALYSIS_DIM
) -> dict[str, Any]:
    measurements = temporal_measurements(values, max_features=max_features)
    length = values.shape[0]
    report: dict[str, Any] = {
        "overall": {name: _summary(value) for name, value in measurements.items()},
        "regions": {},
    }
    bounds = {
        "early": (0, max(3, math.ceil(length / 3))),
        "middle": (length // 3, max(length // 3 + 3, math.ceil(2 * length / 3))),
        "late": (2 * length // 3, length),
    }
    for name, (start, end) in bounds.items():
        region = values[start:min(end, length)]
        if region.shape[0] >= 3:
            region_measurements = temporal_measurements(region, max_features=max_features)
            report["regions"][name] = {
                metric: _summary(value) for metric, value in region_measurements.items()
            }
    return report


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if x.numel() < 2 or x.std() <= EPS or y.std() <= EPS:
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def load_baseline_cost(path: str | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "reason": "No Phase 0 baseline report supplied"}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    batch_one = next((item for item in results if item.get("batch_size") == 1), results[0] if results else None)
    if not batch_one:
        return {"available": False, "reason": "Baseline report contains no benchmark results"}
    components = batch_one.get("components_ms", {})
    if components:
        medians = {
            name: float(value["median"])
            for name, value in components.items()
            if isinstance(value, dict) and value.get("median") is not None
        }
        patch_names = {
            "conditioning",
            "patch_embedding_text_projection",
            "patch_attention",
            "patch_mlp",
            "patch_adaln",
            "patch_residual_tensor_ops",
        }
        patch = sum(value for name, value in medians.items() if name in patch_names)
        total = float(batch_one.get("component_envelopes_ms", {}).get("instrumented_model_envelope_ms", {}).get("median", 0))
        if total <= 0:
            total = sum(medians.values())
        patch = min(patch, total)
        return {
            "available": total > 0 and patch > 0,
            "basis": "instrumented component median latency",
            "total": total,
            "semantic_path": patch,
            "semantic_fraction": patch / total if total else None,
        }
    flops = batch_one.get("estimated_flops", {})
    patch_names = {
        "conditioning",
        "patch_embedding_text_projection",
        "patch_attention",
        "patch_mlp",
        "patch_adaln",
    }
    patch = sum(float(value.get("flops", 0)) for name, value in flops.items() if name in patch_names)
    total = sum(float(value.get("flops", 0)) for value in flops.values())
    return {
        "available": total > 0 and patch > 0,
        "basis": "analytical FLOPs",
        "total": total,
        "semantic_path": patch,
        "semantic_fraction": patch / total if total else None,
    }


def predicted_speedup(cost: dict[str, Any], refresh_interval: int) -> float | None:
    if not cost.get("available") or refresh_interval < 1:
        return None
    fraction = float(cost["semantic_fraction"])
    return 1.0 / ((1.0 - fraction) + fraction / refresh_interval)


def _append_temporal(
    aggregate: dict[str, dict[str, list[torch.Tensor]]],
    name: str,
    values: torch.Tensor,
    analysis_dim: int,
    exact_norms: torch.Tensor | None = None,
) -> None:
    measurements = temporal_measurements(
        values, max_features=analysis_dim, exact_norms=exact_norms
    )
    for metric, value in measurements.items():
        aggregate[name][metric].append(_finite(value))
    length = values.shape[0]
    bounds = {
        "early": (0, max(3, math.ceil(length / 3))),
        "middle": (length // 3, max(length // 3 + 3, math.ceil(2 * length / 3))),
        "late": (2 * length // 3, length),
    }
    for region, (start, end) in bounds.items():
        subset = values[start:min(end, length)]
        if subset.shape[0] < 3:
            continue
        subset_norms = (
            exact_norms[start:min(end, length)] if exact_norms is not None else None
        )
        for metric, value in temporal_measurements(
            subset, max_features=analysis_dim, exact_norms=subset_norms
        ).items():
            aggregate[name][f"regions/{region}/{metric}"].append(_finite(value))


def _median_from_summary(report: dict[str, Any], metric: str) -> float:
    value = report["overall"][metric]["median"]
    return float(value) if value is not None else float("inf")


def _render_markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    status = gate["status"].replace("_", " ").upper()
    lines = [
        "# DualClock Phase 1 analysis",
        "",
        f"Overall gate: **{status}**",
        "",
        (
            f"Traces: {report['trace_count']} shards / "
            f"{report['sample_count']} trajectories / "
            f"{report['minimum_recorded_steps']} minimum recorded steps."
        ),
        f"Activation storage: `{report['activation_storage']}`.",
        "",
        "## Pass conditions",
        "",
        "| Condition | Result |",
        "| --- | --- |",
    ]
    for name, value in gate["conditions"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if value else 'fail'} |")
    if gate["status"] == "insufficient_evidence":
        requirements = gate["evidence_requirements"]
        lines.extend(
            [
                "",
                (
                    "This run is an instrumentation smoke test, not a valid hypothesis "
                    f"gate. The gate requires at least {requirements['min_trajectories']} "
                    f"trajectories and {requirements['min_steps']} solver evaluations."
                ),
            ]
        )
    lines.extend(["", "## Candidate semantic states", ""])
    for name, candidate in gate["candidates"].items():
        speed = candidate.get("predicted_speedup")
        speed_text = "n/a" if speed is None else f"{speed:.3f}x"
        lines.append(
            f"- `{name}`: curvature={candidate['curvature']:.6g}, "
            f"quality interval={candidate.get('quality_refresh_interval')}, speed={speed_text}, "
            f"{'meets conditions' if candidate['meets_conditions'] else 'does not meet conditions'}."
        )
    lines.extend(
        [
            "",
            "All acceleration values are cost-model predictions, not measured Phase 2 speedups.",
            "Sketch-mode activation dynamics are approximate; final semantics and velocities remain exact.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    trace_dir: str | Path,
    baseline_report: str | None,
    max_forecast_relative_rmse: float = 0.25,
    min_forecast_cosine: float = 0.95,
    substantial_factor: float = 0.75,
    min_speedup: float = 1.5,
    analysis_dim: int = DEFAULT_ANALYSIS_DIM,
    min_trajectories: int = 100,
    min_steps: int = 100,
) -> dict[str, Any]:
    directory = Path(trace_dir)
    manifest_path = directory / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError(f"No trace shards found in {manifest_path}")

    temporal: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    substitutions: dict[str, list[torch.Tensor]] = defaultdict(list)
    sensitivity: dict[str, list[torch.Tensor]] = defaultdict(list)
    class_curvature: dict[int, list[torch.Tensor]] = defaultdict(list)
    content_curvature: dict[str, list[torch.Tensor]] = defaultdict(list)
    content_records: list[tuple[float, torch.Tensor]] = []
    branch_curvature: dict[str, list[torch.Tensor]] = defaultdict(list)
    head_variation: dict[str, list[torch.Tensor]] = defaultdict(list)
    activation_modes: set[str] = set()
    recorded_step_counts: list[int] = []
    sample_count = 0

    for shard_info in shards:
        trace = torch.load(directory / shard_info["file"], map_location="cpu", weights_only=False)
        batch_size = int(trace["batch_size"])
        sample_count += batch_size
        activation_modes.add(trace["activation_storage"])
        exact = trace["exact"]
        recorded_step_counts.append(int(exact["timestep"].shape[0]))
        representations: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
            "final_semantic": (exact["semantic"], None),
            "velocity/branches": (exact["velocity_branches"], None),
            "velocity/guided": (exact["velocity_guided"], None),
            "pixel_state": (exact["x_t"], None),
        }
        representations.update(
            {
                name: (value["values"], value.get("norms"))
                for name, value in trace["representations"].items()
            }
        )
        for name, (values, norms) in representations.items():
            if norms is None:
                norms = torch.linalg.vector_norm(
                    values.detach().float().reshape(values.shape[0], values.shape[1], -1),
                    dim=-1,
                )
            _append_temporal(temporal, name, values, analysis_dim, norms)
            if "/head_" in name and values.ndim >= 4:
                for head in range(values.shape[2]):
                    measurement = temporal_measurements(
                        values[:, :, head], max_features=analysis_dim
                    )
                    head_variation[name].append(
                        _finite(measurement["normalized_curvature"])
                    )

        semantic_norms = torch.linalg.vector_norm(
            exact["semantic"].detach().float().reshape(
                exact["semantic"].shape[0], exact["semantic"].shape[1], -1
            ),
            dim=-1,
        )
        semantic_measurement = temporal_measurements(
            exact["semantic"],
            max_features=analysis_dim,
            exact_norms=semantic_norms,
        )
        semantic_curvature = semantic_measurement["normalized_curvature"]
        labels = trace["condition"].tolist()
        high_ratio = exact["image_high_frequency_energy"].mean(dim=0) / (
            exact["image_low_frequency_energy"].mean(dim=0) + EPS
        )
        for sample_index, class_id in enumerate(labels):
            # CFG stream order is [all unconditional, all conditional].
            sample_values = torch.cat(
                (
                    semantic_curvature[:, sample_index],
                    semantic_curvature[:, batch_size + sample_index],
                )
            )
            class_curvature[int(class_id)].append(sample_values)
            content_records.append((float(high_ratio[sample_index]), sample_values))
        branch_curvature["unconditional"].append(semantic_curvature[:, :batch_size])
        branch_curvature["conditional"].append(semantic_curvature[:, batch_size:])

        for name, value in trace["substitutions"].items():
            substitutions[name].append(_finite(value))
        for row in trace.get("sensitivity", []):
            key = f"{row['kind']}/{row['start']}:{row['end']}"
            sensitivity[f"{key}/semantic_relative_l2"].append(_finite(row["semantic_relative_l2"]))
            sensitivity[f"{key}/velocity_relative_rmse"].append(_finite(row["velocity_relative_rmse"]))
            sensitivity[f"{key}/velocity_cosine_error"].append(_finite(row["velocity_cosine_error"]))
            sensitivity[f"{key}/lipschitz"].append(_finite(row["lipschitz"]))

    if content_records:
        content_threshold = float(
            torch.tensor([item[0] for item in content_records]).median()
        )
        for ratio, values in content_records:
            content = (
                "high_frequency_content"
                if ratio > content_threshold
                else "low_frequency_content"
            )
            content_curvature[content].append(values)
    else:
        content_threshold = None

    representation_report: dict[str, Any] = {}
    for name, metrics in temporal.items():
        overall: dict[str, Any] = {}
        regions: dict[str, dict[str, Any]] = defaultdict(dict)
        for metric, values in metrics.items():
            result = _summary(torch.cat(values) if values else torch.tensor([]))
            if metric.startswith("regions/"):
                _, region, metric_name = metric.split("/", 2)
                regions[region][metric_name] = result
            else:
                overall[metric] = result
        representation_report[name] = {"overall": overall, "regions": dict(regions)}
    substitution_report = {
        name: _summary(torch.cat(values) if values else torch.tensor([]))
        for name, values in substitutions.items()
    }
    sensitivity_report = {
        name: _summary(torch.cat(values) if values else torch.tensor([]))
        for name, values in sensitivity.items()
    }

    correlations: dict[str, float | None] = {}
    for method in ("stale_semantic", "linear_semantic"):
        horizons = sorted(
            {
                int(name.split("/")[1][1:])
                for name in substitutions
                if name.startswith(method + "/h")
            }
        )
        for horizon in horizons:
            sem_key = f"{method}/h{horizon}/semantic/relative_l2"
            vel_key = f"{method}/h{horizon}/branches/relative_rmse"
            if sem_key in substitutions and vel_key in substitutions:
                correlations[f"{method}/h{horizon}"] = _correlation(
                    torch.cat(substitutions[sem_key]),
                    torch.cat(substitutions[vel_key]),
                )

    cost = load_baseline_cost(baseline_report)
    minimum_recorded_steps = min(recorded_step_counts, default=0)
    evidence_sufficient = (
        sample_count >= min_trajectories
        and minimum_recorded_steps >= min_steps
    )
    pit_curvatures = [
        _median_from_summary(value, "normalized_curvature")
        for name, value in representation_report.items()
        if name.startswith("pit/") and name.endswith("/output")
    ]
    generic_curvature = min(pit_curvatures, default=float("inf"))
    velocity_curvature = _median_from_summary(
        representation_report["velocity/guided"], "normalized_curvature"
    )
    semantic_names = [
        name
        for name in representation_report
        if name == "final_semantic" or re.fullmatch(r"patch/block_\d+", name)
    ]
    candidates: dict[str, Any] = {}
    for name in semantic_names:
        curvature = _median_from_summary(
            representation_report[name], "normalized_curvature"
        )
        lower_curvature = curvature < generic_curvature and curvature < velocity_curvature
        # Only the final semantic representation has a direct decoder contract
        # in the released model. Earlier layers are reported as candidates for a
        # future split point, but cannot borrow final-state substitution evidence.
        has_decoder_contract = name == "final_semantic"
        useful_predictability = False
        better_than_generic = False
        quality_interval = 1
        method_at_interval: str | None = None
        for horizon in sorted({int(item.split("/")[1][1:]) for item in substitutions if "/h" in item}):
            if not has_decoder_contract:
                break
            for method in ("linear_semantic", "stale_semantic"):
                rmse_key = f"{method}/h{horizon}/guided/relative_rmse"
                cosine_key = f"{method}/h{horizon}/guided/cosine_error"
                generic_key = f"stale_generic/h{horizon}/guided/relative_rmse"
                if rmse_key not in substitution_report:
                    continue
                rmse = substitution_report[rmse_key]["median"]
                cosine_error = substitution_report[cosine_key]["median"]
                generic_rmse = substitution_report.get(generic_key, {}).get("median")
                if rmse is None or cosine_error is None:
                    continue
                acceptable = rmse <= max_forecast_relative_rmse and cosine_error <= 1.0 - min_forecast_cosine
                if horizon >= 2 and acceptable:
                    useful_predictability = True
                if generic_rmse is not None and rmse <= substantial_factor * generic_rmse:
                    better_than_generic = True
                if acceptable and horizon + 1 > quality_interval:
                    quality_interval = horizon + 1
                    method_at_interval = method
        speedup = predicted_speedup(cost, quality_interval)
        speed_condition = speedup is not None and speedup >= min_speedup
        candidate_meets_conditions = (
            lower_curvature
            and useful_predictability
            and better_than_generic
            and speed_condition
        )
        candidates[name] = {
            "curvature": curvature,
            "has_direct_decoder_contract": has_decoder_contract,
            "lower_than_pit_and_velocity": lower_curvature,
            "useful_for_two_intervals": useful_predictability,
            "better_than_stale_generic": better_than_generic,
            "quality_refresh_interval": quality_interval,
            "quality_method": method_at_interval,
            "predicted_speedup": speedup,
            "speed_condition": speed_condition,
            "meets_conditions": candidate_meets_conditions,
            "pass": candidate_meets_conditions and evidence_sufficient,
        }
    hypothesis_pass = any(
        candidate["meets_conditions"] for candidate in candidates.values()
    )
    overall_pass = evidence_sufficient and hypothesis_pass
    status = (
        "insufficient_evidence"
        if not evidence_sufficient
        else ("pass" if overall_pass else "do_not_proceed")
    )
    conditions = {
        "sufficient_evidence": evidence_sufficient,
        "lower_temporal_curvature": any(value["lower_than_pit_and_velocity"] for value in candidates.values()),
        "two_interval_predictability": any(value["useful_for_two_intervals"] for value in candidates.values()),
        "less_error_than_stale_generic": any(value["better_than_stale_generic"] for value in candidates.values()),
        "plausible_1_5x_speedup": any(value["speed_condition"] for value in candidates.values()),
    }
    report = {
        "phase": 1,
        "trace_directory": str(directory.resolve()),
        "trace_count": len(shards),
        "sample_count": sample_count,
        "minimum_recorded_steps": minimum_recorded_steps,
        "activation_storage": sorted(activation_modes),
        "analysis_projection": {
            "max_features": analysis_dim,
            "method": "deterministic signed pooled projection",
            "note": "Full final semantics remain archived; high-dimensional temporal metrics use this bounded projection.",
        },
        "thresholds": {
            "max_forecast_relative_rmse": max_forecast_relative_rmse,
            "min_forecast_cosine": min_forecast_cosine,
            "substantial_error_factor": substantial_factor,
            "min_predicted_speedup": min_speedup,
        },
        "representations": representation_report,
        "per_head_curvature": {
            name: _summary(torch.cat(values) if values else torch.tensor([]))
            for name, values in head_variation.items()
        },
        "semantic_curvature_by_cfg_branch": {
            name: _summary(torch.cat(values) if values else torch.tensor([]))
            for name, values in branch_curvature.items()
        },
        "semantic_curvature_by_class": {
            str(name): _summary(torch.cat(values) if values else torch.tensor([]))
            for name, values in class_curvature.items()
        },
        "semantic_curvature_by_image_content": {
            name: _summary(torch.cat(values) if values else torch.tensor([]))
            for name, values in content_curvature.items()
        },
        "image_content_split_high_low_ratio": content_threshold,
        "substitutions": substitution_report,
        "semantic_velocity_error_correlation": correlations,
        "decoder_sensitivity": sensitivity_report,
        "cost_model": cost,
        "gate": {
            "status": status,
            "pass": overall_pass,
            "hypothesis_conditions_met": hypothesis_pass,
            "conditions": conditions,
            "candidates": candidates,
            "evidence_requirements": {
                "min_trajectories": min_trajectories,
                "min_steps": min_steps,
                "observed_trajectories": sample_count,
                "observed_minimum_steps": minimum_recorded_steps,
            },
            "interpretation": (
                "This run is too small or too short to evaluate the Phase 1 gate."
                if not evidence_sufficient
                else (
                    "Proceed to Phase 2."
                    if overall_pass
                    else "Phase 1 evidence does not satisfy the plan's continuation gate."
                )
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DualClock Phase 1 trajectory traces.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument(
        "--baseline-report",
        help="Phase 0 eager benchmark JSON; defaults to the path recorded in trace run.json.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-forecast-relative-rmse", type=float, default=0.25)
    parser.add_argument("--min-forecast-cosine", type=float, default=0.95)
    parser.add_argument("--substantial-factor", type=float, default=0.75)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--analysis-dim", type=int, default=DEFAULT_ANALYSIS_DIM)
    parser.add_argument("--min-trajectories", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=100)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    baseline_report = args.baseline_report
    if baseline_report is None:
        run_path = trace_dir / "run.json"
        if run_path.exists():
            with open(run_path, "r", encoding="utf-8") as handle:
                recorded = json.load(handle).get("baseline_report")
                baseline_report = (
                    recorded.get("path") if isinstance(recorded, dict) else recorded
                )
    report = analyze(
        trace_dir,
        baseline_report,
        args.max_forecast_relative_rmse,
        args.min_forecast_cosine,
        args.substantial_factor,
        args.min_speedup,
        args.analysis_dim,
        args.min_trajectories,
        args.min_steps,
    )
    output = Path(args.output_dir) if args.output_dir else trace_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "phase1_report.json", report)
    write_json(output / "phase1_gate.json", report["gate"])
    with open(output / "phase1_report.md", "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    print(
        f"Phase 1 {report['gate']['status'].replace('_', ' ').upper()}: "
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()
