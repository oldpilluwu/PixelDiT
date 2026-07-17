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

from .analyze_temporal_dynamics import (
    DEFAULT_ANALYSIS_DIM,
    EPS,
    _correlation,
    _finite,
    _summary,
    temporal_measurements,
)
from .common import write_json
from .phase1 import deterministic_signed_pool, image_frequency_energy


def _regions(length: int) -> dict[str, slice]:
    return {
        "early": slice(0, max(1, math.ceil(length / 3))),
        "middle": slice(length // 3, max(length // 3 + 1, math.ceil(2 * length / 3))),
        "late": slice(2 * length // 3, length),
    }


def _project(values: torch.Tensor, width: int) -> torch.Tensor:
    flat = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    if flat.shape[-1] <= width:
        return flat
    return deterministic_signed_pool(
        flat.reshape(-1, flat.shape[-1]), width
    ).reshape(flat.shape[0], flat.shape[1], width)


def _normalized_first(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    result = torch.full(
        (flat.shape[0], flat.shape[1]), float("nan"), dtype=torch.float32
    )
    result[1:] = torch.linalg.vector_norm(flat[1:] - flat[:-1], dim=-1) / (
        torch.linalg.vector_norm(flat[1:], dim=-1).clamp_min(EPS)
    )
    return result


def _normalized_curvature(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().float().reshape(values.shape[0], values.shape[1], -1)
    result = torch.full(
        (flat.shape[0], flat.shape[1]), float("nan"), dtype=torch.float32
    )
    result[2:] = torch.linalg.vector_norm(
        flat[2:] - 2.0 * flat[1:-1] + flat[:-2], dim=-1
    ) / torch.linalg.vector_norm(flat[2:], dim=-1).clamp_min(EPS)
    return result


def _residual_rank(values: torch.Tensor) -> dict[str, torch.Tensor]:
    projected = values.detach().float().reshape(
        values.shape[0], values.shape[1], -1
    )
    residual = projected[1:] - projected[:-1]
    effective: list[float] = []
    rank_90: list[float] = []
    rank_95: list[float] = []
    rank_99: list[float] = []
    for stream in range(residual.shape[1]):
        matrix = residual[:, stream]
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        eigenvalues = torch.linalg.eigvalsh(matrix @ matrix.T).clamp_min(0).flip(0)
        total = eigenvalues.sum()
        if total <= EPS:
            effective.append(0.0)
            rank_90.append(0.0)
            rank_95.append(0.0)
            rank_99.append(0.0)
            continue
        effective.append(
            float(total.square() / eigenvalues.square().sum().clamp_min(EPS))
        )
        cumulative = eigenvalues.cumsum(0) / total
        rank_90.append(float((cumulative < 0.90).sum() + 1))
        rank_95.append(float((cumulative < 0.95).sum() + 1))
        rank_99.append(float((cumulative < 0.99).sum() + 1))
    return {
        "effective_rank": torch.tensor(effective),
        "rank_90": torch.tensor(rank_90),
        "rank_95": torch.tensor(rank_95),
        "rank_99": torch.tensor(rank_99),
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.detach().float().flatten()
    labels = labels.detach().bool().flatten()
    valid = torch.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(order, dtype=torch.float32)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; tied values receive their average rank.
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[labels].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def _feature_target_report(
    features: dict[str, torch.Tensor],
    target: torch.Tensor,
    harmful_threshold: float,
) -> dict[str, Any]:
    labels = target > harmful_threshold
    report: dict[str, Any] = {
        "target": _summary(target),
        "harmful_threshold": harmful_threshold,
        "harmful_rate": float(labels.float().mean()) if labels.numel() else None,
        "features": {},
    }
    for name, value in features.items():
        valid = torch.isfinite(value) & torch.isfinite(target)
        if not valid.any():
            continue
        auc = _auc(value[valid], labels[valid])
        if auc is None:
            best_auc = None
            direction = None
        elif auc >= 0.5:
            best_auc = auc
            direction = "higher_is_riskier"
        else:
            best_auc = 1.0 - auc
            direction = "lower_is_riskier"
        report["features"][name] = {
            "pearson": _correlation(value[valid], target[valid]),
            "best_univariate_auroc": best_auc,
            "direction": direction,
        }
    return report


def _top_entries(
    totals: torch.Tensor | None,
    count: int,
    limit: int,
    spatial: bool = False,
) -> list[dict[str, Any]]:
    if totals is None or count <= 0:
        return []
    scores = totals / count
    top = torch.topk(scores, min(limit, scores.numel()))
    side = int(round(math.sqrt(scores.numel())))
    result: list[dict[str, Any]] = []
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        row: dict[str, Any] = {"index": int(index), "score": float(value)}
        if spatial and side * side == scores.numel():
            row["row"] = int(index // side)
            row["column"] = int(index % side)
        result.append(row)
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Alternative directions from DualClock trajectories",
        "",
        f"Mode: `{report['mode']}`; trajectories: {report['sample_count']}.",
        "",
        "## Ranked directions",
        "",
    ]
    for item in report["directions"]:
        lines.append(
            f"- **{item['name']}** — {item['status']}: {item['reason']}"
        )
    lines.extend(["", "## Most reusable layers", ""])
    for item in report["layer_rankings"][:10]:
        lines.append(
            f"- `{item['name']}`: curvature={item['curvature']:.6g}, "
            f"lag-1 cosine={item['cosine_lag_1']:.6g}."
        )
    lines.extend(["", "## Raw-patch residual rank", ""])
    rank = report["raw_patch_residual_rank"]
    lines.append(
        f"Median rank-95: `{rank['rank_95']['median']}`; "
        f"effective rank: `{rank['effective_rank']['median']}`."
    )
    lines.extend(["", "## CFG redundancy", ""])
    cfg = report["cfg_redundancy"]
    lines.append(
        f"Conditional-delta/shared norm ratio: "
        f"`{cfg['raw_patch_delta_to_shared_norm']['median']}`."
    )
    lines.extend(["", "## Adaptive error prediction", ""])
    for target, payload in report["error_predictors"].items():
        best = sorted(
            (
                (name, value.get("best_univariate_auroc"))
                for name, value in payload.get("features", {}).items()
                if value.get("best_univariate_auroc") is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if best:
            lines.append(f"- `{target}`: best feature `{best[0][0]}`, AUROC={best[0][1]:.4f}.")
    lines.append("")
    return "\n".join(lines)


def analyze_alternatives(
    trace_dir: str | Path,
    analysis_dim: int = DEFAULT_ANALYSIS_DIM,
    harmful_threshold: float = 0.05,
    top_k: int = 16,
    include_velocity_frequency: bool = True,
) -> dict[str, Any]:
    directory = Path(trace_dir)
    with open(directory / "manifest.json", "r", encoding="utf-8") as handle:
        shards = json.load(handle).get("shards", [])
    if not shards:
        raise ValueError(f"No trajectory shards found in {directory}")

    modes: set[str] = set()
    sample_count = 0
    layer_metrics: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_errors: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    residual_ranks: dict[str, list[torch.Tensor]] = defaultdict(list)
    cfg_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
    token_totals: torch.Tensor | None = None
    channel_totals: torch.Tensor | None = None
    volatility_count = 0
    sensitivity: dict[str, list[torch.Tensor]] = defaultdict(list)
    velocity_hard_counts: torch.Tensor | None = None
    velocity_curve_values: list[torch.Tensor] = []
    frequency: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    predictor_features: dict[str, list[torch.Tensor]] = defaultdict(list)
    predictor_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    group_errors: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for shard_info in shards:
        trace = torch.load(
            directory / shard_info["file"],
            map_location="cpu",
            weights_only=False,
        )
        mode = str(trace.get("mode", "c2i"))
        modes.add(mode)
        batch_size = int(trace["batch_size"])
        sample_count += batch_size
        exact = trace["exact"]
        length = int(exact["timestep"].shape[0])
        width = analysis_dim
        if trace["activation_storage"] == "sketch":
            width = min(width, int(trace.get("sketch_dim", width)))

        layer_pattern = re.compile(
            r"(?:patch|text)/block_\d+$|pit/block_\d+/(?:input|output)$"
        )
        for name, packed in trace["representations"].items():
            if not layer_pattern.fullmatch(name):
                continue
            values = packed["values"]
            measurements = temporal_measurements(
                values,
                max_features=width,
                exact_norms=packed.get("norms"),
            )
            for metric in (
                "normalized_first_difference",
                "normalized_curvature",
                "cosine_lag_1",
                "effective_temporal_rank",
            ):
                layer_metrics[name][metric].append(_finite(measurements[metric]))

        final_layer = max(int(index) for index in trace["patch_layers"])
        raw_packed = trace["representations"].get(
            f"patch/block_{final_layer}"
        )
        if raw_packed is not None:
            raw_projected = raw_packed["values"].detach().float()
        else:
            raw_projected = _project(exact["raw_final_patch"], width)
        semantic_projected = _project(exact["semantic"], width)
        velocity_projected = _project(exact["velocity_guided"], width)

        for name, value in _residual_rank(raw_projected).items():
            residual_ranks[name].append(_finite(value))

        raw_first = _normalized_first(raw_projected)
        semantic_first = _normalized_first(semantic_projected)
        velocity_first = _normalized_first(velocity_projected)
        velocity_curvature = _normalized_curvature(velocity_projected)
        velocity_curve_values.append(_finite(velocity_curvature))
        valid_curve = torch.isfinite(velocity_curvature)
        if valid_curve.any():
            trajectory_thresholds = []
            for sample in range(batch_size):
                finite_curve = velocity_curvature[:, sample][
                    valid_curve[:, sample]
                ]
                trajectory_thresholds.append(
                    torch.quantile(finite_curve, 0.9)
                    if finite_curve.numel()
                    else torch.tensor(float("inf"))
                )
            thresholds = torch.stack(trajectory_thresholds).view(1, batch_size)
            hard = (
                valid_curve & (velocity_curvature >= thresholds)
            ).sum(dim=1).to(torch.float64)
            if velocity_hard_counts is None:
                velocity_hard_counts = torch.zeros(length, dtype=torch.float64)
            velocity_hard_counts[:length] += hard

        raw_exact = exact.get("raw_final_patch")
        if raw_exact is not None:
            for step in range(1, raw_exact.shape[0]):
                current = raw_exact[step].float()
                previous = raw_exact[step - 1].float()
                delta = current - previous
                token_score = torch.linalg.vector_norm(delta, dim=-1) / (
                    torch.linalg.vector_norm(current, dim=-1).clamp_min(EPS)
                )
                channel_score = delta.square().mean(dim=1).sqrt() / (
                    current.square().mean(dim=1).sqrt().clamp_min(EPS)
                )
                token_value = token_score.sum(dim=0).double()
                channel_value = channel_score.sum(dim=0).double()
                token_totals = (
                    token_value if token_totals is None else token_totals + token_value
                )
                channel_totals = (
                    channel_value
                    if channel_totals is None
                    else channel_totals + channel_value
                )
                volatility_count += current.shape[0]

        for sample in range(batch_size):
            uncond = raw_projected[:, sample]
            cond = raw_projected[:, batch_size + sample]
            shared = ((uncond + cond) / 2.0).unsqueeze(1)
            delta = (cond - uncond).unsqueeze(1)
            cfg_metrics["raw_patch_delta_to_shared_norm"].append(
                (
                    torch.linalg.vector_norm(delta.squeeze(1), dim=-1)
                    / torch.linalg.vector_norm(shared.squeeze(1), dim=-1).clamp_min(EPS)
                )
            )
            cfg_metrics["shared_curvature"].append(
                _finite(temporal_measurements(shared)["normalized_curvature"])
            )
            cfg_metrics["delta_curvature"].append(
                _finite(temporal_measurements(delta)["normalized_curvature"])
            )
            cfg_metrics["delta_rank_95"].append(
                _residual_rank(delta)["rank_95"]
            )

        substitutions = trace["substitutions"]
        for key, values in substitutions.items():
            if not key.endswith("/guided/relative_rmse"):
                continue
            for region, region_slice in _regions(length).items():
                region_errors[key][region].append(_finite(values[region_slice]))

        for row in trace.get("sensitivity", []):
            group = f"{row['kind']}/{row['start']}:{row['end']}"
            sensitivity[f"{group}/velocity_relative_rmse"].append(
                _finite(row["velocity_relative_rmse"])
            )
            sensitivity[f"{group}/lipschitz"].append(
                _finite(row["lipschitz"])
            )

        x_low = exact["image_low_frequency_energy"].float()
        x_high = exact["image_high_frequency_energy"].float()
        x_ratio = x_high / x_low.clamp_min(EPS)
        for region, region_slice in _regions(length).items():
            frequency["x_high_to_low"][region].append(_finite(x_ratio[region_slice]))
        if include_velocity_frequency:
            velocity_low: list[torch.Tensor] = []
            velocity_high: list[torch.Tensor] = []
            for step in range(length):
                bands = image_frequency_energy(exact["velocity_guided"][step])
                velocity_low.append(bands["low"])
                velocity_high.append(bands["high"])
            v_low = torch.stack(velocity_low)
            v_high = torch.stack(velocity_high)
            v_ratio = v_high / v_low.clamp_min(EPS)
            for region, region_slice in _regions(length).items():
                frequency["velocity_high_to_low"][region].append(
                    _finite(v_ratio[region_slice])
                )

        features = {
            "timestep": exact["timestep"].float(),
            "x_high_to_low": x_ratio,
            "raw_patch_change": torch.nanmean(
                raw_first.reshape(length, 2, batch_size), dim=1
            ),
            "semantic_change": torch.nanmean(
                semantic_first.reshape(length, 2, batch_size), dim=1
            ),
            "previous_velocity_change": torch.cat(
                (
                    torch.full((1, batch_size), float("nan")),
                    velocity_first[:-1],
                ),
                dim=0,
            ),
            "previous_velocity_curvature": torch.cat(
                (
                    torch.full((1, batch_size), float("nan")),
                    velocity_curvature[:-1],
                ),
                dim=0,
            ),
        }
        target_methods = (
            "stale_generic",
            "stale_raw_patch",
            "linear_raw_patch",
        )
        for method in target_methods:
            target_key = f"{method}/h1/guided/relative_rmse"
            if target_key not in substitutions:
                continue
            target = substitutions[target_key].float()
            predictor_targets[target_key].append(target.flatten())
            for feature_name, feature in features.items():
                predictor_features[f"{target_key}|{feature_name}"].append(
                    feature.flatten()
                )

        if mode == "c2i":
            group_names = [
                f"class_{int(trace['condition'][index])}"
                for index in range(batch_size)
            ]
        else:
            prompts = trace.get("prompts") or [
                item.get("prompt", "") for item in trace.get("samples", [])
            ]
            group_names = [
                (
                    "prompt_short"
                    if len(str(prompt).split()) <= 20
                    else (
                        "prompt_medium"
                        if len(str(prompt).split()) <= 50
                        else "prompt_long"
                    )
                )
                for prompt in prompts
            ]
        for target_key in (
            "stale_generic/h1/guided/relative_rmse",
            "stale_raw_patch/h1/guided/relative_rmse",
        ):
            if target_key not in substitutions:
                continue
            per_sample = torch.nanmean(substitutions[target_key].float(), dim=0)
            for index, group in enumerate(group_names):
                if index < per_sample.numel():
                    group_errors[target_key][group].append(
                        per_sample[index].view(1)
                    )

    if len(modes) != 1:
        raise ValueError(f"Mixed trace modes are unsupported: {sorted(modes)}")
    mode = next(iter(modes))

    layer_report: dict[str, Any] = {}
    rankings: list[dict[str, Any]] = []
    for name, metrics in layer_metrics.items():
        layer_report[name] = {
            metric: _summary(torch.cat(values) if values else torch.tensor([]))
            for metric, values in metrics.items()
        }
        rankings.append(
            {
                "name": name,
                "curvature": layer_report[name]["normalized_curvature"]["median"],
                "first_difference": layer_report[name][
                    "normalized_first_difference"
                ]["median"],
                "cosine_lag_1": layer_report[name]["cosine_lag_1"]["median"],
                "effective_rank": layer_report[name][
                    "effective_temporal_rank"
                ]["median"],
            }
        )
    rankings.sort(
        key=lambda item: (
            float("inf") if item["curvature"] is None else item["curvature"]
        )
    )

    region_report = {
        key: {
            region: _summary(torch.cat(values) if values else torch.tensor([]))
            for region, values in regions.items()
        }
        for key, regions in region_errors.items()
    }
    residual_report = {
        name: _summary(torch.cat(values) if values else torch.tensor([]))
        for name, values in residual_ranks.items()
    }
    cfg_report = {
        name: _summary(torch.cat(values) if values else torch.tensor([]))
        for name, values in cfg_metrics.items()
    }
    sensitivity_report = {
        name: _summary(torch.cat(values) if values else torch.tensor([]))
        for name, values in sensitivity.items()
    }
    frequency_report = {
        signal: {
            region: _summary(torch.cat(values) if values else torch.tensor([]))
            for region, values in regions.items()
        }
        for signal, regions in frequency.items()
    }

    predictor_report: dict[str, Any] = {}
    for target_key, target_values in predictor_targets.items():
        target = torch.cat(target_values)
        features = {
            combined.split("|", 1)[1]: torch.cat(values)
            for combined, values in predictor_features.items()
            if combined.startswith(target_key + "|")
        }
        predictor_report[target_key] = _feature_target_report(
            features, target, harmful_threshold
        )

    hard_schedule: list[dict[str, Any]] = []
    if velocity_hard_counts is not None:
        for step in torch.argsort(velocity_hard_counts, descending=True)[:top_k]:
            hard_schedule.append(
                {
                    "step": int(step),
                    "hard_trajectory_count": int(velocity_hard_counts[step]),
                    "fraction": float(velocity_hard_counts[step] / sample_count),
                }
            )

    grouped_report = {
        target: {
            group: _summary(torch.cat(values) if values else torch.tensor([]))
            for group, values in groups.items()
        }
        for target, groups in group_errors.items()
    }

    generic_key = "stale_generic/h1/guided/relative_rmse"
    raw_key = "stale_raw_patch/h1/guided/relative_rmse"
    generic_overall = (
        torch.cat(
            [
                value
                for region_values in region_errors.get(generic_key, {}).values()
                for value in region_values
            ]
        )
        if region_errors.get(generic_key)
        else torch.tensor([])
    )
    raw_overall = (
        torch.cat(
            [
                value
                for region_values in region_errors.get(raw_key, {}).values()
                for value in region_values
            ]
        )
        if region_errors.get(raw_key)
        else torch.tensor([])
    )
    generic_median = _summary(generic_overall)["median"]
    raw_median = _summary(raw_overall)["median"]
    rank95 = residual_report.get("rank_95", {}).get("median")
    delta_ratio = cfg_report.get(
        "raw_patch_delta_to_shared_norm", {}
    ).get("median")
    best_auroc = max(
        (
            feature.get("best_univariate_auroc") or 0.0
            for target in predictor_report.values()
            for feature in target.get("features", {}).values()
        ),
        default=0.0,
    )
    early_frequency = frequency_report.get("x_high_to_low", {}).get(
        "early", {}
    ).get("median")
    late_frequency = frequency_report.get("x_high_to_low", {}).get(
        "late", {}
    ).get("median")

    directions = [
        {
            "name": "generic PiT/layer caching",
            "status": (
                "supported"
                if generic_median is not None
                and raw_median is not None
                and generic_median < raw_median
                else "not established"
            ),
            "reason": (
                f"stale generic h1 median RMSE={generic_median}; "
                f"raw-patch RMSE={raw_median}"
            ),
        },
        {
            "name": "low-rank semantic residual transport",
            "status": (
                "supported" if rank95 is not None and rank95 <= 10 else "weak"
            ),
            "reason": f"raw-patch residual median rank-95={rank95}",
        },
        {
            "name": "shared CFG base with branch correction",
            "status": (
                "supported"
                if delta_ratio is not None and delta_ratio <= 0.25
                else "weak"
            ),
            "reason": f"conditional delta/shared norm ratio={delta_ratio}",
        },
        {
            "name": "adaptive refresh or solver schedule",
            "status": "supported" if best_auroc >= 0.75 else "weak",
            "reason": f"best cheap univariate harmful-step AUROC={best_auroc:.4f}",
        },
        {
            "name": "progressive resolution/frequency sampling",
            "status": (
                "supported"
                if early_frequency is not None
                and late_frequency is not None
                and late_frequency > 1.25 * early_frequency
                else "not established"
            ),
            "reason": (
                f"x high/low frequency ratio early={early_frequency}, "
                f"late={late_frequency}"
            ),
        },
    ]

    return {
        "phase": 1,
        "analysis": "alternative_directions",
        "mode": mode,
        "trace_directory": str(directory.resolve()),
        "trace_count": len(shards),
        "sample_count": sample_count,
        "analysis_dim": analysis_dim,
        "harmful_threshold": harmful_threshold,
        "layer_dynamics": layer_report,
        "layer_rankings": rankings,
        "region_cache_errors": region_report,
        "raw_patch_residual_rank": residual_report,
        "cfg_redundancy": cfg_report,
        "raw_patch_volatility": {
            "top_tokens": _top_entries(
                token_totals, volatility_count, top_k, spatial=True
            ),
            "top_channels": _top_entries(
                channel_totals, volatility_count, top_k
            ),
            "observation_count": volatility_count,
        },
        "decoder_sensitivity_rankings": sensitivity_report,
        "velocity_dynamics": {
            "curvature": _summary(
                torch.cat(velocity_curve_values)
                if velocity_curve_values
                else torch.tensor([])
            ),
            "most_frequently_hard_steps": hard_schedule,
        },
        "frequency_evolution": frequency_report,
        "error_predictors": predictor_report,
        "error_by_content_group": grouped_report,
        "directions": directions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine existing DualClock trajectories for alternative caching, "
            "solver, CFG, low-rank, and progressive-frequency directions."
        )
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--analysis-dim", type=int, default=DEFAULT_ANALYSIS_DIM)
    parser.add_argument("--harmful-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--skip-velocity-frequency", action="store_true")
    args = parser.parse_args()

    report = analyze_alternatives(
        args.trace_dir,
        analysis_dim=args.analysis_dim,
        harmful_threshold=args.harmful_threshold,
        top_k=args.top_k,
        include_velocity_frequency=not args.skip_velocity_frequency,
    )
    output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.trace_dir) / "alternative_analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "alternative_directions.json", report)
    with open(
        output / "alternative_directions.md", "w", encoding="utf-8"
    ) as handle:
        handle.write(_render_markdown(report))
    print(f"Alternative-direction report: {output.resolve()}")


if __name__ == "__main__":
    main()
