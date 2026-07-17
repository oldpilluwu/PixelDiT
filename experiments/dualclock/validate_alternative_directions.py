from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .analyze_alternative_directions import _auc
from .analyze_temporal_dynamics import EPS, _correlation, _summary
from .common import write_json


CAUSAL_FEATURES = (
    "timestep",
    "step_fraction",
    "x_high_to_low",
    "previous_velocity_change",
    "previous_velocity_curvature",
)


def _previous_velocity_features(
    velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = velocity.detach().float().reshape(
        velocity.shape[0], velocity.shape[1], -1
    )
    change = torch.full(
        flat.shape[:2], float("nan"), dtype=torch.float32
    )
    curvature = torch.full_like(change, float("nan"))
    change[1:] = torch.linalg.vector_norm(flat[1:] - flat[:-1], dim=-1) / (
        torch.linalg.vector_norm(flat[1:], dim=-1).clamp_min(EPS)
    )
    curvature[2:] = torch.linalg.vector_norm(
        flat[2:] - 2.0 * flat[1:-1] + flat[:-2], dim=-1
    ) / torch.linalg.vector_norm(flat[2:], dim=-1).clamp_min(EPS)
    previous_change = torch.cat(
        (torch.full_like(change[:1], float("nan")), change[:-1]), dim=0
    )
    previous_curvature = torch.cat(
        (torch.full_like(curvature[:1], float("nan")), curvature[:-1]), dim=0
    )
    return previous_change, previous_curvature


def load_trajectory_records(trace_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(trace_dir)
    with open(directory / "manifest.json", "r", encoding="utf-8") as handle:
        shards = json.load(handle).get("shards", [])
    if not shards:
        raise ValueError(f"No trajectory shards found in {directory}")

    records: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        trace = torch.load(
            directory / shard["file"],
            map_location="cpu",
            weights_only=False,
        )
        if str(trace.get("mode", "c2i")) != "c2i":
            raise ValueError("Phase 1.5 causal validation currently supports C2I")
        exact = trace["exact"]
        batch_size = int(trace["batch_size"])
        length = int(exact["timestep"].shape[0])
        previous_change, previous_curvature = _previous_velocity_features(
            exact["velocity_guided"]
        )
        low = exact["image_low_frequency_energy"].detach().float()
        high = exact["image_high_frequency_energy"].detach().float()
        feature_matrices = {
            "timestep": exact["timestep"].detach().float(),
            "step_fraction": torch.linspace(
                0.0, 1.0, length, dtype=torch.float32
            )
            .view(length, 1)
            .expand(length, batch_size),
            "x_high_to_low": high / low.clamp_min(EPS),
            "previous_velocity_change": previous_change,
            "previous_velocity_curvature": previous_curvature,
        }
        target_matrices = {
            key: value.detach().float()
            for key, value in trace["substitutions"].items()
            if key.endswith("/guided/relative_rmse")
            and (
                key.startswith("stale_generic/")
                or key.startswith("stale_raw_patch/")
                or key.startswith("linear_raw_patch/")
            )
        }
        conditions = trace.get("condition")
        samples = trace.get("samples", [])
        for sample in range(batch_size):
            if conditions is not None:
                content_group = f"class_{int(conditions[sample])}"
            elif sample < len(samples):
                content_group = f"class_{int(samples[sample].get('class_id', -1))}"
            else:
                content_group = "class_unknown"
            records.append(
                {
                    "trajectory_id": len(records),
                    "shard_index": shard_index,
                    "sample_index": sample,
                    "content_group": content_group,
                    "features": {
                        name: value[:, sample].clone()
                        for name, value in feature_matrices.items()
                    },
                    "targets": {
                        name: value[:, sample].clone()
                        for name, value in target_matrices.items()
                    },
                }
            )
        del trace
    return records


def _stack_rows(
    records: list[dict[str, Any]],
    indices: torch.Tensor,
    target_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in indices.tolist():
        record = records[index]
        target = record["targets"].get(target_key)
        if target is None:
            continue
        matrix = torch.stack(
            [record["features"][name] for name in CAUSAL_FEATURES], dim=-1
        )
        valid = torch.isfinite(target) & torch.isfinite(matrix).all(dim=-1)
        if valid.any():
            features.append(matrix[valid])
            targets.append(target[valid])
    if not features:
        return (
            torch.empty((0, len(CAUSAL_FEATURES)), dtype=torch.float64),
            torch.empty(0, dtype=torch.float64),
        )
    return torch.cat(features).double(), torch.cat(targets).double()


def _fit_ridge_error_model(
    features: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> dict[str, torch.Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0).clamp_min(1e-8)
    normalized = (features - mean) / scale
    design = torch.cat(
        (torch.ones((normalized.shape[0], 1), dtype=torch.float64), normalized),
        dim=1,
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
    }


def _predict_ridge(
    model: dict[str, torch.Tensor], features: torch.Tensor
) -> torch.Tensor:
    normalized = (features.double() - model["mean"]) / model["scale"]
    design = torch.cat(
        (torch.ones((normalized.shape[0], 1), dtype=torch.float64), normalized),
        dim=1,
    )
    return (design @ model["coefficients"]).float()


def _fold_assignments(count: int, folds: int, seed: int) -> torch.Tensor:
    folds = max(2, min(folds, count))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    assignments = torch.empty(count, dtype=torch.long)
    assignments[permutation] = torch.arange(count) % folds
    return assignments


def cross_validated_predictions(
    records: list[dict[str, Any]],
    target_key: str,
    harmful_threshold: float,
    folds: int,
    seed: int,
    ridge: float,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    if len(records) < 2:
        raise ValueError("Held-out validation requires at least two trajectories")
    assignments = _fold_assignments(len(records), folds, seed)
    prediction_rows = [
        torch.full_like(record["targets"][target_key], float("nan"))
        for record in records
    ]
    oriented_feature_rows = {
        name: [
            torch.full_like(record["targets"][target_key], float("nan"))
            for record in records
        ]
        for name in CAUSAL_FEATURES
    }

    for fold in range(int(assignments.max()) + 1):
        train_indices = torch.where(assignments != fold)[0]
        test_indices = torch.where(assignments == fold)[0]
        train_x, train_y = _stack_rows(records, train_indices, target_key)
        if train_x.shape[0] < len(CAUSAL_FEATURES) + 2:
            continue
        model = _fit_ridge_error_model(train_x, train_y, ridge)
        feature_directions = []
        for feature in range(len(CAUSAL_FEATURES)):
            correlation = _correlation(train_x[:, feature], train_y)
            feature_directions.append(
                -1.0 if correlation is not None and correlation < 0 else 1.0
            )

        for record_index in test_indices.tolist():
            record = records[record_index]
            target = record["targets"][target_key]
            matrix = torch.stack(
                [record["features"][name] for name in CAUSAL_FEATURES], dim=-1
            )
            valid = torch.isfinite(target) & torch.isfinite(matrix).all(dim=-1)
            if not valid.any():
                continue
            prediction_rows[record_index][valid] = _predict_ridge(
                model, matrix[valid]
            )
            for feature_index, name in enumerate(CAUSAL_FEATURES):
                oriented_feature_rows[name][record_index][valid] = (
                    matrix[valid, feature_index]
                    * feature_directions[feature_index]
                )

    target_values: list[torch.Tensor] = []
    predicted_values: list[torch.Tensor] = []
    trajectory_ids: list[torch.Tensor] = []
    for index, (record, predictions) in enumerate(
        zip(records, prediction_rows)
    ):
        target = record["targets"][target_key]
        valid = torch.isfinite(target) & torch.isfinite(predictions)
        if valid.any():
            target_values.append(target[valid])
            predicted_values.append(predictions[valid])
            trajectory_ids.append(
                torch.full((int(valid.sum()),), index, dtype=torch.long)
            )
    target = torch.cat(target_values)
    predicted = torch.cat(predicted_values)
    labels = target > harmful_threshold
    auc = _auc(predicted, labels)
    feature_report: dict[str, Any] = {}
    for name in CAUSAL_FEATURES:
        feature_scores = torch.cat(
            [
                values[torch.isfinite(values)]
                for values in oriented_feature_rows[name]
                if torch.isfinite(values).any()
            ]
        )
        feature_targets = torch.cat(
            [
                record["targets"][target_key][torch.isfinite(values)]
                for record, values in zip(records, oriented_feature_rows[name])
                if torch.isfinite(values).any()
            ]
        )
        feature_report[name] = {
            "held_out_auroc": _auc(
                feature_scores, feature_targets > harmful_threshold
            ),
            "pearson": _correlation(feature_scores, feature_targets),
        }
    return (
        {
            "target": target_key,
            "folds": int(assignments.max()) + 1,
            "trajectory_count": len(set(torch.cat(trajectory_ids).tolist())),
            "observation_count": int(target.numel()),
            "harmful_threshold": harmful_threshold,
            "harmful_rate": float(labels.float().mean()),
            "held_out_auroc": auc,
            "held_out_rmse": float(
                torch.mean((predicted - target).square()).sqrt()
            ),
            "prediction_summary": _summary(predicted),
            "target_summary": _summary(target),
            "features": feature_report,
            "feature_contract": (
                "All features are available before the current exact semantic "
                "or raw-patch state is evaluated."
            ),
        },
        prediction_rows,
    )


def _bootstrap_mean_interval(
    values: torch.Tensor,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    values = values.detach().double().flatten()
    generator = torch.Generator().manual_seed(seed)
    means: list[torch.Tensor] = []
    remaining = samples
    while remaining:
        count = min(remaining, 1000)
        indices = torch.randint(
            values.numel(),
            (count, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= count
    distribution = torch.cat(means)
    tail = (1.0 - confidence) / 2.0
    return (
        float(torch.quantile(distribution, tail)),
        float(torch.quantile(distribution, 1.0 - tail)),
    )


def paired_bootstrap_comparisons(
    records: list[dict[str, Any]],
    horizons: tuple[int, ...],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for horizon in horizons:
        generic_key = f"stale_generic/h{horizon}/guided/relative_rmse"
        raw_key = f"stale_raw_patch/h{horizon}/guided/relative_rmse"
        differences: list[torch.Tensor] = []
        generic_values: list[torch.Tensor] = []
        raw_values: list[torch.Tensor] = []
        for record in records:
            generic = record["targets"].get(generic_key)
            raw = record["targets"].get(raw_key)
            if generic is None or raw is None:
                continue
            valid = torch.isfinite(generic) & torch.isfinite(raw)
            if valid.any():
                generic_values.append(generic[valid].mean().view(1))
                raw_values.append(raw[valid].mean().view(1))
                differences.append((raw[valid] - generic[valid]).mean().view(1))
        if not differences:
            continue
        delta = torch.cat(differences)
        lower, upper = _bootstrap_mean_interval(
            delta, samples, confidence, seed + horizon
        )
        generic = torch.cat(generic_values)
        raw = torch.cat(raw_values)
        report[f"h{horizon}"] = {
            "trajectory_count": int(delta.numel()),
            "generic_mean_rmse": float(generic.mean()),
            "raw_patch_mean_rmse": float(raw.mean()),
            "paired_mean_raw_minus_generic": float(delta.mean()),
            "paired_median_raw_minus_generic": float(delta.median()),
            "confidence": confidence,
            "bootstrap_ci": [lower, upper],
            "generic_better_trajectory_fraction": float(
                (delta > 0).float().mean()
            ),
            "generic_advantage_supported": lower > 0.0,
        }
    return report


def _policy_metrics(
    refresh_count: int,
    total_count: int,
    cached_errors: list[float],
    harmful_threshold: float,
) -> dict[str, Any]:
    errors = torch.tensor(cached_errors, dtype=torch.float32)
    refresh_rate = refresh_count / total_count
    return {
        "total_steps": total_count,
        "refresh_count": refresh_count,
        "refresh_rate": refresh_rate,
        "skip_rate": 1.0 - refresh_rate,
        "effective_refresh_interval": total_count / refresh_count,
        "cached_step_count": len(cached_errors),
        "cached_error": _summary(errors),
        "cached_harmful_rate": (
            float((errors > harmful_threshold).float().mean())
            if errors.numel()
            else None
        ),
        "note": (
            "Same-state oracle simulation only; this is not a rolled-out "
            "sampler quality or measured speed result."
        ),
    }


def _simulate_policy(
    records: list[dict[str, Any]],
    scores: dict[int, list[torch.Tensor]] | None,
    thresholds: dict[int, float] | None,
    harmful_threshold: float,
    fixed_interval: int | None = None,
) -> dict[str, Any]:
    refresh_count = 0
    total_count = 0
    cached_errors: list[float] = []
    max_horizon = 3
    for record_index, record in enumerate(records):
        first_target = record["targets"].get(
            "stale_generic/h1/guided/relative_rmse"
        )
        if first_target is None:
            continue
        last_refresh = 0
        refresh_count += 1
        total_count += 1
        for step in range(1, first_target.numel()):
            total_count += 1
            age = step - last_refresh
            should_cache = age <= max_horizon
            if fixed_interval is not None:
                should_cache = should_cache and age < fixed_interval
            else:
                if (
                    not should_cache
                    or scores is None
                    or thresholds is None
                    or age not in scores
                ):
                    should_cache = False
                else:
                    score = scores[age][record_index][step]
                    should_cache = bool(
                        torch.isfinite(score) and score <= thresholds[age]
                    )
            target = record["targets"].get(
                f"stale_generic/h{age}/guided/relative_rmse"
            )
            if target is None or not torch.isfinite(target[step]):
                should_cache = False
            if should_cache:
                cached_errors.append(float(target[step]))
            else:
                refresh_count += 1
                last_refresh = step
    return _policy_metrics(
        refresh_count, total_count, cached_errors, harmful_threshold
    )


def adaptive_policy_curves(
    records: list[dict[str, Any]],
    predictions: dict[int, list[torch.Tensor]],
    harmful_threshold: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"fixed_intervals": {}, "adaptive": []}
    for interval in (2, 3, 4):
        report["fixed_intervals"][f"interval_{interval}"] = _simulate_policy(
            records,
            None,
            None,
            harmful_threshold,
            fixed_interval=interval,
        )
    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
        thresholds: dict[int, float] = {}
        for horizon, rows in predictions.items():
            finite = torch.cat(
                [row[torch.isfinite(row)] for row in rows if torch.isfinite(row).any()]
            )
            thresholds[horizon] = float(torch.quantile(finite, quantile))
        metrics = _simulate_policy(
            records,
            predictions,
            thresholds,
            harmful_threshold,
        )
        metrics["risk_acceptance_quantile"] = quantile
        metrics["score_thresholds"] = {
            f"h{horizon}": value for horizon, value in thresholds.items()
        }
        report["adaptive"].append(metrics)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DualClock Phase 1.5 causal validation",
        "",
        f"Trajectories: {report['trajectory_count']}.",
        f"Recommendation: **{report['recommendation'].upper()}**.",
        "",
        "## Held-out causal prediction",
        "",
    ]
    for key, value in report["held_out_predictors"].items():
        lines.append(
            f"- `{key}`: AUROC={value['held_out_auroc']}, "
            f"harmful rate={value['harmful_rate']:.4f}."
        )
    lines.extend(["", "## Paired generic-vs-raw bootstrap", ""])
    for horizon, value in report["paired_bootstrap"].items():
        lines.append(
            f"- `{horizon}`: raw-generic={value['paired_mean_raw_minus_generic']:.6g}, "
            f"CI={value['bootstrap_ci']}, "
            f"supported={value['generic_advantage_supported']}."
        )
    lines.extend(["", "## Adaptive policies", ""])
    for value in report["policy_curves"]["adaptive"]:
        lines.append(
            f"- acceptance q={value['risk_acceptance_quantile']:.2f}: "
            f"skip={value['skip_rate']:.3f}, "
            f"cached p90={value['cached_error']['p90']}, "
            f"harmful={value['cached_harmful_rate']}."
        )
    lines.extend(
        [
            "",
            "All policy results are same-state oracle simulations, not rolled-out "
            "sample quality or measured acceleration.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_phase1_5(
    trace_dir: str | Path,
    harmful_threshold: float = 0.05,
    folds: int = 5,
    ridge: float = 1e-3,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260718,
) -> dict[str, Any]:
    records = load_trajectory_records(trace_dir)
    predictor_report: dict[str, Any] = {}
    predictions: dict[int, list[torch.Tensor]] = {}
    for horizon in (1, 2, 3):
        target_key = f"stale_generic/h{horizon}/guided/relative_rmse"
        if not all(target_key in record["targets"] for record in records):
            continue
        predictor, rows = cross_validated_predictions(
            records,
            target_key,
            harmful_threshold,
            folds,
            seed + horizon,
            ridge,
        )
        predictor_report[target_key] = predictor
        predictions[horizon] = rows

    bootstrap = paired_bootstrap_comparisons(
        records,
        tuple(predictions),
        bootstrap_samples,
        confidence,
        seed,
    )
    policies = adaptive_policy_curves(
        records, predictions, harmful_threshold
    )
    h1_key = "stale_generic/h1/guided/relative_rmse"
    causal_pass = (
        predictor_report.get(h1_key, {}).get("held_out_auroc") is not None
        and predictor_report[h1_key]["held_out_auroc"] >= 0.75
    )
    bootstrap_pass = bootstrap.get("h1", {}).get(
        "generic_advantage_supported", False
    )
    safe_policies = [
        item
        for item in policies["adaptive"]
        if item["skip_rate"] >= 0.50
        and item["cached_error"]["p90"] is not None
        and item["cached_error"]["p90"] <= harmful_threshold
    ]
    policy_pass = bool(safe_policies)
    if causal_pass and bootstrap_pass and policy_pass:
        recommendation = "run_small_pit_layer_sweep"
    else:
        recommendation = "stop_or_refine_predictor"
    return {
        "phase": 1.5,
        "analysis": "causal_alternative_validation",
        "trace_directory": str(Path(trace_dir).resolve()),
        "trajectory_count": len(records),
        "causal_features": list(CAUSAL_FEATURES),
        "excluded_noncausal_features": [
            "current semantic_change",
            "current raw_patch_change",
            "current substitution error",
        ],
        "harmful_threshold": harmful_threshold,
        "held_out_predictors": predictor_report,
        "paired_bootstrap": bootstrap,
        "policy_curves": policies,
        "gate": {
            "held_out_h1_auroc_at_least_0_75": causal_pass,
            "paired_generic_advantage_ci_above_zero": bootstrap_pass,
            "adaptive_skip_at_least_0_50_with_cached_p90_below_threshold": policy_pass,
        },
        "best_safe_policy": (
            max(safe_policies, key=lambda item: item["skip_rate"])
            if safe_policies
            else None
        ),
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run trajectory-held-out causal prediction, paired bootstrap, "
            "and adaptive refresh-policy validation on C2I Phase 1 traces."
        )
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--harmful-threshold", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    report = validate_phase1_5(
        args.trace_dir,
        harmful_threshold=args.harmful_threshold,
        folds=args.folds,
        ridge=args.ridge,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.trace_dir) / "phase1_5_analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "phase1_5_report.json", report)
    with open(output / "phase1_5_report.md", "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    print(f"Phase 1.5 report: {output.resolve()}")
    print(f"Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
