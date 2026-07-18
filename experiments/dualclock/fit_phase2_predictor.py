from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from .common import git_metadata, write_json
from .phase2 import CAUSAL_FEATURES
from .validate_alternative_directions import (
    _fold_assignments,
    _fit_ridge_error_model,
    _predict_ridge,
    load_trajectory_records,
)


def _stack_feature_rows(
    records: list[dict[str, Any]],
    target_key: str,
    feature_names: Sequence[str],
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    selected = (
        range(len(records)) if indices is None else indices.tolist()
    )
    for index in selected:
        record = records[index]
        target = record["targets"].get(target_key)
        if target is None:
            continue
        matrix = torch.stack(
            [record["features"][name] for name in feature_names], dim=-1
        )
        valid = torch.isfinite(target) & torch.isfinite(matrix).all(dim=-1)
        if valid.any():
            feature_rows.append(matrix[valid])
            target_rows.append(target[valid])
    if not feature_rows:
        raise ValueError(f"No finite rows found for {target_key}")
    return torch.cat(feature_rows).double(), torch.cat(target_rows).double()


def _cross_validated_scores(
    records: list[dict[str, Any]],
    target_key: str,
    feature_names: Sequence[str],
    folds: int,
    seed: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = _fold_assignments(len(records), folds, seed)
    score_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for fold in range(int(assignments.max()) + 1):
        train_indices = torch.where(assignments != fold)[0]
        test_indices = torch.where(assignments == fold)[0]
        train_x, train_y = _stack_feature_rows(
            records, target_key, feature_names, train_indices
        )
        test_x, test_y = _stack_feature_rows(
            records, target_key, feature_names, test_indices
        )
        fitted = _fit_ridge_error_model(train_x, train_y, ridge)
        score_rows.append(_predict_ridge(fitted, test_x))
        target_rows.append(test_y.float())
    return torch.cat(score_rows), torch.cat(target_rows)


def _serializable_head(model: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coefficients": model["coefficients"].tolist(),
    }


def fit_phase2_predictor(
    trace_dir: str | Path,
    *,
    ridge: float = 1e-3,
    quantiles: Sequence[float] = (0.50, 0.75),
    harmful_threshold: float = 0.05,
    folds: int = 5,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Fit deployable full and timestep-only causal ridge predictors."""

    records = load_trajectory_records(trace_dir)
    model_specs = {
        "full": tuple(CAUSAL_FEATURES),
        "timestep_only": ("timestep", "step_fraction"),
    }
    models: dict[str, Any] = {}
    for model_name, feature_names in model_specs.items():
        heads: dict[str, Any] = {}
        score_rows: dict[int, torch.Tensor] = {}
        targets: dict[int, torch.Tensor] = {}
        for horizon in (1, 2, 3):
            target_key = (
                f"stale_generic/h{horizon}/guided/relative_rmse"
            )
            features, target = _stack_feature_rows(
                records, target_key, feature_names
            )
            fitted = _fit_ridge_error_model(features, target, ridge)
            scores, held_out_target = _cross_validated_scores(
                records,
                target_key,
                feature_names,
                folds,
                seed + horizon,
                ridge,
            )
            heads[f"h{horizon}"] = _serializable_head(fitted)
            score_rows[horizon] = scores
            targets[horizon] = held_out_target
        threshold_sets: dict[str, Any] = {}
        for quantile in quantiles:
            threshold_sets[f"q{quantile:.2f}"] = {
                f"h{horizon}": float(torch.quantile(scores, quantile))
                for horizon, scores in score_rows.items()
            }
        models[model_name] = {
            "features": list(feature_names),
            "heads": heads,
            "thresholds": threshold_sets,
            "training": {
                f"h{horizon}": {
                    "observation_count": int(target.numel()),
                    "harmful_rate": float(
                        (target > harmful_threshold).float().mean()
                    ),
                    "rmse": float(
                        (
                            score_rows[horizon] - target
                        ).square().mean().sqrt()
                    ),
                }
                for horizon, target in targets.items()
            },
            "threshold_calibration": (
                f"{folds}-fold trajectory-held-out predictions"
            ),
        }
    return {
        "schema_version": 1,
        "phase": 2,
        "kind": "terminal_pit_causal_ridge_predictor",
        "trace_directory": str(Path(trace_dir).resolve()),
        "trajectory_count": len(records),
        "target": "terminal PiT stale guided relative RMSE",
        "harmful_threshold": harmful_threshold,
        "ridge": ridge,
        "folds": folds,
        "seed": seed,
        "quantiles": list(quantiles),
        "models": models,
        "git": git_metadata(),
        "deployment_contract": (
            "All inputs are available before the current denoiser evaluation; "
            "horizons greater than three always force an exact refresh."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and export the training-free Phase 2 terminal-PiT refresh "
            "predictors from regenerated Phase 1 C2I traces."
        )
    )
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--quantiles", default="0.50,0.75")
    parser.add_argument("--harmful-threshold", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    quantiles = tuple(
        float(value) for value in args.quantiles.split(",") if value.strip()
    )
    report = fit_phase2_predictor(
        args.trace_dir,
        ridge=args.ridge,
        quantiles=quantiles,
        harmful_threshold=args.harmful_threshold,
        folds=args.folds,
        seed=args.seed,
    )
    write_json(args.output, report)
    print(f"Phase 2 predictor: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
