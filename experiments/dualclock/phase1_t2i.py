from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .phase1 import (
    TRACE_SCHEMA_VERSION,
    ActivationRecorder,
    CollectionOptions,
    _append_metric,
    _cpu_tensor,
    _semantic_sensitivity,
    _stack_metric_lists,
    compare_semantics,
    compare_velocity,
    default_patch_layers,
    guided_velocity,
    image_frequency_energy,
    stack_packed,
    temporal_extrapolation_ratio,
)


ROOT = Path(__file__).resolve().parents[2]


class T2ITrajectoryProbe:
    """Model callable used by the official flow DPM-Solver."""

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        options: CollectionOptions,
        patch_layers: Iterable[int],
        text_mask: torch.Tensor | None,
    ) -> None:
        if batch_size != 1:
            raise ValueError(
                "The released T2I DPM wrapper supplies scalar timesteps and is "
                "only shape-safe at batch size 1"
            )
        self.model = model
        self.batch_size = batch_size
        self.options = options
        self.patch_layers = list(patch_layers)
        final_layer = len(model.patch_blocks) - 1
        if final_layer not in self.patch_layers:
            self.patch_layers.append(final_layer)
        generic_layer = options.generic_pit_layer
        self.generic_layer = (
            len(model.pixel_blocks) + generic_layer
            if generic_layer < 0
            else generic_layer
        )
        if not 0 <= self.generic_layer < len(model.pixel_blocks):
            raise ValueError(f"generic_pit_layer {generic_layer} is out of range")
        self.text_mask = text_mask.detach().cpu() if text_mask is not None else None
        self.recorder = ActivationRecorder(
            model,
            batch_size,
            self.patch_layers,
            options.activation_storage,
            options.sketch_dim,
            options.output_dtype,
        )
        self.semantic_history: list[torch.Tensor] = []
        self.raw_patch_history: list[torch.Tensor] = []
        self.time_history: list[torch.Tensor] = []
        self.generic_history: list[torch.Tensor] = []
        self.exact_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.representation_lists: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.substitution_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.sensitivity_rows: list[dict[str, Any]] = []
        self.evaluation_index = 0

    def _nan_method(self, method: str, horizon: int) -> None:
        nan_branches = torch.full((2 * self.batch_size,), float("nan"))
        nan_guided = torch.full((self.batch_size,), float("nan"))
        for metric in ("mse", "rmse", "relative_rmse", "cosine_error"):
            self.substitution_lists[
                f"{method}/h{horizon}/branches/{metric}"
            ].append(nan_branches)
            self.substitution_lists[
                f"{method}/h{horizon}/guided/{metric}"
            ].append(nan_guided)
        if method != "stale_generic":
            for metric in ("l2", "relative_l2"):
                self.substitution_lists[
                    f"{method}/h{horizon}/semantic/{metric}"
                ].append(nan_branches)
            self.substitution_lists[f"{method}/h{horizon}/lipschitz"].append(
                nan_branches
            )

    def _record_semantic_candidate(
        self,
        method: str,
        horizon: int,
        cfg_x: torch.Tensor,
        cfg_model_t: torch.Tensor,
        cfg_text: torch.Tensor,
        semantic: torch.Tensor,
        candidate: torch.Tensor,
        exact_branches: torch.Tensor,
        exact_guided: torch.Tensor,
        continuous_t: torch.Tensor,
    ) -> None:
        output = self.model(
            cfg_x, cfg_model_t, cfg_text, s=candidate, mask=None
        )
        guided = guided_velocity(
            output,
            continuous_t,
            self.options.cfg_scale,
            self.options.guidance_min,
            self.options.guidance_max,
        )
        semantic_error = compare_semantics(semantic, candidate)
        branch_error = compare_velocity(exact_branches, output)
        guided_error = compare_velocity(exact_guided, guided)
        _append_metric(
            self.substitution_lists,
            f"{method}/h{horizon}/semantic",
            semantic_error,
        )
        _append_metric(
            self.substitution_lists,
            f"{method}/h{horizon}/branches",
            branch_error,
        )
        _append_metric(
            self.substitution_lists,
            f"{method}/h{horizon}/guided",
            guided_error,
        )
        self.substitution_lists[f"{method}/h{horizon}/lipschitz"].append(
            branch_error["rmse"] / semantic_error["l2"].clamp_min(1e-12)
        )

    def __call__(
        self,
        cfg_x: torch.Tensor,
        cfg_model_t: torch.Tensor,
        cfg_text: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if cfg_x.shape[0] != 2 * self.batch_size:
            raise RuntimeError(
                "T2I Phase 1 requires CFG to be active at every official model "
                f"evaluation; received batch {cfg_x.shape[0]}"
            )
        if cfg_text.dim() == 4:
            cfg_text = cfg_text.squeeze(1)
        model_dtype = next(self.model.parameters()).dtype
        model_x = cfg_x.to(dtype=model_dtype)
        model_t = cfg_model_t.to(dtype=model_dtype)
        model_text = cfg_text.to(dtype=model_dtype)
        continuous_t = (
            cfg_model_t[: self.batch_size].detach().float() / 1000.0
        )

        self.recorder.capture()
        exact_branches = self.model(
            model_x, model_t, model_text, s=None, mask=None
        )
        captured = self.recorder.finish()
        raw_patch = captured["_final_patch"]
        t_emb = self.model.t_embedder(model_t.reshape(-1)).view(
            2 * self.batch_size, -1, self.model.hidden_size
        )
        semantic = F.silu(t_emb + raw_patch)
        exact_guided = guided_velocity(
            exact_branches,
            continuous_t,
            self.options.cfg_scale,
            self.options.guidance_min,
            self.options.guidance_max,
        )

        x = cfg_x[: self.batch_size]
        self.exact_lists["timestep"].append(continuous_t.cpu())
        self.exact_lists["model_timestep"].append(
            cfg_model_t[: self.batch_size].detach().float().cpu()
        )
        self.exact_lists["x_t"].append(x.detach().cpu())
        self.exact_lists["semantic"].append(
            _cpu_tensor(semantic, self.options.output_dtype)
        )
        self.exact_lists["raw_final_patch"].append(
            _cpu_tensor(raw_patch, self.options.output_dtype)
        )
        self.exact_lists["velocity_branches"].append(
            _cpu_tensor(exact_branches, self.options.output_dtype)
        )
        self.exact_lists["velocity_guided"].append(
            _cpu_tensor(exact_guided, self.options.output_dtype)
        )
        frequency = image_frequency_energy(x)
        self.exact_lists["image_low_frequency_energy"].append(frequency["low"])
        self.exact_lists["image_high_frequency_energy"].append(frequency["high"])
        for name, value in captured.items():
            if not name.startswith("_"):
                self.representation_lists[name].append(value)

        current_generic = captured[f"_pit_full_{self.generic_layer}"]
        for horizon in self.options.stale_horizons:
            available = len(self.semantic_history) >= horizon
            self.substitution_lists[f"available/h{horizon}"].append(
                torch.full((self.batch_size,), available, dtype=torch.bool)
            )
            if not available:
                for method in (
                    "stale_semantic",
                    "linear_semantic",
                    "stale_raw_patch",
                    "linear_raw_patch",
                    "stale_generic",
                ):
                    self._nan_method(method, horizon)
                continue

            stale_semantic = self.semantic_history[-horizon]
            self._record_semantic_candidate(
                "stale_semantic",
                horizon,
                model_x,
                model_t,
                model_text,
                semantic,
                stale_semantic,
                exact_branches,
                exact_guided,
                continuous_t,
            )
            stale_raw_semantic = F.silu(
                t_emb + self.raw_patch_history[-horizon]
            )
            self._record_semantic_candidate(
                "stale_raw_patch",
                horizon,
                model_x,
                model_t,
                model_text,
                semantic,
                stale_raw_semantic,
                exact_branches,
                exact_guided,
                continuous_t,
            )

            if len(self.semantic_history) >= horizon + 1:
                anchor_t = self.time_history[-horizon]
                previous_t = self.time_history[-horizon - 1]
                ratio = temporal_extrapolation_ratio(
                    cfg_model_t, anchor_t, previous_t
                ).view(-1, 1, 1)
                semantic_anchor = self.semantic_history[-horizon]
                semantic_previous = self.semantic_history[-horizon - 1]
                semantic_forecast = semantic_anchor + ratio * (
                    semantic_anchor - semantic_previous
                )
                self._record_semantic_candidate(
                    "linear_semantic",
                    horizon,
                    model_x,
                    model_t,
                    model_text,
                    semantic,
                    semantic_forecast,
                    exact_branches,
                    exact_guided,
                    continuous_t,
                )
                raw_anchor = self.raw_patch_history[-horizon]
                raw_previous = self.raw_patch_history[-horizon - 1]
                raw_forecast = raw_anchor + ratio * (
                    raw_anchor - raw_previous
                )
                raw_forecast_semantic = F.silu(t_emb + raw_forecast)
                self._record_semantic_candidate(
                    "linear_raw_patch",
                    horizon,
                    model_x,
                    model_t,
                    model_text,
                    semantic,
                    raw_forecast_semantic,
                    exact_branches,
                    exact_guided,
                    continuous_t,
                )
            else:
                self._nan_method("linear_semantic", horizon)
                self._nan_method("linear_raw_patch", horizon)

            stale_generic = self.generic_history[-horizon]

            def replace_generic(
                _module: torch.nn.Module,
                _inputs: Any,
                _output: torch.Tensor,
            ) -> torch.Tensor:
                return stale_generic

            handle = self.model.pixel_blocks[
                self.generic_layer
            ].register_forward_hook(replace_generic)
            try:
                generic_output = self.model(
                    model_x,
                    model_t,
                    model_text,
                    s=semantic,
                    mask=None,
                )
            finally:
                handle.remove()
            generic_guided = guided_velocity(
                generic_output,
                continuous_t,
                self.options.cfg_scale,
                self.options.guidance_min,
                self.options.guidance_max,
            )
            _append_metric(
                self.substitution_lists,
                f"stale_generic/h{horizon}/branches",
                compare_velocity(exact_branches, generic_output),
            )
            _append_metric(
                self.substitution_lists,
                f"stale_generic/h{horizon}/guided",
                compare_velocity(exact_guided, generic_guided),
            )

        if (
            self.evaluation_index in self.options.sensitivity_steps
            and self.semantic_history
            and (
                self.options.token_groups > 0
                or self.options.channel_groups > 0
            )
        ):
            for row in _semantic_sensitivity(
                self.model,
                model_x,
                model_t,
                model_text,
                semantic,
                self.semantic_history[-1],
                exact_branches,
                continuous_t,
                self.options,
            ):
                row["step"] = self.evaluation_index
                self.sensitivity_rows.append(row)

        self.semantic_history.append(semantic.detach())
        self.raw_patch_history.append(raw_patch.detach())
        self.time_history.append(cfg_model_t.detach())
        self.generic_history.append(current_generic.detach())
        max_history = max(self.options.stale_horizons, default=1) + 1
        self.semantic_history = self.semantic_history[-max_history:]
        self.raw_patch_history = self.raw_patch_history[-max_history:]
        self.time_history = self.time_history[-max_history:]
        self.generic_history = self.generic_history[-max_history:]
        self.evaluation_index += 1
        return exact_branches

    def finalize(
        self,
        final_sample: torch.Tensor,
        text_embedding_shape: list[int],
    ) -> dict[str, Any]:
        self.recorder.close()
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "mode": "t2i",
            "branch_names": ["unconditional", "conditional"],
            "batch_size": self.batch_size,
            "patch_layers": self.patch_layers,
            "generic_pit_layer": self.generic_layer,
            "activation_storage": self.options.activation_storage,
            "sketch_dim": self.options.sketch_dim,
            "text_embedding_shape": text_embedding_shape,
            "text_mask": self.text_mask,
            "exact": {
                name: torch.stack(items)
                for name, items in self.exact_lists.items()
            },
            "representations": {
                name: stack_packed(items)
                for name, items in self.representation_lists.items()
            },
            "substitutions": _stack_metric_lists(self.substitution_lists),
            "sensitivity": self.sensitivity_rows,
            "final_sample": final_sample.detach().cpu(),
        }

    def close(self) -> None:
        self.recorder.close()


@torch.inference_mode()
def collect_t2i_batch(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    options: CollectionOptions,
    flow_shift: float = 4.0,
    patch_layers: Iterable[int] | None = None,
    text_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if noise.shape[0] != 1:
        raise ValueError("Released T2I-1024 Phase 1 collection requires batch size 1")
    if not (
        options.guidance_min <= 0.0 and options.guidance_max >= 1.0
    ):
        raise ValueError(
            "Accepted T2I-1024 collection requires CFG over the full [0, 1] interval"
        )
    if str(ROOT / "t2i") not in sys.path:
        sys.path.insert(0, str(ROOT / "t2i"))
    from diffusion.model.flow_dpm import DPMS

    selected = list(
        patch_layers or default_patch_layers(len(model.patch_blocks))
    )
    probe = T2ITrajectoryProbe(
        model,
        noise.shape[0],
        options,
        selected,
        text_mask,
    )
    try:
        solver = DPMS(
            probe,
            condition=condition,
            uncondition=uncondition,
            guidance_type="classifier-free",
            cfg_scale=options.cfg_scale,
            model_type="flow",
            model_kwargs={},
            schedule="FLOW",
            interval_guidance=[
                options.guidance_min,
                options.guidance_max,
            ],
        )
        final_sample = solver.sample(
            noise,
            steps=options.num_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )
        return probe.finalize(final_sample, list(condition.shape))
    except Exception:
        probe.close()
        raise
