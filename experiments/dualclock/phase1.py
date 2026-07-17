from __future__ import annotations

import contextlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


TRACE_SCHEMA_VERSION = 2
BRANCH_NAMES = ("unconditional", "conditional")


def parse_indices(value: str | Iterable[int], depth: int) -> list[int]:
    """Parse comma-separated indices, accepting negative Python-style indices."""
    if isinstance(value, str):
        requested = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        requested = [int(item) for item in value]
    resolved: list[int] = []
    for index in requested:
        index = depth + index if index < 0 else index
        if index < 0 or index >= depth:
            raise ValueError(f"Layer index {index} is outside depth {depth}")
        if index not in resolved:
            resolved.append(index)
    return resolved


def default_patch_layers(depth: int) -> list[int]:
    return sorted({0, depth // 2, depth - 1})


def _cpu_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.detach()
    if value.is_floating_point():
        value = value.to(dtype=dtype)
    return value.cpu()


def deterministic_signed_pool(flat: torch.Tensor, width: int) -> torch.Tensor:
    """Project flat features into deterministic signed contiguous buckets."""
    width = min(int(width), flat.shape[-1])
    positions = torch.arange(flat.shape[-1], device=flat.device, dtype=torch.int64)
    signs = (((positions * 1103515245 + 12345) >> 16) & 1).to(flat.dtype)
    signs = signs.mul_(2).sub_(1)
    projected = F.adaptive_avg_pool1d(
        (flat * signs).unsqueeze(1), width
    ).squeeze(1)
    return projected * math.sqrt(flat.shape[-1] / width)


def pack_feature(
    value: torch.Tensor,
    storage: str,
    sketch_dim: int,
    output_dtype: torch.dtype,
) -> dict[str, Any]:
    """Store an activation exactly or as a deterministic temporal sketch.

    The sketch is an adaptive pooled projection of each leading batch item.
    Exact L2 norms are retained beside it so scale statistics remain exact.
    """
    value_float = value.detach().float()
    flat = value_float.reshape(value.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    if storage == "full":
        values = value
    elif storage == "sketch":
        values = deterministic_signed_pool(flat, sketch_dim)
    else:
        raise ValueError(f"Unknown activation storage mode: {storage}")
    return {
        "values": _cpu_tensor(values, output_dtype),
        "norms": norms.cpu(),
        "storage": storage,
        "original_shape": list(value.shape[1:]),
    }


def stack_packed(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot stack an empty packed feature")
    first = items[0]
    return {
        "values": torch.stack([item["values"] for item in items]),
        "norms": torch.stack([item["norms"] for item in items]),
        "storage": first["storage"],
        "original_shape": first["original_shape"],
    }


def compare_velocity(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    ref = reference.detach().float().flatten(1)
    cand = candidate.detach().float().flatten(1)
    delta = cand - ref
    ref_rms = ref.square().mean(dim=-1).sqrt()
    rmse = delta.square().mean(dim=-1).sqrt()
    cosine = F.cosine_similarity(ref, cand, dim=-1, eps=eps)
    return {
        "mse": delta.square().mean(dim=-1).cpu(),
        "rmse": rmse.cpu(),
        "relative_rmse": (rmse / ref_rms.clamp_min(eps)).cpu(),
        "cosine_error": (1.0 - cosine).cpu(),
    }


def compare_semantics(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    ref = reference.detach().float().flatten(1)
    cand = candidate.detach().float().flatten(1)
    error = torch.linalg.vector_norm(cand - ref, dim=-1)
    norm = torch.linalg.vector_norm(ref, dim=-1)
    return {
        "l2": error.cpu(),
        "relative_l2": (error / norm.clamp_min(eps)).cpu(),
    }


def guided_velocity(
    branches: torch.Tensor,
    timestep: torch.Tensor,
    cfg_scale: float,
    guidance_min: float,
    guidance_max: float,
) -> torch.Tensor:
    unconditional, conditional = branches.chunk(2, dim=0)
    scale = cfg_scale if guidance_min < float(timestep[0]) < guidance_max else 1.0
    return unconditional + scale * (conditional - unconditional)


def temporal_extrapolation_ratio(
    current: torch.Tensor,
    anchor: torch.Tensor,
    previous: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    denominator = anchor - previous
    safe = torch.where(
        denominator.abs() < eps,
        torch.where(denominator < 0, -torch.full_like(denominator, eps), torch.full_like(denominator, eps)),
        denominator,
    )
    return (current - anchor) / safe


def image_frequency_energy(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return inexpensive low/high spatial-frequency energy proxies per image."""
    x_float = x.detach().float()
    pooled = F.avg_pool2d(x_float, kernel_size=8, stride=1, padding=4)[:, :, : x.shape[-2], : x.shape[-1]]
    high = x_float - pooled
    return {
        "low": pooled.square().mean(dim=(1, 2, 3)).cpu(),
        "high": high.square().mean(dim=(1, 2, 3)).cpu(),
    }


class ActivationRecorder:
    """Hooks the unmodified C2I core for a single exact model evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        patch_layers: Iterable[int],
        activation_storage: str,
        sketch_dim: int,
        output_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.patch_layers = list(patch_layers)
        self.activation_storage = activation_storage
        self.sketch_dim = int(sketch_dim)
        self.output_dtype = output_dtype
        self.active = False
        self.current: dict[str, Any] = {}
        self.handles: list[Any] = []
        self._install()

    def _pack(self, value: torch.Tensor) -> dict[str, Any]:
        return pack_feature(value, self.activation_storage, self.sketch_dim, self.output_dtype)

    def _install(self) -> None:
        for index in self.patch_layers:
            self.handles.append(
                self.model.patch_blocks[index].register_forward_hook(self._patch_hook(index))
            )
            attention = self.model.patch_blocks[index].attn
            if hasattr(attention, "qkv"):
                self.handles.append(
                    attention.qkv.register_forward_hook(
                        self._qkv_hook("patch", index)
                    )
                )
            else:
                self.handles.append(
                    attention.qkv_x.register_forward_hook(
                        self._qkv_hook("patch", index)
                    )
                )
                self.handles.append(
                    attention.qkv_y.register_forward_hook(
                        self._qkv_hook("text", index)
                    )
                )
        for index, block in enumerate(self.model.pixel_blocks):
            self.handles.append(block.register_forward_pre_hook(self._pit_input_hook(index)))
            self.handles.append(block.register_forward_hook(self._pit_output_hook(index)))
            self.handles.append(
                block.attn.qkv.register_forward_hook(self._qkv_hook("pit", index))
            )

    def _patch_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if self.active:
                image_output = output[0] if isinstance(output, tuple) else output
                self.current[f"patch/block_{index}"] = self._pack(image_output)
                if isinstance(output, tuple) and len(output) > 1:
                    self.current[f"text/block_{index}"] = self._pack(output[1])
                if index == len(self.model.patch_blocks) - 1:
                    self.current["_final_patch"] = image_output.detach()

        return hook

    def _reshape_pit(self, value: torch.Tensor) -> torch.Tensor:
        cfg_batch = 2 * self.batch_size
        if value.shape[0] % cfg_batch:
            raise RuntimeError(
                f"PiT activation leading dimension {value.shape[0]} is not divisible by CFG batch {cfg_batch}"
            )
        patches = value.shape[0] // cfg_batch
        return value.reshape(cfg_batch, patches, *value.shape[1:])

    def _pit_input_hook(self, index: int):
        def hook(_module: torch.nn.Module, inputs: Any) -> None:
            if self.active:
                self.current[f"pit/block_{index}/input"] = self._pack(self._reshape_pit(inputs[0]))

        return hook

    def _pit_output_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if self.active:
                reshaped = self._reshape_pit(output)
                self.current[f"pit/block_{index}/output"] = self._pack(reshaped)
                self.current[f"_pit_full_{index}"] = output.detach()

        return hook

    def _qkv_hook(self, pathway: str, index: int):
        def hook(module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if not self.active:
                return
            heads = (
                self.model.patch_blocks[index].attn.num_heads
                if pathway in {"patch", "text"}
                else self.model.pixel_blocks[index].attn.num_heads
            )
            qkv = output.detach().reshape(output.shape[0], output.shape[1], 3, heads, -1)
            if pathway == "pit":
                cfg_batch = 2 * self.batch_size
                patches = qkv.shape[0] // cfg_batch
                qkv = qkv.reshape(cfg_batch, patches * qkv.shape[1], 3, heads, qkv.shape[-1])
            # Per-head channel summaries are small enough to save exactly.
            mean = qkv.float().mean(dim=1)
            rms = qkv.float().square().mean(dim=1).sqrt()
            stats = torch.stack((mean, rms), dim=-2)
            for qkv_index, name in enumerate(("q", "k", "v")):
                self.current[f"{pathway}/block_{index}/head_{name}"] = pack_feature(
                    stats[:, qkv_index],
                    "full",
                    self.sketch_dim,
                    self.output_dtype,
                )

        return hook

    def capture(self) -> None:
        self.current = {}
        self.active = True

    def finish(self) -> dict[str, Any]:
        self.active = False
        result = self.current
        self.current = {}
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class CollectionOptions:
    num_steps: int = 100
    cfg_scale: float = 2.75
    timeshift: float = 1.0
    guidance_min: float = 0.1
    guidance_max: float = 0.9
    stale_horizons: tuple[int, ...] = (1, 2, 3)
    generic_pit_layer: int = -1
    activation_storage: str = "sketch"
    sketch_dim: int = 1024
    output_dtype: torch.dtype = torch.float16
    sensitivity_steps: tuple[int, ...] = ()
    token_groups: int = 0
    channel_groups: int = 0


def shifted_timesteps(num_steps: int, timeshift: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    last_step = 1.0 / num_steps
    steps = torch.linspace(0.0, 1.0 - last_step, num_steps, device=device, dtype=dtype)
    steps = torch.cat((steps, torch.ones(1, device=device, dtype=dtype)))
    return steps / (steps + (1.0 - steps) * timeshift)


def _append_metric(target: dict[str, list[torch.Tensor]], prefix: str, values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        target[f"{prefix}/{name}"].append(value)


def _stack_metric_lists(values: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack(items) for name, items in values.items() if items}


def _semantic_sensitivity(
    model: torch.nn.Module,
    cfg_x: torch.Tensor,
    cfg_t: torch.Tensor,
    cfg_condition: torch.Tensor,
    exact_semantic: torch.Tensor,
    previous_semantic: torch.Tensor,
    exact_velocity: torch.Tensor,
    timestep: torch.Tensor,
    options: CollectionOptions,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = (
        ("token", 1, options.token_groups),
        ("channel", 2, options.channel_groups),
    )
    for kind, axis, groups in dimensions:
        if groups <= 0:
            continue
        size = exact_semantic.shape[axis]
        for group in range(groups):
            start = group * size // groups
            end = (group + 1) * size // groups
            if start == end:
                continue
            candidate = exact_semantic.clone()
            slices = [slice(None)] * candidate.ndim
            slices[axis] = slice(start, end)
            candidate[tuple(slices)] = previous_semantic[tuple(slices)]
            output = model(cfg_x, cfg_t, cfg_condition, s=candidate)
            sem = compare_semantics(exact_semantic, candidate)
            vel = compare_velocity(exact_velocity, output)
            rows.append(
                {
                    "kind": kind,
                    "start": start,
                    "end": end,
                    "timestep": float(timestep[0]),
                    "semantic_relative_l2": sem["relative_l2"],
                    "velocity_relative_rmse": vel["relative_rmse"],
                    "velocity_cosine_error": vel["cosine_error"],
                    "lipschitz": (
                        vel["rmse"] / sem["l2"].clamp_min(1e-12)
                    ),
                }
            )
    return rows


@torch.inference_mode()
def collect_c2i_batch(
    model: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    options: CollectionOptions,
    patch_layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Collect one exact C2I trajectory batch plus same-state oracle probes."""
    if noise.shape[0] != condition.shape[0] or condition.shape != uncondition.shape:
        raise ValueError("Noise, condition, and uncondition batch sizes must match")
    batch_size = noise.shape[0]
    patch_layers = list(patch_layers or default_patch_layers(len(model.patch_blocks)))
    final_patch_layer = len(model.patch_blocks) - 1
    if final_patch_layer not in patch_layers:
        patch_layers.append(final_patch_layer)
    generic_layer = options.generic_pit_layer
    generic_layer = len(model.pixel_blocks) + generic_layer if generic_layer < 0 else generic_layer
    if not 0 <= generic_layer < len(model.pixel_blocks):
        raise ValueError(f"generic_pit_layer {options.generic_pit_layer} is out of range")

    recorder = ActivationRecorder(
        model,
        batch_size,
        patch_layers,
        options.activation_storage,
        options.sketch_dim,
        options.output_dtype,
    )
    steps = shifted_timesteps(options.num_steps, options.timeshift, noise.device, noise.dtype)
    cfg_condition = torch.cat((uncondition, condition), dim=0)
    x = noise
    previous_velocity: torch.Tensor | None = None
    semantic_history: list[torch.Tensor] = []
    raw_patch_history: list[torch.Tensor] = []
    time_history: list[torch.Tensor] = []
    generic_history: list[torch.Tensor] = []

    exact_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
    representation_lists: dict[str, list[dict[str, Any]]] = defaultdict(list)
    substitution_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
    sensitivity_rows: list[dict[str, Any]] = []
    try:
        for step_index, (t_scalar, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            timestep = t_scalar.repeat(batch_size)
            cfg_timestep = timestep.repeat(2)
            cfg_x = torch.cat((x, x), dim=0)

            recorder.capture()
            exact_branches = model(cfg_x, cfg_timestep, cfg_condition)
            captured = recorder.finish()
            patch_output = captured["_final_patch"]
            t_emb = model.t_embedder(cfg_timestep.reshape(-1)).view(2 * batch_size, -1, model.hidden_size)
            semantic = F.silu(t_emb + patch_output)
            exact_guided = guided_velocity(
                exact_branches,
                timestep,
                options.cfg_scale,
                options.guidance_min,
                options.guidance_max,
            )

            exact_lists["timestep"].append(timestep.detach().float().cpu())
            # The released sampler keeps its solver state in FP32 even while
            # denoiser operations run under BF16 autocast.
            exact_lists["x_t"].append(x.detach().cpu())
            exact_lists["semantic"].append(_cpu_tensor(semantic, options.output_dtype))
            exact_lists["raw_final_patch"].append(
                _cpu_tensor(patch_output, options.output_dtype)
            )
            exact_lists["velocity_branches"].append(_cpu_tensor(exact_branches, options.output_dtype))
            exact_lists["velocity_guided"].append(_cpu_tensor(exact_guided, options.output_dtype))
            frequency = image_frequency_energy(x)
            exact_lists["image_low_frequency_energy"].append(frequency["low"])
            exact_lists["image_high_frequency_energy"].append(frequency["high"])
            for name, value in captured.items():
                if not name.startswith("_"):
                    representation_lists[name].append(value)

            current_generic = captured[f"_pit_full_{generic_layer}"]
            for horizon in options.stale_horizons:
                available = len(semantic_history) >= horizon
                substitution_lists[f"available/h{horizon}"].append(
                    torch.full((batch_size,), available, dtype=torch.bool)
                )
                if not available:
                    nan_batch = torch.full((2 * batch_size,), float("nan"))
                    nan_guided = torch.full((batch_size,), float("nan"))
                    for method in (
                        "stale_semantic",
                        "linear_semantic",
                        "stale_raw_patch",
                        "linear_raw_patch",
                        "stale_generic",
                    ):
                        for metric in ("mse", "rmse", "relative_rmse", "cosine_error"):
                            substitution_lists[f"{method}/h{horizon}/branches/{metric}"].append(nan_batch)
                            substitution_lists[f"{method}/h{horizon}/guided/{metric}"].append(nan_guided)
                    for method in (
                        "stale_semantic",
                        "linear_semantic",
                        "stale_raw_patch",
                        "linear_raw_patch",
                    ):
                        for metric in ("l2", "relative_l2"):
                            substitution_lists[f"{method}/h{horizon}/semantic/{metric}"].append(nan_batch)
                        substitution_lists[f"{method}/h{horizon}/lipschitz"].append(nan_batch)
                    continue

                stale_semantic = semantic_history[-horizon]
                stale_output = model(cfg_x, cfg_timestep, cfg_condition, s=stale_semantic)
                stale_guided = guided_velocity(
                    stale_output, timestep, options.cfg_scale, options.guidance_min, options.guidance_max
                )
                stale_sem_error = compare_semantics(semantic, stale_semantic)
                stale_branch_error = compare_velocity(exact_branches, stale_output)
                stale_guided_error = compare_velocity(exact_guided, stale_guided)
                _append_metric(substitution_lists, f"stale_semantic/h{horizon}/semantic", stale_sem_error)
                _append_metric(substitution_lists, f"stale_semantic/h{horizon}/branches", stale_branch_error)
                _append_metric(substitution_lists, f"stale_semantic/h{horizon}/guided", stale_guided_error)
                substitution_lists[f"stale_semantic/h{horizon}/lipschitz"].append(
                    stale_branch_error["rmse"] / stale_sem_error["l2"].clamp_min(1e-12)
                )

                # The timestep embedding is known at every microstep. Reuse
                # only the raw patch state, then apply the exact current fusion.
                stale_raw_patch = raw_patch_history[-horizon]
                stale_raw_semantic = F.silu(t_emb + stale_raw_patch)
                stale_raw_output = model(
                    cfg_x, cfg_timestep, cfg_condition, s=stale_raw_semantic
                )
                stale_raw_guided = guided_velocity(
                    stale_raw_output,
                    timestep,
                    options.cfg_scale,
                    options.guidance_min,
                    options.guidance_max,
                )
                stale_raw_sem_error = compare_semantics(
                    semantic, stale_raw_semantic
                )
                stale_raw_branch_error = compare_velocity(
                    exact_branches, stale_raw_output
                )
                stale_raw_guided_error = compare_velocity(
                    exact_guided, stale_raw_guided
                )
                _append_metric(
                    substitution_lists,
                    f"stale_raw_patch/h{horizon}/semantic",
                    stale_raw_sem_error,
                )
                _append_metric(
                    substitution_lists,
                    f"stale_raw_patch/h{horizon}/branches",
                    stale_raw_branch_error,
                )
                _append_metric(
                    substitution_lists,
                    f"stale_raw_patch/h{horizon}/guided",
                    stale_raw_guided_error,
                )
                substitution_lists[f"stale_raw_patch/h{horizon}/lipschitz"].append(
                    stale_raw_branch_error["rmse"]
                    / stale_raw_sem_error["l2"].clamp_min(1e-12)
                )

                if len(semantic_history) >= horizon + 1:
                    anchor = semantic_history[-horizon]
                    previous = semantic_history[-horizon - 1]
                    anchor_t = time_history[-horizon]
                    previous_t = time_history[-horizon - 1]
                    ratio = temporal_extrapolation_ratio(
                        cfg_timestep, anchor_t, previous_t
                    )
                    forecast = anchor + ratio.view(-1, 1, 1) * (anchor - previous)
                    forecast_output = model(cfg_x, cfg_timestep, cfg_condition, s=forecast)
                    forecast_guided = guided_velocity(
                        forecast_output, timestep, options.cfg_scale, options.guidance_min, options.guidance_max
                    )
                    forecast_sem_error = compare_semantics(semantic, forecast)
                    forecast_branch_error = compare_velocity(exact_branches, forecast_output)
                    forecast_guided_error = compare_velocity(exact_guided, forecast_guided)
                    _append_metric(substitution_lists, f"linear_semantic/h{horizon}/semantic", forecast_sem_error)
                    _append_metric(substitution_lists, f"linear_semantic/h{horizon}/branches", forecast_branch_error)
                    _append_metric(substitution_lists, f"linear_semantic/h{horizon}/guided", forecast_guided_error)
                    substitution_lists[f"linear_semantic/h{horizon}/lipschitz"].append(
                        forecast_branch_error["rmse"] / forecast_sem_error["l2"].clamp_min(1e-12)
                    )

                    raw_anchor = raw_patch_history[-horizon]
                    raw_previous = raw_patch_history[-horizon - 1]
                    raw_forecast = raw_anchor + ratio.view(-1, 1, 1) * (
                        raw_anchor - raw_previous
                    )
                    raw_forecast_semantic = F.silu(t_emb + raw_forecast)
                    raw_forecast_output = model(
                        cfg_x,
                        cfg_timestep,
                        cfg_condition,
                        s=raw_forecast_semantic,
                    )
                    raw_forecast_guided = guided_velocity(
                        raw_forecast_output,
                        timestep,
                        options.cfg_scale,
                        options.guidance_min,
                        options.guidance_max,
                    )
                    raw_forecast_sem_error = compare_semantics(
                        semantic, raw_forecast_semantic
                    )
                    raw_forecast_branch_error = compare_velocity(
                        exact_branches, raw_forecast_output
                    )
                    raw_forecast_guided_error = compare_velocity(
                        exact_guided, raw_forecast_guided
                    )
                    _append_metric(
                        substitution_lists,
                        f"linear_raw_patch/h{horizon}/semantic",
                        raw_forecast_sem_error,
                    )
                    _append_metric(
                        substitution_lists,
                        f"linear_raw_patch/h{horizon}/branches",
                        raw_forecast_branch_error,
                    )
                    _append_metric(
                        substitution_lists,
                        f"linear_raw_patch/h{horizon}/guided",
                        raw_forecast_guided_error,
                    )
                    substitution_lists[
                        f"linear_raw_patch/h{horizon}/lipschitz"
                    ].append(
                        raw_forecast_branch_error["rmse"]
                        / raw_forecast_sem_error["l2"].clamp_min(1e-12)
                    )
                else:
                    nan_batch = torch.full((2 * batch_size,), float("nan"))
                    for method in ("linear_semantic", "linear_raw_patch"):
                        for metric in ("l2", "relative_l2"):
                            substitution_lists[
                                f"{method}/h{horizon}/semantic/{metric}"
                            ].append(nan_batch)
                        for metric in (
                            "mse",
                            "rmse",
                            "relative_rmse",
                            "cosine_error",
                        ):
                            substitution_lists[
                                f"{method}/h{horizon}/branches/{metric}"
                            ].append(nan_batch)
                            substitution_lists[
                                f"{method}/h{horizon}/guided/{metric}"
                            ].append(torch.full((batch_size,), float("nan")))
                        substitution_lists[
                            f"{method}/h{horizon}/lipschitz"
                        ].append(nan_batch)

                stale_generic = generic_history[-horizon]

                def replace_generic(
                    _module: torch.nn.Module, _inputs: Any, _output: torch.Tensor
                ) -> torch.Tensor:
                    return stale_generic

                handle = model.pixel_blocks[generic_layer].register_forward_hook(replace_generic)
                try:
                    generic_output = model(cfg_x, cfg_timestep, cfg_condition, s=semantic)
                finally:
                    handle.remove()
                generic_guided = guided_velocity(
                    generic_output, timestep, options.cfg_scale, options.guidance_min, options.guidance_max
                )
                _append_metric(
                    substitution_lists,
                    f"stale_generic/h{horizon}/branches",
                    compare_velocity(exact_branches, generic_output),
                )
                _append_metric(
                    substitution_lists,
                    f"stale_generic/h{horizon}/guided",
                    compare_velocity(exact_guided, generic_guided),
                )

            if (
                step_index in options.sensitivity_steps
                and semantic_history
                and (options.token_groups > 0 or options.channel_groups > 0)
            ):
                for row in _semantic_sensitivity(
                    model,
                    cfg_x,
                    cfg_timestep,
                    cfg_condition,
                    semantic,
                    semantic_history[-1],
                    exact_branches,
                    timestep,
                    options,
                ):
                    row["step"] = step_index
                    sensitivity_rows.append(row)

            semantic_history.append(semantic.detach())
            raw_patch_history.append(patch_output.detach())
            time_history.append(cfg_timestep.detach())
            generic_history.append(current_generic.detach())
            max_history = max(options.stale_horizons, default=1) + 1
            semantic_history = semantic_history[-max_history:]
            raw_patch_history = raw_patch_history[-max_history:]
            time_history = time_history[-max_history:]
            generic_history = generic_history[-max_history:]

            dt = t_next - t_scalar
            if previous_velocity is None:
                x = x + exact_guided * dt
            else:
                x = x + dt * (1.5 * exact_guided - 0.5 * previous_velocity)
            previous_velocity = exact_guided
    finally:
        recorder.close()

    exact = {name: torch.stack(items) for name, items in exact_lists.items()}
    representations = {name: stack_packed(items) for name, items in representation_lists.items()}
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "branch_names": list(BRANCH_NAMES),
        "batch_size": batch_size,
        "condition": condition.detach().cpu(),
        "uncondition": uncondition.detach().cpu(),
        "patch_layers": patch_layers,
        "generic_pit_layer": generic_layer,
        "activation_storage": options.activation_storage,
        "sketch_dim": options.sketch_dim,
        "exact": exact,
        "representations": representations,
        "substitutions": _stack_metric_lists(substitution_lists),
        "sensitivity": sensitivity_rows,
        "final_sample": x.detach().cpu(),
    }


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()
