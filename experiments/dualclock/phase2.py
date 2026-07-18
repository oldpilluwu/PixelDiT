from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase1 import shifted_timesteps


EPS = 1e-12
CAUSAL_FEATURES = (
    "timestep",
    "step_fraction",
    "x_high_to_low",
    "previous_velocity_change",
    "previous_velocity_curvature",
)


@dataclass(frozen=True)
class TerminalPiTState:
    """Output of the terminal PiT block plus reconstruction metadata."""

    tokens: torch.Tensor
    image_height: int
    image_width: int
    patch_count: int

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])


class TerminalPiTCacheModel(nn.Module):
    """Experimental C2I fast path that leaves ``pixdit_core`` unchanged.

    Exact refreshes reproduce the released forward pass explicitly and retain
    the output of the last PiT block. Cached evaluations execute only the
    released final projection and image reconstruction.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not getattr(model, "pixel_blocks", None):
            raise ValueError("Terminal PiT caching requires at least one PiT block")
        self.model = model

    def encode_terminal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> TerminalPiTState:
        batch_size, _, height, width = x.shape
        patch_size = int(self.model.patch_size)
        pos = self.model.fetch_pos(
            height // patch_size, width // patch_size, x.device
        )
        x_patches = F.unfold(
            x, kernel_size=patch_size, stride=patch_size
        ).transpose(1, 2)
        t_emb = self.model.t_embedder(t.reshape(-1)).view(
            batch_size, -1, self.model.hidden_size
        )
        y_emb = self.model.y_embedder(y).view(
            batch_size, 1, self.model.hidden_size
        )
        condition = F.silu(t_emb + y_emb)
        semantic = self.model.s_embedder(x_patches)
        for block in self.model.patch_blocks:
            semantic = block(semantic, condition, pos, mask)
        semantic = F.silu(t_emb + semantic)

        length = int(semantic.shape[1])
        semantic_condition = semantic.reshape(
            batch_size * length, self.model.hidden_size
        )
        pixels = self.model.pixel_embedder(
            x,
            img_height=height,
            img_width=width,
            patch_size=patch_size,
        )
        for block in self.model.pixel_blocks:
            pixels = block(
                pixels,
                semantic_condition,
                height,
                width,
                patch_size,
                mask,
            )
        pixels = pixels.reshape(
            batch_size,
            length,
            patch_size * patch_size,
            self.model.pixel_hidden_size,
        )
        return TerminalPiTState(
            tokens=pixels,
            image_height=height,
            image_width=width,
            patch_count=length,
        )

    def decode_terminal(self, state: TerminalPiTState) -> torch.Tensor:
        tokens = state.tokens
        if tokens.ndim != 4:
            raise ValueError(
                "Terminal tokens must have shape [batch, patches, pixels, channels]"
            )
        batch_size, length, pixels_per_patch, channels = tokens.shape
        patch_size = int(self.model.patch_size)
        if length != state.patch_count:
            raise ValueError("Terminal-state patch metadata does not match its tensor")
        if pixels_per_patch != patch_size * patch_size:
            raise ValueError("Terminal-state patch size does not match the model")
        if channels != int(self.model.pixel_hidden_size):
            raise ValueError("Terminal-state channel size does not match the model")
        if (
            state.image_height % patch_size
            or state.image_width % patch_size
            or length
            != (state.image_height // patch_size)
            * (state.image_width // patch_size)
        ):
            raise ValueError("Terminal-state image geometry is inconsistent")

        projected = self.model.final_layer(
            tokens.reshape(
                batch_size * length, pixels_per_patch, channels
            )
        )
        output_channels = int(self.model.out_channels)
        projected = projected.view(
            batch_size, length, pixels_per_patch, output_channels
        )
        projected = projected.permute(0, 3, 2, 1).contiguous()
        projected = projected.view(
            batch_size, output_channels * pixels_per_patch, length
        )
        return F.fold(
            projected,
            (state.image_height, state.image_width),
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward_exact(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TerminalPiTState]:
        state = self.encode_terminal(x, t, y, mask)
        return self.decode_terminal(state), state

    def forward_cached(self, state: TerminalPiTState) -> torch.Tensor:
        return self.decode_terminal(state)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_exact(x, t, y, mask)
        return output


@dataclass(frozen=True)
class RidgeHead:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RidgeHead":
        return cls(
            mean=tuple(float(item) for item in value["mean"]),
            scale=tuple(float(item) for item in value["scale"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
        )

    def predict(
        self, values: torch.Tensor, cache: dict[tuple[str, torch.dtype], Any]
    ) -> torch.Tensor:
        key = (str(values.device), values.dtype)
        tensors = cache.get(key)
        if tensors is None:
            mean = torch.tensor(self.mean, device=values.device, dtype=values.dtype)
            scale = torch.tensor(
                self.scale, device=values.device, dtype=values.dtype
            )
            coefficients = torch.tensor(
                self.coefficients, device=values.device, dtype=values.dtype
            )
            tensors = (mean, scale, coefficients)
            cache[key] = tensors
        mean, scale, coefficients = tensors
        normalized = (values - mean) / scale.clamp_min(1e-8)
        return coefficients[0] + normalized @ coefficients[1:]


class CausalErrorPredictor:
    """Deployable Phase 1.5 ridge predictor and quantile thresholds."""

    def __init__(
        self,
        feature_names: Sequence[str],
        heads: Mapping[int, RidgeHead],
        thresholds: Mapping[int, float],
        source: str | None = None,
    ) -> None:
        unknown = set(feature_names) - set(CAUSAL_FEATURES)
        if unknown:
            raise ValueError(f"Unknown causal features: {sorted(unknown)}")
        self.feature_names = tuple(feature_names)
        self.heads = dict(heads)
        self.thresholds = {
            int(horizon): float(value)
            for horizon, value in thresholds.items()
        }
        self.source = source
        self._tensor_cache: dict[
            int, dict[tuple[str, torch.dtype], Any]
        ] = {horizon: {} for horizon in self.heads}

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        model_name: str = "full",
        quantile: float = 0.75,
    ) -> "CausalErrorPredictor":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = payload.get("models", {})
        if model_name not in models:
            raise ValueError(
                f"Predictor {path} has no model named {model_name!r}"
            )
        model = models[model_name]
        quantile_key = f"q{quantile:.2f}"
        threshold_sets = model.get("thresholds", {})
        if quantile_key not in threshold_sets:
            raise ValueError(
                f"Predictor {path} has no threshold set {quantile_key!r}"
            )
        return cls(
            feature_names=model["features"],
            heads={
                int(name.removeprefix("h")): RidgeHead.from_dict(value)
                for name, value in model["heads"].items()
            },
            thresholds={
                int(name.removeprefix("h")): float(value)
                for name, value in threshold_sets[quantile_key].items()
            },
            source=str(Path(path).resolve()),
        )

    @property
    def max_horizon(self) -> int:
        return max(self.heads, default=0)

    def predict(
        self, horizon: int, feature_values: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if horizon not in self.heads:
            raise KeyError(f"No predictor head for cache horizon {horizon}")
        matrix = torch.stack(
            [feature_values[name] for name in self.feature_names], dim=-1
        ).float()
        return self.heads[horizon].predict(
            matrix, self._tensor_cache[horizon]
        )


def image_frequency_ratio(x: torch.Tensor) -> torch.Tensor:
    """Match the Phase 1 causal frequency proxy without leaving the device."""

    x_float = x.detach().float()
    pooled = F.avg_pool2d(
        x_float, kernel_size=8, stride=1, padding=4
    )[:, :, : x.shape[-2], : x.shape[-1]]
    low = pooled.square().mean(dim=(1, 2, 3))
    high = (x_float - pooled).square().mean(dim=(1, 2, 3))
    return high / low.clamp_min(EPS)


def _normalized_change(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    current_flat = current.detach().float().flatten(1)
    previous_flat = previous.detach().float().flatten(1)
    return torch.linalg.vector_norm(
        current_flat - previous_flat, dim=-1
    ) / torch.linalg.vector_norm(current_flat, dim=-1).clamp_min(EPS)


def _normalized_curvature(
    current: torch.Tensor,
    previous: torch.Tensor,
    previous_previous: torch.Tensor,
) -> torch.Tensor:
    current_flat = current.detach().float().flatten(1)
    previous_flat = previous.detach().float().flatten(1)
    previous_previous_flat = previous_previous.detach().float().flatten(1)
    return torch.linalg.vector_norm(
        current_flat - 2.0 * previous_flat + previous_previous_flat, dim=-1
    ) / torch.linalg.vector_norm(current_flat, dim=-1).clamp_min(EPS)


def causal_features(
    x: torch.Tensor,
    timestep: torch.Tensor,
    step_index: int,
    num_steps: int,
    velocity_history: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Build features available before the current denoiser evaluation."""

    if len(velocity_history) < 3:
        return None
    batch_size = x.shape[0]
    return {
        "timestep": timestep.detach().float(),
        "step_fraction": torch.full(
            (batch_size,),
            step_index / max(num_steps - 1, 1),
            device=x.device,
            dtype=torch.float32,
        ),
        "x_high_to_low": image_frequency_ratio(x),
        "previous_velocity_change": _normalized_change(
            velocity_history[-1], velocity_history[-2]
        ),
        "previous_velocity_curvature": _normalized_curvature(
            velocity_history[-1],
            velocity_history[-2],
            velocity_history[-3],
        ),
    }


@dataclass(frozen=True)
class RefreshDecision:
    refresh: bool
    reason: str
    horizon: int
    max_score: float | None = None
    threshold: float | None = None


@dataclass
class RefreshPolicy:
    mode: str
    fixed_interval: int = 2
    predictor: CausalErrorPredictor | None = None
    forced_refresh_steps: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        supported = {"exact", "fixed", "adaptive"}
        if self.mode not in supported:
            raise ValueError(f"Unknown refresh mode {self.mode!r}")
        if self.fixed_interval < 1:
            raise ValueError("fixed_interval must be at least one")
        if self.mode == "adaptive" and self.predictor is None:
            raise ValueError("Adaptive refresh requires a causal predictor")

    def decide(
        self,
        step_index: int,
        cache_age: int | None,
        x: torch.Tensor,
        timestep: torch.Tensor,
        num_steps: int,
        velocity_history: Sequence[torch.Tensor],
    ) -> RefreshDecision:
        horizon = 0 if cache_age is None else cache_age
        if cache_age is None:
            return RefreshDecision(True, "empty_cache", horizon)
        if step_index in self.forced_refresh_steps:
            return RefreshDecision(True, "forced_guardrail", horizon)
        if self.mode == "exact":
            return RefreshDecision(True, "exact_policy", horizon)
        if self.mode == "fixed":
            refresh = cache_age >= self.fixed_interval
            return RefreshDecision(
                refresh,
                "fixed_interval" if refresh else "fixed_cache",
                horizon,
            )

        assert self.predictor is not None
        if cache_age > self.predictor.max_horizon:
            return RefreshDecision(True, "maximum_horizon", horizon)
        features = causal_features(
            x, timestep, step_index, num_steps, velocity_history
        )
        if features is None:
            return RefreshDecision(True, "insufficient_history", horizon)
        scores = self.predictor.predict(cache_age, features)
        if not torch.isfinite(scores).all():
            return RefreshDecision(True, "nonfinite_score", horizon)
        threshold = self.predictor.thresholds[cache_age]
        max_score = float(scores.max().item())
        refresh = max_score > threshold
        return RefreshDecision(
            refresh,
            "predicted_harmful" if refresh else "predicted_safe",
            horizon,
            max_score=max_score,
            threshold=threshold,
        )


@dataclass
class SamplingResult:
    sample: torch.Tensor
    elapsed_ms: float
    policy_ms: float
    refresh_count: int
    cached_count: int
    decisions: list[dict[str, Any]]
    trajectory: list[torch.Tensor] | None = None

    @property
    def skip_rate(self) -> float:
        total = self.refresh_count + self.cached_count
        return self.cached_count / total if total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "policy_ms": self.policy_ms,
            "policy_overhead_percent": (
                100.0 * self.policy_ms / self.elapsed_ms
                if self.elapsed_ms
                else None
            ),
            "refresh_count": self.refresh_count,
            "cached_count": self.cached_count,
            "skip_rate": self.skip_rate,
            "decisions": self.decisions,
        }


def _guided_velocity(
    branches: torch.Tensor,
    timestep: torch.Tensor,
    cfg_scale: float,
    guidance_min: float,
    guidance_max: float,
) -> torch.Tensor:
    unconditional, conditional = branches.chunk(2, dim=0)
    scale = (
        cfg_scale
        if guidance_min < float(timestep[0]) < guidance_max
        else 1.0
    )
    return unconditional + scale * (conditional - unconditional)


@torch.inference_mode()
def sample_c2i_cached(
    model: TerminalPiTCacheModel,
    noise: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    policy: RefreshPolicy,
    *,
    num_steps: int = 100,
    cfg_scale: float = 2.75,
    timeshift: float = 1.0,
    guidance_min: float = 0.1,
    guidance_max: float = 0.9,
    return_trajectory: bool = False,
) -> SamplingResult:
    """Run the released AB2 sampler with exact or terminal-cache evaluations."""

    if noise.shape[0] != condition.shape[0] or condition.shape != uncondition.shape:
        raise ValueError("Noise, condition, and uncondition batch sizes must match")
    steps = shifted_timesteps(
        num_steps, timeshift, noise.device, noise.dtype
    )
    cfg_condition = torch.cat((uncondition, condition), dim=0)
    x = noise
    previous_solver_velocity: torch.Tensor | None = None
    velocity_history: list[torch.Tensor] = []
    terminal_state: TerminalPiTState | None = None
    last_refresh: int | None = None
    decisions: list[dict[str, Any]] = []
    trajectory = [x.detach().clone()] if return_trajectory else None
    policy_seconds = 0.0
    refresh_count = 0
    cached_count = 0

    use_cuda_events = x.device.type == "cuda"
    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        wall_start = time.perf_counter()

    for step_index, (t_scalar, t_next) in enumerate(
        zip(steps[:-1], steps[1:])
    ):
        timestep = t_scalar.repeat(x.shape[0])
        cache_age = (
            None if last_refresh is None else step_index - last_refresh
        )
        # Adaptive decisions end in a host-visible boolean and therefore
        # synchronize the stream. Drain earlier denoiser work before starting
        # the policy timer so it is not misattributed to predictor overhead.
        if policy.mode == "adaptive" and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        policy_start = time.perf_counter()
        decision = policy.decide(
            step_index,
            cache_age,
            x,
            timestep,
            num_steps,
            velocity_history,
        )
        policy_seconds += time.perf_counter() - policy_start

        if decision.refresh:
            cfg_x = torch.cat((x, x), dim=0)
            cfg_timestep = timestep.repeat(2)
            branches, terminal_state = model.forward_exact(
                cfg_x, cfg_timestep, cfg_condition
            )
            last_refresh = step_index
            refresh_count += 1
        else:
            if terminal_state is None:
                raise RuntimeError("Cache policy skipped with no terminal state")
            branches = model.forward_cached(terminal_state)
            cached_count += 1

        velocity = _guided_velocity(
            branches,
            timestep,
            cfg_scale,
            guidance_min,
            guidance_max,
        )
        decisions.append(
            {
                "step": step_index,
                "timestep": float(t_scalar),
                "refresh": decision.refresh,
                "reason": decision.reason,
                "cache_horizon": decision.horizon,
                "score": decision.max_score,
                "threshold": decision.threshold,
            }
        )
        dt = t_next - t_scalar
        if previous_solver_velocity is None:
            x = x + velocity * dt
        else:
            x = x + dt * (
                1.5 * velocity - 0.5 * previous_solver_velocity
            )
        previous_solver_velocity = velocity
        velocity_history.append(velocity)
        velocity_history = velocity_history[-3:]
        if trajectory is not None:
            trajectory.append(x.detach().clone())

    if use_cuda_events:
        end_event.record()
        torch.cuda.synchronize(x.device)
        elapsed_ms = float(start_event.elapsed_time(end_event))
    else:
        elapsed_ms = 1000.0 * (time.perf_counter() - wall_start)
    return SamplingResult(
        sample=x,
        elapsed_ms=elapsed_ms,
        policy_ms=1000.0 * policy_seconds,
        refresh_count=refresh_count,
        cached_count=cached_count,
        decisions=decisions,
        trajectory=trajectory,
    )


def paired_quality(
    exact: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Paired pixel and frequency diagnostics used before distributional tests."""

    if exact.shape != candidate.shape:
        raise ValueError("Paired outputs must have identical shapes")
    exact_float = exact.detach().float()
    candidate_float = candidate.detach().float()
    delta = candidate_float - exact_float
    flat_exact = exact_float.flatten(1)
    flat_delta = delta.flatten(1)
    rmse = flat_delta.square().mean(dim=-1).sqrt()
    relative_l2 = torch.linalg.vector_norm(
        flat_delta, dim=-1
    ) / torch.linalg.vector_norm(flat_exact, dim=-1).clamp_min(EPS)
    psnr = 20.0 * torch.log10(
        torch.full_like(rmse, 2.0) / rmse.clamp_min(EPS)
    )

    exact_low = F.avg_pool2d(
        exact_float, kernel_size=8, stride=1, padding=4
    )[:, :, : exact.shape[-2], : exact.shape[-1]]
    candidate_low = F.avg_pool2d(
        candidate_float, kernel_size=8, stride=1, padding=4
    )[:, :, : candidate.shape[-2], : candidate.shape[-1]]
    low_rmse = (candidate_low - exact_low).square().mean(
        dim=(1, 2, 3)
    ).sqrt()
    high_rmse = (
        (candidate_float - candidate_low) - (exact_float - exact_low)
    ).square().mean(dim=(1, 2, 3)).sqrt()

    def summary(values: torch.Tensor) -> dict[str, float]:
        values = values.detach().float().cpu()
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(torch.quantile(values, 0.9)),
            "max": float(values.max()),
        }

    return {
        "sample_count": int(exact.shape[0]),
        "pixel_rmse": summary(rmse),
        "relative_l2": summary(relative_l2),
        "psnr_db": summary(psnr),
        "low_frequency_rmse": summary(low_rmse),
        "high_frequency_rmse": summary(high_rmse),
    }
