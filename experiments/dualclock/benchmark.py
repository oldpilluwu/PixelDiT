from __future__ import annotations

import argparse
import contextlib
import gc
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from pixdit_core.pixeldit_c2i import AugmentedDiTBlock, PiTBlock
from pixdit_core.pixeldit_t2i import MMDiTBlockT2I

from .common import (
    DTYPES,
    build_core,
    load_weights,
    make_inputs,
    percentile,
    summarize,
    write_json,
)


def estimate_flops(
    model: torch.nn.Module,
    mode: str,
    batch: int,
    height: int,
    width: int,
    text_length: int,
) -> dict[str, int]:
    """Analytical multiply-add FLOP estimates grouped like the latency profiler."""
    p = model.patch_size
    patches = (height // p) * (width // p)
    pixels_per_patch = p * p
    hidden = model.hidden_size
    pixel_hidden = model.pixel_hidden_size
    channels = model.in_channels
    patch_depth = model.patch_depth
    pixel_depth = model.pixel_depth
    flops: dict[str, int] = defaultdict(int)

    flops["patch_embedding_text_projection"] += 2 * batch * patches * (channels * pixels_per_patch) * hidden
    flops["conditioning"] += 4 * batch * 256 * hidden + 4 * batch * hidden * hidden

    if mode == "c2i":
        semantic_tokens = patches
        flops["patch_attention"] += patch_depth * (
            8 * batch * patches * hidden * hidden + 4 * batch * patches * patches * hidden
        )
        mlp_hidden = int(2 * int(hidden * 4.0) / 3)
        flops["patch_mlp"] += patch_depth * 6 * batch * patches * hidden * mlp_hidden
        flops["patch_adaln"] += patch_depth * 12 * batch * hidden * hidden
    else:
        semantic_tokens = patches + text_length
        flops["patch_embedding_text_projection"] += (
            2 * batch * text_length * model.txt_embed_dim * hidden
        )
        flops["patch_attention"] += patch_depth * (
            8 * batch * semantic_tokens * hidden * hidden
            + 4 * batch * semantic_tokens * semantic_tokens * hidden
        )
        mlp_hidden = int(2 * int(hidden * 4.0) / 3)
        flops["patch_mlp"] += patch_depth * 6 * batch * semantic_tokens * hidden * mlp_hidden
        flops["patch_adaln"] += patch_depth * 24 * batch * hidden * hidden

    flops["pixel_embedding"] += 2 * batch * height * width * channels * pixel_hidden
    for block in model.pixel_blocks:
        attn_hidden = block.attn_dim
        n_mod = 4 if block.adaln_post_modulation else 6
        modulation_width = n_mod * pixel_hidden * pixels_per_patch
        flops["pit_adaln"] += 2 * batch * patches * hidden * modulation_width
        flops["pit_compaction_expansion"] += (
            4 * batch * patches * (pixels_per_patch * pixel_hidden) * attn_hidden
        )
        flops["pit_attention"] += (
            8 * batch * patches * attn_hidden * attn_hidden
            + 4 * batch * patches * patches * attn_hidden
        )
        mlp_width = 4 * pixel_hidden
        flops["pit_mlp"] += (
            4 * batch * patches * pixels_per_patch * pixel_hidden * mlp_width
        )
    flops["final_projection_reconstruction"] += (
        2 * batch * height * width * pixel_hidden * model.out_channels
    )
    return dict(flops)


def module_categories(model: torch.nn.Module) -> dict[torch.nn.Module, str]:
    categories: dict[torch.nn.Module, str] = {
        model.s_embedder: "patch_embedding_text_projection",
        model.t_embedder: "conditioning",
        model.pixel_embedder: "pixel_embedding",
        model.final_layer: "final_projection_reconstruction",
    }
    if hasattr(model, "y_embedder"):
        category = "conditioning" if model.__class__.__name__ == "PixDiT" else "patch_embedding_text_projection"
        categories[model.y_embedder] = category
    for block in model.patch_blocks:
        if isinstance(block, AugmentedDiTBlock):
            categories[block.attn] = "patch_attention"
            categories[block.mlp] = "patch_mlp"
            for module in (block.adaLN_modulation, block.norm1, block.norm2):
                categories[module] = "patch_adaln"
        elif isinstance(block, MMDiTBlockT2I):
            categories[block.attn] = "patch_attention"
            categories[block.mlp_x] = "patch_mlp"
            categories[block.mlp_y] = "patch_mlp"
            for module in (
                block.adaLN_modulation_img,
                block.adaLN_modulation_txt,
                block.norm_x1,
                block.norm_x2,
                block.norm_y1,
                block.norm_y2,
            ):
                categories[module] = "patch_adaln"
    for block in model.pixel_blocks:
        if not isinstance(block, PiTBlock):
            continue
        for module in (block.adaLN_modulation, block.norm1, block.norm2):
            categories[module] = "pit_adaln"
        categories[block.compress_to_attn] = "pit_compaction_expansion"
        categories[block.expand_from_attn] = "pit_compaction_expansion"
        categories[block.attn] = "pit_attention"
        categories[block.mlp] = "pit_mlp"
    return categories


class ComponentTimer:
    def __init__(self, model: torch.nn.Module):
        self.categories = module_categories(model)
        self.handles: list[Any] = []
        self.pending: dict[torch.nn.Module, list[torch.cuda.Event]] = defaultdict(list)
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def __enter__(self) -> "ComponentTimer":
        for module, category in self.categories.items():
            self.handles.append(module.register_forward_pre_hook(self._pre))
            self.handles.append(module.register_forward_hook(self._post))
        return self

    def _pre(self, module: torch.nn.Module, _inputs: Any) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.pending[module].append(event)

    def _post(self, module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((self.categories[module], self.pending[module].pop(), end))

    def clear(self) -> None:
        self.events.clear()

    def elapsed(self) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for category, start, end in self.events:
            result[category] += start.elapsed_time(end)
        return dict(result)

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextlib.contextmanager
def time_patchify(timer: ComponentTimer | None):
    if timer is None:
        yield
        return
    original_unfold = F.unfold
    original_fold = F.fold

    def wrapped(function):
        def call(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            timer.events.append(("patchify_reconstruction", start, end))
            return result

        return call

    F.unfold = wrapped(original_unfold)
    F.fold = wrapped(original_fold)
    try:
        yield
    finally:
        F.unfold = original_unfold
        F.fold = original_fold


@torch.inference_mode()
def benchmark_batch(
    model: torch.nn.Module,
    mode: str,
    batch: int,
    height: int,
    width: int,
    text_length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmup: int,
    repeats: int,
    trials: int,
    component_detail: bool,
) -> dict[str, Any]:
    x, t, y = make_inputs(model, mode, batch, height, width, dtype, device, text_length, seed)
    for _ in range(warmup):
        model(x, t, y)
    torch.cuda.synchronize(device)

    timer_context = ComponentTimer(model) if component_detail else contextlib.nullcontext(None)
    latencies: list[float] = []
    peaks: list[int] = []
    components: dict[str, list[float]] = defaultdict(list)
    trial_medians: list[float] = []
    with timer_context as timer:
        for _trial in range(trials):
            trial_values = []
            for _repeat in range(repeats):
                if timer is not None:
                    timer.clear()
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with time_patchify(timer):
                    model(x, t, y)
                end.record()
                torch.cuda.synchronize(device)
                elapsed = start.elapsed_time(end)
                latencies.append(elapsed)
                trial_values.append(elapsed)
                peaks.append(torch.cuda.max_memory_allocated(device))
                if timer is not None:
                    for category, value in timer.elapsed().items():
                        components[category].append(value)
            trial_medians.append(percentile(trial_values, 0.5))

    latency_summary = summarize(latencies)
    latency_summary["trial_medians_ms"] = trial_medians
    latency_summary["trial_median_variation_percent"] = (
        100.0 * (max(trial_medians) - min(trial_medians)) / latency_summary["median"]
        if latency_summary["median"]
        else float("inf")
    )
    component_summary = {name: summarize(values) for name, values in sorted(components.items())}
    measured_component_ms = sum(item["median"] for item in component_summary.values())
    latency_summary["component_coverage_percent"] = (
        100.0 * measured_component_ms / latency_summary["median"] if latency_summary["median"] else 0.0
    )
    latency_summary["unattributed_ms"] = max(latency_summary["median"] - measured_component_ms, 0.0)
    flops = estimate_flops(model, mode, batch, height, width, text_length)
    total_flops = sum(flops.values())
    return {
        "batch_size": batch,
        "latency_ms": latency_summary,
        "peak_allocated_bytes": max(peaks),
        "throughput_images_per_second": 1000.0 * batch / latency_summary["median"],
        "components_ms": component_summary,
        "estimated_flops": {
            name: {"flops": value, "gflops": value / 1e9, "percent": 100.0 * value / total_flops}
            for name, value in sorted(flops.items())
        },
        "estimated_total_gflops": total_flops / 1e9,
    }


@torch.inference_mode()
def batch_fits(
    model: torch.nn.Module,
    mode: str,
    batch: int,
    height: int,
    width: int,
    text_length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> bool:
    try:
        x, t, y = make_inputs(model, mode, batch, height, width, dtype, device, text_length, seed)
        model(x, t, y)
        torch.cuda.synchronize(device)
        del x, t, y
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def find_largest_batch(
    model: torch.nn.Module,
    mode: str,
    height: int,
    width: int,
    text_length: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    maximum: int,
) -> int:
    low, candidate = 1, 1
    while candidate <= maximum and batch_fits(
        model, mode, candidate, height, width, text_length, dtype, device, seed
    ):
        low = candidate
        candidate *= 2
    high = min(candidate - 1, maximum)
    while low < high:
        middle = math.ceil((low + high) / 2)
        if batch_fits(model, mode, middle, height, width, text_length, dtype, device, seed):
            low = middle
        else:
            high = middle - 1
    return low


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-event PixelDiT Phase 0 component profiler.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--text-length", type=int, default=300)
    parser.add_argument("--batch-sizes", default="1,auto")
    parser.add_argument("--max-auto-batch", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--total-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Phase 0 latency profiler requires a CUDA GPU")
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    model, mode, _ = build_core(args.config)
    weight_report = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    text_length = min(args.text_length, getattr(model, "txt_max_length", args.text_length))

    component_detail = not args.total_only and not args.compile
    compile_report: dict[str, Any] = {"enabled": args.compile, "mode": args.compile_mode}
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as exc:
            compile_report["error"] = f"{type(exc).__name__}: {exc}"
            raise

    requested = [item.strip() for item in args.batch_sizes.split(",") if item.strip()]
    batches = [int(item) for item in requested if item != "auto"]
    if "auto" in requested:
        batches.append(
            find_largest_batch(
                model,
                mode,
                args.height,
                args.width,
                text_length,
                dtype,
                device,
                args.seed,
                args.max_auto_batch,
            )
        )
    batches = sorted(set(batches))
    results = []
    for batch in batches:
        results.append(
            benchmark_batch(
                model,
                mode,
                batch,
                args.height,
                args.width,
                text_length,
                dtype,
                device,
                args.seed,
                args.warmup,
                args.repeats,
                args.trials,
                component_detail,
            )
        )
        latency = results[-1]["latency_ms"]
        print(
            f"batch={batch} median={latency['median']:.3f} ms "
            f"p10={latency['p10']:.3f} p90={latency['p90']:.3f} "
            f"peak={results[-1]['peak_allocated_bytes'] / 2**30:.2f} GiB"
        )

    report = {
        "mode": mode,
        "config": args.config,
        "checkpoint": weight_report,
        "resolution": [args.height, args.width],
        "text_length": text_length if mode == "t2i" else None,
        "dtype": str(dtype),
        "device": torch.cuda.get_device_name(device),
        "compile": compile_report,
        "component_detail": component_detail,
        "warmup": args.warmup,
        "repeats_per_trial": args.repeats,
        "trials": args.trials,
        "results": results,
    }
    write_json(args.output, report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
