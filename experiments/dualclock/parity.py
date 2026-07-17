from __future__ import annotations

import argparse
import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from .common import DTYPES, build_core, load_weights, make_inputs, write_json


def tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@torch.inference_mode()
def extract_and_reinject(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = model.patch_blocks[-1].register_forward_hook(capture)
    try:
        exact = model(x, t, y, s=None, mask=mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one final patch-block activation, captured {len(captured)}")
    batch = x.shape[0]
    t_emb = model.t_embedder(t.reshape(-1)).view(batch, -1, model.hidden_size)
    semantic_state = F.silu(t_emb + captured[0])
    reinjected = model(x, t, y, s=semantic_state, mask=mask)
    return exact, reinjected, semantic_state


def compare(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    delta = (reference - candidate).float()
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    cosine = F.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item()
    return {
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "cosine": cosine,
        "reference_sha256_fp32": tensor_digest(reference),
        "candidate_sha256_fp32": tensor_digest(candidate),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact semantic extraction/reinjection parity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=300)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--check-mask", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    atol = args.atol if args.atol is not None else (2e-2 if dtype == torch.bfloat16 else 2e-3)
    rtol = args.rtol if args.rtol is not None else (2e-2 if dtype == torch.bfloat16 else 2e-3)

    torch.manual_seed(args.seed)
    model, mode, _ = build_core(args.config)
    weight_report = load_weights(model, args.checkpoint, mode)
    model.eval().to(device=device, dtype=dtype)
    x, t, y = make_inputs(
        model,
        mode,
        args.batch_size,
        args.height,
        args.width,
        dtype,
        device,
        min(args.text_length, getattr(model, "txt_max_length", args.text_length)),
        args.seed,
    )
    exact, reinjected, semantic = extract_and_reinject(model, x, t, y)
    repeated = model(x, t, y)
    report: dict[str, Any] = {
        "mode": mode,
        "config": args.config,
        "checkpoint": weight_report,
        "shape": list(x.shape),
        "dtype": str(dtype),
        "semantic_shape": list(semantic.shape),
        "tolerances": {"atol": atol, "rtol": rtol},
        "semantic_reinjection": compare(exact, reinjected, atol, rtol),
        "repeatability": compare(exact, repeated, atol, rtol),
    }
    if args.check_mask and mode == "t2i":
        mask = torch.ones(args.batch_size, y.shape[1], dtype=torch.long, device=device)
        mask[:, y.shape[1] // 2 :] = 0
        try:
            masked = model(x, t, y, mask=mask)
            report["mask_diagnostic"] = {
                "accepted": True,
                "changes_output": not torch.allclose(exact, masked, atol=atol, rtol=rtol),
                "comparison": compare(exact, masked, atol, rtol),
            }
        except Exception as exc:
            report["mask_diagnostic"] = {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    write_json(args.output, report)
    print(
        f"semantic parity allclose={report['semantic_reinjection']['allclose']} "
        f"max_abs={report['semantic_reinjection']['max_abs']:.6g}"
    )
    if not report["semantic_reinjection"]["allclose"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
