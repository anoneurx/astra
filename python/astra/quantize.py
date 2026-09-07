"""Weight quantization preview (docs/ROADMAP.md Phase 3: fp16/bf16; int8 later).

Provides deterministic post-training quantization of LiteLM weights for
inference-time precision studies.  All functions return a lightweight wrapper
around the original model (no weight copy) with quantized weight access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astra.model import LiteLM


def _view_as_uint32(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32)


def _view_as_float32(x: np.ndarray) -> np.ndarray:
    return x.view(np.float32)


def quantize_bf16(w: np.ndarray) -> np.ndarray:
    """Simulate bfloat16 by truncating the lower 16 mantissa bits.

    bfloat16 = sign(1) + exponent(8) + mantissa(7), total 16 bits.
    We keep the upper 16 bits of a float32 and zero the lower 16.
    """
    u = _view_as_uint32(w.astype(np.float32))
    # clear lower 16 bits
    u = u & np.uint32(0xFFFF0000)
    return _view_as_float32(u)


def quantize_fp16(w: np.ndarray) -> np.ndarray:
    """Quantize to IEEE 754 float16 (5-bit exponent, 10-bit mantissa)."""
    return w.astype(np.float16).astype(np.float32)


@dataclass
class QuantResult:
    """Result of a quantization round-trip."""
    weights_before: np.ndarray
    weights_after: np.ndarray
    max_abs_error: float
    mean_abs_error: float
    relative_error: float

    @staticmethod
    def from_pair(before: np.ndarray, after: np.ndarray) -> QuantResult:
        diff = np.abs(before.astype(np.float32) - after.astype(np.float32))
        return QuantResult(
            weights_before=before,
            weights_after=after,
            max_abs_error=float(diff.max()),
            mean_abs_error=float(diff.mean()),
            relative_error=float(diff.mean() / (np.abs(before).mean() + 1e-12)),
        )


def quantize_model_weights(
    model: LiteLM,
    method: str = "bf16",
) -> dict[str, QuantResult]:
    """Quantize every weight in the model in-place; return per-weight error metrics.

    Supported ``method`` values: ``"fp16"``, ``"bf16"``, ``"int8"``.
    int8 uses symmetric per-tensor quantization with float32 scale.
    """
    from astra.model.core import all_params

    quant_fn = {
        "fp16": quantize_fp16,
        "bf16": quantize_bf16,
        "int8": _quantize_int8,
    }[method]

    results: dict[str, QuantResult] = {}
    for name, w, _g in all_params(model):
        before = w.copy()
        w[...] = quant_fn(w)
        results[name] = QuantResult.from_pair(before, w)
    return results


def quantize_int8(w: np.ndarray) -> np.ndarray:
    """Symmetric per-tensor int8 quantization (dequantized back to float32).

    scale = max(abs(w)) / 127, then clamp to [-127, 127].
    """
    w32 = w.astype(np.float32)
    scale = np.max(np.abs(w32)) / 127.0 if w32.size > 0 else 1.0
    w8 = np.clip(np.round(w32 / scale), -127, 127).astype(np.int8)
    return (w8.astype(np.float32) * scale).astype(np.float32)


_quantize_int8 = quantize_int8
