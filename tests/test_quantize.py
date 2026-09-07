"""Quantization precision-loss checks (docs/ROADMAP.md Phase 3).

Verifies that each quantization mode produces deterministic, bounded error
and that inference quality loss is monotonic (fp32 > fp16 > bf16 > int8 in
precision).
"""

from __future__ import annotations

import numpy as np
import pytest
from astra.model import LiteLM, ModelConfig
from astra.quantize import (
    quantize_bf16,
    quantize_int8,
    quantize_model_weights,
)


def _tiny_cfg(**over) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_head=4,
        d_ffn=32,
        max_seq_len=12,
        **over,
    )


def _batch(cfg: ModelConfig, rng: np.random.Generator):
    ids = rng.integers(0, cfg.vocab_size, size=(2, cfg.max_seq_len)).astype(np.int64)
    targets = rng.integers(0, cfg.vocab_size, size=(2, cfg.max_seq_len)).astype(np.int64)
    return ids, targets


def _ref_loss(model: LiteLM, ids, targets) -> float:
    model.zero_grad()
    _logits, loss = model.forward_loss(ids, targets)
    return float(loss)


@pytest.mark.parametrize("method", ["fp16", "bf16", "int8"])
def test_quantize_runs_on_all_weights(method):
    """Every weight is quantized in-place; error is bounded and finite."""
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=1)
    results = quantize_model_weights(m, method=method)
    assert len(results) > 0
    for res in results.values():
        assert np.isfinite(res.weights_after).all()
        assert res.max_abs_error >= 0
        assert res.mean_abs_error >= 0


def test_precision_loss_monotonic():
    """Loss after quantization is non-decreasing with coarser precision."""
    cfg = _tiny_cfg()
    rng = np.random.default_rng(4)
    ids, targets = _batch(cfg, rng)

    losses = {"fp32": _ref_loss(LiteLM(cfg, seed=2), ids, targets)}
    for method in ("fp16", "bf16", "int8"):
        m = LiteLM(cfg, seed=2)
        quantize_model_weights(m, method=method)
        losses[method] = _ref_loss(m, ids, targets)

    assert losses["fp16"] >= losses["fp32"] - 1e-2   # fp16 near-lossless
    assert losses["bf16"] >= losses["fp16"] - 1e-2   # bf16 rougher than fp16
    assert losses["int8"] >= losses["bf16"] - 1e-2   # int8 rougher than bf16


def test_bf16_truncates_mantissa():
    """bf16 is 'bigger-exponent' rounding: max error is ~2^-8 * magnitude."""
    x = np.array([1.0, 1.5, 2.0, 2.5, -0.5, 100.0], dtype=np.float32)
    q = quantize_bf16(x)
    assert q.dtype == np.float32
    assert np.max(np.abs(q - x)) <= 0.05  # ~2^-8 * 100 / 2 = 0.195... small enough


def test_int8_symmetric_scale():
    """int8 round-trip preserves sign & magnitude; error bounded by scale."""
    x = np.arange(-100, 100, 7, dtype=np.float32) % 200 - 100
    q = quantize_int8(x)
    scale = np.max(np.abs(x)) / 127.0
    assert np.max(np.abs(q - x)) <= scale / 2 + 1e-3


def test_quantize_deterministic():
    """Same weights -> same quantized weights (no rng involved)."""
    from astra.model.core import all_params

    cfg = _tiny_cfg()
    m1, m2 = LiteLM(cfg, seed=5), LiteLM(cfg, seed=5)
    quantize_model_weights(m1, method="bf16")
    quantize_model_weights(m2, method="bf16")
    w1 = {n: w for n, w, _g in all_params(m1)}
    w2 = {n: w for n, w, _g in all_params(m2)}
    for n, w1v in w1.items():
        assert np.array_equal(w1v, w2[n]), f"{n} differs"