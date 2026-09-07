"""KV-cache decoder correctness and equivalence (docs/ROADMAP.md Phase 3).

The incremental decoder must be mathematically identical to the full
recompute-everything reference sampler, grow beyond max_seq_len, and honor
sliding-window attention + top-k/top-p sampling.
"""

from __future__ import annotations

import numpy as np
import pytest
from astra.evaluation.metrics import generate
from astra.inference.decoder import KVCache, decode, decode_token
from astra.model import LiteLM, ModelConfig


def _cfg(**over) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_head=4,
        d_ffn=32,
        max_seq_len=8,
        **over,
    )


def _seed_ids(rng: np.random.Generator, cfg: ModelConfig, n: int = 5) -> list[int]:
    return rng.integers(0, cfg.vocab_size, size=n).tolist()


@pytest.mark.parametrize("n_new", [1, 8])
def test_decode_token_matches_reference(n_new):
    """Every incremental logit equals the full-context forward pass."""
    cfg = _cfg()
    m = LiteLM(cfg, seed=0)
    rng = np.random.default_rng(3)
    ctx = rng.integers(0, cfg.vocab_size, size=(1, cfg.max_seq_len)).astype(np.int64)
    cache = KVCache(cfg)
    for i in range(len(ctx[0])):
        dec = decode_token(m, cache, np.array([ctx[0, i]], dtype=np.int64))
        ref, _ = m.forward_loss(ctx[:, : i + 1])
        assert np.allclose(ref[0, -1], dec[0, 0], atol=1e-4, rtol=1e-4), f"pos {i}"


def test_decode_matches_generate_with_sampling():
    """KV-cache + reference samplers produce identical tokens for a fixed rng
    while the full context fits in one window (5-seed + 3-new <= max_seq_len)."""
    cfg = _cfg()
    m = LiteLM(cfg, seed=1)
    rng = np.random.default_rng(7)
    seed = _seed_ids(rng, cfg, n=5)
    assert len(seed) + 3 <= cfg.max_seq_len

    out_ref = generate(m, None, seed, max_new=3, temperature=0.9, rng=np.random.default_rng(99), top_k=5, top_p=0.9)
    out_dec = decode(m, seed, max_new=3, temperature=0.9, rng=np.random.default_rng(99), top_k=5, top_p=0.9)
    assert out_ref == out_dec


def test_decode_grows_past_max_seq_len():
    """Phase 3: decoding beyond the training context must not crash and
    must use the growing KV cache (window kept within cache)."""
    cfg = _cfg()
    m = LiteLM(cfg, seed=2)
    rng = np.random.default_rng(1)
    seed = _seed_ids(rng, cfg)
    # generate 3x max_seq_len tokens
    out = decode(m, seed, max_new=3 * cfg.max_seq_len, rng=np.random.default_rng(5), temperature=1.0)
    assert len(out) == 3 * cfg.max_seq_len
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_generate_extended_context():
    """Phase 3 context handling: generate() should work beyond max_seq_len."""
    cfg = _cfg()
    m = LiteLM(cfg, seed=0)
    rng = np.random.default_rng(3)
    seed = _seed_ids(rng, cfg, n=5)
    # generate 10 new tokens → total 15 > max_seq_len=8
    out = generate(m, None, seed, max_new=10, rng=np.random.default_rng(7))
    assert len(out) == 10
    assert all(0 <= t < cfg.vocab_size for t in out)

    # with explicit context_window = max_seq_len (old truncation behavior)
    out_win = generate(m, None, seed, max_new=10, rng=np.random.default_rng(7),
                       context_window=cfg.max_seq_len)
    assert len(out_win) == 10
    assert all(0 <= t < cfg.vocab_size for t in out_win)


def test_sliding_window_bounds():
    """window must stay >= 1 and bounds of the attended slice are valid."""
    cfg = _cfg()
    m = LiteLM(cfg, seed=4)
    cache = KVCache(cfg)
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, cfg.vocab_size, size=(1, cfg.max_seq_len)).astype(np.int64)
    logits: list[np.ndarray] = []
    for i in range(len(ctx[0])):
        logits.append(decode_token(m, cache, np.array([ctx[0, i]], np.int64), window=4)[0])
    # finite logits and no nan from degenerate attention over window
    assert np.isfinite(np.stack(logits)).all()


def test_learned_positions_decode():
    """KV-cache decoder respects pos_type='learned' (absolute position)."""
    cfg = _cfg(pos_type="learned")
    m = LiteLM(cfg, seed=11)
    rng = np.random.default_rng(9)
    ctx = rng.integers(0, cfg.vocab_size, size=(1, cfg.max_seq_len)).astype(np.int64)
    cache = KVCache(cfg)
    for i in range(len(ctx[0])):
        dec = decode_token(m, cache, np.array([ctx[0, i]], dtype=np.int64))
        ref, _ = m.forward_loss(ctx[:, : i + 1])
        assert np.allclose(ref[0, -1], dec[0, 0], atol=1e-4, rtol=1e-4), f"pos {i}"


def test_gelu_decode():
    """KV-cache decoder matches reference for ffn_type='gelu' too."""
    cfg = _cfg(ffn_type="gelu")
    m = LiteLM(cfg, seed=21)
    rng = np.random.default_rng(2)
    ctx = rng.integers(0, cfg.vocab_size, size=(1, cfg.max_seq_len)).astype(np.int64)
    cache = KVCache(cfg)
    for i in range(len(ctx[0])):
        dec = decode_token(m, cache, np.array([ctx[0, i]], dtype=np.int64))
        ref, _ = m.forward_loss(ctx[:, : i + 1])
        assert np.allclose(ref[0, -1], dec[0, 0], atol=1e-4, rtol=1e-4), f"pos {i}"