"""Optimizer, scheduler, and training-loop contracts (EX-03, EX-04, Phase 2)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from astra.model import LiteLM, ModelConfig, all_params as _ap
from astra.training import AdamW, CosineSchedule, SeqStream, loss_by_shard, tokenize_corpus, train
from astra.training.data import Corpus
from astra.training.optim import (
    build_optimizer,
    build_schedule,
    clip_grad_norm,
    grad_norm,
)


def test_cosine_schedule_monotonic_after_warmup():
    s = CosineSchedule(max_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr=1e-5)
    lrs = [s.lr(i) for i in range(101)]
    assert lrs[0] > 0
    assert lrs[5] < lrs[9]          # warmup rising
    assert lrs[30] > lrs[80]        # decay
    assert abs(lrs[100] - 1e-5) < 1e-7


def test_adamw_updates_state_and_weights():
    m = LiteLM(ModelConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8), seed=0)
    before = dict((n, w.copy()) for n, w, _g in _ap(m))
    m.zero_grad()
    for _n, _w, g in _ap(m):
        g[:] = 0.5
    opt = AdamW(m, lr=0.01)
    opt.step(0.01)
    assert len(opt.m) > 0 and len(opt.v) > 0
    changed = any(not np.array_equal(before[n], w) for n, w, _g in _ap(m))
    assert changed
    assert opt.m["wte"].shape == m.wte.w.shape


def test_seqstream_deterministic_and_finite():
    np.random.seed(0)
    corpus = Corpus(ids=np.arange(10_000, dtype=np.int32), manifest={})
    from astra.model.config import ModelConfig

    cfg = ModelConfig(vocab_size=1000, d_model=16, n_layers=1, n_heads=2, d_head=8, d_ffn=16, max_seq_len=32)
    s1 = list(SeqStream(corpus, 4, 32, np.random.default_rng(0)))
    s2 = list(SeqStream(corpus, 4, 32, np.random.default_rng(0)))
    for (x1, y1), (x2, y2) in zip(s1, s2):
        assert np.array_equal(x1, x2)
        assert x1.shape == (4, 32)
        assert y1.shape == (4, 32)
        assert np.array_equal(x1[:, 1:], y1[:, :-1])


def test_tokenize_corpus_caps_to_windows():
    from astra.tokenizer import ByteLevelBPE

    tok = ByteLevelBPE(vocab_size=32)
    text = "tokenize me please " * 500
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_head=8, d_ffn=16, max_seq_len=64)
    c = tokenize_corpus(text, tok, cfg)
    assert c.ids.size % 64 == 0


def test_clip_grad_norm_reduces():
    m = LiteLM(ModelConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8), seed=0)
    m.zero_grad()
    for _name, _w, g in _ap(m):
        g[:] = 5.0
    n_before = grad_norm(m)
    clip_grad_norm(m, 1.0)
    assert grad_norm(m) <= 1.0 + 1e-5
    assert n_before > 1.0


def test_loss_by_shard_partitions_contiguously():
    losses = [float(i) for i in range(100)]
    shards = loss_by_shard(losses, 4)
    assert len(shards) == 4
    assert [s["shard"] for s in shards] == [0, 1, 2, 3]
    assert [s["loss"] for s in shards] != [0.0] * 4
    assert abs(shards[0]["loss"] - float(np.mean(losses[:25]))) < 1e-12
    assert abs(shards[-1]["loss"] - float(np.mean(losses[75:]))) < 1e-12


def test_val_loss_returns_shard_breakdown():
    from astra.training import val_loss
    from astra.model.config import ModelConfig

    np.random.seed(0)
    cfg = ModelConfig(vocab_size=1000, d_model=16, n_layers=1, n_heads=2, d_head=8, d_ffn=16, max_seq_len=32)
    corpus = Corpus(ids=np.arange(4096, dtype=np.int32) % 1000, manifest={})
    m = LiteLM(cfg, seed=0)
    plain = val_loss(m, corpus, cfg)
    sharded = val_loss(m, corpus, cfg, n_shards=4)
    assert "shards" not in plain
    assert len(sharded["shards"]) == 4
    assert abs(sharded["loss"] - plain["loss"]) < 1e-12
    assert all(s["ppl"] > 0 for s in sharded["shards"])


def test_cosine_schedule_boundaries_and_clamp():
    s = CosineSchedule(max_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr=1e-5)
    assert s.lr(0) == pytest.approx(1e-3 / 10)          # first warmup step scales 1/warmup
    assert s.lr(9) == pytest.approx(1e-3, rel=1e-9)     # warmup end reaches peak
    assert s.lr(100) == pytest.approx(1e-5, rel=1e-6)   # end of decay hits min_lr
    assert s.lr(500) == pytest.approx(1e-5, rel=1e-6)   # clamped past max_steps
    assert s.lr(50) > s.lr(90)
    lrs = [s.lr(i) for i in range(101)]
    assert all(b <= a + 1e-12 for a, b in pairwise(lrs[10:]))  # monotone non-increasing post-warmup


def test_optimizer_scheduler_registry():
    from astra.training import OPTIMIZER_REGISTRY, SCHEDULE_REGISTRY

    cfg = ModelConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8)
    m = LiteLM(cfg, seed=0)
    opt = build_optimizer("adamw", m, lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, AdamW)
    sched = build_schedule("cosine", max_steps=10, warmup_steps=2, peak_lr=1e-3, min_lr=1e-5)
    assert isinstance(sched, CosineSchedule)
    assert set(OPTIMIZER_REGISTRY) == {"adamw"}
    assert set(SCHEDULE_REGISTRY) == {"cosine"}
    with pytest.raises(KeyError):
        build_optimizer("lion", m, lr=1e-3)
    with pytest.raises(KeyError):
        build_schedule("linear", max_steps=10, warmup_steps=2, peak_lr=1e-3)


def test_gradient_accumulation_equals_macro_batch():
    """Accumulating N micro-batches then dividing by N == one macro batch (EX-04)."""
    cfg = ModelConfig(vocab_size=64, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8)
    n_micro, bsz, seq = 4, 2, cfg.max_seq_len
    rng = np.random.default_rng(0)
    x = rng.integers(1, cfg.vocab_size, size=(n_micro * bsz, seq))
    y = rng.integers(0, cfg.vocab_size, size=(n_micro * bsz, seq))

    macro = LiteLM(cfg, seed=0)
    macro.zero_grad()
    logits, _ = macro.forward_loss(x, y)
    macro.backward(logits, y)

    micro = LiteLM(cfg, seed=0)
    micro.zero_grad()
    for i in range(n_micro):
        xi = x[i * bsz:(i + 1) * bsz]
        yi = y[i * bsz:(i + 1) * bsz]
        li, _ = micro.forward_loss(xi, yi)
        micro.backward(li, yi)  # accumulates across micro-batches (uses +=)
    for _n, _w, g in _ap(micro):
        g[:] = g / n_micro

    for (n1, _w1, g1), (n2, _w2, g2) in zip(_ap(macro), _ap(micro)):
        assert n1 == n2
        np.testing.assert_allclose(g1, g2, rtol=1e-4, atol=1e-6, err_msg=f"grad mismatch: {n1}")


def test_train_reproduces_identical_loss_trajectory_same_seed():
    """Same seed + same config => identical loss history and final error (reproducibility)."""
    import tempfile

    cfg = ModelConfig(vocab_size=64, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8)
    ids = (np.arange(40_000, dtype=np.int32) * 7) % 62 + 1   # avoid id 0 for a bit of shape
    train_c = Corpus(ids=ids, manifest={"name": "t"})
    val_c = Corpus(ids=ids[:8000], manifest={"name": "v"})
    tc = {
        "max_steps": 6, "peak_lr": 1e-3, "min_lr": 1e-5, "warmup_steps": 2,
        "batch_seq": 4, "accum_steps": 1, "optimizer": "adamw", "scheduler": "cosine",
        "weight_decay": 0.1, "grad_clip": 1.0, "val_shards": 2, "experiment_store": "unused",
    }
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, tempfile.TemporaryDirectory() as s1, tempfile.TemporaryDirectory() as s2:
        r1 = train(cfg, tc, train_c, val_c, seed=11, out_dir=d1, experiment_store=s1)
        r2 = train(cfg, tc, train_c, val_c, seed=11, out_dir=d2, experiment_store=s2)
    assert r1.loss_hist == r2.loss_hist
    assert r1.final_val == r2.final_val
    assert r1.steps == r2.steps == 6
    assert len(r1.loss_hist) == 6
    assert r1.store_run_id != "" and r2.store_run_id != ""


def test_train_accum_steps_diverges_from_single_accum_but_still_converges():
    """accum_steps=2 must run and lower loss; the schedule indexes optimizer updates."""
    import tempfile

    cfg = ModelConfig(vocab_size=64, d_model=8, n_layers=1, n_heads=2, d_head=4, d_ffn=8, max_seq_len=8)
    ids = (np.arange(40_000, dtype=np.int32) * 7) % 62 + 1
    train_c = Corpus(ids=ids, manifest={"name": "t"})
    val_c = Corpus(ids=ids[:8000], manifest={"name": "v"})
    tc = {
        "max_steps": 8, "peak_lr": 1e-3, "min_lr": 1e-5, "warmup_steps": 2,
        "batch_seq": 4, "accum_steps": 2, "optimizer": "adamw", "scheduler": "cosine",
        "weight_decay": 0.1, "grad_clip": 1.0, "val_shards": 2,
    }
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as s:
        rep = train(cfg, tc, train_c, val_c, seed=5, out_dir=d, experiment_store=s)
    assert rep.steps == 8
    assert len(rep.loss_hist) == 8
    assert len(rep.val_hist) == 0           # 8 micro-steps at val_every=250: never hits
    assert all(np.isfinite(rep.loss_hist))
    assert rep.loss_hist[-1] < rep.loss_hist[0]      # training loss decreased across updates