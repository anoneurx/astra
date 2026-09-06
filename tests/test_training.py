"""Optimizer, scheduler, and training-loop contracts (EX-03, EX-04)."""

from __future__ import annotations

import numpy as np
import pytest

from astra.training import AdamW, CosineSchedule, SeqStream, tokenize_corpus
from astra.training.data import Corpus
from astra.training.optim import grad_norm, clip_grad_norm, all_params
from astra.model import LiteLM, ModelConfig, all_params as _ap


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