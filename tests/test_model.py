"""Model gradient checks (EX-02).

Central finite differences over a tiny model validate every hand-derived
backward pass (RMSNorm, attention w/ RoPE, SwiGLU, tied head).
"""

from __future__ import annotations

import numpy as np
import pytest

from astra.model import LiteLM, ModelConfig, all_params


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


def _double_model(model: LiteLM) -> LiteLM:
    """Deep float64 copy used for accurate finite-difference numerics."""
    import copy

    d = copy.deepcopy(model)
    f64 = {n: w.astype(np.float64) for n, w, _g in all_params(d)}
    for name, arr in f64.items():
        root, _, attr = name.partition(".")
        if root == "wte":
            d.wte.w = arr
        elif root == "ln_f":
            d.ln_f.g = arr
        elif root.startswith("attn"):
            b = d.attn[int(root[4:])]
            if attr == "qkv":
                b.qkv.w = arr
            elif attr == "out":
                b.out_w.w = arr
            elif attr == "ln1":
                b.ln1.g = arr
        elif root.startswith("ffn"):
            b = d.ffn[int(root[3:])]
            if attr == "wg":
                b.wg.w = arr
            elif attr == "wu":
                b.wu.w = arr
            elif attr == "wd":
                b.wd.w = arr
            elif attr == "ln2":
                b.ln2.g = arr
    return d


def _numeric_grad(f64: LiteLM, ids, targets, name, idx, eps=1e-4):
    w, _g = dict((n, (ww, gg)) for n, ww, gg in all_params(f64))[name]
    orig = w[idx].copy()
    w[idx] = orig + eps
    lp = f64.forward_loss(ids, targets)[1]
    w[idx] = orig - eps
    lm = f64.forward_loss(ids, targets)[1]
    w[idx] = orig
    return (lp - lm) / (2 * eps)


def _analytic_grad(model: LiteLM, ids, targets):
    model.zero_grad()
    logits, _ = model.forward_loss(ids, targets)
    model.backward(logits, targets)
    return dict((n, (w, g)) for n, w, g in all_params(model))


@pytest.mark.parametrize("seed", [0, 7])
def test_gradcheck_all_layers(seed):
    cfg = _tiny_cfg()
    rng = np.random.default_rng(seed)
    m = LiteLM(cfg, seed=seed)
    ids, targets = _batch(cfg, rng)
    f64 = _double_model(m)
    analytic = _analytic_grad(m, ids, targets)
    checked = 0
    for name in analytic:
        w, g = analytic[name]
        if w.ndim == 1:
            idxs: list = [0, 1] if w.shape[0] > 2 else [0]
        else:
            idxs = [(0, 0), (1, 2)] if name not in ("wte",) else [(12, 3), (5, 0)]
        for idx in idxs:
            if w.ndim == 1:
                if idx >= w.shape[0]:
                    continue
            elif idx[0] >= w.shape[0] or idx[1] >= w.shape[1]:
                continue
            num = _numeric_grad(f64, ids, targets, name, idx)
            ana = float(g[idx])
            denom = max(abs(ana), abs(num), 1e-9)
            rel = abs(ana - num) / denom
            assert rel < 1e-2, f"{name}{idx}: analytic={ana} numeric={num} rel={rel:.4f}"
            checked += 1
    assert checked >= 8


def test_loss_reduces_toward_truth():
    """A memorization task must push loss down (sanity that backward helps)."""
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=3)
    rng = np.random.default_rng(3)
    ids, _ = _batch(cfg, rng)
    # target = fixed shift, model must learn it
    targets = np.roll(ids, 1, axis=1)
    from astra.training.optim import AdamW, clip_grad_norm

    opt = AdamW(m, lr=1e-2, weight_decay=0.0)
    losses = []
    for _ in range(60):
        m.zero_grad()
        logits, loss = m.forward_loss(ids, targets)
        m.backward(logits, targets)
        clip_grad_norm(m, 1.0)
        opt.step()
        losses.append(loss)
    assert losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}"


def test_forward_shape_and_bounds():
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=1)
    ids = np.array([[1, 2, 3]])
    logits, _ = m.forward_loss(ids)
    assert logits.shape == (1, 3, cfg.vocab_size)
    assert np.isfinite(logits).all()


def test_param_count_matches_config():
    cfg = _tiny_cfg()
    m = LiteLM(cfg, seed=0)
    n = 0
    for _n, w, _g in all_params(m):
        n += w.size
    assert m.num_params == n


def test_num_params_report():
    # docs/MODEL.md § 5 requirement — measurable params per config
    cfg = ModelConfig(vocab_size=800, d_model=64, n_layers=2, n_heads=4, d_head=16, d_ffn=128)
    assert LiteLM(cfg, seed=0).num_params > 60_000