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
    w, _g = {n: (ww, gg) for n, ww, gg in all_params(f64)}[name]
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
    return {n: (w, g) for n, w, g in all_params(model)}


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


def test_variant_arch_differences():
    """Ablation variants trade known params and keep a valid forward pass."""
    base = _tiny_cfg()
    variants = {
        "layernorm": {"norm_type": "layernorm"},
        "gelu": {"ffn_type": "gelu"},
        "learned": {"pos_type": "learned"},
    }
    rng = np.random.default_rng(0)
    ids, targets = _batch(base, rng)
    base_model = LiteLM(base, seed=0)
    base_params = base_model.num_params
    for name, over in variants.items():
        m = LiteLM(_tiny_cfg(**over), seed=0)
        lp, _ = m.forward_loss(ids, targets)
        assert lp.shape == (2, base.max_seq_len, base.vocab_size)
        assert np.isfinite(lp).all()
        assert m.num_params != base_params, f"{name} should change param count"
    lay = LiteLM(_tiny_cfg(norm_type="layernorm"), seed=0)
    gelu = LiteLM(_tiny_cfg(ffn_type="gelu"), seed=0)
    # layernorm adds 2 biases/block + final; gelu drops FFN's wu (d_model x d_ffn)
    assert lay.num_params > base_params
    assert gelu.num_params < base_params


def _twin_f64(model: LiteLM) -> LiteLM:
    """Deep copy with every trainable buffer genuinely float64 (f64->f32 loss
    free), copying exact values from the fp32 model."""
    import copy

    twin = copy.deepcopy(model)
    f64buf = {n: w.astype(np.float64) for n, w, _g in all_params(twin)}
    _set_buffers(twin, f64buf)
    return twin


def _set_buffers(model: LiteLM, f64: dict[str, np.ndarray]) -> None:
    """Reassign param buffer arrays (promotes storage to float64)."""
    for name, arr in f64.items():
        root, _dot, tail = name.partition(".")
        if root == "wte":
            model.wte.w = arr
        elif root == "pos_emb":
            model.pos_emb.w = arr
        elif root == "ln_f":
            if tail == "b":
                model.ln_f.b = arr
            else:
                model.ln_f.g = arr
        elif root.startswith("attn"):
            b = model.attn[int(root[4:])]
            if tail == "qkv":
                b.qkv.w = arr
            elif tail == "out":
                b.out_w.w = arr
            elif tail == "ln1":
                b.ln1.g = arr
            elif tail == "ln1.b":
                b.ln1.b = arr
        elif root.startswith("ffn"):
            b = model.ffn[int(root[3:])]
            if tail == "wg":
                b.wg.w = arr
            elif tail == "wu":
                b.wu.w = arr
            elif tail == "wd":
                b.wd.w = arr
            elif tail == "ln2":
                b.ln2.g = arr
            elif tail == "ln2.b":
                b.ln2.b = arr


def _numeric_grad_generic(f64: LiteLM, ids, targets, name, idx, eps=1e-4):
    """numeric grad using a float64 twin over any all_params entry."""
    w = {n: w for n, w, _g in all_params(f64)}[name]
    orig = w[idx].copy()
    w[idx] = orig + eps
    lp = f64.forward_loss(ids, targets)[1]
    w[idx] = orig - eps
    lm = f64.forward_loss(ids, targets)[1]
    w[idx] = orig
    return (lp - lm) / (2 * eps)


@pytest.mark.parametrize(
    "over, names",
    [
        ({"norm_type": "layernorm"}, ["attn0.ln1.b", "ffn0.ln2.b", "ln_f.b"]),
        ({"ffn_type": "gelu"}, ["ffn0.wg", "ffn0.wd"]),
        ({"pos_type": "learned"}, ["pos_emb"]),
        ({"norm_type": "layernorm", "ffn_type": "gelu", "pos_type": "learned"},
         ["attn0.ln1.b", "ffn0.wg", "pos_emb", "ln_f"]),
    ],
)
def test_gradcheck_variants(over, names):
    """Phase 3 ablation variants must have correct backward passes."""
    cfg = _tiny_cfg(**over)
    rng = np.random.default_rng(0)
    m = LiteLM(cfg, seed=0)
    ids, targets = _batch(cfg, rng)

    analytic = _analytic_grad(m, ids, targets)
    f64 = _twin_f64(m)

    checked = 0
    for name in names:
        w, g = analytic[name]
        idx: int | tuple = (0, 0) if w.ndim == 2 else 0
        num = _numeric_grad_generic(f64, ids, targets, name, idx)
        ana = float(g[idx])
        denom = max(abs(ana), abs(num), 1e-9)
        rel = abs(ana - num) / denom
        assert rel < 2e-2, f"{name}: analytic={ana} numeric={num} rel={rel:.4f}"
        checked += 1
    assert checked == len(names)


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