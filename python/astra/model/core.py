"""NumPy Transformer core with hand-derived backward passes.

Architecture per docs/MODEL.md baseline, backend-agnostic so a PyTorch
reference can replace it in Phase 1 without changing the pipeline contract
(ADR-0002):

    pre-norm blocks:  x -> RMSNorm -> MHA(RoPE) -> +x
                                  -> RMSNorm -> SwiGLU -> +x
    final RMSNorm -> tied output head (logits)

Block backward methods return the PATH gradient only; callers add the
residual-stream gradient. All operations are float32 ndarrays.
    x, hidden: (B, T, d_model); logits: (B, T, V)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal, overload

import numpy as np

from astra.model.config import ModelConfig

# ---------------------------------------------------------------- norms


class RmsNorm:
    def __init__(self, size: int, eps: float = 1e-6):
        self.g = np.ones(size, dtype=np.float32)
        self.grad = np.zeros_like(self.g)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean_sq = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(mean_sq + self.eps)
        return x * inv * self.g

    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        mean_sq = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(mean_sq + self.eps)
        self.grad += np.sum(grad * x * inv, axis=(0, 1))
        d_inv = np.sum(grad * (x * self.g), axis=-1, keepdims=True)
        dx = grad * (inv * self.g) - d_inv * (mean_sq + self.eps) ** -1.5 * (x / x.shape[-1])
        return dx


class LayerNorm:
    """Standard per-feature LayerNorm with scale + bias (ablation variant).

    docs/MODEL.md research goal: RMSNorm vs LayerNorm (docs/ROADMAP.md Phase 3).
    """

    def __init__(self, size: int, eps: float = 1e-6):
        self.g = np.ones(size, dtype=np.float32)
        self.b = np.zeros(size, dtype=np.float32)
        self.grad = np.zeros_like(self.g)
        self.bgrad = np.zeros_like(self.b)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        self._cache = (x, mean, var)
        return (x - mean) / np.sqrt(var + self.eps) * self.g + self.b

    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        xm, mean, var = self._cache
        inv = 1.0 / np.sqrt(var + self.eps)
        n = x.shape[-1]
        g = grad * self.g
        self.grad += np.sum(grad * (xm - mean) * inv, axis=(0, 1))
        self.bgrad += np.sum(grad, axis=(0, 1))
        dx = inv * (g - np.mean(g, axis=-1, keepdims=True)
                  - (xm - mean) * inv ** 2 * np.sum(g * (xm - mean), axis=-1, keepdims=True) / n)
        return dx


def _make_norm(cfg: ModelConfig):
    if cfg.norm_type == "layernorm":
        return LayerNorm(cfg.d_model, cfg.eps)
    return RmsNorm(cfg.d_model, cfg.eps)


# ------------------------------------------------------------ operators


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_prime(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU (tanh approximation, docs/ROADMAP.md Phase 3 ablation)."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def gelu_prime(x: np.ndarray) -> np.ndarray:
    u = 0.7978845608 * (x + 0.044715 * x**3)
    du = 0.7978845608 * (1.0 + 0.134145 * x**2)
    return 0.5 * (1.0 + np.tanh(u)) + 0.5 * x * (1.0 - np.tanh(u) ** 2) * du


class Linear:
    def __init__(self, in_f: int, out_f: int, rng: np.random.Generator, scale: float = 1.0):
        limit = math.sqrt(6.0 / (in_f + out_f))
        self.w = (rng.uniform(-limit, limit, size=(in_f, out_f)) * scale / limit).astype(
            np.float32
        )
        self.grad = np.zeros_like(self.w)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w

    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        self.grad += x.reshape(-1, x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        return grad @ self.w.T


class Embedding:
    def __init__(self, vocab: int, dim: int, rng: np.random.Generator):
        self.w = (rng.normal(0.0, 0.02, size=(vocab, dim))).astype(np.float32)
        self.grad = np.zeros_like(self.w)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        self._ids = ids
        return self.w[ids]

    def backward(self, grad: np.ndarray) -> None:
        gradf = grad.reshape(-1, grad.shape[-1])
        np.add.at(self.grad, self._ids.reshape(-1), gradf)


class RoPE:
    """Rotary positional embedding applied to per-head q,k vectors."""

    def __init__(self, head_dim: int, max_seq: int, theta: float):
        self.head_dim = head_dim
        self.theta = theta
        inv = 1.0 / (theta ** (np.arange(0, head_dim - 1, 2) / head_dim))
        self._inv = inv
        pos = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(pos, inv)  # (T, head_dim//2)
        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)

    def _angles(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        if t <= len(self.cos):
            return self.cos[:t][None, None], self.sin[:t][None, None]
        # on-the-fly extension beyond precomputed table (inference context > max_seq_len)
        pos = np.arange(t, dtype=np.float32)
        freqs = np.outer(pos, self._inv)
        return np.cos(freqs).astype(np.float32)[None, None], np.sin(freqs).astype(np.float32)[None, None]

    def rotate(self, x: np.ndarray) -> np.ndarray:
        d2 = np.arange(0, x.shape[-1], 2)
        even, odd = x[..., d2], x[..., d2 + 1]
        c, s = self._angles(x.shape[2])
        out = np.empty_like(x)
        out[..., d2] = even * c - odd * s
        out[..., d2 + 1] = even * s + odd * c
        return out

    def unrotate_grad(self, grad: np.ndarray) -> np.ndarray:
        d2 = np.arange(0, grad.shape[-1], 2)
        ge, go = grad[..., d2], grad[..., d2 + 1]
        c, s = self._angles(grad.shape[2])
        out = np.zeros_like(grad)
        out[..., d2] = ge * c + go * s
        out[..., d2 + 1] = -ge * s + go * c
        return out


# ------------------------------------------------------------------ blocks


class AttentionBlock:
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, layer_idx: int):
        d, dh, h = cfg.d_model, cfg.d_head, cfg.n_heads
        self.d, self.dh, self.h = d, dh, h
        self.ln1 = _make_norm(cfg)
        s = 1.0 / math.sqrt(layer_idx + 1)
        self.qkv = Linear(d, 3 * d, rng, scale=s)
        self.out_w = Linear(d, d, rng, scale=s)
        self.rope = RoPE(dh, cfg.max_seq_len, cfg.rope_theta)
        self.causal = np.triu(np.full((cfg.max_seq_len, cfg.max_seq_len), -1e9), k=1).astype(
            np.float32
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, d = x.shape
        h = self.ln1.forward(x)
        q, k, v = np.split(self.qkv.forward(h), 3, axis=-1)
        q = q.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        q, k = self.rope.rotate(q), self.rope.rotate(k)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.dh**-0.5)
        # causal mask: extend dynamically for T > max_seq_len
        causal = self.causal[:T, :T] if T <= self.causal.shape[0] else np.triu(
            np.full((T, T), -1e9, dtype=np.float32), k=1
        )
        scores = scores + causal[None, None]
        att = self._softmax(scores)
        z = att @ v
        self._cache = (x, h, q, k, v, att, z)
        z = z.transpose(0, 2, 1, 3).reshape(B, T, d)
        return x + self.out_w.forward(z)

    @staticmethod
    def _softmax(s: np.ndarray) -> np.ndarray:
        e = np.exp(s - s.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        xm, h, q, k, v, att, z = self._cache
        B, T, d = xm.shape
        self.out_w.backward(grad, z.transpose(0, 2, 1, 3).reshape(B, T, d))
        dz = (grad @ self.out_w.w.T).reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        dv = att.transpose(0, 1, 3, 2) @ dz
        ds = dz @ v.transpose(0, 1, 3, 2)
        dlogits = att * (ds - (ds * att).sum(axis=-1, keepdims=True)) * (self.dh**-0.5)
        dq = self.rope.unrotate_grad(dlogits @ k)
        dk = self.rope.unrotate_grad(dlogits.transpose(0, 1, 3, 2) @ q)
        dqkv = np.concatenate(
            [
                dq.transpose(0, 2, 1, 3).reshape(B, T, d),
                dk.transpose(0, 2, 1, 3).reshape(B, T, d),
                dv.transpose(0, 2, 1, 3).reshape(B, T, d),
            ],
            axis=-1,
        )
        dh = self.qkv.backward(dqkv, h)
        return self.ln1.backward(dh, xm)


class SwiGLUBLock:
    """Feed-forward block; ``ffn_type='gelu'`` uses a 2-projection GELU MLP,
    ``'swiglu'`` the default gated 3-projection variant (docs/MODEL.md)."""

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, layer_idx: int):
        d, ff = cfg.d_model, cfg.d_ffn
        s = 1.0 / math.sqrt(layer_idx + 1)
        self.ffn_type = cfg.ffn_type
        self.ln2 = _make_norm(cfg)
        self.wg = Linear(d, ff, rng, scale=s)
        if cfg.ffn_type == "swiglu":
            self.wu = Linear(d, ff, rng, scale=s)
        self.wd = Linear(ff, d, rng, scale=s)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.ln2.forward(x)
        if self.ffn_type == "swiglu":
            gate = silu(h @ self.wg.w)
            up = h @ self.wu.w
            ff_in = gate * up
        else:
            ff_in = gelu(h @ self.wg.w)
        out = ff_in @ self.wd.w
        self._cache = (h, ff_in, x)
        return x + out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        h, ff_in, xm = self._cache
        d_ff = self.wd.backward(grad, ff_in)
        if self.ffn_type == "swiglu":
            gate_in = h @ self.wg.w
            d_gate_h = self.wg.backward(d_ff * (h @ self.wu.w) * silu_prime(gate_in), h)
            d_up_h = self.wu.backward(d_ff * silu(gate_in), h)
            return self.ln2.backward(d_gate_h + d_up_h, xm)
        gate_in = h @ self.wg.w
        d_h = self.wg.backward(d_ff * gelu_prime(gate_in), h)
        return self.ln2.backward(d_h, xm)


# ------------------------------------------------------------------- model


class LiteLM:
    """Reference implementation of docs/MODEL.md.

    forward(ids) -> logits
        (B, T, id) int array in [0, vocab)
    backward(logits, targets) -> full loss
        accumulates parameter gradients in place (caller must zero them).
    """

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.wte = Embedding(cfg.vocab_size, cfg.d_model, rng)
        if cfg.pos_type == "learned":
            self.pos_emb = Embedding(cfg.max_seq_len, cfg.d_model, rng)
        self.attn = [AttentionBlock(cfg, rng, i) for i in range(cfg.n_layers)]
        self.ffn = [SwiGLUBLock(cfg, rng, i) for i in range(cfg.n_layers)]
        self.ln_f = _make_norm(cfg)

    @overload
    def forward(self, ids: np.ndarray, hidden: Literal[False] = False) -> np.ndarray: ...

    @overload
    def forward(self, ids: np.ndarray, hidden: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def forward(self, ids: np.ndarray, hidden: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Forward pass — ``forward(ids) -> logits`` (B, T, V).

        With ``hidden=True`` also returns the pre-output-head last-layer
        activations ``(B, T, d_model)`` (post final RMSNorm), used as the
        sentence-embedding source by the memory engine (docs/MEMORY.md § 5).
        """
        x = self.wte.forward(ids)
        if self.cfg.pos_type == "learned":
            pos = np.arange(ids.shape[1], dtype=np.int64)[None, :]
            x = x + self.pos_emb.forward(np.broadcast_to(pos, ids.shape))
        for a, f in zip(self.attn, self.ffn):
            x = a.forward(x)
            x = f.forward(x)
        self._final_act = x
        self._ln_out = self.ln_f.forward(x)
        logits = self._ln_out @ self.wte.w.T  # tied output head
        if hidden:
            return logits, self._ln_out
        return logits

    def zero_grad(self) -> None:
        self.wte.grad[:] = 0
        if self.cfg.pos_type == "learned":
            self.pos_emb.grad[:] = 0
        self.ln_f.grad[:] = 0
        if getattr(self.ln_f, "bgrad", None) is not None:
            self.ln_f.bgrad[:] = 0
        for a, f in zip(self.attn, self.ffn):
            a.qkv.grad[:] = 0
            a.out_w.grad[:] = 0
            a.ln1.grad[:] = 0
            if getattr(a.ln1, "bgrad", None) is not None:
                a.ln1.bgrad[:] = 0
            f.wg.grad[:] = 0
            if getattr(f, "wu", None) is not None:
                f.wu.grad[:] = 0
            f.wd.grad[:] = 0
            f.ln2.grad[:] = 0
            if getattr(f.ln2, "bgrad", None) is not None:
                f.ln2.bgrad[:] = 0

    def forward_loss(self, ids: np.ndarray, targets: np.ndarray | None = None):
        logits = self.forward(ids)
        if targets is None:
            return logits, 0.0
        return logits, mean_cross_entropy(log_softmax(logits), targets)

    def backward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        B, T, V = logits.shape
        d = self.cfg.d_model
        logp = log_softmax(logits)
        loss = float(-np.take_along_axis(logp, targets[..., None], axis=-1).mean())
        onehot = np.zeros((B, T, V), dtype=np.float32)
        onehot[np.arange(B)[:, None], np.arange(T)[None, :], targets] = 1.0
        dlogits = (np.exp(logp) - onehot) / (B * T)
        # tied head: accumulate head-path gradient into wte, then through final norm
        self.wte.grad += dlogits.reshape(-1, V).T @ self._ln_out.reshape(-1, d)
        d_ln = dlogits @ self.wte.w
        g = self.ln_f.backward(d_ln, self._final_act)
        for f, a in zip(reversed(self.ffn), reversed(self.attn)):
            g = g + f.backward(g)
            g = g + a.backward(g)
        self.wte.backward(g)
        if self.cfg.pos_type == "learned":
            self.pos_emb.backward(g)
        return loss

    @property
    def num_params(self) -> int:
        n = self.wte.w.size
        if self.cfg.pos_type == "learned":
            n += self.pos_emb.w.size
        for a, f in zip(self.attn, self.ffn):
            n += a.qkv.w.size + a.out_w.w.size + a.ln1.g.size
            if getattr(a.ln1, "b", None) is not None:
                n += a.ln1.b.size
            n += f.wg.w.size + f.wd.w.size + f.ln2.g.size
            if getattr(f, "wu", None) is not None:
                n += f.wu.w.size
            if getattr(f.ln2, "b", None) is not None:
                n += f.ln2.b.size
        n += self.ln_f.g.size
        if getattr(self.ln_f, "b", None) is not None:
            n += self.ln_f.b.size
        return n


def mean_cross_entropy(logp: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.take_along_axis(logp, targets[..., None], axis=-1).mean())


def ce_and_grad(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Cross-entropy loss and its gradient w.r.t. logits.

    Shared by ``LiteLM.backward`` and ``preference_backward`` so that DPO
    training can compute good/bad CE losses and gradients without mutating the
    model's cached activation state between forward-backward pairs.
    """
    B, T, V = logits.shape
    logp = log_softmax(logits)
    loss = float(-np.take_along_axis(logp, targets[..., None], axis=-1).mean())
    onehot = np.zeros((B, T, V), dtype=np.float32)
    onehot[np.arange(B)[:, None], np.arange(T)[None, :], targets] = 1.0
    grad = (np.exp(logp) - onehot) / (B * T)
    return loss, grad


def _preference_backprop(model: LiteLM, x: np.ndarray, dlogits: np.ndarray, ln_out: np.ndarray) -> None:
    """Push ``dlogits`` through the tied head + residual stream.

    ``model._final_act`` and the per-block caches must already hold this
    sequence's forward activations (the caller re-forwards before each call).
    Accumulates into every trainable parameter's ``.grad``.
    """
    V = dlogits.shape[-1]
    d = model.cfg.d_model
    model.wte.grad += dlogits.reshape(-1, V).T @ ln_out.reshape(-1, d)
    d_ln = dlogits @ model.wte.w
    g = model.ln_f.backward(d_ln, model._final_act)
    for f, a in zip(reversed(model.ffn), reversed(model.attn)):
        g = g + f.backward(g)
        g = g + a.backward(g)
    model.wte.backward(g)
    if model.cfg.pos_type == "learned":
        model.pos_emb.backward(g)


def preference_backward(
    model: LiteLM,
    x_good: np.ndarray,
    y_good: np.ndarray,
    x_bad: np.ndarray,
    y_bad: np.ndarray,
    beta: float = 0.1,
) -> float:
    """DPO-style preference backward pass (docs/LEARNING.md § 1.8).

    ``(x_good, y_good)`` and ``(x_bad, y_bad)`` are shifted next-token
    sequences (x = ids[:-1], y = ids[1:]). Computes the CE losses for both,
    then the DPO weighting::

        w = β · σ(β·(L_bad − L_good))
        dL/dlogits = w · (softmax(good) − onehot_good) − w · (softmax(bad) − onehot_bad)

    Each sequence's gradient is backpropagated through its *own* cached
    activations (the model re-forwards each sequence before backprop, since a
    single forward cache cannot hold both). All gradients accumulate into the
    model parameter grads; the returned loss is the (signed) DPO objective for
    logging.
    """
    logits_good = model.forward(x_good)            # cache = good
    loss_good, grad_good = ce_and_grad(logits_good, y_good)
    logits_bad = model.forward(x_bad)              # cache = bad
    loss_bad, grad_bad = ce_and_grad(logits_bad, y_bad)

    weight = float(1.0 / (1.0 + np.exp(-beta * (loss_bad - loss_good))))
    w = beta * weight

    model.forward(x_good)                          # refresh cache = good
    _preference_backprop(model, x_good, w * grad_good, model._ln_out)
    model.forward(x_bad)                           # refresh cache = bad
    _preference_backprop(model, x_bad, -w * grad_bad, model._ln_out)

    return w * loss_good - w * loss_bad


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def all_params(model: LiteLM) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (name, weights, grads) for every trainable parameter."""
    for name in ("wte",):
        w = getattr(model, name)
        yield name, w.w, w.grad
    if model.cfg.pos_type == "learned":
        yield "pos_emb", model.pos_emb.w, model.pos_emb.grad
    for i, a in enumerate(model.attn):
        yield f"attn{i}.qkv", a.qkv.w, a.qkv.grad
        yield f"attn{i}.out", a.out_w.w, a.out_w.grad
        yield f"attn{i}.ln1", a.ln1.g, a.ln1.grad
        if getattr(a.ln1, "b", None) is not None:
            yield f"attn{i}.ln1.b", a.ln1.b, a.ln1.bgrad
    for i, f in enumerate(model.ffn):
        yield f"ffn{i}.wg", f.wg.w, f.wg.grad
        if getattr(f, "wu", None) is not None:
            yield f"ffn{i}.wu", f.wu.w, f.wu.grad
        yield f"ffn{i}.wd", f.wd.w, f.wd.grad
        yield f"ffn{i}.ln2", f.ln2.g, f.ln2.grad
        if getattr(f.ln2, "b", None) is not None:
            yield f"ffn{i}.ln2.b", f.ln2.b, f.ln2.bgrad
    yield "ln_f", model.ln_f.g, model.ln_f.grad
    if getattr(model.ln_f, "b", None) is not None:
        yield "ln_f.b", model.ln_f.b, model.ln_f.bgrad