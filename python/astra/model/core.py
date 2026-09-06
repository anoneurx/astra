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


# ------------------------------------------------------------ operators


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_prime(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


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
        inv = 1.0 / (theta ** (np.arange(0, head_dim - 1, 2) / head_dim))
        pos = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(pos, inv)  # (T, head_dim//2)
        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)

    def _angles(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.cos[:t][None, None], self.sin[:t][None, None]

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
        self.ln1 = RmsNorm(d, cfg.eps)
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
        scores = scores + self.causal[:T, :T][None, None]
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
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, layer_idx: int):
        d, ff = cfg.d_model, cfg.d_ffn
        s = 1.0 / math.sqrt(layer_idx + 1)
        self.ln2 = RmsNorm(d, cfg.eps)
        self.wg = Linear(d, ff, rng, scale=s)
        self.wu = Linear(d, ff, rng, scale=s)
        self.wd = Linear(ff, d, rng, scale=s)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.ln2.forward(x)
        gate = silu(h @ self.wg.w)
        up = h @ self.wu.w
        self._cache = (h, gate, up, x)
        return x + ((gate * up) @ self.wd.w)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        h, gate, up, xm = self._cache
        d_ff = self.wd.backward(grad, gate * up)
        gate_in = h @ self.wg.w
        d_gate_h = self.wg.backward(d_ff * up * silu_prime(gate_in), h)
        d_up_h = self.wu.backward(d_ff * gate, h)
        return self.ln2.backward(d_gate_h + d_up_h, xm)


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
        self.attn = [AttentionBlock(cfg, rng, i) for i in range(cfg.n_layers)]
        self.ffn = [SwiGLUBLock(cfg, rng, i) for i in range(cfg.n_layers)]
        self.ln_f = RmsNorm(cfg.d_model, cfg.eps)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        x = self.wte.forward(ids)
        for a, f in zip(self.attn, self.ffn):
            x = a.forward(x)
            x = f.forward(x)
        self._final_act = x
        self._ln_out = self.ln_f.forward(x)
        return self._ln_out @ self.wte.w.T  # tied output head

    def zero_grad(self) -> None:
        self.wte.grad[:] = 0
        self.ln_f.grad[:] = 0
        for a, f in zip(self.attn, self.ffn):
            a.qkv.grad[:] = 0
            a.out_w.grad[:] = 0
            a.ln1.grad[:] = 0
            f.wg.grad[:] = 0
            f.wu.grad[:] = 0
            f.wd.grad[:] = 0
            f.ln2.grad[:] = 0

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
        return loss

    @property
    def num_params(self) -> int:
        n = self.wte.w.size
        for a, f in zip(self.attn, self.ffn):
            n += a.qkv.w.size + a.out_w.w.size + a.ln1.g.size
            n += f.wg.w.size + f.wu.w.size + f.wd.w.size + f.ln2.g.size
        return n + self.ln_f.g.size


def mean_cross_entropy(logp: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.take_along_axis(logp, targets[..., None], axis=-1).mean())


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def all_params(model: LiteLM) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (name, weights, grads) for every trainable parameter."""
    for name in ("wte",):
        w = getattr(model, name)
        yield name, w.w, w.grad
    for i, a in enumerate(model.attn):
        yield f"attn{i}.qkv", a.qkv.w, a.qkv.grad
        yield f"attn{i}.out", a.out_w.w, a.out_w.grad
        yield f"attn{i}.ln1", a.ln1.g, a.ln1.grad
    for i, f in enumerate(model.ffn):
        yield f"ffn{i}.wg", f.wg.w, f.wg.grad
        yield f"ffn{i}.wu", f.wu.w, f.wu.grad
        yield f"ffn{i}.wd", f.wd.w, f.wd.grad
        yield f"ffn{i}.ln2", f.ln2.g, f.ln2.grad
    yield "ln_f", model.ln_f.g, model.ln_f.grad