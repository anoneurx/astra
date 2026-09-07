"""Incremental (KV-cache) autoregressive decoder for LiteLM.

Feed tokens one at a time and reuse cached K/V states instead of recomputing
the whole prefix, and cap the attended window with an optional sliding-window
(can decode past ``max_seq_len`` by growing the cache). Verified against the
recompute-everything reference sampler (evaluation/metrics.generate).
"""

from __future__ import annotations

import numpy as np

from astra.model.config import ModelConfig
from astra.model.core import LiteLM, gelu, silu


class KVCache:
    """Per-layer key/value states used by decode_token.

    Grows on demand so decoding may exceed max_seq_len (context handling).
    ``seen`` = number of stored tokens.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.seen = 0
        self._k: list[np.ndarray | None] = [None] * cfg.n_layers
        self._v: list[np.ndarray | None] = [None] * cfg.n_layers

    def _ensure(self, layer: int, needed: int) -> None:
        if self._k[layer] is not None and self._k[layer].shape[2] >= needed:
            return
        B, H, dh = 1, self.cfg.n_heads, self.cfg.d_head
        cur = self._k[layer].shape[2] if self._k[layer] is not None else max(self.cfg.max_seq_len, needed)
        slots = max(2 * cur, needed)
        k = np.zeros((B, H, slots, dh), dtype=np.float32)
        v = np.zeros((B, H, slots, dh), dtype=np.float32)
        if self._k[layer] is not None:
            k[:, :, : self._k[layer].shape[2], :] = self._k[layer]
            v[:, :, : self._v[layer].shape[2], :] = self._v[layer]
        self._k[layer], self._v[layer] = k, v

    def store(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        self._ensure(layer, self.seen + 1)
        self._k[layer][:, :, self.seen, :] = k[:, :, 0, :]
        self._v[layer][:, :, self.seen, :] = v[:, :, 0, :]


def _headify(x: np.ndarray, heads: int, dh: int) -> np.ndarray:
    B, T, _d = x.shape
    return x.reshape(B, T, heads, dh).transpose(0, 2, 1, 3)


def _rope_at(cfg: ModelConfig, pos: int) -> tuple[np.ndarray, np.ndarray]:
    dh = cfg.d_head
    freqs = pos / (cfg.rope_theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _rotate(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    d2 = np.arange(0, x.shape[-1], 2)
    even, odd = x[..., d2], x[..., d2 + 1]
    out = np.empty_like(x)
    out[..., d2] = even * c - odd * s
    out[..., d2 + 1] = even * s + odd * c
    return out


def decode_token(model: LiteLM, cache: KVCache, token: np.ndarray, window: int = 0) -> np.ndarray:
    """Process one new token; return (B, V) logits for the next-token sample.

    Args:
        model: trained LiteLM (no grads used).
        cache: mutable KVCache; updated in place.
        token: (1,) int64 array of the new token id.
        window: >0 cap on attended keys (sliding window), 0 = full context.
    """
    cfg = model.cfg
    pos = cache.seen
    x = model.wte.forward(token.reshape(1, 1))
    if cfg.pos_type == "learned":
        x = x + model.pos_emb.forward(np.array([[pos]], dtype=np.int64))

    c, s = _rope_at(cfg, pos)
    for li, (attn, ffn) in enumerate(zip(model.attn, model.ffn)):
        h = attn.ln1.forward(x)
        q, k, v = np.split(attn.qkv.forward(h), 3, axis=-1)
        q = _headify(q, cfg.n_heads, cfg.d_head)
        k = _headify(k, cfg.n_heads, cfg.d_head)
        v = _headify(v, cfg.n_heads, cfg.d_head)
        q = _rotate(q, c[None, None], s[None, None])
        k = _rotate(k, c[None, None], s[None, None])
        

        cache.store(li, k, v)
        lo = max(0, cache.seen + 1 - window) if window > 0 else 0
        K = cache._k[li][:, :, lo : cache.seen + 1, :]
        V = cache._v[li][:, :, lo : cache.seen + 1, :]
        scores = (q @ K.transpose(0, 1, 3, 2)) * (cfg.d_head**-0.5)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        z = p @ V
        x = x + attn.out_w.forward(z.reshape(1, 1, cfg.d_model))

        hf = ffn.ln2.forward(x)
        if ffn.ffn_type == "swiglu":
            ff_in = silu(hf @ ffn.wg.w) * (hf @ ffn.wu.w)
        else:
            ff_in = gelu(hf @ ffn.wg.w)
        x = x + (ff_in @ ffn.wd.w)

    cache.seen += 1
    ln = model.ln_f.forward(x)
    return ln @ model.wte.w.T  # tied output head


def decode(
    model: LiteLM,
    seed_ids: list[int],
    max_new: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
    top_k: int = 0,
    top_p: float = 1.0,
    window: int = 0,
    cache: KVCache | None = None,
) -> list[int]:
    """Sample ``max_new`` tokens after ``seed_ids`` using decode_token.

    Returns the generated tokens (without the seed). ``top_k`` (0 = off) and
    ``top_p`` (1.0 = off) are applied together with temperature before the
    final multinomial draw. Reuse ``cache`` across calls to continue a dialog.
    """
    rng = rng or np.random.default_rng(0)
    cache = cache or KVCache(model.cfg)
    v = model.cfg.vocab_size
    out: list[int] = []
    for tok in seed_ids:
        logits = decode_token(model, cache, np.array([tok], dtype=np.int64), window=window)
    for _ in range(max_new):
        lp = logits[0, 0] / temperature
        lp = lp - lp.max()
        p = np.exp(lp)
        p = p / p.sum()
        if top_k > 0:
            k = min(top_k, v)
            keep = np.zeros(v, dtype=bool)
            keep[np.argsort(p)[-k:]] = True
            p = np.where(keep, p, 0.0)
            p = p / p.sum()
        if top_p < 1.0:
            order = np.argsort(p)[::-1]
            cp = np.cumsum(p[order])
            cutoff = cp > top_p
            cutoff[1:] = cutoff[1:] | cutoff[:-1]
            p[order[~cutoff]] = 0.0
            p = p / p.sum()
        nxt = int(rng.choice(v, p=p))
        out.append(nxt)
        logits = decode_token(model, cache, np.array([nxt], dtype=np.int64), window=window)
    return out