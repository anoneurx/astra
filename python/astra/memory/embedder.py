"""Embedding layer for the memory engine (docs/MEMORY.md § 5).

Production default is ``LiteLMExtractor``: the Astra encoder (last-layer
activations, mean-pooled over the sequence, L2-normalized). ``HashEmbedder``
is a deterministic, model-free implementation used for tests and engine-only
evaluation; alternatives plug in behind the same ``Embedder`` protocol.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from astra.model.config import ModelConfig
from astra.model.core import LiteLM
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json, stable_seed

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` → float32 array (n, dim), rows L2-normalized."""

    @property
    @abstractmethod
    def config(self) -> dict:
        """Immutable enough to detect embedding-version drift (re-index)."""


class LiteLMExtractor(Embedder):
    """Mean-pooled last-layer activations of the trained Astra encoder."""

    def __init__(self, model: LiteLM, tokenizer: ByteLevelBPE, max_len: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len or model.cfg.max_seq_len

    @property
    def dim(self) -> int:
        return self.model.cfg.d_model

    def embed(self, texts: list[str]) -> np.ndarray:
        cfg: ModelConfig = self.model.cfg
        pad = cfg.vocab_size - 1          # internal pad slot (avoid special-id layout)
        high = cfg.vocab_size - 2         # content ids clamped below the pad slot
        vectors: list[np.ndarray] = []
        for text in texts:
            ids = self.tokenizer.encode(text)[: self.max_len]
            ids = [min(i, high) for i in ids]  # clamp out-of-vocab
            ids = ids + [pad] * (self.max_len - len(ids))
            arr = np.asarray([ids], dtype=np.int64)
            _logits, h = self.model.forward(arr, hidden=True)
            mask = np.asarray(ids, dtype=np.float32) != pad
            vec = (h[0] * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
            n = float(np.linalg.norm(vec))
            vectors.append(vec / n if n > 0 else np.zeros(cfg.d_model, dtype=np.float32))
        return np.stack(vectors) if vectors else np.zeros((0, cfg.d_model), dtype=np.float32)

    @property
    def config(self) -> dict:
        return {
            "name": "LiteLMExtractor",
            "d_model": self.model.cfg.d_model,
            "max_len": self.max_len,
            "pos_type": self.model.cfg.pos_type,
        }

    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: str) -> LiteLMExtractor:
        raw = read_json(config_path)
        tok = ByteLevelBPE.load(raw["tokenizer"])
        cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})
        model = LiteLM(cfg, seed=0)
        load_checkpoint(checkpoint_path, model, opt=None, schedule=None)
        return cls(model, tok)


class HashEmbedder(Embedder):
    """Deterministic model-free embedding (tests / engine-only retrieval eval).

    Signed bag-of-unigrams over lowercased words, hashed into a fixed dim.
    Deterministic given the seed; NOT the production encoder.
    """

    def __init__(self, dim: int = 64, seed: int = 1):
        self.dim = dim
        self._seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for term in _WORD.findall(text.lower()):
                idx = stable_seed(term) % self.dim
                sign = 1.0 if (stable_seed(term + "$") & 1) else -1.0
                v[idx] += sign
            n = float(np.linalg.norm(v))
            vecs.append(v / n if n > 0 else v)
        return np.stack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    @property
    def config(self) -> dict:
        return {"name": "HashEmbedder", "dim": self.dim, "seed": self._seed}