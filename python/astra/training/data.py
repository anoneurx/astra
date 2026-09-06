"""Deterministic corpus -> token-array -> batch stream.

Pipeline per docs/TRAINING.md § 2: tokenize -> deterministic shuffle -> batches
of contiguous context windows. Val split is carved from the same manifest but
frozen and never shuffled with training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astra.model.config import ModelConfig


@dataclass
class Corpus:
    ids: np.ndarray          # int32
    manifest: dict           # hashes etc.

    def __len__(self) -> int:
        return int(self.ids.size)


def tokenize_corpus(text: str, tokenizer, cfg: ModelConfig) -> Corpus:
    ids = np.array(tokenizer.encode(text), dtype=np.int32)
    cap = len(ids) - (len(ids) % cfg.max_seq_len)
    return Corpus(ids=ids[:cap], manifest={"tokenizer": "bytebpe-v0", "tokens": int(cap)})


class SeqStream:
    """Streams overlapping-free context windows from token ids.

    Deterministic: given the same rng state, order is identical.
    """

    def __init__(
        self,
        corpus: Corpus,
        batch_seq: int,
        seq_len: int,
        rng: np.random.Generator,
        epoch: int = 0,
    ):
        self.ids = corpus.ids
        self.batch_seq = batch_seq
        self.seq_len = seq_len
        self.rng = rng
        n_windows = (len(self.ids) - 1) // seq_len
        self.order = np.arange(n_windows)
        self._shuffle(epoch)

    def _shuffle(self, epoch: int) -> None:
        # epoch-scoped but deterministic: use a local generator seeded per epoch
        r = np.random.default_rng(epoch + 1_000_003)
        r.shuffle(self.order)

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pos >= len(self.order):
            raise StopIteration
        end = min(self._pos + self.batch_seq, len(self.order))
        idx = self.order[self._pos : end]
        self._pos = end
        # assemble (B, T) input windows + (B, T) shifted targets
        starts = idx * self.seq_len
        x = np.stack([self.ids[s : s + self.seq_len] for s in starts])
        y = np.stack([self.ids[s + 1 : s + self.seq_len + 1] for s in starts])
        return x, y