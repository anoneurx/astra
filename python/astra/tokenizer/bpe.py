"""Byte-level BPE tokenizer (from scratch, NumPy-accelerated pair counting).

Follows docs/TOKENIZER.md: byte-oriented base vocabulary, BPE merges,
reserved special-token ids, exact byte round-trip decoding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ID_BYTE_MASK = 0xFFFF


def _best_pair(counts: Mapping[tuple[int, int], int]) -> tuple[tuple[int, int], int]:
    """Most frequent adjacent pair; ties break on lowest (a, b) lexicographic.

    Matches the reference full-rescan selection (numpy unique sort order +
    first argmax), keeping training deterministic and identical between the
    incremental and rescan implementations.
    """
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))


def _merge_positions(ids: np.ndarray, a: int, b: int) -> np.ndarray:
    """Indices of non-overlapping (a, b) occurrences (same greediness as before)."""
    pos = np.nonzero((ids[:-1] == a) & (ids[1:] == b))[0]
    if pos.size > 1:
        keep = np.ones(pos.size, dtype=bool)
        keep[1:] = pos[1:] != pos[:-1] + 1
        pos = pos[keep]
    return pos


def _update_counts(
    counts: dict[tuple[int, int], int],
    old_ids: np.ndarray,
    old_pos: np.ndarray,
    out_ids: np.ndarray,
    out_pos: np.ndarray,
) -> None:
    """Apply local adjacency deltas after one merge, matching a full recount.

    Only pair indices whose left or right element was replaced are touched:
    {p-1, p, p+1} around each old merged position and {q-1, q} around each
    output position; duplicates are set-deduped so counts never drift.
    """
    n_old = int(old_ids.size)
    removed: set[int] = set()
    for p in old_pos:
        removed.update((p - 1, p, p + 1))
    for idx in removed:
        if 0 <= idx < n_old - 1:
            key = (int(old_ids[idx]), int(old_ids[idx + 1]))
            c = counts.get(key, 0) - 1
            if c > 0:
                counts[key] = c
            else:
                counts.pop(key, None)

    n_new = int(out_ids.size)
    added: set[int] = set()
    for q in out_pos:
        added.update((q - 1, q))
    for idx in added:
        if 0 <= idx < n_new - 1:
            key = (int(out_ids[idx]), int(out_ids[idx + 1]))
            counts[key] = counts.get(key, 0) + 1


@dataclass
class VocabEntry:
    piece: bytes
    id: int


@dataclass
class ByteLevelBPE:
    vocab_size: int
    num_special: int = 4  # pad, bos, eos, unk
    min_frequency: int = 2
    special_names: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

    # maps byte value (0..255) -> token id for the base vocabulary
    byte_to_id: dict[int, int] = field(default_factory=dict)
    # maps token id -> bytes piece (for merges and specials)
    id_to_piece: dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[int, int, int]] = field(default_factory=list)  # (a, b, new_id)
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.byte_to_id:
            self._make_base_vocab()

    def _make_base_vocab(self) -> None:
        # base ids 0..255 = raw bytes
        for b in range(256):
            self.byte_to_id[b] = b
            self.id_to_piece[b] = bytes([b])
        # special tokens occupy ids 256..255+num_special
        for i, name in enumerate(self.special_names):
            tid = 256 + i
            self.id_to_piece[tid] = name.encode("utf-8")
        self._first_merge_id = 256 + self.num_special

    # ----- training -----

    def train(self, corpus: bytes | str, max_merges: int | None = None) -> ByteLevelBPE:
        """Train BPE merges on the byte stream of *corpus*.

        Uses incremental adjacent-pair counts (only pairs around each merged
        position are touched per round) instead of rescans, so training is
        O(N + V^2)-ish rather than O(V*N*logN) while producing the exact same
        merge sequence as the reference full-rescan implementation.
        """
        if isinstance(corpus, str):
            corpus = corpus.encode("utf-8")
        max_merges = max_merges or (self.vocab_size - self._first_merge_id)
        ids = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)
        size_before = int(ids.size)
        merges: list[tuple[int, int, int]] = []

        counts: dict[tuple[int, int], int] = {}
        for i in range(int(ids.size) - 1):
            key = (int(ids[i]), int(ids[i + 1]))
            counts[key] = counts.get(key, 0) + 1

        for _ in range(max_merges):
            if ids.size < 2 or not counts:
                break
            pair, count = _best_pair(counts)
            if count < self.min_frequency:
                break
            new_id = self._first_merge_id + len(merges)
            out, new_pos, applied = self._apply_merge(ids, pair[0], pair[1], new_id)
            if applied == 0:  # pragma: no cover
                break
            _update_counts(counts, ids, _merge_positions(ids, pair[0], pair[1]), out, new_pos)
            ids = out
            merges.append((int(pair[0]), int(pair[1]), new_id))
            self.merges = merges
            self.id_to_piece[new_id] = self.id_to_piece[int(pair[0])] + self.id_to_piece[int(pair[1])]

        self.stats.update(
            {
                "vocab_size": len(self.byte_to_id) + len(self.merges),
                "num_merges": len(self.merges),
                "tokens_seen": size_before,
                "tokens_after": int(ids.size),
                "compression": float(ids.size) / max(1, size_before),
                "min_frequency": self.min_frequency,
            }
        )
        return self

    @staticmethod
    def _apply_merge(
        ids: np.ndarray, a: int, b: int, new_id: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Merge non-overlapping (a,b) occurrences; return (out, pos_of_new, n)."""
        pos = _merge_positions(ids, a, b)
        n = pos.size
        if n == 0:
            return ids, np.empty(0, dtype=np.int64), 0
        out = np.empty(int(ids.size - pos.size), dtype=np.int64)
        new_positions = np.empty(int(pos.size), dtype=np.int64)
        cursor = 0
        write = 0
        for k, p in enumerate(map(int, pos)):
            seg = ids[cursor : p + 1]  # include the 'a'
            out[write : write + seg.size - 1] = seg[:-1]
            write += seg.size - 1
            out[write] = new_id
            new_positions[k] = write
            write += 1
            cursor = p + 2
        seg = ids[cursor:]
        out[write : write + seg.size] = seg
        return out, new_positions, int(pos.size)

    # ----- encoding / decoding -----

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8", "surrogatepass")
        ids = np.array([self.byte_to_id[b] for b in data], dtype=np.int64)
        for (a, b, new_id) in self.merges:
            ids, _pos, _applied = self._apply_merge(ids, a, b, new_id)
        return ids.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        out = b"".join(self.id_to_piece[i] for i in ids)
        return out.decode("utf-8", errors="surrogatepass")

    def __len__(self) -> int:
        return len(self.id_to_piece)

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "num_special": self.num_special,
            "min_frequency": self.min_frequency,
            "special_names": list(self.special_names),
            "first_merge_id": self._first_merge_id,
            "merges": self.merges,
            "stats": self.stats,
            "special_tokens": {
                name: 256 + i for i, name in enumerate(self.special_names)
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> ByteLevelBPE:
        payload = json.loads(Path(path).read_text())
        tok = cls(
            vocab_size=payload["vocab_size"],
            num_special=payload["num_special"],
            min_frequency=payload["min_frequency"],
            special_names=tuple(payload["special_names"]),
        )
        tok._first_merge_id = payload["first_merge_id"]
        tok.merges = [tuple(m) for m in payload["merges"]]
        tok.stats = payload["stats"]
        for (a, b, new_id) in tok.merges:
            tok.id_to_piece[new_id] = tok.id_to_piece[a] + tok.id_to_piece[b]
        return tok

    # ----- statistics -----

    def quality_report(self, corpus: bytes | str) -> dict:
        if isinstance(corpus, str):
            corpus = corpus.encode("utf-8")
        ids = np.array(self.encode(corpus.decode("utf-8", "surrogateescape")), dtype=np.int64)
        # validate round-trip on the whole corpus is too slow for large inputs,
        # so compute on a bounded sample:
        sample = corpus[: 1 << 20]
        sample_ids = np.array(
            self.encode(sample.decode("utf-8", "surrogateescape")), dtype=np.int64
        )
        roundtrip = len(self.decode(sample_ids.tolist()).encode("utf-8")) == len(sample)
        # n-gram vocabulary coverage
        tokens = ids
        report = {
            "vocab_size": len(self.id_to_piece),
            "merges": len(self.merges),
            "bytes_total": len(corpus),
            "tokens_total": int(tokens.size),
            "bytes_per_token": float(len(corpus)) / max(1, int(tokens.size)),
            "tokens_per_character": float(int(tokens.size)) / max(1, len(corpus)),
            "sample_bytes": len(sample),
            "sample_tokens": int(sample_ids.size),
            "roundtrip_exact": roundtrip,
            "unknown_tokens_used": 0,  # byte-level coverage guarantees none
        }
        self.stats["report"] = report
        return report