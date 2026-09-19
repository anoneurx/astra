"""Word-level tokenizer (whole English words as tokens, from scratch).

Motivation (docs/TOKENIZER.md rationale for a word-level variant):
the byte-level BPE shatters English words into pieces like ``ed``, ``ing``,
``stra``, so a 6M-param LM at ppl ~2500 emits fragments instead of words.
A word-level tokenizer maps each common word to ONE token id, which gives a
small network a far stronger signal per step and, crucially, clean decoded
text (whole words, no mid-word byte fragments).

Design:
  * special tokens first (pad/bos/eos/unk), ids 0..3
  * one token per whole word, one token per space char, one per newline,
    one for each observed punctuation/other char
  * exact byte round-trip: decode(encode(s)) == s for vocab-covered text;
    out-of-vocab units map to <unk>

Interface matches astra.tokenizer.bpe.ByteLevelBPE (encode/encode_batch/decode/
__len__/save/load/quality_report), so training and inference work unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_WORD_RE = re.compile(r"[A-Za-z0-9']+| |\n|.")


@dataclass
class WordLevel:
    vocab_size: int
    num_special: int = 4  # pad, bos, eos, unk
    min_frequency: int = 2
    special_names: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

    # maps token id -> bytes piece (utf-8 of a word/char/special)
    id_to_piece: dict[int, bytes] = field(default_factory=dict)
    # maps bytes piece -> token id (words and single characters)
    piece_to_id: dict[bytes, int] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id_to_piece and not self.piece_to_id:
            self._make_special_vocab()

    def _make_special_vocab(self) -> None:
        for i, name in enumerate(self.special_names):
            self.id_to_piece[i] = name.encode("utf-8")
            self.piece_to_id[name.encode("utf-8")] = i

    # ----- training (vocabulary selection) -----

    def train(
        self,
        corpus: str,
        max_vocab: int | None = None,
        min_frequency: int | None = None,
    ) -> WordLevel:
        """Build the vocab from *corpus*: every observed char plus the most
        frequent whole words, up to ``vocab_size`` total tokens."""
        if min_frequency is not None:
            self.min_frequency = min_frequency
        cap = max_vocab or self.vocab_size
        counts: Counter[bytes] = Counter()
        total = 0
        for unit in _WORD_RE.findall(corpus):
            total += 1
            if _is_word(unit):
                counts[unit.encode("utf-8")] += 1
            else:
                # keep every distinct char (space, newline, punctuation, unicode)
                counts[unit.encode("utf-8")] += 1

        # chars (non-words) always included, id order: space, newline, then sorted
        chars = [k for k in counts if not _is_word(k.decode("utf-8", "replace"))]
        chars.sort()

        words = [(p, c) for p, c in counts.items() if _is_word(p.decode("utf-8", "replace"))]
        words.sort(key=lambda kv: (-kv[1], kv[0]))  # most frequent first, tie by bytes
        words = [(p, c) for p, c in words if c >= self.min_frequency]

        budget = cap - self.num_special - len(chars)
        if budget <= 0:
            raise ValueError(f"vocab_size={cap} too small for {len(chars)} chars")

        selected = [(p, min(c, budget)) for p, c in words]
        selected = selected[:budget]

        self.id_to_piece = {
            i: self.special_names[i].encode("utf-8") for i in range(self.num_special)
        }
        self.piece_to_id = {
            self.special_names[i].encode("utf-8"): i for i in range(self.num_special)
        }
        tid = self.num_special
        for p in chars:
            self.id_to_piece[tid] = p
            self.piece_to_id[p] = tid
            tid += 1
        for p, _c in selected:
            self.id_to_piece[tid] = p
            self.piece_to_id[p] = tid
            tid += 1

        covered = sum(c for p, c in counts.items() if p in self.piece_to_id)
        self.stats.update(
            {
                "vocab_size": len(self.id_to_piece),
                "num_words": len(selected),
                "num_chars": len(chars),
                "tokens_seen": total,
                "coverage": float(covered) / max(1, total),
                "min_frequency": self.min_frequency,
            }
        )
        return self

    # ----- encoding / decoding -----

    def encode(self, text: str) -> list[int]:
        return [self.piece_to_id.get(unit.encode("utf-8"), self._unk_id()) for unit in _WORD_RE.findall(text)]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_piece.get(i, b"?") for i in ids).decode(
            "utf-8", errors="replace"
        )

    def _unk_id(self) -> int:
        name = self.special_names.index("<unk>")
        return name if 0 <= name < self.num_special else 0

    def __len__(self) -> int:
        return len(self.id_to_piece)

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class": "WordLevel",
            "vocab_size": self.vocab_size,
            "num_special": self.num_special,
            "min_frequency": self.min_frequency,
            "special_names": list(self.special_names),
            "tokens": [
                [str(tid), piece.decode("utf-8", errors="replace")] for tid, piece in sorted(self.id_to_piece.items())
            ],
            "stats": self.stats,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> WordLevel:
        payload = json.loads(Path(path).read_text())
        tok = cls(
            vocab_size=payload["vocab_size"],
            num_special=payload["num_special"],
            min_frequency=payload["min_frequency"],
            special_names=tuple(payload["special_names"]),
        )
        tok.id_to_piece = {}
        tok.piece_to_id = {}
        for tid, piece in payload["tokens"]:
            pb = piece.encode("utf-8")
            tok.id_to_piece[int(tid)] = pb
            tok.piece_to_id[pb] = int(tid)
        tok.stats = payload.get("stats", {})
        return tok

    # ----- statistics -----

    def quality_report(self, corpus: str) -> dict:
        n = len(corpus)
        sample = corpus[: 1 << 20]
        report = {
            "vocab_size": len(self.id_to_piece),
            "bytes_total": n,
            "tokens_total": len(self.encode(corpus)),
            "bytes_per_token": float(n) / max(1, len(self.encode(corpus))),
            "tokens_per_character": float(len(self.encode(corpus))) / max(1, n),
            "sample_bytes": len(sample),
            "sample_tokens": len(self.encode(sample)),
            "words": self.stats.get("num_words", 0),
            "coverage": self.stats.get("coverage", 0.0),
            "unk_chars_in_sample": sample.count("<unk>"),
        }
        self.stats["report"] = report
        return report


def _is_word(unit: str) -> bool:
    if not unit or unit == " " or unit == "\n":
        return False
    return all(
        ch.isalnum() or ch == "'" for ch in unit
    ) and any(ch.isalnum() for ch in unit)