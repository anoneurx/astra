"""Build the word-level tokenizer artifact over prose + chat corpora.

Usage: python tools/build_word_tokenizer.py
Output: tokenizer/artifacts/prose_chat_word.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.tokenizer.word import WordLevel

VOCAB_SIZE = 16384
ARTIFACT = Path("tokenizer/artifacts/prose_chat_word.json")


def main() -> None:
    prose = (
        Path("datasets/prose/train.txt").read_text(encoding="utf-8")
        + "\n"
        + Path("datasets/prose/val.txt").read_text(encoding="utf-8")
    )
    chat = (
        Path("datasets/chat/train.txt").read_text(encoding="utf-8")
        + "\n"
        + Path("datasets/chat/val.txt").read_text(encoding="utf-8")
    )
    corpus = chat + "\n" + prose

    print(f"[word ] corpus chars: {len(corpus):,}")
    tok = WordLevel(vocab_size=VOCAB_SIZE).train(corpus)
    qr = tok.quality_report(corpus)
    print(f"[word ] {len(tok)} tokens, {tok.stats['num_words']} words, "
          f"coverage={tok.stats['coverage']:.4f}")
    tok.save(ARTIFACT)
    print(f"[word ] artifact -> {ARTIFACT}")

    # round-trip sanity on the chat framing
    s = "You: Hello, my name is Astra.\nAstra: Nice to meet you!"
    assert tok.decode(tok.encode(s)) == s, "round-trip failed"
    print("[word ] round-trip OK:", repr(tok.decode(tok.encode(s))))
    print("[word ] quality_report:", {k: v for k, v in qr.items() if k != "sample"}, sep="\n  ")


if __name__ == "__main__":
    main()