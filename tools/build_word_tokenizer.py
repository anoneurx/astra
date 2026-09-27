"""Build the word-level tokenizer artifact over prose + chat corpora.

Usage: python tools/build_word_tokenizer.py [--extra PATH ...]
Output: tokenizer/artifacts/prose_chat_word.json

`--extra` folds additional corpora into the training text. Needed for any corpus the artifact
will be used on: the default chat+prose vocab leaves 16.02% of the distilled security
vocabulary as `<unk>`, because words like "exfil-check" or "serviceaccount" appear nowhere in
the prose/chat mix. Training a vocab only on a small extra corpus is the opposite mistake - it
would starve the corpora the artifact was originally built for - so extras are added to the
union rather than replacing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.tokenizer.word import WordLevel

VOCAB_SIZE = 16384
ARTIFACT = Path("tokenizer/artifacts/prose_chat_word.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra corpus files to fold into the tokenizer's training text")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--out", default=str(ARTIFACT))
    args = ap.parse_args()

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
    for path in args.extra:
        extra = Path(path).read_text(encoding="utf-8")
        print(f"[word ] extra corpus {path}: {len(extra):,} chars")
        corpus += "\n" + extra

    out = Path(args.out)
    print(f"[word ] corpus chars: {len(corpus):,}")
    tok = WordLevel(vocab_size=args.vocab_size).train(corpus)
    qr = tok.quality_report(corpus)
    print(f"[word ] {len(tok)} tokens, {tok.stats['num_words']} words, "
          f"coverage={tok.stats['coverage']:.4f}")
    tok.save(out)
    print(f"[word ] artifact -> {out}")

    # round-trip sanity on the chat framing
    s = "You: Hello, my name is Astra.\nAstra: Nice to meet you!"
    assert tok.decode(tok.encode(s)) == s, "round-trip failed"
    print("[word ] round-trip OK:", repr(tok.decode(tok.encode(s))))
    print("[word ] quality_report:", {k: v for k, v in qr.items() if k != "sample"}, sep="\n  ")

    # OOV on each extra corpus, which is the number --extra exists to move
    for path in args.extra:
        text = Path(path).read_text(encoding="utf-8")
        ids = tok.encode(text)
        unk = tok.special_names.index("<unk>")
        oov = sum(1 for i in ids if i == unk)
        print(f"[word ] OOV on {path}: {100.0 * oov / max(1, len(ids)):.2f}%")


if __name__ == "__main__":
    main()
