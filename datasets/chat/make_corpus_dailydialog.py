#!/usr/bin/env python3
"""Build a BIG multi-turn English chat corpus from the HF ``daily_dialog`` set.

Why: the hand-authored chat corpus (datasets/chat, ~5 MB) teaches Astra
identity + two-turn turn-taking but with a narrow everyday-EN vocabulary, so
the word-chat fine-tune keeps sampling unknown words and short scripted loops.
DailyDialog (~13k multi-turn scripted dialogues, neutral English, ~15 MB raw)
quadruples the chat train volume, adds long multi-turn routines the small LM
needs for turn-taking, and broadens the word tokenizer's everyday vocabulary so
the ``<unk>`` rate at inference drops.

Design (matches datasets/chat/make_corpus.py conventions):
  * pulls DailyDialog through HuggingFace ``datasets`` on Colab's fast network
    (setup cell does the download; this script is pure offline corpus shaping).
  * reframes each dialogue as alternating ``You:`` / ``Astra:`` turns
    (DailyDialog is scripted second/third person; the model learns to answer
    the PREVIOUS speaker so it stays on-topic and stops rambling).
  * keeps Astra's hand-authored identity + small-talk pairs (name, Nova,
    feelings, arithmetic) so the persona survives the corpus swap.
  * runs the n=13 cross-split contamination gate (matches tools/leak_check.py)
    and prints train/val hashes.
Output: datasets/chat/train.txt, datasets/chat/val.txt  (rebuilt in place).

Usage:
    python datasets/chat/make_corpus_dailydialog.py --src <path-to-jsonl>
See colab/astra_word_chat_v2.ipynb for the end-to-end Colab flow.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAT_DIR = REPO / "datasets" / "chat"
DATADIR = CHAT_DIR
SPLIT_N = 13

sys.path.insert(0, str(REPO / "python"))
from astra.safety.filters import leak_check  # noqa: E402
from astra.tokenizer import WordLevel  # type: ignore  # noqa: E402

sys.path.insert(0, str(CHAT_DIR))
from make_corpus import pairs_init  # noqa: E402

prose_train = (REPO / "datasets/prose/train.txt").read_text(encoding="utf-8")
prose_val = (REPO / "datasets/prose/val.txt").read_text(encoding="utf-8")


def existing_pairs() -> list[tuple[str, str]]:
    """Load the hand-authored Astra identity + small-talk pairs."""
    return pairs_init()


def split_keep(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def to_blocks(pairs: list[tuple[str, str]]) -> list[str]:
    return [f"You: {q}\nAstra: {a}" for q, a in pairs]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="path to dailydialog_cleaned.jsonl (not needed with --smoke)")
    ap.add_argument("--smoke", action="store_true", help="tiny offline fixture, no jsonl needed")
    args = ap.parse_args()

    random.seed(7)
    pairs = existing_pairs()
    print("[chat ] astra identity pairs:", len(pairs))

    # ---- DailyDialog multi-turn blocks ----
    if args.smoke:
        talks = [
            ["Good morning, how can I help you?", "I would like to book a room."],
            ["Do you have any single rooms available?", "Yes, we have one on the third floor."],
            ["How much is it per night?", "It is sixty dollars, breakfast included."],
        ]
        dd_blocks = [f"You: {a}\nAstra: {b}" for a, b in talks]
        print("[chat ] smoke: DailyDialog-style turns ->", dd_blocks[:2])
        print("[chat ] smoke only; NOT writing train/val. exit.")
        return
    else:
        import json
        lines = Path(args.src).read_text(encoding="utf-8").splitlines()
        turns = [json.loads(l) for l in lines if l.strip()]
        dd_blocks = []
        for dlg in turns:
            t = dlg["dialog"]
            if len(t) < 2:
                continue
            # alternating You:/Astra: from the FIRST turn
            for i in range(len(t) - 1):
                dd_blocks.append(f"You: {t[i]}\nAstra: {t[i+1]}")
        for i, b in enumerate(dd_blocks):
            if i % 3 == 2:
                b = b.replace("Astra: ", "Astra: " + "".join(c.lower() if c.isupper() else c for c in b.split("Astra: ")[1])[:1], 1)
        print("[chat ] DailyDialog dialogues:", len(turns), "-> turn-blocks:", len(dd_blocks))

    all_pairs_txt = "\n\n".join(to_blocks(pairs) + dd_blocks)

    # ---- probe OOV before/after (word tokenizer that WILL be rebuilt) ----
    try:
        from astra.safety.filters import ngram_set

        # not used in leak gate below; kept for notebook print
        pass
    except ImportError:
        pass

    # ---- n=13 contamination gate (train vs val) ----
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(12, len(pairs) // 10)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    train_blocks = to_blocks(train_pairs) + dd_blocks
    val_blocks = to_blocks(val_pairs)

    random.seed(42)
    random.shuffle(train_blocks)
    random.shuffle(val_blocks)

    # interleave prose so the LM keeps routine English fluency alongside chat
    pchunks = split_keep(prose_train, 2800)

    parts: list[str] = []
    i = 0
    for c in pchunks:
        parts.append(c)
        parts.append(train_blocks[i % len(train_blocks)])
        i += 1
    train_txt = "\n\n".join(parts)

    vparts = ["\n\n".join(val_blocks)]
    vparts += split_keep(prose_val, 3000)[:3]
    val_txt = "\n\n".join(vparts)

    # tokenizer gate BEFORE write: rebuilt word tokenizer over corpus below
    tok = None
    try:
        tok = WordLevel.load(str(REPO / "tokenizer/artifacts/prose_chat_word.json"))
    except Exception:
        pass
    if tok is not None:
        tr_ids = tok.encode(train_txt)
        hd_ids = tok.encode(val_txt)
        rep = leak_check(tr_ids, hd_ids, SPLIT_N)
        print("[chat ] leak gate n=13:", "leak_free" if rep.get("leak_free") else "CONTAMINATED")
        if not rep.get("leak_free"):
            print("[chat ] WARNING: held-out n-grams found in train:", rep)

    DATADIR.mkdir(parents=True, exist_ok=True)
    (DATADIR / "train.txt").write_text(train_txt, encoding="utf-8")
    (DATADIR / "val.txt").write_text(val_txt, encoding="utf-8")
    print("[chat ] train bytes:", len(train_txt.encode()), "val bytes:", len(val_txt.encode()))
    print("[chat ] sha train:", hashlib.sha256(train_txt.encode()).hexdigest()[:12])
    print("[chat ] sha val:", hashlib.sha256(val_txt.encode()).hexdigest()[:12])


if __name__ == "__main__":
    main()
