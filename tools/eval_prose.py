#!/usr/bin/env python3
"""Sample a trained prose model and score spelling/sentence quality.

Usage:
    python tools/eval_prose.py --config configs/astra5m_prose.json \
        --checkpoint checkpoints/astra5m_prose/final.npz \
        --tokenizer tokenizer/artifacts/prose_bpe.json \
        --out experiments/phase7/prose_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import decode
from astra.learning.evaluate import load_model
from astra.model import ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus
from astra.training.loop import val_loss
from astra.utils import read_json

SENT_END = re.compile(r"[.!?][\"')\]]*$")
WORD_RE = re.compile(r"[A-Za-z']+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    raw = read_json(args.config)
    tok = ByteLevelBPE.load(args.tokenizer)
    cfg = ModelConfig.from_dict(raw["model"])
    cfg.vocab_size = len(tok)

    model = load_model(cfg, args.checkpoint)
    val_ids = np.asarray(tok.encode(Path(raw["data"]["val"]).read_text(encoding="utf-8")), dtype=np.int32)
    val_words = set(WORD_RE.findall(Path(raw["data"]["val"]).read_text(encoding="utf-8").lower()))

    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, max(1, len(val_ids) - rng.integers(8, 64)), size=args.samples)

    decoded_words: list[str] = []
    ends = 0
    for s in starts:
        ctx = val_ids[int(s) : int(s) + 8]
        new = decode(model, ctx, max_new=args.max_new, temperature=args.temperature, rng=rng)
        text = tok.decode(new)
        decoded_words += WORD_RE.findall(text.lower())
        if SENT_END.search(text.strip()):
            ends += 1

    total = len(decoded_words)
    report = {
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "max_new": args.max_new,
        "temperature": args.temperature,
        "word_like": sum(w in val_words for w in decoded_words) / max(1, total),
        "sentence_end": ends / args.samples,
        "avg_word_len": sum(len(w) for w in decoded_words) / max(1, total),
        "decoded_words": total,
    }

    val_corpus = Corpus(ids=val_ids, manifest={"name": "prose-val", "kind": "validation"})
    v = val_loss(model, val_corpus, cfg, n_shards=1)
    report["val_ce"] = v["loss"]
    report["val_ppl"] = v["ppl"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()