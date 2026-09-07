#!/usr/bin/env python3
"""Interactive text generation for Astra 0.1.

Usage:
    python inference/generate.py
    python inference/generate.py --checkpoint checkpoints/phase0/final.npz
    python inference/generate.py --checkpoint checkpoints/phase0/final.npz --temperature 0.8

Type your prompt and press Enter. Astra responds. Type 'quit' or Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from astra.evaluation.metrics import generate
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Astra 0.1 text generation")
    ap.add_argument("--checkpoint", default="checkpoints/phase0/final.npz")
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = read_json(args.config)
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})

    model = LiteLM(cfg, seed=0)
    step, _hist, meta = load_checkpoint(args.checkpoint, model, opt=None, schedule=None)
    print(f"Astra 0.1 loaded (step {step}, {model.num_params} params)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}")
    print()

    rng = np.random.default_rng(args.seed)

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        seed_ids = tok.encode(prompt)
        if len(seed_ids) < 2:
            seed_ids = seed_ids + [tok.byte_to_id.get(ord(" "), 0)] * (2 - len(seed_ids))

        gen_ids = generate(
            model, tok, seed_ids,
            max_new=args.max_new,
            temperature=args.temperature,
            rng=rng,
        )
        try:
            response = tok.decode(gen_ids)
        except (UnicodeDecodeError, KeyError):
            response = "".join(chr(b) if 32 <= b < 127 else "." for b in
                              b"".join(tok.id_to_piece.get(i, b"?") for i in gen_ids))
        print(f"Astra: {response}")
        print()


if __name__ == "__main__":
    main()
