#!/usr/bin/env python3
"""Interactive text generation for Astra.

Chat using the incremental KV-cache decoder (astra.inference.decode) with
top-k + temperature sampling. Each turn starts a fresh cache (the toy corpus
windows are independent), which keeps answers tightly focused on the prompt.

Usage:
    python inference/generate.py
    python inference/generate.py --checkpoint checkpoints/name/resumed/final.npz \
        --config configs/toy_name.json
    python inference/generate.py --checkpoint checkpoints/phase0/final.npz --temperature 0.8

Type your prompt and press Enter. Astra responds. Type 'quit' or Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import KVCache, decode
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Astra text generation")
    ap.add_argument("--checkpoint", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = read_json(args.config)
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})

    model = LiteLM(cfg, seed=0)
    step, _hist, _meta = load_checkpoint(args.checkpoint, model, opt=None, schedule=None)
    print(f"Astra loaded (step {step}, {model.num_params} params)")
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
        if len(seed_ids) < 1:
            continue

        cache = KVCache(model.cfg)
        gen_ids = decode(
            model, seed_ids, args.max_new,
            temperature=args.temperature,
            rng=rng,
            top_k=args.top_k,
            cache=cache,
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