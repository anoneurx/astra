#!/usr/bin/env python3
"""Run (and optionally re-run) toy pretraining.

Usage: python -m astra.training.train --config configs/toy_pretrain.json
Run twice with identical args to verify EX-04 reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.model import LiteLM, ModelConfig
from astra.safety import leak_check
from astra.tokenizer import ByteLevelBPE, WordLevel, load_tokenizer
from astra.training import Corpus, train
from astra.utils import read_json, sha256_file


def _encoded_ids(
    tok: ByteLevelBPE | WordLevel,
    text: str,
    tok_path: str,
    split: str,
    corpus_path: Path,
    cache_dir: str | None,
) -> list[int]:
    """Encode a split, memoized to ``cache_dir`` keyed by content hashes.

    Chunked/resumed training re-invokes this on every run; caching the encoded
    ids (corpus sha + tokenizer sha) turns a repeated ~minute-scale encode into
    an instant npy load.
    """
    if cache_dir:
        key = f"{split}-{sha256_file(corpus_path)[:16]}-{sha256_file(tok_path)[:16]}"
        cached = Path(cache_dir) / key / f"{split}.npy"
        if cached.exists():
            print(f"[data ] cache hit: {cached}")
            return np.load(cached).tolist()
    ids = tok.encode(text)
    if cache_dir:
        cached = Path(cache_dir) / key / f"{split}.npy"
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, np.asarray(ids, dtype=np.int32))
        print(f"[data ] cache write: {cached}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_pretrain.json")
    ap.add_argument("--seed", type=int, default=None, help="override data seed in config")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--experiments", default=None, help="override experiment_store dir")
    ap.add_argument("--resume", default=None, help="resume from checkpoint .npz")
    ap.add_argument("--reset-step", action="store_true",
                    help="warm-start from checkpoint but restart step/LR/optimizer at 0 (fine-tuning)")
    ap.add_argument("--cache-dir", default=None, help="tokenized-corpus cache (external tmp)")
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    tr = raw["training"]
    seed = args.seed if args.seed is not None else raw.get("seed", 0)
    if args.steps:
        tr = {**tr, "max_steps": args.steps}
    out_dir = args.out or raw["out_dir"]
    if args.experiments:
        tr = {**tr, "experiment_store": args.experiments}

    tok = load_tokenizer(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())  # re-validate after vocab change

    train_path = Path(raw["data"]["train"])
    val_path = Path(raw["data"]["val"])

    train_ids = _encoded_ids(
        tok,
        train_path.read_text(encoding="utf-8"),
        raw["tokenizer"],
        "train",
        train_path,
        args.cache_dir,
    )
    val_ids = _encoded_ids(
        tok,
        val_path.read_text(encoding="utf-8"),
        raw["tokenizer"],
        "val",
        val_path,
        args.cache_dir,
    )

    # safety gate: contamination check between train and val (docs/DATA.md § 3)
    # For corpora split at the row/document level (cyber telemetry) the n-gram
    # gate is disabled via config and the row-level disjointness is recorded in
    # the corpus manifest instead.
    safety = raw.get("safety", {})
    if safety.get("leak_check", True):
        leak = leak_check(train_ids, val_ids, n=13)
        print(f"[safety] leak_check(train, val, n=13) = {leak}")
        if not leak["leak_free"]:
            raise SystemExit(f"contamination detected: {leak}")
    else:
        print(f"[safety] n-gram leak gate disabled: {safety.get('note', '')}")

    train_corpus = Corpus(
        ids=np.array(train_ids, dtype=np.int32),
        manifest={"name": "toy-train", "kind": "training", "sha256": sha256_file(train_path)},
    )
    val_corpus = Corpus(
        ids=np.array(val_ids, dtype=np.int32),
        manifest={"name": "toy-val", "kind": "validation", "sha256": sha256_file(val_path)},
    )

    model = LiteLM(cfg, seed=seed)
    print(f"[model] params={model.num_params} model={cfg.name} vocab={cfg.vocab_size}")
    print(f"[data ] train_tokens={len(train_ids)} val_tokens={len(val_ids)}")

    report = train(
        cfg,
        train_config=tr,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        seed=seed,
        out_dir=out_dir,
        resume_from=args.resume,
        reset_step=args.reset_step,
    )
    print("[final val]", json.dumps(report.final_val, indent=2))
    print(f"run manifest -> {out_dir}/report.json")


if __name__ == "__main__":
    main()