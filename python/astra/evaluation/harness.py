"""Evaluation harness (docs/EVALUATION.md).

Produces a versioned JSON eval report tied to a checkpoint checksum, recording
environment, metrics, and artifact hashes. Used by the release gates.
"""

from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from astra.evaluation.metrics import generate, next_token_accuracy, repetition_fraction
from astra.model.core import LiteLM
from astra.training.checkpoint import load_checkpoint
from astra.training.data import Corpus, SeqStream
from astra.utils import sha256_file, write_json


def evaluate_checkpoint(
    ckpt_path: str,
    tokenizer,
    val_corpus: Corpus,
    model_cfg,
    out_json: str,
    sample_seed: int = 0,
    gen_len: int = 200,
) -> dict:
    model = LiteLM(model_cfg, seed=0)
    step, _hist, meta = load_checkpoint(ckpt_path, model, opt=None, schedule=None)

    # 1. frozen val split: loss, perplexity, next-token accuracy
    stream = SeqStream(val_corpus, batch_seq=4, seq_len=model_cfg.max_seq_len,
                       rng=np.random.default_rng(0))
    total, acc_hits, n = 0.0, 0, 0
    t_start = time.time()
    for x, y in stream:
        logits, loss = model.forward_loss(x, y)
        total += loss * (x.shape[0] * x.shape[1])
        pred = np.argmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
        acc_hits += int(np.sum(pred == y.reshape(-1)))
        n += x.shape[0] * x.shape[1]
    val_loop_s = time.time() - t_start
    val_loss = total / max(1, n)

    # 2. generation sanity (decode throughput proxy + degenerate-text probe)
    rng = np.random.default_rng(sample_seed)
    seed_ids = val_corpus.ids[: model_cfg.max_seq_len].tolist()
    t0 = time.time()
    gen_tokens = generate(model, tokenizer, seed_ids, max_new=gen_len, rng=rng)
    gen_s = time.time() - t0

    report = {
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "eval_seed": sample_seed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        },
        "metrics": {
            "val_loss": float(val_loss),
            "val_ppl": float(math.exp(min(val_loss, 30.0))),
            "next_token_accuracy": float(acc_hits / max(1, n)),
            "gen_tokens": len(gen_tokens),
            "gen_unique_tokens": len(set(gen_tokens)),
            "gen_repetition_frac_4gram": float(repetition_fraction(gen_tokens, n=4)),
            "decode_tokens_per_sec": float(len(gen_tokens) / max(1e-6, gen_s)),
            "val_loop_sec": float(val_loop_s),
        },
        "val_data": val_corpus.manifest,
        "model_step": step,
        "sample_snippet": tokenizer.decode(gen_tokens)[:256],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, report)
    return report