#!/usr/bin/env python3
"""Phase-5 learning-loop driver (docs/LEARNING.md).

End-to-end: curated feedback (validated) -> ExperienceStore -> candidate
training off the active checkpoint -> loss-based candidate-vs-active eval on a
held-out target set and a regression (prior-corpus) set. Deterministic per
(seed, config).

The target set is *held out* — it never appears in the experience store, so any
loss reduction proves generalization, not memorization of the training lines.
The regression set is a slice of the original pretraining data, so a rising
loss there flags forgetting.

Usage: python tools/learning_loop.py --config configs/toy_name.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.learning.candidate import CandidateConfig, train_candidate
from astra.learning.experience import ExperienceStore, make_example
from astra.learning.feedback import make_feedback, validate_feedback
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json, write_json


def corpus_loss(model: LiteLM, tok: ByteLevelBPE, sentences: list[str]) -> float:
    """Mean next-token CE over the sentences (predict-each-token-after-first)."""
    tot = n = 0.0
    for s in sentences:
        ids = tok.encode(s)
        if len(ids) < 2:
            continue
        x = np.array([ids[:-1]], dtype=np.int64)
        y = np.array([ids[1:]], dtype=np.int64)
        _logits, loss = model.forward_loss(x, y)
        tot += float(loss) * len(ids)
        n += len(ids)
    return tot / max(1.0, n)


def load_model(cfg: ModelConfig, ckpt: str) -> LiteLM:
    m = LiteLM(cfg, seed=0)
    load_checkpoint(ckpt, m, opt=None, schedule=None)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--base", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--out-dir", default="checkpoints/candidate")
    ap.add_argument("--store", default="learning/store")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    args = ap.parse_args()

    raw = read_json(args.config)
    cfg = ModelConfig.from_dict(raw["model"])
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())

    store = ExperienceStore(root=args.store)

    # --- 1. feedback intake: each Nova training line arrives as human-verified
    #        feedback, passes validation, and routes to a (deduped) SFT example.
    novel_lines = Path("datasets/learning/nova_train.txt").read_text(encoding="utf-8").splitlines()
    examples: list[dict] = []
    for i, line in enumerate(novel_lines):
        fb = make_feedback(
            "human",
            "human_verification",
            line,
            confidence=0.98,
            verification_status="human_verified",
            context="Astra fact correction",
        )
        res = validate_feedback(fb, store=store)
        if res["result"] != "accepted":
            print(f"[intake] quarantined/duplicate: {res['result']} for {line!r}")
            continue
        payload = {"input": "", "output": line}
        ex = make_example(
            "sft", payload,
            feedback_id=fb["id"], source=fb["source"], confidence=fb["confidence"],
            verification_status=fb["verification_status"], trust_tier=fb["tier"],
        )
        ex_id, added = store.add(ex)
        if added:
            examples.append(ex)
        else:
            examples.append(store.get(ex_id))
    print(f"[intake] accepted {len(examples)} unique validated experiences into store")

    # --- 2. safety gate: experience lines must not leak into the held-out eval set
    heldout = Path("datasets/learning/nova_heldout.txt").read_text(encoding="utf-8").splitlines()

    trained = {ex["payload"]["output"] for ex in examples}
    leak = [s for s in heldout if s in trained]
    if leak:
        raise SystemExit(f"held-out eval set leaked into training: {leak}")

    # --- 3. candidate training off the active checkpoint with replay
    replay = np.array(tok.encode(Path("datasets/name/train.txt").read_text(encoding="utf-8")), dtype=np.int32)
    cc = CandidateConfig(
        max_steps=args.steps,
        peak_lr=args.peak_lr,
        warmup_steps=max(5, args.steps // 15),
        batch_seq=4,
        replay_ratio=args.replay_ratio,
    )
    res = train_candidate(
        base_checkpoint=args.base,
        tokenizer=tok,
        model_config=cfg,
        experiences=examples,
        replay_ids=replay,
        config=cc,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    print(f"[train ] {res.steps} steps, final CE {res.final_loss:.4f}, {res.checkpoint}")

    # --- 4. candidate-vs-active eval on held-out target + regression corpora
    base = load_model(cfg, args.base)
    cand = load_model(cfg, res.checkpoint)
    astra_lines = Path("datasets/name/train.txt").read_text(encoding="utf-8").splitlines()[:20]
    scores = {
        "target_nova_loss": {
            "base": corpus_loss(base, tok, heldout),
            "candidate": corpus_loss(cand, tok, heldout),
        },
        "regression_astra_loss": {
            "base": corpus_loss(base, tok, astra_lines),
            "candidate": corpus_loss(cand, tok, astra_lines),
        },
    }
    for k, v in scores.items():
        delta = v["candidate"] - v["base"]
        print(f"[eval  ] {k:26s} base={v['base']:.3f} cand={v['candidate']:.3f} delta={delta:+.3f}")

    target_delta = scores["target_nova_loss"]["candidate"] - scores["target_nova_loss"]["base"]
    regress_delta = scores["regression_astra_loss"]["candidate"] - scores["regression_astra_loss"]["base"]
    decision = "ACCEPT" if target_delta < -0.1 and regress_delta < 0.1 else "REJECT"
    print(f"[decision] {decision} (target CE delta {target_delta:+.3f}, regression delta {regress_delta:+.3f})")

    write_json(
        f"{args.out_dir}/learning_report.json",
        {
            "base": args.base,
            "candidate": res.checkpoint,
            "seed": args.seed,
            "decision": decision,
            "scores": scores,
            "target_delta": target_delta,
            "regress_delta": regress_delta,
            "experience_ids": [e["id"] for e in examples],
            "candidate_manifest": res.manifest,
        },
    )
    print(f"[report] -> {args.out_dir}/learning_report.json")


if __name__ == "__main__":
    main()