#!/usr/bin/env python3
"""Long-form QA measurement: RAG (memory-augmented) vs baseline (docs/BENCHMARKS.md §2.6).

For each item in the memory QA eval set (datasets/memory/qa_v1.json):
  - BASELINE: generate an answer from the question alone.
  - RAG: retrieve memories for the question, prepend the <|memory|> block, generate.
  Answers are scored by fact-recall: a gold fact counts as recalled when any of its
  distinctive content words (non-stopword tokens) appears in the generated text.

The eval-set facts are tagged ``eval-quarantine`` and retrieval opts in via
``allow_quarantine=True`` so demonstration memories never leak into the inference
store by default (docs/MEMORY.md § 7).

Generation is greedy (temperature 0) to keep the comparison deterministic and to
isolate the effect of retrieval on answer content.

Usage:
    python tools/memory_qa.py
    python tools/memory_qa.py --embedder litelm
    python tools/memory_qa.py --manifest datasets/memory/qa_v1.json --out benchmarks/results/memory/qa-vs-baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import KVCache, decode
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import environment, git_commit, read_json

STOPWORDS = {
    "a", "an", "the", "of", "is", "are", "isn't", "was", "to", "in", "on", "at",
    "what", "what's", "which", "its", "it's", "than", "as", "and", "or", "for",
    "with", "into", "that", "this", "by", "over", "after", "only",
}


def distinctive_words(content: str) -> set[str]:
    return {w for w in content.lower().split() if w not in STOPWORDS}


def fact_nll(model, context_ids: list[int], target_ids: list[int]) -> float:
    """Mean next-token nats of ``target_ids`` conditioned on ``context_ids``."""
    import numpy as np

    if not target_ids:
        return 0.0
    ids = np.array([context_ids + target_ids], dtype=np.int64)
    C, T = len(context_ids), len(ids[0])
    logits = model.forward(ids)[0]
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    start, end = C, T - 1
    nats = -np.take_along_axis(logp[start:end], ids[0, start + 1:end + 1, None], axis=-1)
    return float(nats.mean())


def fact_recall(generated: str, gold_facts: list[str]) -> tuple[float, list[bool]]:
    gen_lower = generated.lower()
    hits = [bool(distinctive_words(f) & set(gen_lower.split())) for f in gold_facts]
    return (sum(hits) / len(hits) if hits else 0.0), hits


def safe_decode(tok: ByteLevelBPE, ids: list[int]) -> str:
    try:
        return tok.decode(ids)
    except (UnicodeDecodeError, KeyError):
        return "".join(chr(b) if 32 <= b < 127 else "." for b in
                       b"".join(tok.id_to_piece.get(i, b"?") for i in ids))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--checkpoint", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--manifest", default="datasets/memory/qa_v1.json")
    ap.add_argument("--embedder", default="hash", choices=["hash", "litelm"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget-tokens", type=int, default=192)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.manifest).read_text())
    facts = data["facts"]
    items = data["items"]

    raw = read_json(args.config)
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})
    model = LiteLM(cfg, seed=0)
    step, _hist, _meta = load_checkpoint(args.checkpoint, model, opt=None, schedule=None)

    from astra.memory import (
        HashEmbedder,
        LiteLMExtractor,
        MemoryRecord,
        MemoryStore,
        build_memory_block,
    )

    embedder = HashEmbedder(seed=args.seed) if args.embedder == "hash" else LiteLMExtractor(model, tok)
    store = MemoryStore(name=f"qa-{args.seed}", directory=tempfile.mkdtemp(prefix="astra-mem-qa-"),
                        embedder=embedder)
    for f in facts:
        store.add(MemoryRecord(content=f["content"], kind=f.get("kind", "fact"), source="auto_extract",
                               tags=[f["id"], "eval-quarantine"], verification_status="verified",
                               confidence=f.get("confidence", 0.95)))

    per_item = []
    for it in items:
        gold_contents = [next(f["content"] for f in facts if f["id"] == gid) for gid in it["gold_fact_ids"]]
        seed_ids = tok.encode(it["question"])
        cache = KVCache(model.cfg)
        baseline = decode(model, seed_ids, args.max_new, temperature=1e-3,
                          rng=np.random.default_rng(args.seed), top_k=1, cache=cache)
        baseline_text = safe_decode(tok, baseline)

        hits = store.search(query=it["question"], k=args.k, budget_tokens=args.budget_tokens,
                            allow_quarantine=True)
        block = build_memory_block(hits, tok, budget_tokens=args.budget_tokens)
        rag_ids = tok.encode(block.prepend(it["question"]))
        cache = KVCache(model.cfg)
        rag = decode(model, rag_ids, args.max_new, temperature=1e-3,
                     rng=np.random.default_rng(args.seed), top_k=1, cache=cache)
        rag_text = safe_decode(tok, rag)

        base_recall, base_hits = fact_recall(baseline_text, gold_contents)
        rag_recall, rag_hits = fact_recall(rag_text, gold_contents)
        gold_ids = set(it["gold_fact_ids"])
        recalled_into_block = [h.record.tags[0] for h in block.included if h.record.tags[0] in gold_ids]

        probe = {}
        for gid, gcon in zip(it["gold_fact_ids"], gold_contents):
            tgt = tok.encode(gcon)
            ctx_b = tok.encode(it["question"])
            ctx_r = tok.encode(block.prepend(it["question"]))
            nll_b = fact_nll(model, ctx_b, tgt)
            nll_r = fact_nll(model, ctx_r, tgt)
            probe[gid] = {"baseline_nats": round(nll_b, 4), "rag_nats": round(nll_r, 4),
                          "delta_nats": round(nll_b - nll_r, 4)}
        per_item.append({
            "id": it["id"],
            "question": it["question"],
            "gold_fact_ids": it["gold_fact_ids"],
            "baseline": {"text": baseline_text, "fact_recall": base_recall, "facts_hit": base_hits},
            "rag": {"text": rag_text, "fact_recall": rag_recall, "facts_hit": rag_hits},
            "retrieved": [h.record.tags[0] for h in block.included],
            "gold_retrieved": recalled_into_block,
            "probe_nats_per_fact": probe,
        })

    n = max(len(per_item), 1)
    baseline_recall = sum(p["baseline"]["fact_recall"] for p in per_item) / n
    rag_recall = sum(p["rag"]["fact_recall"] for p in per_item) / n
    total_gold = sum(len(p["gold_fact_ids"]) for p in per_item) or 1
    gold_retrieved = sum(len(p["gold_retrieved"]) for p in per_item) / total_gold
    all_deltas = [d for p in per_item for d in p["probe_nats_per_fact"].values()]
    probe_delta = sum(d["delta_nats"] for d in all_deltas) / max(len(all_deltas), 1)

    metrics = {
        "baseline_fact_recall": round(baseline_recall, 4),
        "rag_fact_recall": round(rag_recall, 4),
        "delta": round(rag_recall - baseline_recall, 4),
        "gold_retrieval_rate": round(gold_retrieved, 4),
        "conditioning_probe_delta_nats": round(probe_delta, 4),
    }
    report = {
        "task": "long-form-QA RAG-vs-baseline",
        "commit": git_commit(),
        "environment": environment(),
        "embedder": embedder.config,
        "model": {"params": model.num_params, "step": step},
        "params": {"max_new": args.max_new, "k": args.k, "budget_tokens": args.budget_tokens,
                   "seed": args.seed, "temperature": 0.0},
        "corpus": {"facts": len(facts), "items": len(items), "manifest": args.manifest},
        "method": "greedy generation (T=1e-3, top_k=1); fact recall = any distinctive gold "
                  "content word in the generated continuation; conditioning probe = mean "
                  "next-token nats of the gold fact with (RAG) vs without (baseline) the "
                  "memory block",
        "quarantine": {"tags": ["eval-quarantine"], "opt_in": "allow_quarantine=True"},
        "metrics": metrics,
        "per_item": per_item,
    }
    print(json.dumps(metrics, indent=2))

    out = args.out
    if out is None:
        out = "benchmarks/results/memory/qa-vs-baseline.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()