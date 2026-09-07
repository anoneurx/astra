# Astra Internal Benchmark Suite

> The canonical benchmark definitions used for release evaluation and promotion gating.

**STATUS: ACTIVE (v1, incremental)** — harness (`astra/evaluation/bench.py`),
suite manifests (`benchmarks/suites/`), and gate tool (`tools/benchmark.py`)
built for the `core-basic` suite; `core-reason`+ and promotion thresholds to be
assembled for the released Phase-3 model.

---

## 1. Purpose & Rules

- The suite is the **single source of truth** for model capability claims.
- All items are **quarantined** (never present in any training slice) — enforced by the leakage scanner in `tools/`.
- Benchmarked only on registered model artifacts (by checksum).
- A benchmark result without a run manifest is not a valid result.
- A suite without recorded thresholds is **Advisory**; promotion requires **ACTIVE(thresholded)** suites only (§ 4).

---

## 2. Suite Catalog

### 2.1 `core-basic`

| Item ID | Scorer | Metric | Threshold | Notes |
|---|---|---|---|---|
| `val_ppl` | `val_ppl` | Frozen validation perplexity | `< 5.0` | Frozen slice, never opened |
| `decode_tokens_per_sec` | `decode_tokens_per_sec` | Decode throughput | `> 50.0` | Incremental KV-cache decode |
| `repetition_fraction` | `repetition_fraction` | 4-gram self-repetition | `< 0.5` | Detects collapsed/degenerate text |

Manifest: `benchmarks/suites/core-basic.json` (version 1). Run:

```sh
python tools/benchmark.py --checkpoint <ckpt.npz> --config configs/toy_pretrain.json \
  --out benchmarks/results
```

Results land in `benchmarks/results/<checksum>/<suite>/result.json` (git-ignored)
tied to the checkpoint sha256, with environment + scorers per item.
Reference runs (2026-09-07):
- `checkpoints/phase0/final.npz`: **PASS** (val_ppl 2.535, 182.9 tok/s, repetition 0.0)
- `checkpoints/name/resumed/final.npz` (Astra-name fine-tune): **PASS**
  (val_ppl 4.567, 361.9 tok/s, repetition 0.0)

### 2.2 `core-reason`

| Item | Measures | Metric |
|---|---|---|
| Arithmetic (held-out) | Numeric inference | accuracy |
| Symbolic/boolean logic | Rule application | accuracy |
| Multi-step word problems | Compositional reasoning | accuracy graded by final answer |
| Self-consistency sanity | Robustness | agreement rate (research) |

### 2.3 `core-instruction`

| Item | Measures | Metric |
|---|---|---|
| Rubric instruction set | Adherence | rubric score 0-1 |
| Constraint following | Spec compliance | pass rate |
| Formatting compliance | Output structure | format pass rate |

### 2.4 `core-code`

| Item | Measures | Metric |
|---|---|---|
| Code generation (held-out) | Syntax+semantics | pass@k |
| Bugfix tasks | Localization+repair | patch pass rate |
| Explanation fidelity | Doc-comment correctness | human-audited sample score |

Code items are run-verified (unit tests), not judged by LLM only.

### 2.5 `core-retrieval` (Phase 4+)

| Item | Measures | Metric |
|---|---|---|
| Long-form QA with injected facts | Retrieval recall | hit@5, MRR, answer ROUGE/BLEU |
| Contradiction detection | Fact-consistency | accuracy |
| Memory lifecycle probes | Correction/deletion/expiry | accuracy |

### 2.6 `core-safety`

| Item | Measures | Metric |
|---|---|---|
| Poisoning probe suite | Robustness to injected examples | detection rate |
| Refusal probe suite | Behavior on harmful prompts | refusal rate (policy-specified) |
| Prompt-injection probes (incl. retrieved text) | Injection resistance | attack success rate (lower better) |
| PII-protection checks | Filtering | leak rate |

### 2.7 `core-resource`

| Item | Measures | Metric |
|---|---|---|
| Decode latency | TTFT + per-token | ms per decode step |
| Throughput | tokens/s at fixed batch | tok/s |
| Peak RAM | process resident set | MB |
| GPU utilization & VRAM | compute efficiency | % / GB |
| CPU decode run (Rust runtime) | CPU-only feasibility | tok/s on CPU |

---

## 3. Manifest Schema (per suite)

Harness v1 manifests are scorer-driven:

```json
{
  "suite": "core-basic",
  "version": 1,
  "items": [
    {
      "id": "val_ppl",
      "scorer": "val_ppl",
      "args": {"seed": 0},
      "threshold": {"op": "<", "value": 5.0}
    }
  ]
}
```

`scorer` names a function in `SCORERS` (`val_ppl`, `decode_tokens_per_sec`,
`repetition_fraction`); `args` are projected per scorer; `threshold` uses
`_satisfies(metric, threshold)` with a comparison op (`<`, `<=`, `>`, `>=`).
Suites are versioned; a change to a suite bumps version and records rationale.

---

## 4. Threshold Policy

- Thresholds are **set from measurements**, recorded after a baseline run, and documented before adoption.
- A suite without recorded thresholds is Advisory; promotion requires ACTIVE(thresholded) suites only.
- Changing a threshold requires a PR updating the decision record.

---

## 5. Reporting

Results are written to `benchmarks/results/<checkpoint-sha256-prefix>/<suite>/result.json`
(git-ignored) as one JSON report per run with full environment info and the
`passed` gate; each item records `metric`, `threshold`, and `threshold_met`.
The registry entry links its eval report (Phase-3 milestone).

---

## 6. Benchmark Hygiene

- Never tune on benchmark items.
- Never train on benchmark items.
- Report failures honestly in the release eval report.
- Re-run entire suites on every registration; partial re-runs are flagged.