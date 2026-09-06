# Astra Internal Benchmark Suite

> The canonical benchmark definitions used for release evaluation and promotion gating.

**STATUS: PROPOSED** — suite structure defined; item contents and thresholds to be assembled during Phase 1–3.

---

## 1. Purpose & Rules

- The suite is the **single source of truth** for model capability claims.
- All items are **quarantined** (never present in any training slice) — enforced by the leakage scanner in `tools/`.
- Benchmarked only on registered model artifacts (by checksum).
- A benchmark result without a run manifest is not a valid result.

---

## 2. Suite Catalog

### 2.1 `core-basic`

| Item | Measures | Metric | Notes |
|---|---|---|---|
| Frozen validation perplexity | LM quality | ppl | Frozen slice, never opened |
| Duplicate-generation stability | Sampling sanity | self-BLEU on samples | Detects collapsed/degenerate text |
| Tokenizer round-trip | Coverage | bytes lossless % | Encoding correctness |

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

```json
{
  "suite": "core-reason",
  "version": 2,
  "items": [
    {
      "id": "arith-m05",
      "split": "eval",
      "content": "<prompt/mode>",
      "scorer": "exact",
      "expected_metric": "accuracy",
      "note": ""
    }
  ],
  "status": "ACTIVE"
}
```

Suites are versioned; a change to a suite bumps version and records rationale.

---

## 4. Threshold Policy

- Thresholds are **set from measurements**, recorded after a baseline run, and documented before adoption.
- A suite without recorded thresholds is Advisory; promotion requires ACTIVE(thresholded) suites only.
- Changing a threshold requires a PR updating the decision record.

---

## 5. Reporting

Results tables are stored under `benchmarks/results/<model-checksum>/<date>.json` with full environment info. The registry entry links its eval report.

---

## 6. Benchmark Hygiene

- Never tune on benchmark items.
- Never train on benchmark items.
- Report failures honestly in the release eval report.
- Re-run entire suites on every registration; partial re-runs are flagged.