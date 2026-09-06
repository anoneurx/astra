# Astra Evaluation System

> Metrics, benchmark harness, gates.

**STATUS: VALIDATED (harness + toy loop)** — `evaluation/evaluate.py` measured on the Phase-0 toy model (val_loss 2.535, accuracy 27.1%, ppl 12.6) and writes a versioned, checksum-tied JSON report; metric thresholds for later phases remain to be set from measurement.

---

## 1. Principles

1. **Improvement is never defined by training loss alone.** Training loss measures fit to training data, not capability.
2. **Evaluate on frozen, quarantined data** (see `docs/DATA.md` contamination policy).
3. **Evaluate the model, not the repository version** — candidate scores are attached to the artifact checksum.
4. **Report uncertainty** — a metric is a number with spread (repeat `n` runs, confidence intervals) when stochastic.
5. **Every claim maps to a measured artifact.**

---

## 2. Metrics

### 2.1 Core / Training Metrics
| Metric | Definition | Reported when |
|---|---|---|
| Training loss | mean CE over in-loop stream | every N steps |
| Validation loss | frozen split | every N steps |
| Perplexity | `exp(validation_loss)` | every N steps |

### 2.2 Capability Metrics
| Metric | Definition |
|---|---|
| Accuracy | Task-level accuracy on benchmark subsets |
| Reasoning | Scaled/step-verifiable reasoning score on reasoning suite |
| Instruction following | rubric-scored adherence on instruction set |
| Coding ability | pass@k on held-out coding tasks |
| Hallucination rate | verified-claim correctness on sampled completions |
| Retrieval accuracy | hit-rate@k / MRR on memory eval queries |
| Memory accuracy | recall/precision of corrected facts after memory write |

### 2.3 Resource / Performance Metrics
| Metric | Definition | Unit |
|---|---|---|
| Latency | TTFT, tokens/s (decoding) | ms, tok/s |
| RAM usage | peak resident set | MB / GB |
| GPU utilization | SM utilization %, VRAM used | % / GB |
| CPU utilization | avg cores during decode | count / % |

See `docs/BENCHMARKS.md` for the benchmark manifest; see `docs/HARDWARE.md` for hardware scoping.

---

## 3. The Internal Astra Benchmark Suite

Suites live in `benchmarks/` and are defined by a JSON manifest (id, split, prompt set, scorer, expected metric, threshold placeholder, version hash). Categories:

| Suite | Contents |
|---|---|
| `core-basic` | Language-model sanity: perplexity on frozen held-out sample, round-trip generation stability |
| `core-reason` | Arithmetic, logic, step-by-step (held-out) |
| `core-instruction` | Instruction-following rubric tasks |
| `core-code` | RU (run-and-verify) code tasks, pass@k |
| `core-retrieval` | Memory QA requiring retrieved facts (Phase 4+) |
| `core-safety` | Poisoning probe, refusal of harmful requests, prompt-injection probes |
| `core-resource` | Latency/RAM/GPU budget checks at fixed env |

Every suite is **quarantined**: membership in benchmark files is tracked; contamination scanning runs against all training data.

---

## 4. Inference-Time Evaluation Methodology

- Evaluation uses fixed inference settings (temperature, sampling seed if applicable, max tokens) per benchmark; greedy for deterministic subsets.
- TTFT/latency measured over repeated runs with interleaving to average cache warmth.
- Results table published per model artifact with timestamps and environment (GPU model, driver).

---

## 5. Gating Rules

The gate engine computes a pass/fail report:

- All-suite execution required for promotion; skip of a suite is a fail.
- Critical suite regression → automatic fail → rollback path.
- Candidate must beat active model on ≥1 primary benchmark with statistical significance, and must not regress beyond tolerance on any core suite.

---

## 6. Reporting Format (Release Eval Report)

```
Model: astra-0.5-300M-<checksum>
Config: configs/astra-300M.yaml
Data: manifest datasets/train-v3.json
Eval date:...
Environment: ...
Suites: [results table]
Significance: [method: paired bootstrap, n, ci]
Regressions: [none] / [list]
Gate: PASS / FAIL
```

A report is machine-readable plus human-readable; it is attached to the registry entry at promotion time.