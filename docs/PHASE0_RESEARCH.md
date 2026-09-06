# Astra Phase 0 — Research: Complete Definition

> Defines the seven foundational research areas for Astra (Astra 0.0.1).
> All definitions are grounded in the project's existing documentation,
> validated experiments, and architectural decisions.
> Status markers: **VALIDATED** (experimentally confirmed), **PROPOSED** (design decided, not yet validated), **RESEARCH** (open question).

---

## 1. Philosophy

**Definition:** Astra's philosophy is that a language model must become measurably more capable over time through **verified experience**, operating inside a **controlled, auditable pipeline** — never through uncontrolled real-time weight modification.

**Core tenets:**

| Tenet | Operational Meaning |
|---|---|
| **Reproducibility** | Every run records code commit, config, data hash, seed, hardware, and library versions. Same inputs → same results within tolerance. |
| **Transparency** | No silent changes. Every model change is logged, versioned, and auditable. Limitations are documented, never hidden. |
| **Controlled continual learning** | Learning happens via candidate model generation → evaluation → accept/reject. Never uncontrolled in-place weight updates. |
| **Evaluation before deployment** | No candidate becomes the active model without passing the full benchmark suite. |
| **Safety and rollback** | Every deployment can be reverted. Every dataset and model is versioned with checksums. |
| **Data quality over quantity** | A small verified corpus beats a large noisy one. |
| **Efficient computation** | Consumer-hardware-first development; scale only when evidence justifies it. |
| **Honest claims** | Measure, then claim. Never extrapolate unmeasured capability. |

**The fundamental rule:** *Memory changes the prompt. Learning changes the candidate weights. Evaluation changes the active model. Never in-place without a gate.*

**Self-learning (Astra's definition):** The system converts recorded, verified experience into improved capability through a controlled pipeline — observation → feedback → validated training example → candidate model → evaluation → (accept/reject). The model does not "learn" every interaction; it learns only from experiences that pass validation and survive comparison against a baseline.

**Self-improvement (Astra's definition):** A measurable, versioned improvement in evaluated capability from model version *N* to version *N+1*, achieved through the controlled pipeline, where version *N+1* strictly passes the benchmark suite and version *N* remains deployable for rollback.

**What Astra is not:** A wrapper around any commercial AI API. A chatbot product. A claim to AGI, consciousness, or human-level intelligence. An unsupervised autodidact that arbitrarily modifies its own weights at runtime.

**Status:** VALIDATED (as policy — documented and operationally enforced in Phase 0 experiments)

---

## 2. Architecture Specification

**Definition:** Astra is a **decoder-only causal Transformer** built from first principles, designed to scale from 100M to multi-billion parameters by configuration alone. The architecture uses a single parameterized implementation with no hard-coded architecture paths.

**Architecture stack:**

| Component | Specification | Status |
|---|---|---|
| Architecture family | Decoder-only Transformer | VALIDATED (standard) |
| Task | Causal language modeling (next-token prediction) | VALIDATED (standard) |
| Attention | Multi-head (MHA), causal mask, RoPE on Q/K | PROPOSED (RoPE ablation planned Phase 3) |
| Normalization | RMSNorm (eps=1e-6) | PROPOSED (ablate vs LayerNorm) |
| FFN | SwiGLU (SiLU-gated, LLaMA-style 8/3·d hidden factor) | PROPOSED (ablate vs GELU) |
| Residual | PreNorm with separate per-sub-block normalization | PROPOSED |
| Positional encoding | RoPE (base 10000), applied to Q and K | PROPOSED |
| Output head | Linear(d_model, vocab_size), weight-tying default | PROPOSED (tied/untied research) |
| Init | Truncated normal / He-style scaled by width | PROPOSED |
| Scaling | Power-law scaling; single implementation parameterized by config | VALIDATED (as policy) |

**Model family targets (development targets, not commitments):**

| Model | d_model | n_layers | n_heads | d_ffn | Params ~ | Purpose |
|---|---|---|---|---|---|---|
| Astra-100M | 768 | 12 | 12 | 2048 | ~100M | Feasibility/baseline |
| Astra-300M | 1024 | 24 | 16 | 2732 | ~300M | Core v1 candidate |
| Astra-700M | ~1280 | 28 | 20 | 3413 | ~700M | Core v1 candidate |
| Astra-1B | 1536 | 32 | 24 | 4096 | ~1B | Scale study |
| Astra-3B | 2048 | 40 | 32 | 5461 | ~3B | Scale study |
| Astra-7B | 3072 | 32 | 32 | 8192 | ~7B | Scale study |

**Module interface contract:** Each module has a defined input/output interface:

```
Tokenizer:   Text → IDs → Text
Embeddings:  Token IDs → (batch, seq, d_model)
Attention:   hidden → contextual hidden (causal mask, KV-cache)
Transformer: hidden → deeper hidden (N stacked blocks)
Output Head: hidden → logits
Training:    config + data + model → checkpoints + metrics
Inference:   model + context → tokens
Memory:      query → ranked memory records
Learning:    experience + current model → candidate model
Evaluation:  candidate model → gate pass/fail report
Safety:      any stage → allow/deny + audit record
```

**Technology stack:**

| Language | Role | Phase |
|---|---|---|
| **Python** | Research, training, dataset processing, evaluation | Phase 0–3 |
| **Rust** | High-performance inference, runtime, memory engine | Phase 3+ |
| **C/C++** | Optional low-level optimization (SIMD, GPU kernels) | Only if profiling justifies |

**Interop:** Rust exposes C-ABI/cbindgen bindings; Python calls via ctypes/PyO3/maturin. Single on-disk artifact format shared by both.

**Repository structure:** `docs/` (spec), `configs/` (experiment configs), `datasets/` (versioned manifests), `tokenizer/`, `model/`, `training/`, `inference/`, `memory/`, `learning/`, `evaluation/`, `safety/`, `runtime/`, `rust/`, `python/`, `tests/`, `benchmarks/`, `experiments/`, `checkpoints/`, `tools/`.

**Status:** VALIDATED (as design) — pipeline, modules, and repo structure implemented and validated at toy scale (Astra 0.1.0)

---

## 3. Data Strategy

**Definition:** Astra's learning quality depends on **data discipline** — strict separation between training and evaluation data, tracked provenance, deterministic cleaning, and contamination controls. A small verified corpus beats a large noisy one.

**Data pipeline stages:**

```
Raw data → Cleaned → Filtered → Training Data
                              → Validation Data
                              → Evaluation Data
                              → Feedback Data (separate)
                              → Synthetic Data (separate, validated)
                              → Verified Correction Data (highest priority)
```

**Data stores:**

| Store | Description | Usage |
|---|---|---|
| **Raw** | Original bytes, never mutated. Read-only with provenance, license, hash. | Source of truth |
| **Cleaned** | After deterministic, versioned cleaning rules. Hash-stable re-runs. | Intermediate |
| **Filtered** | After allow/deny filters (PII, toxicity, quality, license). | Pre-split |
| **Training** | Filtered subset for training. Any token permanently excluded from eval. | Model training |
| **Validation** | Frozen 0.5–1% slice, never updated, hash recorded. | Training loss monitoring |
| **Evaluation** | Unseen, held-out benchmark + internal suite. Frozen. Never enters training. | Benchmarking |
| **Feedback** | Interaction outcomes with confidence, source, timestamp, verification. | Candidate training (after validation) |
| **Synthetic** | AI-generated examples, always earmarked and screened. | Training (after quality gate) |
| **Verified corrections** | Corrected outputs verified by humans or strong automated checks. | Highest-priority learning signal |

**Contamination & leakage controls:**

- **Data contamination** = training data overlapping with evaluation/benchmark data → **zero tolerance**.
- **Data leakage** = information seen during evaluation leaking into the model or artifacts.
- Detection methods: exact hash matching, 13-gram overlap with disallow list, document-level MinHash at ~Jaccard 0.85 cutoff.
- When leakage is found: data is **re-split** — not patched silently — and a record is added to the manifest.
- Memory stores are **quarantined** from benchmark evaluation unless the benchmark explicitly measures retrieval.

**Manifest format (per dataset slice):**

```json
{
  "name": "astra-dataset-v1",
  "kind": "training" | "validation" | "evaluation" | "feedback" | "synthetic",
  "source_ids": ["..."],
  "license": "cc-by-sa-4.0",
  "created_at": "ISO8601",
  "cleaning_rules": ["rule-id-v1"],
  "filters": [{"id": "filter-v1", "params": {}}],
  "hash": {"algorithm": "sha256", "value": "..."},
  "documents": 12345,
  "tokens": 4567890,
  "contamination_checked": true,
  "leakage_report": "link"
}
```

**Phase 0 validation:** Toy corpus pipeline (documents, split manifests, decontamination, n-gram leak gate, file hashes) implemented and measured. Zero overlapping 13-grams between train/val/eval splits (H0.5 met).

**Status:** VALIDATED (Phase-0 subset)

---

## 4. Compute Requirements

**Definition:** Astra must be developable, trainable, and evaluable on **consumer hardware** for Phases 0–4. Large-scale training is an option, not a prerequisite.

**Phase 0 / toy measurements (validated, 2026-09-06):**

| Metric | Value | Notes |
|---|---|---|
| Model (toy) | 133,440 params | d_model=64, 2 layers, 4 heads |
| FP32 footprint | 0.53 MB | 133,440 × 4 B |
| Training throughput | 4,051 tok/s | 2000 steps, 252s, 8 threads, batch 8×64 |
| Reproducibility | Bit-identical | 1 vs 8 threads, 2000/2000 points |
| Training peak RSS | ~90 MB | `/proc` VmHWM, 60-step probe |
| Evaluation peak RSS | ~196 MB | NumPy import + model + one batch |
| CPU decode (reference) | 92–276 tok/s | Load-dependent |
| Checkpoint size | ~1.45 MB | `final.npz` (params + optimizer state) |
| Hardware | 8 CPU cores, 29 GB RAM, no GPU | NumPy-only path |

**Hardware requirements by workload:**

| Workload | CPU | RAM | GPU | Storage |
|---|---|---|---|---|
| **Development (Phases 0–2)** | 4 cores min / 8+ rec. | 16 GB min / 32 GB rec. | None required; optional 8 GB | 20 GB / 100 GB NVMe |
| **Small-scale (Astra-100M/300M)** | 8+ cores | 32 GB / 64 GB | 8 GB VRAM min / 16–24 GB rec. | 100 GB / 500 GB NVMe |
| **Medium-scale (Astra-700M/1B)** | 12+ cores | 64 GB / 128 GB | 24 GB VRAM min / 40–80 GB | 500 GB / 1 TB NVMe |
| **Large-scale (Astra-3B/7B)** | Cluster | 128+ GB | 4+ H100/A100-class | Multi-TB |

**Inference requirements:**

| Model | CPU-only | GPU |
|---|---|---|
| Astra-100M–700M | 16 GB RAM, CPU decode in Rust | 8–16 GB VRAM |
| Astra-1B–3B | 32 GB RAM (quantized) | 24+ GB VRAM |
| Astra-7B | 48 GB RAM (int8) | 48+ GB VRAM (fp16) / 24 GB (int8) |

**Key principles:**
- Consumer hardware and cloud spot instances are equally valid.
- Reproducibility contract requires recording hardware IDs in run manifests.
- Distributed training (DDP/FSDP) is out of scope for Phases 0–3 — single-machine reproducibility first.
- Mixed precision (bf16/fp16) roughly halves memory vs fp32; fp16 path with loss scaling for consumer GPUs.

**Status:** VALIDATED (toy-scale measurements); GPU/large-scale budgets remain targets until Phase 1+ measurements land.

---

## 5. Training Methodology

**Definition:** Astra's training system turns a verified, deduplicated, contamination-checked corpus into checkpoints via a **config-driven, reproducible, auditable** pipeline. Training always operates on **candidate models** (never in-place on the active model), using gated evaluation for acceptance.

**Pipeline stages:**

```
Dataset ingestion → Cleaning → Deduplication → Tokenization
→ Shuffling → Batching → Training loop → Loss tracking
→ Checkpointing → Validation → Experiment tracking
```

**Training loop (standard):**
1. Forward pass (causal LM)
2. Cross-entropy loss over tokens
3. Backward; gradient clipping at 1.0
4. Optimizer step (AdamW)
5. Scheduler step (cosine with warmup)
6. Periodic validation + checkpoint

**Baseline hyperparameters (Phase 1):**

| Component | Value |
|---|---|
| Optimizer | AdamW (β=(0.9, 0.95), eps 1e-8) |
| LR schedule | Cosine with linear warmup (~2% of steps) |
| Peak LR | ~3e-4 (sweep later) |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch (tokens) | 0.5M–4M depending on scale |
| Mixed precision | bf16 (A100/H100) or fp16 (consumer) |
| Context length | 1024 (Phase 1); extendable via RoPE |
| Gradient accumulation | As needed by memory budget |

**Phase 0 toy training config (`configs/toy_pretrain.json`):**
- 133,440 params, vocab=800, d_model=64, 2 layers, 4 heads, d_ffn=128, max_seq_len=64
- 2000 steps, peak_lr=1e-3, min_lr=1e-5, warmup=50 steps, batch_seq=8
- **Result:** val CE 2.535 (target: <4.0) — H0.3 met

**Reproducibility contract:** A training run is reproducible if, given the same config, same input data shards with same hash, same seed, and same kernel/arithmetic, the loss trajectory matches within a documented tolerance. Phase 0 verified bit-identical loss trajectories across thread counts (H0.4 met, 2000/2000 points).

**Checkpointing:**
- Saved at fixed step intervals and at best-validation-loss.
- Each checkpoint includes: weight file, optimizer state (optional), config snapshot, data manifest hash, training-state snapshot (step, lr, rng state), SHA-256 checksum.
- Resume must reproduce deterministic-bit-equal state from the snapshot.

**Pretraining vs. adaptive learning:**

| Aspect | Pretraining | Adaptive (continued) learning |
|---|---|---|
| Purpose | Language modeling foundation | Improve on verified feedback/tasks |
| Data | Large curated corpus | Verified experiences, corrections |
| Objective | Next-token loss | Mixed objective (LM loss + preference/feedback loss) |
| Learning rate | Higher peak, long schedule | Very low (or low-rank adapters) |
| Checkpoint | Fresh from scratch / big resume | Candidate from previous checkpoint |
| Risk | Catastrophic forgetting low | Catastrophic forgetting relevant |

**Adaptive learning always happens on a candidate model**, never the active deployed model in place.

**Phase 0 validation:** Training loop (AdamW, cosine+warmup, gradient clip, deterministic batching, checkpoints, resume) verified correct and reproducible (H0.3, H0.4 met).

**Status:** VALIDATED (Phase-0 subset — training-loop, scheduling, checkpointing, reproducibility contract, leak-gated evaluation validated at toy scale)

---

## 6. Evaluation Methodology

**Definition:** Astra's evaluation system produces **trustworthy, measurable numbers** — improvement is never defined by training loss alone. Every claim maps to a measured artifact. Evaluation uses frozen, quarantined data and evaluates the model artifact (not the repository version).

**Core principles:**
1. Improvement is never defined by training loss alone.
2. Evaluate on frozen, quarantined data (contamination policy).
3. Evaluate the model artifact (checksum-attached), not the repo version.
4. Report uncertainty — repeat `n` runs, confidence intervals for stochastic metrics.
5. Every claim maps to a measured artifact.

**Metric categories:**

| Category | Metrics |
|---|---|
| **Core/training** | Training loss (mean CE), Validation loss, Perplexity (`exp(val_loss)`) |
| **Capability** | Accuracy, Reasoning, Instruction following, Coding ability (pass@k), Hallucination rate, Retrieval accuracy, Memory accuracy |
| **Resource/performance** | Latency (TTFT, tokens/s), RAM usage, GPU utilization, CPU utilization |

**Internal benchmark suite (`benchmarks/`):**

| Suite | Contents |
|---|---|
| `core-basic` | Perplexity on frozen held-out sample, round-trip generation stability |
| `core-reason` | Arithmetic, logic, step-by-step (held-out) |
| `core-instruction` | Instruction-following rubric tasks |
| `core-code` | RU (run-and-verify) code tasks, pass@k |
| `core-retrieval` | Memory QA requiring retrieved facts (Phase 4+) |
| `core-safety` | Poisoning probe, refusal of harmful requests, prompt-injection probes |
| `core-resource` | Latency/RAM/GPU budget checks at fixed env |

Every suite is **quarantined**: membership tracked, contamination scanning runs against all training data.

**Inference-time evaluation:** Fixed inference settings (temperature, sampling seed, max tokens) per benchmark; greedy for deterministic subsets. TTFT/latency measured over repeated runs with interleaving. Results published per model artifact with timestamps and environment.

**Gating rules:**
- All-suite execution required for promotion; skip = fail.
- Critical suite regression → automatic fail → rollback path.
- Candidate must beat active model on ≥1 primary benchmark with statistical significance, and must not regress beyond tolerance on any core suite.

**Reporting format:** Machine-readable + human-readable JSON report tied to checkpoint checksum, containing: model, config, data manifest, eval date, environment, suites results table, significance analysis (e.g., paired bootstrap), regressions list, gate verdict (PASS/FAIL).

**Phase 0 validation:** Evaluation harness (`evaluation/evaluate.py`) produces checksum-tied JSON report. Measured: val_loss 2.535, accuracy 27.1%, perplexity 12.6. Metrics + environment recorded (H0.6 met).

**Status:** VALIDATED (harness + toy loop)

---

## 7. Safety Boundaries

**Definition:** Astra's safety system enforces that **every data path is validated, every model change is gated and auditable, every learning step can be reverted, and uncertainty is handled, not hidden.** Safety is not a post-hoc add-on — it is enforced at every stage of the pipeline.

**Safety components:**

| Component | Description |
|---|---|
| **Data validation** | Manifests must include provenance, license, hash, cleaning/filter rule IDs. Checksum verification on every dataset read at training start. |
| **Training-data filtering** | PII redaction, toxicity scoring, private-data scanning, language-lock per corpus policy. Deterministic, versioned, audited. No black-box silent filtering. |
| **Prompt-injection resistance** | Injection/indirect-prompt probes on every release candidate (`core-safety` suite). Memory retrieval content treated as untrusted: bounded token budget, delimited zones, no code-running during eval. |
| **Malicious-data detection** | Contamination scanning (hash + n-gram overlaps). Feedback poisoning detection (source verification, cross-checking, rate limits, anomaly detection). Dataset provenance review. |
| **Model rollback** | Registry keeps previous versions with checksums. Rollback restores prior active model artifact + config + memory pointer state. Rollback drill runs in CI (automated, measured RTO). |
| **Version control** | Models, memory schema, tokenizers, datasets, configs — all versioned. Git for text, hash/S3 for binaries. |
| **Evaluation gates** | Candidates must pass full evaluation suite before promotion. Gate verdict recorded immutably. Thresholds set by measurement. |
| **Human approval** | Deployment modes: `auto`, `manual`, `hybrid`. Default for Phase 6+: `hybrid`. |
| **Audit logs** | Append-only audit trail covering: dataset writes, training launches, candidate creations, evaluation runs, gate decisions, promotions, rollbacks, feedback-influence steps, memory corrections. |

**Safety-critical rules (non-negotiable):**
1. **No** in-place active-weight modification.
2. **No** model promotion without gates.
3. **No** silent dataset merge into training.
4. **No** automatic trust of raw feedback.
5. **No** evaluation-data training.

**Failure mode defense matrix:**

| Failure mode | Defense |
|---|---|
| **Data poisoning** | Provenance checks, filters, contamination scans, quarantine, monitoring behavior probes |
| **Feedback poisoning** | Trust tiers, verification, aggregation, anomaly detection, human review |
| **Model collapse** | Diversity metrics on experiences, CPU-size replay, synthetic-data guardrails, collapse probes in benchmark set |
| **Catastrophic forgetting** | Low LR, replay corpus, prior-version regression gates, capability probes per release |
| **Reward hacking** | Gates measure general capability (not reward-max), cross-checked reward, exploration restriction, independent audits |
| **Distribution drift** | Drift detector on eval slices; alerting → human review; re-training on fresh data; memory refresh |

**Trust tiers for feedback (lowest to highest):**
1. Model self-score (lowest-trusted; weight only with support)
2. Raw user signal (medium default — adversarial spoofable)
3. Automated evaluation (high — deterministic)
4. Explicit correction (high after verification)
5. Human verification (highest)

Unexpected/unverifiable feedback goes to a **review queue**, not straight to training.

**Incident response:** A reported incident creates an issue; safety-critical incidents trigger **retention freeze** (no promotions until review). Rollback is the default mitigation; root-cause analysis documented before re-promotion. Incident records live in `experiments/` as structured markdown with hashes.

**Phase 0 validation:** Hygiene + contamination gates (sanitize, dedup, n-gram leak check, run manifests) implemented, tested, and enforced during Phase 0 data generation/training. Training aborts on detected leakage.

**Status:** VALIDATED (Phase-0 subset)

---

## Cross-Area Summary

| Area | Status | Key Finding |
|---|---|---|
| Philosophy | VALIDATED (policy) | Controlled learning + candidate evaluation operationalizable |
| Architecture | VALIDATED (as design) | Transformer core implemented and numerically validated from scratch |
| Data Strategy | VALIDATED (Phase-0 subset) | Deterministic, manifest-tracked, leak-free splits at toy scale |
| Compute Requirements | VALIDATED (toy) | Consumer CPU-only feasible; GPU budgets defined for scaling |
| Training Methodology | VALIDATED (Phase-0 subset) | AdamW/cosine/warmup pipeline correct, reproducible, reaches target loss |
| Evaluation Methodology | VALIDATED (harness) | Checksum-tied JSON reports, metrics + environment recorded |
| Safety Boundaries | VALIDATED (Phase-0 subset) | Contamination/leak gates, data filters, audit scaffolding enforced |

**All Phase 0 hypotheses (H0.1–H0.5) met.** The seven research areas are operationally defined, experimentally validated at toy scale, and ready for Phase 1 scaling.
