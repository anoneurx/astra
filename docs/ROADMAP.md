# Astra Roadmap

> Official project phases, releases, milestones, and exit criteria. Design decided as the guiding plan; individual milestones are adjusted by evidence.

**STATUS: PROPOSED** (phase scheduling and content are targets, not guarantees)

**STATUS: VALIDATED** — Phase 0 (Astra 0.0.1) research and Phase 2 (Astra 0.2.0)
Training Foundation completed per their exit criteria (`docs/releases/v0.0.1.md`
status `v0.1.0.md`, and `docs/releases/v0.2.0.md`).

---

## Version Mapping

| Phase | Name | Release | Focus |
|---|---|---|---|
| 0 | Research | Astra 0.0.1 | Foundations, spec, tooling |
| 1 | Minimal Language Model | Astra 0.1 | Working small LM checkpoint |
| 2 | Training Foundation | Astra 0.2 | Reproducible training pipeline |
| 3 | Stable Neural Core | Astra 0.3 | Released core model |
| 4 | Memory | Astra 0.5 | External memory system |
| 5 | Learning | Astra 0.7 | Feedback-driven candidate training |
| 6 | Self-Improvement | Astra 0.9 | Continuous controlled improvement |
| 7 | First Stable System | Astra 1.0 | Integrated, gated, auditable system |
| 8 | Continuous Evolution | Astra 1.x | Deployed learning in controlled setting |
| 9 | Advanced Architecture | Astra 2.0 | Architecture research branch |

Version semantics are defined in `docs/VERSIONING.md`. Patch-level releases (0.0.1-style) occur within phases for fixes; the sub-phase release number above is the minor/major milestone.

---

## Phase 0 — Research

- **Objective:** Stand up the project: spec, tooling, tokenizer experiments, first tiny model feasibility.
- **Features:** Repository skeleton, CI, doc tree, developer environment.
- **Architecture changes:** None (specification phase).
- **Research goals:** Validate tokenizer choice on a small corpus; validate training on a toy dataset (~1M tokens).
- **Required components:** Tokenizer scaffold, toy training loop, evaluation scaffold, docs.
- **Success criteria:** A toy model trains to loss < 4.0 on a toy corpus reproducibly; tokenizer quality report exists.
- **Tests:** Unit tests for tokenizer and training loop; nightly reported toy-run metrics.
- **Risks:** Scope creep; premature scaling.
- **Exit criteria:** Spec frozen at 0.0.1; CI green; toy training reproducible.

## Phase 1 — Minimal Language Model

- **Objective:** Produce a working Astra-100M-class model checkpoint on a curated public corpus.
- **Features:** Pretraining loop, checkpointing, validation split, perplexity reporting.
- **Architecture changes:** Tokenizer v1, model v0.1, embedding, transformer core, output head.
- **Research goals:** Establish baseline scaling curve slope; measure tokens/sec on commodity GPU.
- **Required components:** Dataset pipeline (cleaning, dedup, splits), tokenizer, training engine.
- **Success criteria:** Pretrained Astra-100M with validation perplexity within reasonable band of a reference small model on same data; reproducibility run under same seed.
- **Tests:** Dataset pipeline tests, model forward/backward tests, checkpoint round-trip test.
- **Risks:** Data licensing; unreproducible training; tokenizer regressions.
- **Exit criteria:** Checkpoint registered, evaluation published, config recorded.

## Phase 2 — Training Foundation

- **Objective:** Harden training into a reproducible, configurable, audited pipeline.
- **Features:** Config-driven runs, experiment tracking, learning-rate sweeps, gradient accumulation, mixed precision, distributed-ready hooks.
- **Architecture changes:** Training engine v1; optimizer/scheduler registry.
- **Research goals:** Optimization hyper-parameter study (lr, warmup, batch).
- **Required components:** Experiment tracker, log schema, run-manifest generator.
- **Success criteria:** Two identical configs reproduce within reported tolerance; experiment store queryable.
- **Tests:** Reproduction test (same-seed determinism), scheduler tests, accumulation equivalence tests.
- **Risks:** Over-engineering; determinism-difficulty on new hardware.
- **Exit criteria:** Training Foundation documented and used for all subsequent runs.
- **STATUS: COMPLETE (Astra 0.2.0)** — exit criteria met (reproduction, scheduler,
  and accumulation-equivalence tests; config-driven; experiment store queryable).
  Release notes: `docs/releases/v0.2.0.md`. Deferred within phase: LR sweeps,
  mixed precision, distributed hooks (`docs/TRAINING.md` §§ 3–4, 2.12).

## Phase 3 — Stable Neural Core

- **Objective:** Produce and release the first "real" core model (Astra-300M/700M) as a productized artifact.
- **Features:** Core model checkpoint, inference service (Python + Rust), model registry with checksums, quantization preview (fp16/bf16; int8 later).
- **Architecture changes:** Inference engine v1 (Rust), model registry v1.
- **Research goals:** RMSNorm vs LayerNorm ablation; SwiGLU vs GELU ablation; RoPE vs learned positional.
- **Required components:** Registry, checksum tooling, benchmark harness v1, release gates.
- **Success criteria:** Core model passes benchmark suite (documented in `docs/BENCHMARKS.md`); no critical regressions; reproducible training run recorded.
- **Tests:** Benchmark reproducibility, registry round-trip, inference equivalence Python↔Rust.
- **Risks:** Long compute; benchmark leakage from eval data.
- **Exit criteria:** Astra 0.3 released with model artifact + eval report + limitations doc.

## Phase 4 — Memory

- **Objective:** Add external memory system.
- **Features:** Memory records, embeddings, retrieval/ranking, short/long-term stores, memory lifecycle (expiry, correction, deletion, conflict resolution).
- **Architecture changes:** Memory engine v1; memory store (Rust) + Python reference; retrieval integration into inference context.
- **Research goals:** Relevance-scoring study; how much retrieved context helps/degrades generation; memory conflict policies.
- **Required components:** Memory store, retrieval API, memory evaluation set (long-form QA).
- **Success criteria:** Retrieval-augmented inference improves long-form QA accuracy over baseline on memory eval set without hurting core benchmarks.
- **Tests:** Retrieval ranking tests, memory lifecycle tests, leakage tests (memory never blends into train data).
- **Risks:** Retrieval quality noise; memory overfitting; data contamination.
- **Exit criteria:** Astra 0.5 released; memory benchmarks documented.

## Phase 5 — Learning

- **Objective:** Turn feedback into validated training examples and candidate-model training.
- **Features:** Feedback system with confidence scores; human-verification workflow; automated reward signals; experience store; candidate training from experiences.
- **Architecture changes:** Learning engine v1; feedback schemas; safety filtering of feedback.
- **Research goals:** How to weigh low-confidence feedback; helpful versus harmful feedback balance; curriculum of experiences.
- **Required components:** Feedback intake, validation, experience store, candidate trainer.
- **Success criteria:** A candidate trained from a small validated feedback set measurably improves one target benchmark without regressing others; every improvement traceable to validated examples.
- **Tests:** Feedback poisoning tests; candidate-vs-baseline evaluation; attribution tests.
- **Risks:** Feedback poisoning; overfitting to feedback distribution; reward hacking.
- **Exit criteria:** Astra 0.7 released; learning pipeline documented and reproducibly run.

## Phase 6 — Self-Improvement

- **Objective:** Automate the candidate → evaluate → accept/reject → deploy loop with human approval gates.
- **Features:** Candidate model worker, automated evaluation gates, automatic rollback upon regression, audit trail for every promotion.
- **Architecture changes:** Evaluation engine v2 (gates), deployment orchestration; version registry promotion logic.
- **Research goals:** Evaluation-gate thresholds; prevention of improvement-through-drift; automatic rollback response time.
- **Required components:** Gate engine, registry promotion, audit log, alerting.
- **Success criteria:** The loop runs unattended for a supervised evaluation window; any regression triggers rollback; every promotion has a full audit trail.
- **Tests:** Gate logic tests, rollback drills, audit-log completeness tests, adversarial gate tests.
- **Risks:** Reward hacking on gates; evaluation gaming; silent distribution drift; model collapse from low-diversity feedback.
- **Exit criteria:** Astra 0.9 released with a supervised continuous-improvement demonstration.

## Phase 7 — First Stable System

- **Objective:** Integrate all subsystems into a single, gated, auditable release.
- **Features:** Full-stack Astra system: tokenizer → model → memory → learning → gates → registry; documented deployment.
- **Architecture changes:** Final interfaces freeze; full audit story; release process automated.
- **Research goals:** End-to-end system reliability; measuring the full loop latency/cost.
- **Required components:** All engines integrated; E2E tests; release checklist automation.
- **Success criteria:** Full E2E test suite green; a supervised continuous-improvement run over >1 week with zero silent regressions; complete document set.
- **Tests:** E2E, integration, adversarial, rollback drills, documentation completeness checks.
- **Risks:** Integration complexity; maintainability; drift.
- **Exit criteria:** Astra 1.0 released.

## Phase 8 — Continuous Evolution

- **Objective:** Operate controlled continuous improvement over long periods.
- **Features:** Scheduled candidate training, durable memory growth, regression watch, drift detection, human review queues.
- **Architecture changes:** Operational tooling; model lineage graphing; long-term telemetry.
- **Research goals:** Long-horizon stability; forgetting measurement over many iterations; memory lifecycle at scale.
- **Required components:** Ops tooling, long-term metrics store, drift detectors.
- **Success criteria:** Successive minor releases show monotonic non-regression on core suite across ≥ 1 year simulated; rollback ever available.
- **Tests:** Long-run soak tests, drift-simulation tests.
- **Risks:** Cumulative drift; cost growth; incentive misalignment.
- **Exit criteria:** Astra 1.x series demonstrates sustained controlled improvement.

## Phase 9 — Advanced Architecture

- **Objective:** Branch to architecture research validated under the same rigor.
- **Features:** Attention-alternative experiments, efficient attention, advanced memory architectures, compression, quantization, tool use.
- **Architecture changes:** Research branch in `experiments/`; parallel architecture track.
- **Research goals:** As defined in `docs/RESEARCH.md`.
- **Success criteria:** At least one experimental architecture beats the established core on a defined benchmark with the same data budget, plus a documented ablation.
- **Tests:** The full benchmark suite plus architecture-specific probes.
- **Risks:** Perspective drift; losing eval discipline.
- **Exit criteria:** Astra 2.0 released — optionally adopting a superior validated architecture.

---

## Guiding Rules

- Minor releases are feature milestones; patches are fixes.
- Nothing advances to the next phase without the previous phase's exit criteria being met.
- Exit criteria are minimums; evidence may warrant additional gates.
- Phases may run partially in parallel (e.g., research track independent of ops track) but never without recording the split.