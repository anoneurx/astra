# Astra Phases

> High-level phase plan with step-level completion status. Source of truth for
> verified state: `docs/PHASE_STATUS.md`, `docs/ROADMAP.md`, and
> `docs/CHANGELOG.md`. Each phase's steps below are checked against the repo
> evidence (release notes under `docs/releases/`).

## Status at a glance

| Phase | Release | Status |
|---|---|---|
| Phase 0 — Research | Astra 0.0.1 | ✅ Completed (validated) |
| Phase 1 — Birth | Astra 0.1 | ✅ Completed (validated) |
| Phase 2 — Foundation | Astra 0.2.0 | ✅ Completed (validated) |
| Phase 3 — Core | Astra 0.3.0 / 0.4.0 | 🟡 In progress (engine complete; full-size model parked) |
| Phase 4 — Memory | Astra 0.5.0 | ✅ Completed (G1–G6 closed) |
| Phase 5 — Learning | Astra 0.7.0 | ✅ Core completed (preference path landed in Phase 7) |
| Phase 6 — Self-Improvement | Astra 0.9.0 | ✅ Completed (exit criteria all met) |
| Phase 7 — Stable | Astra 1.0.0 | 🟡 In progress (increments 1–5 done; supervised window started) |
| Phase 8 — Evolution | Astra 1.x | 🟡 Engineering complete (operator toolkit + compaction; wall-clock/compute items open) |
| Phase 9 — Advanced | Astra 2.0 | 🔬 Research |

Legend: ✅ done · 🟡 in progress · ⬜ not started · 🔬 research/planned

---

## Phase 0 — Research · `Astra 0.0.1` · ✅ Completed

Foundation, specification, and tooling (see `experiments/phase0/REPORT.md`).

- [x] Define Astra's philosophy
- [x] Architecture specification
- [x] Data strategy
- [x] Compute requirements
- [x] Training methodology
- [x] Evaluation methodology
- [x] Safety boundaries

## Phase 1 — Birth · `Astra 0.1.0` · ✅ Completed

Working small language model checkpoint (toy corpus; loss < 4.0 reproducible).

- [x] Custom tokenizer
- [x] Embeddings
- [x] Transformer
- [x] Attention
- [x] Feed-forward layers
- [x] Training loop
- [x] Checkpoints
- [x] Basic text generation

## Phase 2 — Foundation · `Astra 0.2.0` · ✅ Completed

Reproducible, configurable training pipeline (`docs/releases/v0.2.0.md`).
Deferred within phase: LR sweeps, mixed precision, distributed hooks.

- [x] Better tokenizer
- [x] Better dataset pipeline
- [x] Improved training stability
- [x] Validation system
- [x] Loss/perplexity monitoring
- [x] Reproducible experiments
- [ ] LR sweeps / mixed precision / distributed hooks (deferred)

## Phase 3 — Core · `Astra 0.3.0 / 0.4.0` · 🟡 In progress

Stable neural core: config variants, inference, context handling, quantization
preview, benchmarking, and the productization tail (registry v1, inference
service skeleton, ACTIVE gates). **Remaining:** full-size core model
(Astra-300M/700M, parked per `docs/PLAN-ASTRA-5M.md`).

- [x] More capable Transformer (config variants + ablation rig)
- [x] Better inference (KV-cache decoder, top-k/top-p, streaming, extended context)
- [x] Context handling (RoPE theta-extension, sliding window)
- [x] Quantization experiments (fp16/bf16/int8 preview)
- [x] Model benchmarking (harness v1 + suites + gates)
- [ ] Full-size core model release (Astra-300M/700M, parked on Astra-5M plan)

## Phase 4 — Memory · `Astra 0.5.0` · ✅ Completed

External memory system; gaps G1–G6 closed (`docs/releases/v0.5.0.md`).
Positive RAG-generation signal is measured on the Astra-5M model (parked).

- [x] Short-term memory (`SessionMemory`)
- [x] Long-term memory (durable store + lifecycle)
- [x] Semantic retrieval (embedding + hybrid ranking + token budget)
- [x] Experience storage (append-only store + audit)
- [x] Knowledge management (correction revisions, expiry, dispute/resolve)
- [x] Retrieval evaluation (long-form QA set + `core-retrieval` suite)
- [ ] RAG generation-benefit measured on Astra-5M (parked)

## Phase 5 — Learning · `Astra 0.7.0` · ✅ Core completed

Feedback → validated training examples → candidate training
(`docs/releases/v0.7.0.md`). The preference-pair trainer, a Phase-5 exit gap,
was completed in Phase 7.

- [x] User feedback (intake cascade + trust tiers + quarantine)
- [x] Corrections (correction → validated example → store)
- [x] Reward signals (automated-verification tier)
- [x] Experience → training examples (`ExperienceStore`, dedup + provenance)
- [x] Dataset filtering (validated-only training, no-leak replay); *live leak coverage on `learning/store` still open*
- [x] Preference-pair (ranking) trainer (*completed in Phase 7*)
- [ ] Feedback-confidence reward-weight research (open research)

## Phase 6 — Self-Improvement · `Astra 0.9.0` · ✅ Completed

Automated candidate → evaluate → accept/reject → deploy loop with human
approval gates, audit trail, and auto-rollback (`docs/releases/v0.9.0.md`).
Exit criteria all met.

- [x] Automatic evaluation (GateEngine: `gain` / `no_regress` / `absolute`)
- [x] Candidate model training (off active checkpoint from validated examples)
- [x] Model comparison (deterministic base-vs-candidate CE)
- [x] Regression testing (no-forgetting gate)
- [x] Automatic checkpoint selection (registry `promote`)
- [x] Rollback (`--rollback-drill` + append-only audit trail)
- [x] Demo: target −0.775 CE, regression +0.009 CE → ACCEPT → promoted 0.9.0

## Phase 7 — Stable · `Astra 1.0.0` · 🟡 In progress

First stable, integrated, gated, auditable release. Increments 1–5 done; the
long supervised window (continuous improvement > 1 week, zero silent
regressions) is the remaining exit item.

- [x] Stable model architecture (interface freeze, `__all__` everywhere)
- [x] Stable training pipeline (`make lint` / `make typecheck` green)
- [x] Memory (memory-aware candidate training via `search_memory_block`)
- [x] Learning pipeline (DPO preference path + memory context)
- [x] Evaluation (full E2E walk + 8-test E2E suite + gates)
- [x] Versioning (semver + registry promote/rollback/history)
- [x] Documentation (docs-completeness gate in `make release`)
- [x] Reproducible builds (release automation `tools/release.py` + checksums)
- [x] Preference training path (DPO-style margin objective, verified E2E)
- [x] Release checklist automation (`tools/release.py check|tag`, 7 gates)
- [ ] Long supervised window (started with `tools/continuous.py`; > 1 week wall-clock, Astra-5M)

## Phase 8 — Evolution · `Astra 1.x` · 🟡 Engineering in progress

Operate continuous controlled improvement over long periods. The operator
toolkit (coding phase) is done — ``astra/learning/evolution.py`` (append-only
long-term metrics store, drift detector, lineage) + ``tools/continuous.py``
(drives iterations, records promotions, watches for regression/silent drift,
rollback always available) + 7 tests. The six aspirational tracks below require
real data/compute beyond the in-repo toy dataset and stay open.

- [x] Operator tooling (continuous loop, drift watch, regression watch)
- [x] Long-term metrics store (append-only, per-promotion snapshots)
- [x] Drift detection (improving/stable/regressing + silent-drift flag)
- [x] Human review queues (supervised: alert-and-stop on drift by default)
- [x] Model lineage graphing (`lineage()` from registry `superseded_by` chain)
- [x] Rollback automation (`--auto-rollback`, audited)
- [ ] Scheduled candidate training (continuous supervised window, > 1 week)
- [x] Durable memory growth (`MemoryStore.compact()`: prune + dedupe, audited)
- [ ] Larger models
- [ ] Better datasets
- [ ] Improved reasoning
- [ ] Better memory
- [ ] More efficient learning
- [ ] Hardware optimization

## Phase 9 — Advanced · `Astra 2.0` · 🔬 Research

Architecture research branch, validated under the same rigor.

- [ ] Advanced reasoning
- [ ] Tool use
- [ ] Planning
- [ ] Multimodal research
- [ ] More sophisticated continual learning
- [ ] Distributed training/research