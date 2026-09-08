# Astra Changelog

> Record of notable changes per release. Keep it accurate; detail lives in release notes and ADRs.

**STATUS: VALIDATED** — Phase 0 experiments all meet their acceptance criteria (see `experiments/phase0/REPORT.md`); Phase 2 Training Foundation (Astra 0.2.0) and Phase-3 core components (Astra 0.3.0) implemented and tested.

---

## 0.4.0 (2026-09-08) — Phase-3 Tail / Registry & Service

**Incremental release closing the Phase-3 productization tail: model registry
v1, inference service skeleton (Python + Rust), ACTIVE benchmark gates.**

- **Model registry v1** (`astra/registry.py`, `tools/registry.py`): immutable
  artifact registry (semver, commit, content-addressed config_id, data/eval
  manifest ids, sha256); `register` refuses silent re-registration unless
  `--force`; `verify` / `show` / `list`; JSON persistence. Registered
  `astra-name 0.4.0` (sha256 `dd9432cc…310`) and `astra-phase0 0.3.0`.
- **Inference service (Python)** (`service/inference.py`): stdlib HTTP
  `ThreadingHTTPServer` over the KV-cache decoder — `GET /health`,
  `POST /generate`, stateless per request.
- **Inference service (Rust skeleton)** (`service/rust`, crate `astra-rt`):
  dependency-free SHA-256 checkpoint fingerprinting + verify CLI; cross-checks
  the Python registry sha.
- **Benchmark gates ACTIVE**: `core-basic` v2 promoted ADVISORY→ACTIVE with
  adopted thresholds (`threshold_decision` in manifest); gate passes only when
  all ACTIVE suites pass.
- **Phase-3 tail tests**: 12 registry + 9 service tests. **90/90 tests pass**;
  ruff clean; Rust `cargo test` 3/3. Birth test PASS (val_loss 3.9158).
- Release notes: `docs/releases/v0.4.0.md`.

---

## 0.3.0 (2026-09-07) — Core

**Phase-3 core-improvement release: variant transformer, KV-cache inference,
extended context, quantization preview, benchmark harness v1.**

- **Transformer variants** (`astra/model/core.py`, `config.py`): `norm_type`
  (rmsnorm default / layernorm), `ffn_type` (swiglu default / gelu), `pos_type`
  (rope default / learned). Default config is bit-identical to 0.2.0 (logits,
  loss, all overlapping grads); every variant backward is float64-gradchecked.
- **Ablation rig** (`tools/ablate.py`): trains baseline-vs-variant and writes
  `experiments/ablations/` comparison summaries + `index.json`.
- **Inference service (Python)** (`astra/inference/decoder.py`): KV cache grows
  on demand; `decode_token` / `decode` with temperature / top-k / top-p /
  sliding-window decode matches reference per-position logits (~1e-7).
- **Context handling**: RoPE angles + causal mask extend on-the-fly past
  `max_seq_len`; `generate(..., context_window=...)` added to `metrics.py`.
- **Quantization preview** (`astra/quantize.py`, `tools/quantize.py`): in-place
  fp16 / bf16 / int8 (per-tensor symmetric) with bounded-error report; int8
  quarters weights.
- **Benchmark harness v1** (`astra/evaluation/bench.py`, `tools/benchmark.py`,
  `benchmarks/suites/core-basic.json`): manifest suites, pluggable scorers,
  thresholds, sha256-tied JSON reports; `core-basic` PASS 100% on both the
  phase-0 checkpoint and the Astra-name fine-tune (`checkpoints/name/resumed/final.npz`).
- **Phase-3 tests**: variant gradchecks, decoder equivalence, quantize error
  bounds, benchmark gates + determinism. **69/69 tests pass**; birth test PASS
  (val_loss 3.9158 unchanged). Ruff clean on new files.
- Release notes: `docs/releases/v0.3.0.md`.

---

## 0.2.0 (2026-09-07) — Training Foundation

**Phase-2 hardening of the training pipeline: reproducible, configurable, audited.**

- **Optimizer/scheduler registry** (`astra/training/optim.py`): `OPTIMIZER_REGISTRY`,
  `SCHEDULE_REGISTRY`, `build_optimizer`/`build_schedule`; config-driven via
  `optimizer`/`scheduler` keys. Unknown names fail fast.
- **Gradient accumulation** (`accum_steps` in `astra/training/loop.py`): gradients
  accumulated across micro-batches, averaged before clip/step; `accum_steps=1` is
  bit-identical to the Phase-0 loop.
- **Experiment tracking** (`astra/experiments/store.py`, `tools/experiments.py`):
  `ExperimentStore` with `list`/`show`/`query`; every `train()` run auto-logged.
- **Run-manifest audit fields**: `git_commit` + `environment` recorded in
  `report.json` and checkpoint manifests.
- **Phase-2 exit tests**: same-seed reproduction (bit-identical trajectory),
  scheduler boundary/clamp/registry, and gradient-accumulation equivalence
  (N micro-batches ÷ N == one macro batch). **42/42 tests pass**.
- Release notes: `docs/releases/v0.2.0.md`.

---

## 0.1.0 (2026-09-06) — Birth

**First tagged release.** Working end-to-end LM stack implemented from scratch.

- Custom byte-level BPE tokenizer (vocab 800; round-trip exact; 6.84 B/token).
- Transformer core (RoPE, causal MHA, RMSNorm, SwiGLU, PreNorm, tied head) with
  float64 gradcheck-validated backward passes.
- Training loop (warmup + cosine decay, weight decay, grad clip, leak-gated val).
- Deterministic checkpoints + run manifests; bit-identical across thread counts.
- Basic text generation + evaluation harness → checksum-tied JSON eval report.
- Release notes: `docs/releases/v0.1.0.md`; artifact registry:
  `experiments/phase0/artifact_registry.json`.

---

## 0.0.1 (2026-09-06)

**Initial research foundation.** Project scaffolding, documentation master spec, toy training scaffold.

- Project structure created (`docs/ARCHITECTURE.md` § 7).
- Master technical specification authored.
- Environment/installation docs drafted.
- CI: unit-test workflow (`.github/workflows/ci.yml`) + `Makefile` added.
- **Phase 0 experiments validated:**
  - Byte-level BPE tokenizer (EX-01, H0.1 met) — round-trip exact, 6.84 B/token.
  - Finite-difference gradcheck of every layer (EX-02, H0.2 met) — 3 backward bugs found and fixed
    (attention output projection, SwiGLU norm-input routing, test float64 FDM).
  - Toy transformer (133,440 params) trains to val CE 2.535 < 4.0 (EX-03, H0.3 met).
  - Bit-identical loss trajectories across thread counts (EX-04, H0.4 met).
  - Leak-free train/val/eval splits — 0 overlapping 13-grams (EX-05, H0.5 met).
  - Evaluation harness → checksum-tied JSON report (EX-06, H0.6 met): val loss 2.535, acc 27.1%.
- Tokenizer quality report written (`experiments/phase0/tokenizer_report.md`).

---

*Format: [SemVer](https://semver.org) per `docs/VERSIONING.md`; backward-incompatible changes noted explicitly.*