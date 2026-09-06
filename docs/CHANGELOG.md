# Astra Changelog

> Record of notable changes per release. Keep it accurate; detail lives in release notes and ADRs.

**STATUS: VALIDATED** — Phase 0 experiments all meet their acceptance criteria (see `experiments/phase0/REPORT.md`).

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