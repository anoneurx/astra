# Phase 0 — Research Plan

> Concrete, falsifiable plan for the Phase 0 "Research" milestone (Astra 0.0.1).
> This doc operationalizes the seven Phase 0 areas into hypotheses, experiments,
> and acceptance criteria. Results are recorded per experiment in `experiments/phase0/`.

**STATUS: VALIDATED** — all hypotheses (H0.1–H0.5) met; see `experiments/phase0/REPORT.md`.
VALIDATED / REJECTED when its measurement exists.

---

## 0.1 Scope

Phase 0 proves that the Astra stack is **buildable, reproducible, and honest**
at toy scale before any real corpus or GPU work begins. It requires no GPU and
works on consumer hardware with a from-scratch NumPy implementation (current
environment: 8 CPU cores, 29 GB RAM, no GPU, no PyTorch available).

The NumPy implementation is a deliberate first-principles choice for Phase 0:
it exercises tokenizer, model math, training loop, evaluation, and
reproducibility contracts with zero heavy dependencies. The architecture
decision record (ADR-0002) documents the backend-agnostic interface that lets a
PyTorch reference implementation replace it in Phase 1 without changing the
pipeline contract.

## 0.2 The Seven Areas → Concrete Questions

| # | Area | Concrete Phase-0 question |
|---|---|---|
| 1 | Philosophy | Is the controlled-training + candidate-evaluation philosophy operationalizable, i.e., can we define gates whose results we can actually measure today? |
| 2 | Architecture | Can we implement and numerically validate the proposed transformer core (RMSNorm, SwiGLU, RoPE, pre-norm residuals, tied head) from scratch? |
| 3 | Data strategy | Can we build a deterministic, manifest-tracked, leak-checked train/val/eval split pipeline at toy scale? |
| 4 | Compute requirements | What does a reproducible toy run cost on CPU-only hardware (time, RAM)? Baseline for later GPU budgets. |
| 5 | Training methodology | Is the training loop (AdamW, cosine+warmup, gradient clip, deterministic batching, checkpoints, resume) correct and reproducible? |
| 6 | Evaluation methodology | Can evaluation produce trustworthy numbers (perplexity/accuracy with environment recorded) and does "improvement" tracking work end-to-end? |
| 7 | Safety boundaries | Are the contamination/leak gates, data filters, and audit-lite scaffolding implemented and testable? |

## 0.3 Hypotheses (falsifiable, Phase 0)

- **H0.1 (tokenizer):** A from-scratch byte-level BPE over the toy corpus reaches
  an effective, reversible vocabulary (round-trip exactness on 100% of test
  strings) and produces a lower-bytes-per-token than raw bytes.
- **H0.2 (model):** Gradient checks (central finite difference) of every layer in
  the NumPy transformer agree to within 1e-2 relative, validating the backward pass.
- **H0.3 (training):** A ~200k-parameter toy transformer trained on the
  structured toy corpus reaches **CrossEntropy < 4.0** on the frozen validation
  split within 4000 optimizer steps on CPU.
- **H0.4 (reproducibility):** Two training runs with identical seed/config/data
  hash/thread-count reproduce bit-identical loss trajectories.
- **H0.5 (leakage):** The leakage gate (n-gram overlap between train and
  val/eval splits) reports zero cross-split collisions at n=13 on generated toy data.

## 0.4 Experiments (each is a numbered artifact in `experiments/phase0/`)

| Exp | Tests | Acceptance |
|---|---|---|
| EX-01 tokenizer | `tests/test_tokenizer.py` | H0.1 met; quality report emitted |
| EX-02 model gradcheck | `tests/test_model.py` | H0.2 met |
| EX-03 toy training | `training/train.py` (config `configs/toy_pretrain.json`) | H0.3 met |
| EX-04 reproducibility | run EX-03 twice, diff loss curves | H0.4 met (2000/2000 points bit-identical) |
| EX-05 leak gate | `tools/leak_check.py` on generated splits | H0.5 met |
| EX-06 eval harness | `evaluation/evaluate.py` → JSON eval report | metrics + env recorded |

## 0.5 Addressed questions & data-format rules for Phase 0

- Every generated dataset slice carries a manifest (`datasets/*/manifest.json`)
  with hash, license stub, cleaning rules, contamination status.
- Every training run writes a run manifest (`experiments/phase0/runs/…/manifest.json`)
  with repo commit, config, data hash, seed, hardware, thread count, metrics stream.
- Every evaluation writes a JSON eval report tied to a checkpoint checksum.
- Contamination check is part of the dataset build, not an afterthought.

## 0.6 Success criteria (exit criteria for Phase 0, from ROADMAP.md)

1. Toy model trains to **loss < 4.0** on the toy corpus **reproducibly**.
2. Tokenizer quality report exists (`experiments/phase0/tokenizer_report.md`).
3. CI green (`pytest tests/` local and GitHub Actions).
4. Spec frozen at 0.0.1; docs updated to match measured reality (no PROPOSED
   status where VALIDATED now applies).

## 0.7 Risks

- Network-bound dependency installs (torch unavailable) — mitigated by NumPy-only path.
- Nondeterminism in BLAS — mitigated by single-thread reproducibility mode for
  the definitive EX-04 measurement.
- Scope creep — all Phase-0 work tracks the seven areas above; anything beyond
  goes to `docs/RESEARCH.md` backlog.

## 0.8 Deliverables checklist

- [x] docs/PHASE0.md (this plan)
- [x] `python/astra` package: tokenizer, model, training, evaluation, safety
- [x] `configs/toy_pretrain.json`, `configs/tokenizer.json`
- [x] entrypoints: `tokenizer/train_tokenizer.py`, `training/train.py`, `evaluation/evaluate.py`
- [x] corpus generator + manifests (`datasets/toy/`)
- [x] tests (tokenizer, model gradcheck, optim, data, safety)
- [x] tools: `param_count.py`, `checksum.py`, `leak_check.py`
- [x] CI workflow + Makefile
- [x] experiments/phase0/reports (tokenizer, training, reproducibility), ADRs