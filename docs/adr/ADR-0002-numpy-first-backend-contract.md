# ADR-0002 — NumPy-first reference implementation with a backend-agnostic contract

Status: ACCEPTED · Date: 2026-09-06

## Context

Phase 0 must validate the full training stack (tokenizer, model math, training
loop, evaluation, safety, reproducibility) on commodity hardware. The runtime
has **no GPU and no PyTorch install available**; installing heavy deps is
network-bound and slow. Later phases (1+) will scale on GPU/PyTorch (or a
native backend).

## Decision

Implement the reference stack from scratch on **NumPy only**, while keeping a
backend-agnostic interface (`python/astra/model/core.py`) so that a PyTorch
implementation can replace the math kernels without changing the pipeline
contract (config → data → model → training loop → checkpoint → eval report).

## Consequences

- Positive: zero heavyweight dependencies; gradient math is derived by hand and
  made auditable; every construct (RMSNorm, RoPE, causal MHA, SwiGLU, tied head)
  is checked against finite differences before any training signal is trusted.
- Positive: deterministic, bit-identical reproducibility even across BLAS thread
  counts (verified: 2000/2000 loss points identical, EX-04).
- Negative: slower than optimized backends (~218 tok/s inference at 133k params);
  larger-scale phases will use a faster backend behind the same interface.
- Evidence: `experiments/phase0/REPORT.md` (H0.2, H0.3, H0.4).