# Phase 0 — Experiment Report

Date: 2026-09-06 · Repo state: uncommitted working tree (Phase 0 code) ·
Machine: 8-core CPU, 29 GB RAM, no GPU · Python 3.13.12 · NumPy-only implementation.

## Environment

| item | value |
|---|---|
| python | 3.13.12 (system interpreter, no torch) |
| numpy | 2.3.x |
| CPU | 8 cores, OPENBLAS_NUM_THREADS=8 (dynamical runs) and =1 (EX-04 definitive) |
| backend | from-scratch NumPy reference (ADR-0002: backend-agnostic contract) |

## Hypothesis results

### H0.1 — Tokenizer (EX-01): VALIDATED

- Byte-level BPE, vocab 800 (256 bytes + 4 specials + 540 merges), trained on
  `datasets/toy/train.txt` (1,307,660 bytes).
- Round-trip exact on 100% of test strings (incl. lone surrogates): **True**.
- 6.837 bytes/token vs 1.0 for raw bytes → reversible + compression. Tokens/char 0.1463.
- Unknown tokens used: 0. Evidence: `tests/test_tokenizer.py`,
  `experiments/phase0/tokenizer_report.md`.

### H0.2 — Model gradients (EX-02): VALIDATED

- Every hand-derived backward pass (RMSNorm, Linear, attention w/ RoPE + causal
  mask, SwiGLU, tied head) checked vs central finite differences in float64.
- 3 real bugs found & fixed, then all params agree to rel < 1e-2:
  1. Attention output projection was treated as identity in its own backward
     (`dz` did not propagate through `out_w`).
  2. SwiGLU passed the norm *output* instead of the norm *input* to the norm
     backward (`ln2.backward(dh, h)` → corrected to `xm`).
  3. Finite-difference checks in the test ran in float32 (cancellation); now run
     on a float64 model copy.
- Evidence: `tests/test_model.py::test_gradcheck_all_layers` (2 seeds), plus the
  training/loss-reduction sanity test.

### H0.3 — Toy training (EX-03): VALIDATED

- Config: `configs/toy_pretrain.json` — d_model 64, 2 layers, 4 heads, d_ffn 128,
  seq 64, batch_seq 8, vocab 800 → **133,440 params**.
- Frozen val split reaches **CE 2.5352 (ppl 12.62)**, below the 4.0 bar, at step
  2000 (≤ 4000 allowed). Loss curve: 6.70 → 2.44 train; val 4.94 → 2.54.
- Evidence: `checkpoints/phase0/report.json`, `checkpoints/phase0/final.npz`.

### H0.4 — Reproducibility (EX-04): VALIDATED

- Two runs, identical seed/config/data, OPENBLAS_NUM_THREADS=8 vs =1.
- All 2000/2000 loss points bit-identical; val_history and final val identical.
- Evidence: `/tmp/runs/run1.json`, `/tmp/runs/run2.json`.

### H0.5 — Leakage gate (EX-05): VALIDATED

- n=13 token-gram overlap between train and held-out splits:
  - train vs val: **0 / 15196** held-out 13-grams (`leak_free: true`)
  - train vs eval: **0 / 6880** held-out 13-grams (`leak_free: true`)
- The gate correctly *caught* 69 overlapping 13-grams before corpus hygiene
  (template boilerplate); decontamination removed 37/200 val and 25/100 eval docs.
- Evidence: `tools/leak_check.py` output, `datasets/toy/corpus.py --decontaminate`,
  safety gate inside `training/train.py`.

### H0.6 — Eval harness (EX-06): VALIDATED

- `evaluation/evaluate.py` loads `checkpoints/phase0/final.npz`, frozen val loop:
  **val_loss 2.5352, ppl 12.62, next-token accuracy 27.1%** (matches training's val).
- Generation sanity: 200 tokens, 132 unique, 4-gram repetition fraction **0.0**,
  decode ~218 tok/s (NumPy reference inference).
- Bug found & fixed: val loss was under-weighted by seq-len in the harness
  (`total += loss * B` → factor `B*T`).
- Evidence: `checkpoints/phase0/eval_report.json`.

## Exit criteria (docs/PHASE0.md § 0.6)

1. Toy model CE < 4.0 on frozen val, reproducibly: **met (2.5352, twice)**.
2. Tokenizer quality report: **met** (`experiments/phase0/tokenizer_report.md`).
3. CI green: **pending** (workflow added in this phase; local `pytest tests/` green).
4. Spec frozen at 0.0.1, docs updated (PROPOSED → VALIDATED where measured):
   **this report + doc-status sweep**.