# Overnight Training Plan — Astra-5M (Phase 3 release candidate)

> Status: ON HOLD — not started. Plan drafted 2026-09-07. Follow when ready.
> Hardware decision: train on this CPU-only box (8 cores / 29 GB RAM / no GPU).
> No hardcoded replies; model-driven chat only.

## The binding constraint

CPU-only machine. Compute scales ~linearly with params: the 133k-param toy trains at
~4,051 tok/s, so HARDWARE.md's Astra-100M/300M class (wants 8–24 GB VRAM) would run at
~5–20 tok/s here — infeasible. Astra-5M is the right-size model for this box.

## Decisions already locked

- **Architecture** (from ablations, all variants beat baseline at 100 steps):
  `norm_type=rmsnorm`, `ffn_type=swiglu`, `pos_type=rope`, `tie_embeddings=true`.
- **Model config** (~5.3M params):

  | param | value |
  |---|---|
  | vocab_size | 8192 (new tokenizer) |
  | d_model / n_layers / n_heads | 256 / 6 / 8 |
  | d_head / d_ffn / max_seq_len | 32 / 512 / 128 |
  | tie_embeddings | true |

- **Training** (TRAINING.md defaults): batch_seq 16 × ctx 128 = 2048 tok/step,
  ~15k steps over ~30M tokens, AdamW, wd 0.1, clip 1.0, peak_lr 3e-4,
  cosine → 1e-5, warmup ~300 steps, `val_every 2000`, resume-friendly.

## Work packages

1. **Data**: source ~30–50M tokens of licensed real text (public-domain books +
   permissive docs corpus). Per-source manifest: provenance, license, hash, dedup,
   n=13 leak check (existing `corpus.py` / `leak_check.py`). Frozen train/val/eval.
2. **Tokenizer**: train ByteLevelBPE at vocab 8192 on cleaned corpus
   (TOKENIZER.md); version-pinned. Do NOT reuse `toy_bpe.json` (vocab 800).
3. **Config + train**: new configs/<name>.json wiring data + tokenizer + model;
   overnight/weekend runs with resume (Phase 2 infra already supports).
4. **Gate + registry**: pass `core-basic` (val_ppl, decode tok/s, repetition) on
   the real checkpoint, checksum/registry entry, release notes v0.3.0 →
   satisfies Phase 3 exit criteria.

## Runtime estimate (honest)

~90–140 tok/s predicted on 8 cores → ~30M tokens ≈ 2.5–4 days of compute,
spread over overnight chunks (~1 week wall-clock). Run with
`OPENBLAS_NUM_THREADS=8`, nothing else on the box.

## Not in scope (parallel tracks)

- Astra-100M+ on cloud GPU (16–24 GB VRAM) — later phase / separate box.
- Hardcoded/scripted chat replies — rejected.