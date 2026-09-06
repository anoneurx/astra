# Astra Model Specification

> The neural architecture of Astra's Transformer-based language model.

**STATUS: VALIDATED (as design)** — architecture baseline implemented in the NumPy reference; component mechanics (embeddings, RoPE, causal attention, RMSNorm, SwiGLU FFN, PreNorm residuals, tied head) were empirically validated at toy scale in Phase 0 (docs/PHASE0.md, H0.2–H0.5) before scaling.

---

## 1. Overview

Astra's initial architecture is a standard **decoder-only causal Transformer** with:
- Multi-head self-attention with causal masking
- RMSNorm normalization
- SwiGLU feed-forward networks
- Gemma-style or LLaMA-style residual connections (separate pre-/post-norm per sub-block per config)
- Rotary positional embeddings (RoPE) applied to Q and K
- Pre-normalized blocks (PreNorm)
- Tied or untied output embeddings (per config, default tied v1)
- Parameter scaling following a modified Chinchilla-style heuristic, adjusted small-scale

The design is intentionally **scalable from 100M to multi-billion parameters** by config, not by code path. There is a single model implementation parameterized by config.

---

## 2. Decisions & Rationale

| Area | Decision | Rationale | Status |
|---|---|---|---|
| Architecture family | Decoder-only Transformer | Proven for causal LM training | VALIDATED (standard) |
| Task | Causal language modeling | Next-token prediction; foundation for all downstream | VALIDATED (standard) |
| Attention | Multi-head (MHA), causal mask, RoPE on Q/K | MHA baseline; RoPE gives length generalization without learned position params | PROPOSED (RoPE to be ablated) |
| Normalization | RMSNorm | No mean-centering; simpler, faster; standard for modern LMs | PROPOSED (ablate vs LayerNorm) |
| FFN | SwiGLU | Improved learning dynamics vs plain GELU MLP; standard | PROPOSED (ablate vs GELU) |
| Residual | PreNorm with separate per-sub-block normalization | Matches LLaMA-style; stable training | PROPOSED |
| Positional encoding | RoPE | Length generalization, no position-embedding parameters | PROPOSED |
| Init | Truncated normal/He-style scaled by width; standard GPT/NanoGPT recommendations | | PROPOSED |
| Activation | SiLU inside SwiGLU; GELU (approx) as alternative | | PROPOSED |
| Scaling | Power-law scaling; start small | See § 7 | VALIDATED (as policy) |

---

## 3. Module Details

### 3.1 Embeddings

- Token embeddings: `nn.Embedding(vocab_size, d_model)` — config `wte`.
- Non-tied output head variant: `Linear(d_model, vocab_size)` on final norm output (config `tie_embeddings: true` default).
- Embedding init: normal with small std relative to width to avoid variance blowup at forward.

**STATUS: VALIDATED (mechanics; toy scale)** — tied/untied head path gradient verified by
finite-difference gradcheck (H0.2) and toy training reached target loss (H0.3).

### 3.2 Positional Encoding (RoPE)

Rotary embeddings per `ROPE` (Su et al. 2021) applied to query/key pairs:

```
d = d_head
θ_i = base^(-2i/d)   for i in 0..d/2
Rot(q, pos) = apply rotation by pos*θ_i per pair
```

Default `rope_theta = 10000`, extendable for length generalization. Config must record θ base, whether `freqs_cis` are precomputed, and truncation strategy for long context.

**STATUS: VALIDATED (mechanics; toy scale)** — RoPE forward/backward agree with
finite differences (H0.2); token positions correctly permuted under training (H0.3).
Ablation vs learned absolute positional embeddings still planned at Phase 3.

### 3.3 Attention

- Head count `n_heads`; optionally GQA later (Grouped Query Attention) at larger scales (research).
- KV-cache supported (CPU/Rust; optional GPU).
- Causal mask enforced. Attention bias: none (masking only).
- Dropout on attention probs during training specified in config (default 0.0 for pretraining of small models; can be enabled).

**STATUS: VALIDATED (mechanics; toy scale)** — causal MHA, masking-only bias, and
tied-head gradient paths pass finite-difference gradcheck (H0.2).
GQA/KV-cache remain later-scale research.

### 3.4 Normalization (RMSNorm)

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * γ
```

- `eps` configurable (default 1e-6). Applied before attention and before FFN (PreNorm blocks).

**STATUS: VALIDATED (mechanics; toy scale)** — RMSNorm forward/backward agreed with
finite differences for arbitrary inputs (dense-gradient test, 1.5e-9 max error, H0.2).

### 3.5 Feed-Forward Network (SwiGLU)

```
FFN(x):
    gate = SiLU(x W_g)          # d_out
    up   = x W_u                # d_out
    down = (gate ⊙ up) W_d      # d_in
```
with `d_out = 2/3 * 4 * d_model` (LLaMA-style hidden factor). Standard SwiGLU has internal dimensions ~ `8/3 · d_model` total.

**STATUS: VALIDATED (mechanics; toy scale)** — SwiGLU gate/up/down gradient paths
(the two Phase-0 backward bugs) pass finite-difference gradcheck (H0.2); LLaMA-style
8/3·d hidden factor remains a larger-scale target.

### 3.6 Residual Connections

- PreNorm: `x_out = x + SubBlock(Norm(x))` for both attention and FFN sub-blocks.

**STATUS: VALIDATED (mechanics; toy scale)** — PreNorm residual paths incl. per-block
norm/input routing verified by finite-difference gradcheck (H0.2).

### 3.7 Output Head

- Final norm, then `Linear(d_model, vocab_size)`.
- Optional weight tying between head and `wte` (default tied in v1 for parameter efficiency); untied variant under research.

**STATUS: VALIDATED (mechanics; toy scale)** — final-norm + Linear head with default
weight tying trained to target loss (H0.3); untied variant under research.

---

## 4. Initialization

Target stable training at the smallest scale first. Baseline (from standard practice):

- Weights: `nn.init.normal_(std = sqrt(2 / (fan_in + fan_out)))` for linear layers in FFN/attn; embed head normal with std `1/sqrt(d_model)`; final output projection scaled by `1/sqrt(num_layers)` (GPT-2 style residual scaling is an option per config; default simple).
- Biases: disabled by default (RMSNorm none; linear no bias).

Revisit with a small initialization sweep experiment at Phase 1.

**STATUS: PROPOSED**

---

## 5. Parameter Scaling

- Scale d_model, n_layers, n_heads, d_ffn following a smooth config matrix rather than hard-coded architectures.
- Config template below for the initial family; values chosen for feasibility on commodity hardware, not arbitrary size.

### Astra model family (development targets, NOT guaranteed requirements)

| Model | d_model | n_layers | n_heads | d_ffn (SwiGLU up-gate total) | Params ~est (non-tied) | Purpose |
|---|---|---|---|---|---|---|
| Astra-100M | 768 | 12 | 12 | 2048 | ~100M | Feasibility/baseline |
| Astra-300M | 1024 | 24 | 16 | 2732 | ~300M | Core v1 candidate |
| Astra-700M | 1229 (≈) / 1280 | 28 | 20 | 3413 | ~700M | Core v1 candidate |
| Astra-1B | 1536 | 32 | 24 | 4096 | ~1B | Scale study |
| Astra-3B | 2048 | 40 | 32 | 5461 | ~3B | Scale study |
| Astra-7B | 3072 | 32 | 32 | 8192 | ~7B | Scale study |

Exact numbers are to be recomputed via the parameter-count script (tools/param_count.py) before training. These are **development targets** set for tractability, not commitments.

---

## 6. Training Configuration (Baseline v0.1)

| Key | Value (Phase 1 baseline) |
|---|---|
| Optimizer | AdamW (β=(0.9,0.95), eps 1e-8) |
| LR schedule | Cosine with linear warmup (warmup_frac 0.02 target) |
| Peak LR | ~3e-4 (sweep later) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch (tokens) | 0.5M – 4M depending on scale |
| Mixed precision | bf16 (A100/H100) or fp16 (consumer) |
| Context length | 1024 (Phase 1); extend later (RoPE supports longer precompute) |
| Gradient accumulation | as needed by memory budget |

Full configs live in `configs/`.

---

## 7. Scaling Approach

- Start at Astra-100M. Verify the training loop is stable and loss decreases monotonically.
- Extrapolate next scale from loss-vs-compute data, not guesswork.
- All six candidate families are targets. A run reports params, tokens/sec, loss, perplexity.

---

## 8. Future Model Architecture Research

- Attention alternatives, efficient attention, MQA/GQA, sliding windows — see `docs/RESEARCH.md`.
- Memory-augmented attention (research, Phase 9).
- Any architecture change requires a benchmark-backed decision record.

---

## 9. Open Questions

- RoPE base and capability at >1024 context (research).
- RMSNorm eps effect at very low precision.
- Whether to tie embeddings at what scale (parameter vs quality trade—measured, not assumed).