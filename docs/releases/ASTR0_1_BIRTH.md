# Astra 0.1 — Birth

**Astra 0.1.0** (2026-09-06): the first tagged release — a working, verifiable end-to-end language model stack implemented from scratch in NumPy, measured and reproducible.

Every component is defined below with its specification, implementation, and measured results.

---

## 1. Custom Tokenizer

**Specification:** Byte-level BPE, vocabulary built from 256 raw byte tokens + reserved special tokens + learned merges. Text → UTF-8 bytes → byte-token IDs → BPE merge application → final token IDs. Lossless round-trip decode guaranteed.

**Implementation** (`python/astra/tokenizer/bpe.py` — `ByteLevelBPE`):
- **Base vocabulary:** IDs 0–255 map to raw bytes; IDs 256+ reserved for special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`).
- **Training:** Iteratively finds the most frequent adjacent byte-pair (via packed `np.bincount`), applies merges, creates new merge-token IDs. Merges list persisted as JSON artifact.
- **Encoding:** UTF-8 encode with `surrogatepass`, assign byte IDs, apply merges in order.
- **Decoding:** IDs → bytes → UTF-8 string with `surrogatepass` error handling.
- **Quality:** `quality_report()` computes bytes/token, tokens/character, round-trip exactness, vocabulary coverage.

**Config** (`configs/tokenizer.json`): vocab_size=800, num_special=4, min_frequency=2, special_names=[`<pad>`, `<bos>`, `<eos>`, `<unk>`].

**Measured results** (toy corpus, vocab 800):
| Metric | Value |
|---|---|
| Vocab size (final) | 800 (256 bytes + 4 specials + 540 merges) |
| Bytes/token | **6.84** (vs raw 1.0 byte/byte; compression via merges) |
| Round-trip exact | **100%** (incl. lone surrogates, CJK, emoji, control chars) |
| Unknown tokens | 0 (byte-level guarantees full coverage) |

**Artifact:** `tokenizer/artifacts/toy_bpe.json` — SHA-256 `e964964d3d658766684a76c052bcdbe926d1f21b9d099976e519dd2596bc3d`.

**Correctness contract** (`tests/test_tokenizer.py`): round-trip identity over ASCII, accented Latin, CJK, RTL, emoji, control chars, lone surrogates, null bytes, very long strings; determinism; version stability.

---

## 2. Embeddings

**Specification:** Token embeddings (`wte`) map token IDs to dense vectors of dimension `d_model`. Positional information encoded via RoPE (rotary positional embeddings) applied to Q and K vectors, not via learned position embeddings. Output head is optionally tied to `wte`.

**Implementation** (`python/astra/model/core.py`):
- **`Embedding` class:** `nn.Embedding(vocab_size, d_model)` equivalent. Weight matrix `w` of shape `(vocab, d_model)`, initialized `normal(0, 0.02)`. Forward: `w[ids]`. Backward: `np.add.at` accumulation into `w.grad`.
- **`RoPE` class:** Computes rotation angles `θ_i = base^(-2i/d)` for `i ∈ 0..d/2`. Precomputes `cos` and `sin` matrices of shape `(max_seq_len, d_head/2)`. `rotate(x)` applies Givens rotation per pair: `out[even] = even*cos - odd*sin`, `out[odd] = even*sin + odd*cos`. `unrotate_grad()` computes the transposed rotation for backward pass.
- **Tied output head:** Final `Linear(d_model, vocab_size)` uses `wte.w.T` as its weight matrix — the same embedding matrix serves dual purpose.

**Config** (toy): `d_model=64`, `n_heads=4`, `d_head=16`, `max_seq_len=64`, `rope_theta=10000.0`, `tie_embeddings=true`.

**Key design decisions:**
- RoPE gives length generalization without learned position parameters.
- Tied head reduces parameter count and improves parameter efficiency.
- Initialization: He-normal for Linear layers (`std = sqrt(6/(fan_in+fan_out))`), `std=0.02` for embeddings.

**Measured:** Embedding lookup + RoPE rotation + tied-head gradient paths all pass float64 finite-difference gradcheck (H0.2, max error 1.5e-9).

---

## 3. Transformer

**Specification:** Decoder-only causal Transformer with `n_layers` stacked blocks. Each block consists of a **PreNorm attention sub-block** and a **PreNorm SwiGLU feed-forward sub-block**. Final normalization followed by a tied output head.

**Architecture per block:**
```
x → RMSNorm → MultiHeadAttention(RoPE) → +x          (attention sub-block)
  → RMSNorm → SwiGLU → +x                            (FFN sub-block)
x → final RMSNorm → Linear(vocab_size, d_model) [tied] (output head)
```

**Implementation** (`python/astra/model/core.py` — `LiteLM`):
- **`LiteLM` class:** Composes `Embedding`, `n_layers` × `AttentionBlock`, `n_layers` × `SwiGLUBLock`, `RmsNorm(ln_f)`. `forward(ids)` → `logits` shape `(B, T, vocab_size)`.
- **`RmsNorm`:** `x / sqrt(mean(x²) + eps) * γ`. Forward and backward analytically derived; eps=1e-6 default.
- **Parameter count:** Formula: `wte(w*vocab*d_model) + Σ attn{qkv, out, ln1} + Σ ffn{wg, wu, wd, ln2} + ln_f`. Toy config: **133,440 params**.

**Model family config template:**
| Model | d_model | n_layers | n_heads | d_ffn | Params ~ |
|---|---|---|---|---|---|
| Astra-100M | 768 | 12 | 12 | 2048 | ~100M |
| Astra-300M | 1024 | 24 | 16 | 2732 | ~300M |
| Astra-1B | 1536 | 32 | 24 | 4096 | ~1B |
| Astra-7B | 3072 | 32 | 32 | 8192 | ~7B |

Single implementation parameterized by config — no hard-coded architecture paths.

**Measured:** Full transformer forward + backward passes validated against float64 finite-difference gradient checks on every layer (H0.2). Three backward bugs found and fixed during gradcheck (attention output projection routing, SwiGLU norm-input routing, test float64 FDM).

---

## 4. Attention

**Specification:** Multi-head causal self-attention with RoPE positional encoding. Causal masking via lower-triangular bias matrix (`-1e9` for masked positions). No attention bias beyond masking. KV-cache architecture supported.

**Implementation** (`python/astra/model/core.py` — `AttentionBlock`):
- **Forward:**
  1. `h = RMSNorm(x)` — pre-normalization
  2. `q, k, v = Linear(h)` split into 3 heads of `d_head` each
  3. `q, k` rotated by RoPE
  4. `scores = (q @ k^T) * (d_head^-0.5)` — scaled dot-product
  5. `scores = scores + causal_mask[:T, :T]` — causal masking
  6. `att = softmax(scores)` — per-head attention weights
  7. `z = att @ v` — weighted value aggregation
  8. `z = z.transpose(0,2,1,3).reshape(B,T,d)` → `x + Linear(z)` — output projection with residual
- **Backward:** Computes gradients for `out_w`, `v`, `k`, `q`, `qkv` projection, and residual stream through `RmsNorm`. RoPE gradients computed via `unrotate_grad()`.
- **Causal mask:** `np.triu(fill(-1e9), k=1)` of shape `(max_seq_len, max_seq_len)`, applied as additive bias.
- **Softmax:** Numerically stable — subtract max before exp.

**Config:** `n_heads=4`, `d_head=16` (= `d_model/n_heads`), causal masking enforced.

**Key properties:**
- Causal: position `t` can only attend to positions `≤ t`.
- Multi-head: 4 independent attention heads, each with `d_head=16` dimensions.
- RoPE applied to Q and K independently; gradients flow through rotation.

**Measured:** Causal MHA, masking-only bias, and tied-head gradient paths all pass float64 finite-difference gradcheck (H0.2).

---

## 5. Feed-Forward Layers

**Specification:** SwiGLU (Sigmoid-Gated Linear Unit) feed-forward network. Three weight matrices: `wg` (gate projection), `wu` (up projection), `wd` (down projection). Internal dimension `d_ffn = 8/3 * d_model` (LLaMA-style hidden factor).

**Implementation** (`python/astra/model/core.py` — `SwiGLUBLock`):
- **Forward:**
  1. `h = RMSNorm(x)` — pre-normalization
  2. `gate = SiLU(h @ wg.w)` — gated activation (SiLU = x·σ(x))
  3. `up = h @ wu.w` — up projection
  4. `out = x + (gate ⊙ up) @ wd.w` — gated product with residual
- **Backward:**
  1. `d_ff = wd.backward(grad, gate * up)` — gradient through down projection
  2. `d_gate_h = wg.backward(d_ff * up * silu_prime(gate_in), h)` — gradient through SiLU gate
  3. `d_up_h = wu.backward(d_ff * gate, h)` — gradient through up projection
  4. `return RMSNorm.backward(d_gate_h + d_up_h, x)` — residual + norm
- **`silu(x) = x / (1 + exp(-x))`**, `silu_prime(x) = σ(x) * (1 + x * (1 - σ(x)))`

**Config:** `d_ffn=128` for toy (`8/3 * 64 ≈ 170`, truncated to 128); `d_ffn=2048` for Astra-100M; `d_ffn=2732` for Astra-300M.

**Key properties:**
- Gate mechanism allows the model to selectively filter information through the FFN.
- Pre-Norm placement ensures stable gradients.
- LLaMA-style `8/3·d_model` total internal dimension (`wg` + `wu` = `16/3·d_model`, `wd` = `d_model`).

**Measured:** SwiGLU gate/up/down gradient paths pass float64 finite-difference gradcheck (H0.2) — two Phase-0 backward bugs identified and fixed here.

---

## 6. Training Loop

**Specification:** Config-driven, reproducible training pipeline. AdamW optimizer with cosine learning rate decay and linear warmup. Gradient clipping at 1.0. Validation on frozen split every `val_every` steps. Checkpointing at fixed intervals and at best validation loss. Deterministic batching.

**Implementation** (`python/astra/training/loop.py` — `train()`):
- **Optimizer:** `AdamW` (`python/astra/training/optim.py`) — decoupled weight decay, betas=(0.9, 0.95), eps=1e-8. Weight decay disabled for norm parameters (`ln1`, `ln2`, `ln_f`). Gradient norm clipping via `clip_grad_norm()`.
- **LR Schedule:** `CosineSchedule` — linear warmup for first `warmup_steps` (2% of max_steps), then cosine decay from `peak_lr` to `min_lr`.
- **Data pipeline:** `Corpus` (token ID array + manifest) → `SeqStream` (deterministic windowed batching with epoch-scoped shuffle). Each window produces `(B, T)` input + `(B, T)` shifted target.
- **Training step:** `model.zero_grad()` → `forward_loss(ids, targets)` → `model.backward(logits, targets)` → `clip_grad_norm(model, 1.0)` → `opt.step(schedule.lr(step))`.
- **Validation:** Frozen val corpus, deterministic (`rng` seed=0), reports loss + perplexity every `val_every` steps.
- **Run manifest:** Records seed, model config, train config, data manifests, params, thread count.

**Config** (`configs/toy_pretrain.json`): max_steps=2000, peak_lr=1e-3, min_lr=1e-5, warmup_steps=50, batch_seq=8, weight_decay=0.1, grad_clip=1.0, val_every=250.

**Measured results** (toy, 133,440 params, 8 CPU threads):
| Metric | Value |
|---|---|
| Final val loss | **2.535** (target: < 4.0 ✓) |
| Perplexity | **12.62** |
| Next-token accuracy | **27.1%** |
| 4-gram repetition | **0.0** |
| Throughput | **4,051 tok/s** (2000 steps, 252s) |
| Reproducibility | **Bit-identical** (2000/2000 points, 1 vs 8 threads) |
| Training peak RSS | ~90 MB |

---

## 7. Checkpoints

**Specification:** Immutable, checksummed `.npz` weight snapshots. Every checkpoint includes weight file, optimizer state, scheduler state, config snapshot, data manifest hash, training-state snapshot (step, lr, rng state), and SHA-256 checksum. Resume restores deterministic-bit-equal state.

**Implementation** (`python/astra/training/checkpoint.py`):
- **`save_checkpoint()`:** Iterates `all_params(model)` to save weights. Saves AdamW `m` (first moment) and `v` (second moment) state dicts. Saves `CosineSchedule` opt_t. Writes companion `.manifest.json` with step, loss history, config, data manifests, params. Computes SHA-256 checksum of `.npz` file.
- **`load_checkpoint()`:** Loads `.npz` into model weights. Restores optimizer state, scheduler state, training step, loss history, and manifest metadata. Enables deterministic resume from any checkpoint.

**Checkpoint format:** Compressed NumPy `.npz` with keys:
```
w:attn0.qkv.w    (attention QKV projection weights)
w:attn0.out.w    (attention output projection weights)
m:attn0.qkv.w    (AdamW first moment)
v:attn0.qkv.w    (AdamW second moment)
... (all params + optimizer state)
```

**Manifest format:**
```json
{
  "seed": 42,
  "model_config": {"d_model": 64, "n_layers": 2, ...},
  "train_config": {"max_steps": 2000, "peak_lr": 0.001, ...},
  "data_manifest": {"train": {...}, "val": {...}},
  "params": 133440,
  "step": 2000,
  "opt_t": 2000,
  "loss_hist": [4.5, 3.8, ...],
  "checkpoint_sha256": "88e5bc5392733ce07268618d9dd0404c36cf0ee5eaa54ce5937bb2820043cae7"
}
```

**Artifact:** `checkpoints/phase0/final.npz` ≈ 1.45 MB (133k params × 4 B + optimizer state). SHA-256: `88e5bc5392733ce07268618d9dd0404c36cf0ee5eaa54ce5937bb2820043cae7`.

**Reproducibility:** Two runs with identical seed/config/data hash/thread-count produce bit-identical loss trajectories (EX-04, 2000/2000 points).

---

## 8. Basic Text Generation

**Specification:** Reference inference using greedy-with-temperature autoregressive sampling. The model generates tokens one at a time, feeding each generated token back as context. Generation bounded by `max_seq_len` sliding window. Basic text generation used for generation sanity and evaluation.

**Implementation** (`python/astra/evaluation/metrics.py` — `generate()`):
- **Algorithm:** Autoregressive sampling from the last `max_seq_len` context window.
  1. Initialize context with `seed_ids` (first `max_seq_len` tokens from validation corpus).
  2. For each step up to `max_new`:
     - Run `model.forward_loss(ctx[:, -max_seq_len:])` to get logits.
     - Take last token logits, divide by `temperature`.
     - Apply temperature scaling: `p = softmax(logits / temperature)`.
     - Sample next token from `p` (greedy when `temperature=1.0`).
     - Append to context.
  3. Return generated tokens (excluding seed_ids).
- **`repetition_fraction(tokens, n=4)`:** Measures degenerate text — fraction of overlapping 4-grams that repeat. Lower = more diverse generation.

**Evaluation harness** (`evaluation/evaluate.py` → `evaluation/harness.py` — `evaluate_checkpoint()`):
- Loads checkpoint, reconstructs model.
- Computes frozen val loss, perplexity, next-token accuracy over entire val corpus.
- Runs `generate()` for 200 tokens, computes: `gen_tokens`, `gen_unique_tokens`, `gen_repetition_frac_4gram`, `decode_tokens_per_sec`.
- Produces **checksum-tied JSON eval report** with environment metadata (Python version, NumPy version, CPU count, OpenBLAS threads).

**Measured results** (toy model, 200 generated tokens, temperature=1.0):
| Metric | Value |
|---|---|
| Generated tokens | 200 |
| Unique tokens | varies (vocab 800) |
| 4-gram repetition | **0.0** (no degenerate repetition) |
| Decode throughput | **92–276 tok/s** (CPU, load-dependent) |
| Val loss | 2.535 |
| Perplexity | 12.62 |
| Next-token accuracy | 27.1% |

---

## Cross-Component Summary

| Component | File | Key Function | Measured |
|---|---|---|---|
| **Tokenizer** | `python/astra/tokenizer/bpe.py` | `ByteLevelBPE.train/encode/decode` | 6.84 B/token, round-trip exact |
| **Embeddings** | `python/astra/model/core.py` | `Embedding`, `RoPE` | Gradcheck pass, 1.5e-9 max error |
| **Transformer** | `python/astra/model/core.py` | `LiteLM.forward/backward` | 133,440 params, val CE 2.535 |
| **Attention** | `python/astra/model/core.py` | `AttentionBlock.forward/backward` | Causal MHA + RoPE, gradcheck pass |
| **Feed-Forward** | `python/astra/model/core.py` | `SwiGLUBLock.forward/backward` | SwiGLU, gradcheck pass |
| **Training Loop** | `python/astra/training/loop.py` | `train()` | 4K tok/s, bit-identical |
| **Checkpoints** | `python/astra/training/checkpoint.py` | `save/load_checkpoint` | Resume deterministic |
| **Text Generation** | `python/astra/evaluation/metrics.py` | `generate()` | 92–276 tok/s, 0.0 repetition |

**All Phase 0 hypotheses met (H0.1–H0.5).** The system is built, measured, reproducible, and ready for Phase 1 scaling.

---

## Release Artifacts

| Artifact | Version | SHA-256 |
|---|---|---|
| `checkpoints/phase0/final.npz` | astra-toy-51k (133,440 params) | `88e5bc53...` |
| `tokenizer/artifacts/toy_bpe.json` | bpe-v1 | `e964964d...` |
| `experiments/phase0/eval_report-v0.1.0.json` | v0.1.0 | — |
| `experiments/phase0/REPORT.md` | v0.1.0 | — |

**Release notes:** Full details in `docs/releases/v0.1.0.md`. Changelog in `docs/CHANGELOG.md`.
