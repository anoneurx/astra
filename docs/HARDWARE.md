# Astra Hardware Requirements

> Hardware recommendations by workload. Designed to be consumer-hardware-friendly for early phases.

**STATUS: VALIDATED (Phase-0 / toy subset)** — toy budgets measured (see §4); GPU-based and Astra-100M+ budgets remain targets until Phases 1–2 measurements land.

---

## 1. Guiding Principle

Astra must be developable, trainable, and evaluable on **consumer hardware** for Phases 0–4. Large-scale training is an option, not a prerequisite.

---

## 2. Workload Requirements

### 2.1 Development (Phases 0–2)

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores, x86-64 or arm64 | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | None appropriate; optional 6–8 GB VRAM for toy runs | Any ≥ 8 GB VRAM GPU (consumer or cloud) |
| Storage | 20 GB free (SDD) | 100 GB NVMe |
| OS | Linux (recommended), macOS, WSL2 | Linux |

### 2.2 Small-Scale Training (Astra-100M/300M)

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 12+ cores |
| RAM | 32 GB | 64 GB |
| GPU | 8 GB VRAM (fp16/bf16) | 16–24 GB VRAM |
| Storage | 100 GB NVMe | 500 GB NVMe |
| Software | CUDA for NVIDIA or ROCm for AMD | Consistent driver stack |

### 2.3 Medium-Scale Training (Astra-700M/1B)

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | 24 GB VRAM | 40–80 GB VRAM (e.g., A100/H100/consumer 24GB+) |
| RAM | 64 GB | 128 GB |
| Storage | 500 GB NVMe | 1 TB NVMe + scratch |
| Footprint | Single GPU, gradient accumulation | Multi-GPU DDP later |

### 2.4 Large-Scale Training (Astra-3B/7B — later phases)

| Resource | Notes |
|---|---|
| GPU | 4+ GPUs H100/A100-class or memory-optimized clusters |
| Interconnect | NVLink/InfiniBand for efficiency (recommended, not required day one) |
| Storage | Multi-TB NVMe/network store |
| Notes | Sharded training (FSDP) + possible data/tensor parallel — **research** |

### 2.5 Inference (all phases)

| Resource | CPU-only (small) | GPU (fast) |
|---|---|---|
| Astra-100M–700M | 16 GB RAM, CPU decode in Rust runtime | 8–16 GB VRAM |
| Astra-1B–3B | 32 GB RAM (quantized), slower | 24+ GB VRAM |
| Astra-7B | 48 GB RAM int8-quantized, slow | 48+ GB VRAM (fp16) / 24 GB (int8) |

---

## 3. Discussion by Component

### 3.1 CPU
- Training data prep and tokenization benefit from many cores.
- Rust inference runtime uses all cores for CPU decode — a key consumer path.

### 3.2 RAM
- Binder for dataset shards and memory store; 32 GB is the comfort zone for Phases 0–3.

### 3.3 GPU VRAM
- Dominant for training. Model size × (parameter memory + optimizer + activations) determines VRAM; gradient accumulation trades speed for memory. Mixed precision (bf16/fp16) roughly halves memory vs fp32.

### 3.4 Storage
- Tokenized shards and checkpoints grow fast. NVMe strongly recommended for training I/O; HDD acceptable for cold archives.

### 3.5 NVMe
- Reduces data-loading stalls dramatically vs SATA/SDD at scale. Needed for stable reproducibility runs.

### 3.6 CUDA/ROCm
- NVIDIA CUDA is the default first-class target. ROCm (AMD) supported behind a backend switch; both measured with the same metrics.

### 3.7 Multi-GPU
- DDP first; FSDP later (research). Single-GPU path remains primary for early phases.

### 3.8 Distributed Training
- Out of scope for Phases 0–3 by policy (runs must be reproducible on single machine first). Research in Phase 2+ (hooks only).

---

## 4. Measured Budgets (to be filled by experiments)

Planned tables (populated from real runs):
- `tokens/s` per GPU model.
- Peak VRAM per model/config.
- Checkpoint size, shard I/O rates.
- CPU decode t/s per model (Rust runtime).

### Phase 0 / 0.1.0 toy measurements (NumPy reference, 8-core x86_64, no GPU)

Measured 2026-09-06 on `docs/releases/v0.1.0.md` (astra-toy-51k, 133,440 params,
`configs/toy_pretrain.json`):

| quantity | value |
|---|---|
| Model fp32 footprint | 0.53 MB (133,440 params × 4 B) |
| Training throughput | 4,051 tok/s (2000 steps, 252 s, 8 threads; batch 8×64) |
| Reproducibility | bit-identical loss trajectories at 1 vs 8 threads |
| Training peak RSS | ~90 MB (`/proc` VmHWM, 60-step probe) |
| Evaluation peak RSS | ~196 MB (NumPy import + model + one batch) |
| CPU decode (reference inference) | 92–276 tok/s (load-dependent) |
| CPU decode (KV-cache decoder) | 182.9 tok/s (measured 2026-09-07, phase-0 ckpt) |
| CPU decode (KV-cache decoder) | 543.9 tok/s (measured 2026-09-08, `astra-name`, core-basic gate run) |
| HTTP service (`service/inference.py`) | stdlib ThreadingHTTPServer; per-request KV cache; single-threaded decode behind server threads |
| Quantized weights | fp16: 0.27 MB; int8: 0.13 MB vs 0.53 MB fp32 (worst-error bounded) |
| Checkpoint size | `final.npz` ≈ 1.45 MB (133k params × 4 B + optimizer state) |

GPU/hardware budgets for Astra-100M and larger remain targets until Phases 1–2
measurements land (see §§ 2.2–2.4).

**STATUS: RESEARCH** — populate from Phase 1/2 measurements before releasing hardware guidance as validated numbers.

---

## 5. Notes on Cloud

- Consumer hardware and cloud spot instances are equally valid; nothing in the project depends on enterprise infrastructure.
- Reproducibility contract requires recording hardware ids (GPU name, driver, CUDA/ROCm version) in the run manifest (`docs/TRAINING.md`).