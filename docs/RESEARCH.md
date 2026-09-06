# Astra Research Track

> The research agenda. Clearly separates established engineering from experimental research.

**STATUS: RESEARCH** — items below are proposals with hypotheses, not commitments.

---

## 1. Principle: Established Engineering vs. Experimental Research

| Category | Definition | Where it lives |
|---|---|---|
| **Established engineering** | Components accepted into the active model/stack behind benchmark gates | `model/`, `training/`, `inference/`, `memory/`, `evaluation/` (active paths) |
| **Experimental research** | Untested or unproven components/ideas, kept isolated | `experiments/` (sandbox), features behind flags |

No experimental component enters the established stack without its own validated experiment and gate report.

---

## 2. Research Themes

### 2.1 Architecture Experiments
- Block-structure ablations: PreNorm/PostNorm, RMSNorm ep-sensitivity, tied/un-tied embeddings.
- Depth vs width trade at fixed parameter budget.
- Attention head interactions & head-count scaling.

### 2.2 Attention Alternatives
- GQA/MQA (Grouped/Multi-Query Attention) at larger scales.
- Sliding-window + global attention hybrid.
- Linear/async attention kernels (e.g., FFT variants) — long-term.

### 2.3 Efficient Attention
- FlashAttention backends (compatibility + profiling).
- KV-cache compression techniques (quantized cache, cache eviction policies).
- Long-context via RoPE extrapolation or context-window scaling.

### 2.4 Memory Architectures
- Retrieval-augmentation benefits and degradation analysis.
- Memory record curation (what to store, when, at what confidence).
- End-to-end memory module training (memory-augmented attention) — long-term.

### 2.5 Continual Learning
- Replay/mixture approaches to avoid catastrophic forgetting.
- Parameter-efficient fine-tuning (LoRA-class) for adaptation.
- EWC/GEM-style regularizers vs replay.

### 2.6 Catastrophic Forgetting
- Quantify forgetting under successive candidate training rounds.
- Benchmarks to detect forgetting (previous-version regression gates).
- Mitigation comparison (replay rate/curriculum/regularizers).

### 2.7 Retrieval
- Ranking function ablation (hybrid scoring weights).
- Query reformulation for better memory hits.
- Long-horizon relevance (cross-session).

### 2.8 Synthetic Data
- Self-generated curriculum; diversity protocols; collapse detection.
- Human-in-the-loop validation of synthetic examples.

### 2.9 Self-Evaluation
- Calibration of model confidence (self-consistency, P(True)).
- Trust-tier weighting of feedback.
- Using self-eval as a gate feature, not as ground truth.

### 2.10 Automated Curriculum Generation
- Ordering/mixing experiences for candidate training.
- Difficulty estimators for experience ranking.

### 2.11 Model Compression
- Pruning/sparsification research at Astra scale.
- Embedding/output-head compression.

### 2.12 Quantization
- Post-training INT8/INT4 for inference (consumer friendliness).
- QAT (quantization-aware training) at small scale; measure quality drop.

### 2.13 Efficient Inference
- Speculative decoding (draft+verify) — measured.
- Prefix caching, prompt caching.
- Batching policies for throughput.

---

## 3. Experiment Lifecycle

Each research item becomes an experiment (`experiments/`), runs the methodology loop (`docs/DEVELOPMENT.md`), produces a measured result, and routes to one of:

- **Adopt** → merge into established stack with gate report + doc update.
- **Reject** → record negative result (valuable) + rationale.
- **Defer** → park with open questions.

## 4. Review Cadence

- Quarterly research review: reassess theme priorities.
- Priority driver: roadmap windows (Phases 3, 9).

## 5. Open Problems (known hard)

These are long-horizon research objectives — **not** guarantees:
- True continual learning without any forgetting.
- Reliable self-evaluation without external truth.
- Reward hacking resistant design under aggressive optimization.