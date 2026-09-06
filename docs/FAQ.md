# Astra FAQ

**STATUS: PROPOSED** — fill in answers from policy; refine with experience.

---

### What is Astra?
Astra is an open-source AI research project building a self-learning, continuously improving language model from scratch — not a wrapper around an existing commercial API.

### Is Astra trying to achieve AGI?
No. Astra's objectives are specific and measurable: learning from high-quality data, verified experience, memory, and controlled self-improvement. Advanced capabilities are research objectives, not guarantees.

### What does "self-learning" mean?
The system converts verified experience into better capability through a controlled pipeline: observe → feedback → validated example → candidate model → evaluation → accept/reject. It does not blindly learn every interaction.

### Does Astra modify its own weights at runtime?
No, not as a primary learning mechanism. Adaptive behavior happens through external memory and gated candidate-model training. This guards against forgetting, poisoning, and non-rollback states.

### Does Astra have memory like humans?
Astra has an external, structured, auditable memory system (facts, episodes, semantics). "Memory" does not change model weights.

### What hardware do I need to contribute?
Consumer hardware is fine for early phases: 4+ core CPU, 16–32 GB RAM, optional 8 GB+ VRAM GPU. See `docs/HARDWARE.md`.

### What language is Astra written in?
Python for the training/research stack; Rust for the inference runtime, memory engine, and model management; optional C/C++ for targeted low-level optimization.

### What is the difference between validation loss and the benchmark suite?
Validation loss is a training-time diagnostic. Real capability claims come only from the quarantined benchmark suite (`docs/BENCHMARKS.md`), which is never trained on.

### How does Astra prevent data contamination?
Strict split discipline, hash + n-gram overlap scanning, MinHash near-duplicate detection, and quarantine of all benchmark items. See `docs/DATA.md`.

### What happens if a new model version regresses?
It is rejected at the gate; if a regression is discovered post-deploy, automatic rollback restores the previous registered version. See `docs/SAFETY.md`.

### Can I train Astra at home?
Early phases yes (Astra-100M/300M on consumer GPUs). Larger models are later-phase/research and are not prerequisites for contributing.

### Who decides what goes into a release?
Gates are automatic and objective (CI + benchmark suite + gate engine). Human approval options exist; escalation goes to maintainers.

### How are claims validated in this project?
Every claim requires a measured artifact: run manifest, eval report, benchmark result. Unmeasured claims are rejected from docs and releases.

### What license will Astra use?
Decision pending (`STATUS: PROPOSED`); an open license (e.g., Apache-2.0) is recommended.

### I found a bug in documentation. What do I do?
Open an issue or PR. Docs must stay current; all changes are reviewed.

### What are Astra's known limitations right now?
The project is at Phase 0 (research). No model, memory, or learning component is yet validated. See `docs/ROADMAP.md` for status.