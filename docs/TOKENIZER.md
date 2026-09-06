# Astra Tokenizer

> The tokenizer subsystem: vocabulary, algorithm choice, encoding/decoding, and quality controls.

**STATUS: PROPOSED** — algorithm choice below is the recommendation; a validation experiment is required in Phase 0.

---

## 1. Approach Comparison

| Approach | Pros | Cons | Verdict for Astra |
|---|---|---|---|
| **BPE (Byte-level / GPT-style)** | Proven; robust to unseen bytes; complete Unicode via byte fallback; mature libraries (`tokenizers`/`tiktoken`); no UNK for fair text; produces stable merges | Slightly larger vocab for CJK/latin mix; suboptimal morphological segmentation vs some alternatives | **Recommended** for v1 |
| **Unigram** | Probabilistic; nice subword smoothness; length normalization; good for morphologically rich langs | Requires EM training framework; slightly more moving parts | Secondary candidate |
| **SentencePiece-style** | Direct BPE/unigram with raw text; excellent Unicode | Raw-text pre-tokenization pitfalls (whitespace rules); relies on a library; MbPE nuances | Considered via the library; **not a separate framework** |
| **Byte-level (pure bytes)** | Fully lossless, no UNK at all | Longer sequences for non-Latin text; higher compute per token | Fallback / research path |

**Recommendation:** Byte-level BPE with a ~32k–50k vocab, following the GPT/SentencePiece byte-encode + 256 base bytes pattern. Rationale:
- Universal byte coverage ⇒ no unknown-token class for arbitrary text.
- Exact reversibility (every Unicode string round-trips exactly).
- Mature reference alignment for correctness verification.

**STATUS: PROPOSED** — Phase 0 experiment must compare BPE vs Unigram on a multilingual sample measuring tokens/character and reconstruction exactness.

---

## 2. Vocabulary

### 2.1 Size

- **v1 target:** 48,000 (base option 32k/50k). Larger = better coverage, more embedding memory; smaller = denser. Config-selected via train run.
- Include: 256 byte-initial tokens + merge-based pieces, plus reserved special slots.

### 2.2 Special Tokens

| Token | Use |
|---|---|
| `<bos>` | Optional start marker (v1: not required) |
| `<eos>` | End marker (structured text) |
| `<pad>` | Padding (fixed-length batching), optional |
| `<unk>` | Reserved legacy; should be unused by construction |
| `<|user|>` / `<|assistant|>` | Chat/conversation framing (Phase 5+) |
| `<|memory|>` | Memory-context framing (Phase 4+) |
| `<|tool|>` | Tool-use framing (research, Phase 9) |
| `<reserved N>` | Reallocation buffer for future specials |

Special tokens are non-splittable, excluded from merges, and enumerated by ID in the tokenizer config.

### 2.3 Encoding

Text → UTF-8 → bytes → byte-token(s) → BPE merges → IDs. Standard GPT-style pipeline.

### 2.4 Decoding

IDs → byte strings → bytes → UTF-8 string. Require byte-exact round-trip `decode(encode(s)) == s` for all strings (tested).

### 2.5 Unknown Tokens

- By design, `<unk>` is **never produced** for valid UTF-8: byte fallback guarantees coverage.
- If produced (e.g., corrupt input), a warning in the pipeline; `<unk>` maps to a designated replacement (255-byte). Logging policy documented.

### 2.6 Unicode Handling

- Pre-tokenization is **byte-oriented** (no Unicode-aware script splitting in v1) — most robust.
- Optional script-aware pre-tokenization is a research option; keeps raw text analysis honest (report tokens/character per script).

### 2.7 Token Statistics (reported per corpus & run)

- Vocabulary size; OOV rate; bytes/token; tokens/character (split by script/class); merge frequency top-N; compression ratio on the corpus.
- These live in the tokenizer training report artifact under `tokenizer/`.

---

## 3. Correctness Contract

Test suite (`tests/test_tokenizer.py`) must assert:
1. `roundtrip(encode/decode) == identity` over: ASCII, accented Latin, CJK, RTL, emoji, control chars, lone surrogates (decoded safely), very long strings, null bytes.
2. Deterministic (same input → same IDs).
3. Encodes/decodes remain stable across tokenizer versions (compatibility policy).

---

## 4. Configuration (v1)

| Key | Value |
|---|---|
| Algorithm | Byte-level BPE |
| Vocab size | 48,000 (config-testable) |
| Corpus | Initial public corpus sample (as per `datasets/`) |
| min_frequency | configurable (default 2) |
| special handling | reserved IDs list |
| normalization | NFC optional, recorded per config; default NONE (byte-exact) |

Tokenizer artifacts are versioned and immutable (hash + semantic version), and any tokenizer change updates the affected dataset manifests (`docs/DATA.md`).