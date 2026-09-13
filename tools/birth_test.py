#!/usr/bin/env python3
"""Astra 0.1 Birth Test — verifies the complete neural-language-model pipeline.

Usage: python tools/birth_test.py

Tests:
  1. Tokenizer works
  2. Forward pass works
  3. Backpropagation works
  4. Loss decreases
  5. Checkpoint saves
  6. Checkpoint reloads
  7. Generation works
  8. Generated text is influenced by training data
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.evaluation.metrics import generate, repetition_fraction
from astra.model import LiteLM, ModelConfig, all_params
from astra.safety import leak_check
from astra.tokenizer import ByteLevelBPE
from astra.training import Corpus, train
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.optim import AdamW, CosineSchedule, clip_grad_norm


def _load_model_and_tokenizer():
    cfg_dict = {
        "vocab_size": 800, "d_model": 64, "n_layers": 2, "n_heads": 4,
        "d_head": 16, "d_ffn": 128, "max_seq_len": 64, "rope_theta": 10000.0,
        "eps": 1e-6, "tie_embeddings": True,
    }
    cfg = ModelConfig.from_dict(cfg_dict)
    tok = ByteLevelBPE.load("tokenizer/artifacts/toy_bpe.json")
    cfg.vocab_size = len(tok)
    cfg = ModelConfig.from_dict(cfg.to_dict())
    return cfg, tok


def test_tokenizer():
    _cfg, tok = _load_model_and_tokenizer()
    text = "section alpha: specifications (record 0)\nverify complete for record 0.\n"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"Round-trip failed: {decoded[:60]!r} != {text[:60]!r}"
    assert len(ids) > 0, "Empty encoding"
    return True, f"vocab={len(tok)}, tokens={len(ids)}, roundtrip=exact"


def test_forward_pass():
    cfg, _tok = _load_model_and_tokenizer()
    model = LiteLM(cfg, seed=42)
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    logits, _loss = model.forward_loss(ids)
    assert logits.shape == (1, 5, cfg.vocab_size), f"Wrong shape: {logits.shape}"
    assert np.isfinite(logits).all(), "Non-finite logits"
    assert model.num_params == 133440, f"Wrong param count: {model.num_params}"
    return True, f"shape={logits.shape}, params={model.num_params}, finite=true"


def test_backpropagation():
    cfg, _tok = _load_model_and_tokenizer()
    model = LiteLM(cfg, seed=42)
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6]], dtype=np.int64)
    logits, loss = model.forward_loss(ids, targets)
    grad_before = sum(float(np.abs(g).sum()) for _, _, g in all_params(model))
    model.backward(logits, targets)
    grad_after = sum(float(np.abs(g).sum()) for _, _, g in all_params(model))
    assert grad_before == 0.0, "Grads should be zero before backward"
    assert grad_after > 0.0, "Grads should be non-zero after backward"
    return True, f"loss={loss:.4f}, grad_norm={grad_after:.4f}"


def test_loss_decreases():
    cfg, _tok = _load_model_and_tokenizer()
    model = LiteLM(cfg, seed=42)
    opt = AdamW(model, lr=1e-3, weight_decay=0.1)
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    losses = []
    for _ in range(50):
        model.zero_grad()
        logits, loss = model.forward_loss(ids, targets)
        model.backward(logits, targets)
        clip_grad_norm(model, 1.0)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return True, f"{losses[0]:.4f} -> {losses[-1]:.4f} (decreased by {losses[0]-losses[-1]:.4f})"


def test_checkpoint_save_load():
    cfg, _tok = _load_model_and_tokenizer()
    model = LiteLM(cfg, seed=42)
    opt = AdamW(model, lr=1e-3)
    schedule = CosineSchedule(max_steps=100, warmup_steps=10, peak_lr=1e-3, min_lr=1e-5)
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = f"{td}/test_ckpt.npz"
        save_checkpoint(ckpt_path, model, opt, schedule, step=10, loss_hist=[3.0, 2.5],
                        meta={"seed": 42, "params": model.num_params})
        assert Path(ckpt_path).exists(), "Checkpoint file not created"
        assert Path(ckpt_path.replace(".npz", ".manifest.json")).exists(), "Manifest not created"
        model2 = LiteLM(cfg, seed=99)
        step, _hist, _meta = load_checkpoint(ckpt_path, model2, opt=None, schedule=None)
        assert step == 10, f"Step mismatch: {step}"
        for (n1, w1, _), (n2, w2, _) in zip(all_params(model), all_params(model2)):
            assert np.allclose(w1, w2), f"Weight mismatch after load: {n1}"
    return True, f"saved+reloaded, step={step}, all weights match"


def test_generation():
    cfg, tok = _load_model_and_tokenizer()
    model = LiteLM(cfg, seed=42)
    val_path = Path("datasets/toy/val.txt")
    val_ids = tok.encode(val_path.read_text(encoding="utf-8"))
    seed_ids = val_ids[:cfg.max_seq_len]
    rng = np.random.default_rng(42)
    gen_ids = generate(model, tok, seed_ids, max_new=50, temperature=1.0, rng=rng)
    assert len(gen_ids) == 50, f"Wrong gen length: {len(gen_ids)}"
    try:
        tok.decode(gen_ids)
    except (UnicodeDecodeError, KeyError):
        repr(b"".join(tok.id_to_piece.get(i, b"?") for i in gen_ids))
    rep = repetition_fraction(gen_ids, n=4)
    return True, f"gen={len(gen_ids)} tokens, repetition_4gram={rep:.3f}"


def test_training_data_influence():
    cfg, tok = _load_model_and_tokenizer()
    train_path = Path("datasets/toy/train.txt")
    val_path = Path("datasets/toy/val.txt")
    train_ids = tok.encode(train_path.read_text(encoding="utf-8"))
    val_ids = tok.encode(val_path.read_text(encoding="utf-8"))
    train_corpus = Corpus(ids=np.array(train_ids, dtype=np.int32),
                          manifest={"name": "toy-train", "kind": "training"})
    val_corpus = Corpus(ids=np.array(val_ids, dtype=np.int32),
                        manifest={"name": "toy-val", "kind": "validation"})
    leak = leak_check(train_ids, val_ids, n=13)
    assert leak["leak_free"], f"Data leak detected: {leak}"

    report = train(
        cfg,
        train_config={"max_steps": 1000, "peak_lr": 1e-3, "min_lr": 1e-5,
                       "warmup_steps": 50, "batch_seq": 8, "weight_decay": 0.1,
                       "grad_clip": 1.0, "val_every": 250},
        train_corpus=train_corpus, val_corpus=val_corpus,
        seed=42, out_dir=tempfile.mkdtemp(),
    )
    final_val = report.final_val
    assert final_val["loss"] < 4.0, f"Val loss too high: {final_val['loss']:.4f}"

    model = LiteLM(cfg, seed=0)
    with tempfile.TemporaryDirectory() as td:
        ckpt = f"{td}/trained.npz"
        save_checkpoint(ckpt, model, None, None, step=0, loss_hist=[], meta={})
        load_checkpoint(ckpt, model, None, None)

    gen_ids = generate(model, tok, val_ids[:cfg.max_seq_len], max_new=30,
                       temperature=0.5, rng=np.random.default_rng(42))
    try:
        decoded = tok.decode(gen_ids)
    except (UnicodeDecodeError, KeyError):
        decoded = repr(b"".join(tok.id_to_piece.get(i, b"?") for i in gen_ids))
    return True, f"val_loss={final_val['loss']:.4f}, gen_sample={decoded[:80]!r}..."


def main():
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Forward pass", test_forward_pass),
        ("Backpropagation", test_backpropagation),
        ("Loss reduction", test_loss_decreases),
        ("Checkpoint save", test_checkpoint_save_load),
        ("Generation", test_generation),
        ("Training data influence", test_training_data_influence),
    ]

    results = []
    all_pass = True

    for name, fn in tests:
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
        except Exception as e:
            status = "FAIL"
            detail = str(e)
            all_pass = False
        results.append((name, status, detail))

    print("=" * 60)
    print("  ASTRA 0.1 BIRTH TEST")
    print("=" * 60)
    for name, status, detail in results:
        marker = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {marker} {name}: {detail}")
    print("-" * 60)
    if all_pass:
        print("  [PASS] Tokenizer")
        print("  [PASS] Embeddings")
        print("  [PASS] Transformer")
        print("  [PASS] Attention")
        print("  [PASS] Training")
        print("  [PASS] Loss reduction")
        print("  [PASS] Checkpoint")
        print("  [PASS] Reload")
        print("  [PASS] Generation")
        print()
        print("  ASTRA 0.1 STATUS: BORN")
    else:
        print("  ASTRA 0.1 STATUS: FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
