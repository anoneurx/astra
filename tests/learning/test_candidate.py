"""Tests for the candidate trainer (docs/LEARNING.md § 1.9).

Runs a real (tiny) model: trains a base on concept A, then a candidate off it
on concept B with replay, and asserts the candidate learned B without
forgetting A — the Phase-5 learning-loop contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from astra.learning.candidate import CandidateConfig, train_candidate
from astra.model.config import ModelConfig
from astra.model.core import LiteLM
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.training.data import Corpus
from astra.training.loop import train

CONCEPT_A = "The bird is called Sky."
CONCEPT_B = "The bird is called Nova."
TOY = {"vocab_size": 300, "d_model": 24, "n_layers": 1, "n_heads": 2, "d_head": 12,
       "d_ffn": 48, "max_seq_len": 32, "rope_theta": 10000.0, "eps": 1e-6,
       "tie_embeddings": True}


@pytest.fixture(scope="module")
def infra(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cand")
    tok = ByteLevelBPE(vocab_size=300)
    cfg = ModelConfig.from_dict(TOY)
    cfg.vocab_size = 300
    cfg = ModelConfig.from_dict(cfg.to_dict())

    train_ids = np.array(tok.encode(CONCEPT_A * 40), dtype=np.int32)
    val_ids = np.array(tok.encode(CONCEPT_A * 5), dtype=np.int32)
    train_corpus = Corpus(ids=train_ids, manifest={"name": "a-train"})
    val_corpus = Corpus(ids=val_ids, manifest={"name": "a-val"})

    tcfg = {
        "max_steps": 30, "peak_lr": 3e-3, "min_lr": 1e-4, "warmup_steps": 4,
        "batch_seq": 2, "accum_steps": 1, "optimizer": "adamw",
        "scheduler": "cosine", "weight_decay": 0.02, "grad_clip": 1.0,
        "val_every": 1000, "val_shards": 1, "experiment_store": str(tmp / "runs"),
    }
    base_out = str(tmp / "base")
    train(cfg, tcfg, train_corpus, val_corpus, seed=0, out_dir=base_out)
    base_ck = f"{base_out}/final.npz"
    return {"tok": tok, "cfg": cfg, "base_ck": base_ck, "tmp": tmp}


def _loss(model, tok, text):
    ids = tok.encode(text)
    x = np.array([ids[:-1]], dtype=np.int64)
    y = np.array([ids[1:]], dtype=np.int64)
    _logits, loss = model.forward_loss(x, y)
    return float(loss)


def test_candidate_learns_target_without_forgetting(infra):
    tok, cfg = infra["tok"], infra["cfg"]
    base_ck = infra["base_ck"]

    b_model = LiteLM(cfg, seed=0)
    load_checkpoint(base_ck, b_model, None, None)
    b_A = _loss(b_model, tok, CONCEPT_A)
    b_B = _loss(b_model, tok, CONCEPT_B)

    # candidate on NEW concept B + replay A
    experiences = [
        {
            "id": "e1", "kind": "sft",
            "payload": {"input": "", "output": CONCEPT_B},
            "feedback_id": "f1", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "b1",
        },
        {
            "id": "e2", "kind": "sft",
            "payload": {"input": "", "output": "The bird lives on Nova."},
            "feedback_id": "f2", "source": "human", "confidence": 0.99,
            "verification_status": "human_verified", "trust_tier": "human_verification",
            "status": "active", "dedup_key": "b2",
        },
    ]
    replay = np.array(tok.encode(CONCEPT_A * 40), dtype=np.int32)
    out = str(infra["tmp"] / "candidate")
    cc = CandidateConfig(max_steps=60, peak_lr=5e-3, warmup_steps=6, batch_seq=2, replay_ratio=0.4)
    res = train_candidate(
        base_checkpoint=base_ck, tokenizer=tok, model_config=cfg,
        experiences=experiences, replay_ids=replay, config=cc, out_dir=out, seed=0,
    )

    c_model = LiteLM(cfg, seed=0)
    load_checkpoint(res.checkpoint, c_model, None, None)
    c_A = _loss(c_model, tok, CONCEPT_A)
    c_B = _loss(c_model, tok, CONCEPT_B)

    # learned B: candidate loss on B much lower than base's (which never saw B)
    assert c_B < b_B - 0.3, f"expected learning B: base={b_B:.3f} cand={c_B:.3f}"
    # didn't forget A: candidate loss on A not much higher than base's
    assert c_A < b_A + 0.1, f"expected no forgetting: base={b_A:.3f} cand={c_A:.3f}"
    # provenance manifest recorded
    assert res.manifest["base_checkpoint"] == base_ck
    assert res.manifest["num_experiences"] == 2
    assert res.manifest["replay_tokens"] == replay.size
    assert res.manifest["git_commit"]
    assert res.steps == 60


def test_candidate_checkpoint_is_loadable(infra):
    tok, cfg = infra["tok"], infra["cfg"]
    out = str(infra["tmp"] / "candidate2")
    cc = CandidateConfig(max_steps=10, peak_lr=1e-3, warmup_steps=2, batch_seq=2)
    res = train_candidate(
        base_checkpoint=infra["base_ck"], tokenizer=tok, model_config=cfg,
        experiences=[], replay_ids=None, config=cc, out_dir=out, seed=1,
    )
    m = LiteLM(cfg, seed=0)
    step, _hist, meta = load_checkpoint(res.checkpoint, m, None, None)
    assert step >= 0
    assert meta["candidate_config"]["max_steps"] == 10