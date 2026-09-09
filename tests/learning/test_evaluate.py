"""Tests for the shared candidate-vs-active measurement (astra/learning/evaluate.py).

End-to-end tiny-model check: a base trained on concept A, a candidate that also
saw concept B, then a measure that B-loss drops for the candidate while A-loss
stays flat. This is the exact measurement the Phase-6 gate consumes.
"""

from __future__ import annotations

import numpy as np
import pytest
from astra.learning.evaluate import corpus_loss, load_model, partition_metrics
from astra.model.config import ModelConfig
from astra.model.core import LiteLM
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.training.data import Corpus
from astra.training.loop import train

TOY = {"vocab_size": 300, "d_model": 24, "n_layers": 1, "n_heads": 2, "d_head": 12,
       "d_ffn": 48, "max_seq_len": 32, "rope_theta": 10000.0, "eps": 1e-6,
       "tie_embeddings": True}
A = "The bird is called Sky."
B = "The bird is called Nova."


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    tok = ByteLevelBPE(vocab_size=300)
    cfg = ModelConfig.from_dict(TOY)
    cfg.vocab_size = 300
    cfg = ModelConfig.from_dict(cfg.to_dict())
    tcfg = {
        "max_steps": 30, "peak_lr": 3e-3, "min_lr": 1e-4, "warmup_steps": 4,
        "batch_seq": 2, "accum_steps": 1, "optimizer": "adamw",
        "scheduler": "cosine", "weight_decay": 0.02, "grad_clip": 1.0,
        "val_every": 1000, "val_shards": 1, "experiment_store": str(tmp / "runs"),
    }
    tr_corpus = Corpus(ids=np.array(tok.encode(A * 40), dtype=np.int32), manifest={"name": "a"})
    val_corpus = Corpus(ids=np.array(tok.encode(A * 5), dtype=np.int32), manifest={"name": "a"})
    base_out = str(tmp / "base")
    train(cfg, tcfg, tr_corpus, val_corpus, seed=0, out_dir=base_out)
    base_ck = f"{base_out}/final.npz"
    return {"tok": tok, "cfg": cfg, "base_ck": base_ck, "tmp": tmp, "B_ck": None}


def test_partition_metrics_drops_target_loss(pair):
    tok, cfg, base_ck = pair["tok"], pair["cfg"], pair["base_ck"]
    tmp = pair["tmp"]

    # candidate fine-tuned on B (reuse Phase-5 trainer quickly)
    from astra.learning.candidate import CandidateConfig, train_candidate

    ex = [{
        "id": "e1", "kind": "sft", "payload": {"input": "", "output": B},
        "feedback_id": "f1", "source": "human", "confidence": 0.99,
        "verification_status": "human_verified", "trust_tier": "human_verification",
        "status": "active", "dedup_key": "b1",
    }]
    replay = np.array(tok.encode(A * 40), dtype=np.int32)
    cand_out = str(tmp / "cand")
    res = train_candidate(
        base_checkpoint=base_ck, tokenizer=tok, model_config=cfg,
        experiences=ex, replay_ids=replay,
        config=CandidateConfig(max_steps=60, peak_lr=5e-3, warmup_steps=6, batch_seq=2,
                               replay_ratio=0.4),
        out_dir=cand_out, seed=0,
    )

    metrics = partition_metrics(
        base_ck, res.checkpoint, tok, cfg,
        {"target_B": [B, "The bird was born on Nova."], "regress_A": [A]},
    )
    # target improves: candidate CE on B clearly below base CE on B
    assert metrics["target_B"]["candidate"] < metrics["target_B"]["base"] - 0.3
    # regression stays flat: A-loss not much higher
    assert metrics["regress_A"]["candidate"] < metrics["regress_A"]["base"] + 0.1


def test_corpus_loss_and_load_model(pair):
    tok, cfg, base_ck = pair["tok"], pair["cfg"], pair["base_ck"]
    model = load_model(cfg, base_ck)
    assert isinstance(model, LiteLM)
    la = corpus_loss(model, tok, [A, A, A])
    lb = corpus_loss(model, tok, [B])
    assert 0.0 < la < lb  # unseen B is harder than seen A
    load_checkpoint(base_ck, model, None, None)  # model is reusable after load_model