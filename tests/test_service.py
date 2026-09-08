"""Tests for the Python inference service skeleton (service/inference.py)."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from threading import Thread

import pytest
from astra.inference import decode
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json

from service.inference import InferenceApp, make_handler

NACKPT = "checkpoints/name/resumed/final.npz"
NCFG = "configs/toy_name.json"


@pytest.fixture(scope="module")
def app() -> InferenceApp:
    return InferenceApp(NACKPT, NCFG)


def test_health(app):
    h = app.health()
    assert h["status"] == "ok"
    assert h["params"] == 133440
    assert h["step"] == 2600
    assert len(h["checksum"]) == 64


def test_generate_matches_decoder_directly(app):
    r = app.generate({"prompt": "My name is Astra", "seed": 7, "max_new": 12, "top_k": 5})
    raw = read_json(NCFG)
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(tok)})
    model = LiteLM(cfg, seed=0)
    load_checkpoint(NACKPT, model, opt=None, schedule=None)
    import numpy as np

    gen = decode(model, tok.encode("My name is Astra"), 12,
                 temperature=0.6, rng=np.random.default_rng(7), top_k=5)
    assert r["text"] == tok.decode(gen)
    assert r["tokens"] == 12


def test_generate_requires_prompt(app):
    with pytest.raises(ValueError, match="prompt"):
        app.generate({"seed": 0})


def test_empty_prompt_rejected(app):
    with pytest.raises(ValueError, match="prompt"):
        app.generate({"prompt": "   ", "seed": 0})


def test_deterministic_given_seed(app):
    a = app.generate({"prompt": "field voltage =", "seed": 3, "max_new": 8})
    b = app.generate({"prompt": "field voltage =", "seed": 3, "max_new": 8})
    assert a["text"] == b["text"]


def test_caps_max_new(app):
    r = app.generate({"prompt": "x", "seed": 1, "max_new": 9999})
    assert r["tokens"] <= 256


class _QuietServer:
    """Minimal harness to exercise the real HTTP handler."""

    def __init__(self, app):
        from http.server import ThreadingHTTPServer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        self.thread = Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.thread.join()


@pytest.fixture()
def live_server(app):
    srv = _QuietServer(app)
    yield srv
    srv.close()


def test_http_health(live_server):
    conn = HTTPConnection("127.0.0.1", live_server.port, timeout=30)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["status"] == "ok"
    conn.close()


def test_http_generate(live_server):
    conn = HTTPConnection("127.0.0.1", live_server.port, timeout=60)
    body = json.dumps({"prompt": "Astra", "seed": 2, "max_new": 10}).encode()
    conn.request("POST", "/generate", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    out = json.loads(resp.read())
    assert out["tokens"] == 10
    assert out["prompt"] == "Astra"
    conn.close()


def test_http_errors(live_server):
    conn = HTTPConnection("127.0.0.1", live_server.port, timeout=30)
    conn.request("GET", "/nope")
    assert conn.getresponse().status == 404
    conn.request("POST", "/generate", body=b"{}",
                 headers={"Content-Type": "application/json"})
    assert conn.getresponse().status == 400
    conn.close()