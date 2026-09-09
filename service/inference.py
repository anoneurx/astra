#!/usr/bin/env python3
"""Astra inference service (Python reference) — Phase-3 service skeleton.

Thin HTTP wrapper over the incremental KV-cache decoder
(astra.inference.decode). Stateless per request: each call builds a fresh
KVCache so responses stay focused on the prompt. The Rust service (see
service/rust/) is the long-term runtime; this Python reference defines the
wire API the Rust engine must match (docs/releases/v0.4.0.md).

Endpoints:
    GET  /health          -> {"status": "ok", "model": ..., "step": ...}
    POST /generate        -> {"text": "...", "tokens": n, "tok/s": ...}
        body: {"prompt": ..., "max_new": 32, "temperature": 0.6,
               "top_k": 8, "top_p": 0.0, "seed": 0, "window": 0}

Usage:
    python service/inference.py --checkpoint checkpoints/name/resumed/final.npz \
        --config configs/toy_name.json --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import KVCache, decode
from astra.model import LiteLM, ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.training.checkpoint import load_checkpoint
from astra.utils import read_json, sha256_file


class InferenceApp:
    """Plain functions wrapping the decoder; also used directly by tests."""

    def __init__(
        self,
        checkpoint: str,
        config: str,
        memory: str | None = None,
        memory_dir: str = "memory/store",
        memory_embedder: str = "hash",
        memory_budget_tokens: int = 256,
        memory_k: int = 8,
    ):
        raw = read_json(config)
        self.tokenizer = ByteLevelBPE.load(raw["tokenizer"])
        cfg = ModelConfig.from_dict({**raw["model"], "vocab_size": len(self.tokenizer)})
        self.model = LiteLM(cfg, seed=0)
        self.step, _hist, _meta = load_checkpoint(checkpoint, self.model, opt=None, schedule=None)
        self.checkpoint = checkpoint
        self.checksum = sha256_file(checkpoint)
        self.memory = None
        if memory:
            from astra.memory import HashEmbedder, LiteLMExtractor, MemoryStore

            if memory_embedder == "litelm":
                embedder = LiteLMExtractor(self.model, self.tokenizer)
            else:
                embedder = HashEmbedder()
            self.memory = MemoryStore.open(memory, memory_dir, embedder=embedder)
            self.memory_budget_tokens = int(memory_budget_tokens)
            self.memory_k = int(memory_k)

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": "LiteLM",
            "params": self.model.num_params,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "checksum": self.checksum,
            "memory": self.memory.name if self.memory else None,
            "memory_embedding": self.memory.meta["embedding_config"]["name"] if self.memory else None,
        }

    def _memory_block(self, prompt: str) -> tuple[str, dict]:
        """Retrieve memories for ``prompt`` and prepend the ``<|memory|>`` block."""
        from astra.memory import build_memory_block

        hits = self.memory.search(
            query=prompt,
            k=self.memory_k,
            budget_tokens=self.memory_budget_tokens,
            tokenizer=self.tokenizer,
        )
        block = build_memory_block(hits, self.tokenizer, budget_tokens=self.memory_budget_tokens)
        text = block.prepend(prompt)
        info = {
            "included": [h.record.id for h in block.included],
            "dropped": [h.record.id for h in block.dropped],
            "block_tokens": len(block.tokens),
        }
        return text, info

    def generate(self, payload: dict) -> dict:
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            raise ValueError("'prompt' is required and must be non-empty")
        memory_info = None
        use_memory = bool(payload.get("memory", self.memory is not None))
        if use_memory and self.memory is not None:
            prompt, memory_info = self._memory_block(prompt)
        seed = int(payload.get("seed", 0))
        rng = np.random.default_rng(seed)
        seed_ids = self.tokenizer.encode(prompt)
        max_new = max(1, min(int(payload.get("max_new", 32)), 256))
        cache = KVCache(self.model.cfg)
        window = int(payload.get("window", 0))
        gen = decode(
            self.model,
            seed_ids,
            max_new,
            temperature=float(payload.get("temperature", 0.6)),
            rng=rng,
            top_k=int(payload.get("top_k", 8)),
            top_p=float(payload.get("top_p", 0.0)),
            window=window,
            cache=cache,
        )
        try:
            text = self.tokenizer.decode(gen)
        except (UnicodeDecodeError, KeyError):
            text = "".join(chr(b) if 32 <= b < 127 else "." for b in
                           b"".join(self.tokenizer.id_to_piece.get(i, b"?") for i in gen))
        result = {"text": text, "tokens": len(gen), "prompt": prompt, "seed": seed}
        if memory_info is not None:
            result["memory"] = memory_info
        return result


def make_handler(app: InferenceApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quiet stdlib logging
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/health":
                self._json(200, app.health())
            else:
                self._json(404, {"error": f"unknown path {self.path!r}"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/generate":
                self._json(404, {"error": f"unknown path {self.path!r}"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = app.generate(payload)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - enforce JSON error boundary
                self._json(500, {"error": f"internal: {exc}"})
                return
            self._json(200, result)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Astra inference service (Python reference)")
    ap.add_argument("--checkpoint", default="checkpoints/name/resumed/final.npz")
    ap.add_argument("--config", default="configs/toy_name.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--memory", default=None, help="memory store name (memory/store/<name>.json)")
    ap.add_argument("--memory-dir", default="memory/store")
    ap.add_argument("--memory-embedder", default="hash", choices=["hash", "litelm"])
    ap.add_argument("--memory-budget-tokens", type=int, default=256)
    ap.add_argument("--memory-k", type=int, default=8)
    args = ap.parse_args()

    app = InferenceApp(args.checkpoint, args.config, memory=args.memory,
                       memory_dir=args.memory_dir, memory_embedder=args.memory_embedder,
                       memory_budget_tokens=args.memory_budget_tokens, memory_k=args.memory_k)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Astra inference service on http://{args.host}:{args.port} "
          f"({app.model.num_params} params, step {app.step})")
    if app.memory:
        print(f"memory: {app.memory.name} ({app.memory.meta['embedding_config']['name']}, "
              f"k={app.memory_k}, budget={app.memory_budget_tokens} tokens)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()