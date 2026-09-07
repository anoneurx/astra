"""Inference utilities (docs/MODEL.md § inference, docs/ROADMAP.md Phase 3)."""

from astra.inference.decoder import KVCache, decode, decode_token

__all__ = ["KVCache", "decode", "decode_token"]