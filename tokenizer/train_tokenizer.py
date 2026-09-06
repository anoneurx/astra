#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer on a corpus and emit a quality report.

Usage: python -m astra.tokenizer.train --config configs/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.tokenizer import ByteLevelBPE
from astra.utils import read_json, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.json")
    args = ap.parse_args()

    cfg = read_json(args.config)
    corpus_path = Path(cfg["corpus"])
    text = corpus_path.read_text(encoding="utf-8")

    tok = ByteLevelBPE(
        vocab_size=cfg.get("vocab_size", 800),
        num_special=cfg.get("num_special", 4),
        min_frequency=cfg.get("min_frequency", 2),
        special_names=tuple(cfg.get("special_names", ["<pad>", "<bos>", "<eos>", "<unk>"])),
    )
    tok.train(text)
    report = tok.quality_report(text)

    out = Path(cfg["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)

    report["artifact"] = str(out)
    report["artifact_sha256"] = sha256_file(out)
    report["corpus"] = str(corpus_path)
    report["corpus_sha256"] = sha256_file(corpus_path)
    report["vocab_size_final"] = len(tok)

    rpath = Path(cfg.get("report_path", "tokenizer_report.json"))
    rpath.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0 — Tokenizer quality report",
        "",
        f"- Tokenizer: {report.get('artifact')}",
        f"- Artifact SHA-256: `{report['artifact_sha256']}`",
        f"- Vocab size (final): {report['vocab_size_final']} (merges: {report['merges']})",
        f"- Corpus: {report['corpus']} (`{report['corpus_sha256'][:12]}…`)",
        f"- Tokens: {report['tokens_total']}",
        f"- Bytes/token: **{report['bytes_per_token']:.3f}**",
        f"- Tokens/char: {report['tokens_per_character']:.4f}",
        f"- Round-trip exact on sample: **{report['roundtrip_exact']}**",
        f"- Unknown tokens used: {report['unknown_tokens_used']}",
        "",
        "**STATUS: VALIDATED**" if report["roundtrip_exact"] else "**STATUS: REJECTED**",
        "",
    ]
    rpath.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"quality report -> {rpath}")


if __name__ == "__main__":
    main()