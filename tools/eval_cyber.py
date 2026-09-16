#!/usr/bin/env python3
"""Evaluate the Astra-5M cyber model: early-warning label accuracy + sentence quality.

For each eval row, scores the model by generating the label continuation from
the alert prefix and comparing against the verified NSL-KDD label. Also reports
sentence-style metrics (predictive text must stay on-line, terminated cleanly).

Usage:
    python tools/eval_cyber.py --config configs/astra5m_cyber.json \
        --checkpoint checkpoints/astra5m_cyber/final.npz --out experiments/cyber/eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
from astra.inference import decode
from astra.learning.evaluate import load_model
from astra.model import ModelConfig
from astra.tokenizer import ByteLevelBPE
from astra.utils import read_json

LABEL_RE = re.compile(r"EARLY_WARNING\s+([a-zA-Z0-9_]+)")
SPACE = " "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-file", default="datasets/cyber/eval.txt",
                    help="corpus file to score against (e.g. datasets/cyber/test21.txt)")
    ap.add_argument("--max-new", type=int, default=16, help="tokens to generate for label")
    ap.add_argument("--samples", type=int, default=2000, help="eval rows to score")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    raw = read_json(args.config)
    tok = ByteLevelBPE.load(raw["tokenizer"])
    cfg = ModelConfig.from_dict(raw["model"])
    cfg.vocab_size = len(tok)

    model = load_model(cfg, args.checkpoint)

    eval_path = Path(args.eval_file)
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()][: args.samples]

    rng = np.random.default_rng(args.seed)
    correct = 0
    detected = 0  # attack labeled as attack (any attack class)
    false_positive = 0
    n_normal = 0
    n_attack = 0
    per_class: dict[str, list[int]] = {}

    for line in lines:
        label_m = LABEL_RE.search(line)
        if not label_m:
            continue
        true_label = label_m.group(1)
        prefix = line[: line.index("| EARLY_WARNING")]
        is_attack = true_label != "normal"
        if is_attack:
            n_attack += 1
        else:
            n_normal += 1
        per_class.setdefault(true_label, [0, 0])

        ids = tok.encode(prefix)
        ids = ids[-96:]  # keep within input window
        new = decode(model, np.asarray(ids, dtype=np.int64), max_new=args.max_new,
                     temperature=args.temperature, rng=rng)
        text = tok.decode(new)
        pred_m = LABEL_RE.search(text)
        pred = pred_m.group(1) if pred_m else ""
        if pred:
            hit = pred == true_label
            per_class[true_label][0] += int(hit)
            per_class[true_label][1] += 1
            correct += int(hit)
            if is_attack:
                detected += int(pred != "normal")
            else:
                false_positive += int(pred != "normal")

    scored = sum(v[1] for v in per_class.values())
    acc = correct / max(1, scored)
    det_rate = detected / max(1, n_attack)
    fp_rate = false_positive / max(1, n_normal)

    report = {
        "checkpoint": args.checkpoint,
        "eval_file": str(eval_path),
        "rows_scored": scored,
        "rows_total": len(lines),
        "accuracy": round(acc, 4),
        "attack_detection_rate": round(det_rate, 4),
        "normal_false_positive_rate": round(fp_rate, 4),
        "class_balance": {"normal": n_normal, "attack": n_attack},
        "per_class_entries": {k: v for k, v in sorted(per_class.items()) if v[1] > 0},
        "sentence_quality": {
            "on_line_continuations": 0,  # placeholder; updated by sample viewer
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()