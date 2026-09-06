#!/usr/bin/env python3
"""Cross-split contamination gate (docs/DATA.md § 3.2).

Usage: python tools/leak_check.py --tokenizer tokenizer/artifacts/toy_bpe.json \
        --train datasets/toy/train.txt --heldout datasets/toy/val.txt --n 13
Exits non-zero if any held-out n-gram also appears in training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.safety import leak_check  # noqa: E402
from astra.tokenizer import ByteLevelBPE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=13)
    args = ap.parse_args()

    tok = ByteLevelBPE.load(args.tokenizer)
    tr = tok.encode(Path(args.train).read_text(encoding="utf-8"))
    hd = tok.encode(Path(args.heldout).read_text(encoding="utf-8"))
    result = leak_check(tr, hd, args.n)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["leak_free"] else 1)


if __name__ == "__main__":
    main()