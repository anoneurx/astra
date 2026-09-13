#!/usr/bin/env python3
"""Print SHA-256 for artifacts (models, datasets, tokenizers).

Usage: python tools/checksum.py <path...>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from astra.utils import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        print(f"{sha256_file(p)}  {p}")


if __name__ == "__main__":
    main()