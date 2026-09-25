#!/usr/bin/env python3
"""Import Colab checkpoints into checkpoints/.

Watches ~/Downloads for astra final.npz files produced by
colab/astra_gpu_training.ipynb and installs them at the repo locations the
inference path expects:

    checkpoints/astra5m_word_prose/resumed/final.npz
    checkpoints/astra5m_word_chat/resumed/final.npz

Usage: python tools/import_colab_checkpoints.py

Run it after downloading from Colab; ids the files by size and step manifest.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"

TARGETS = {
    "word_prose": REPO / "checkpoints/astra5m_word_prose/resumed",
    "word_chat": REPO / "checkpoints/astra5m_word_chat/resumed",
}


def candidates() -> list[Path]:
    if not DOWNLOADS.is_dir():
        return []
    return sorted(
        (p for p in DOWNLOADS.rglob("*.npz") if p.is_file() and "final" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def manifest(path: Path) -> dict:
    mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
    if Path(mpath).exists():
        return json.loads(Path(mpath).read_text())
    return {}


def target_for(path: Path) -> tuple[str, Path]:
    m = manifest(path)
    cfg = m.get("model_config", {})
    n_layers = cfg.get("n_layers", 0)
    max_steps = (m.get("train_config") or {}).get("max_steps", 0)
    name = path.name
    if "word_chat" in name or (n_layers == 6 and max_steps >= 3000):
        return "word_chat", TARGETS["word_chat"] / "final.npz"
    if "word_prose" in name or (n_layers == 6 and max_steps >= 5000):
        return "word_prose", TARGETS["word_prose"] / "final.npz"
    return "", TARGETS["word_prose"] / "unknown.npz"


def main() -> None:
    found = candidates()
    if not found:
        print("no final.npz found in", DOWNLOADS)
        sys.exit(1)
    placed = []
    for path in found:
        name, dst = target_for(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        mpath = str(path).rsplit(".npz", 1)[0] + ".manifest.json"
        if Path(mpath).exists():
            shutil.copy2(mpath, str(dst).rsplit(".npz", 1)[0] + ".manifest.json")
        placed.append((name, dst))
        print(f"[ok] {name}: {path.name} ({path.stat().st_size} bytes) -> {dst}")
    if not placed:
        print("no recognized checkpoints; check the files in", DOWNLOADS)


if __name__ == "__main__":
    main()