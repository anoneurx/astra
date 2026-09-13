#!/usr/bin/env python3
"""Build the public-domain prose corpus (docs/DATA.md, docs/PLAN-ASTRA-5M.md).

Downloads classic public-domain books (Project Gutenberg, US public domain),
strips the Gutenberg boilerplate, scrubs control characters, de-duplicates
paragraphs, and writes frozen train/val/eval splits with a provenance + hashes
manifest. All raw+intermediate work lands on the external tmp drive; only the
frozen texts and manifest are written into the repo (datasets/prose/).

Usage:
    python tools/build_prose_corpus.py \
        --tmp /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/prose \
        --out datasets/prose \
        --eval-books "etext1342" "etext11"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

BOOKS = {
    # id -> (url, title) — public-domain prose with well-formed spelling/sentences
    "1342": ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
    "2701": ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick"),
    "11": ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
    "1661": ("https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes"),
    "98": ("https://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities"),
    "84": ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein"),
    "74": ("https://www.gutenberg.org/files/74/74-0.txt", "The Adventures of Tom Sawyer"),
    "43": ("https://www.gutenberg.org/files/43/43-0.txt", "Dracula"),
    "408": ("https://www.gutenberg.org/files/408/408-0.txt", "Heart of Darkness"),
    "132": ("https://www.gutenberg.org/files/132/132-0.txt", "The Art of War"),
    "1080": ("https://www.gutenberg.org/files/1080/1080-0.txt", "A Modest Proposal"),
}

HEADER_RE = re.compile(
    r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .*?\*\*\*\s*",
    re.DOTALL,
)
FOOTER_RE = re.compile(
    r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .*?\*\*\*\s*",
    re.DOTALL,
)
DASHED_LINE = re.compile(r"^(-{5,}|_{5,}|={5,}|#{5,}|\.{5,})$")


def clean_text(raw: str) -> str:
    """Strip Gutenberg boilerplate and scrubbed control characters."""
    raw = HEADER_RE.sub("", raw)
    raw = FOOTER_RE.sub("", raw)
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or DASHED_LINE.match(ln):
            continue
        if ln.startswith("[Illustration"):
            continue
        if ln in {"Project Gutenberg", "www.gutenberg.org"}:
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def dedupe_paragraphs(text: str) -> str:
    seen: set[str] = set()
    paras = []
    for para in text.split("\n\n"):
        p = " ".join(para.split()).strip()
        if not p:
            continue
        canon = p.lower()
        if canon in seen:
            continue
        seen.add(canon)
        paras.append(p)
    return "\n\n".join(paras) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "astra-data/0.4 (docs)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"  downloaded {url} -> {dest} ({len(data)} bytes)")
            return
        except Exception as exc:
            if dest.exists():
                dest.unlink()  # discard partial download
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            print(f"  retry {attempt}/{retries} for {url} ({exc})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True, help="external tmp dir for raw + intermediates")
    ap.add_argument("--out", default="datasets/prose", help="repo output dir")
    ap.add_argument("--skip-download", action="store_true", help="reuse raw files on tmp")
    args = ap.parse_args()

    tmp = Path(args.tmp)
    raw_dir = tmp / "raw"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("[prose] downloading books")
        for book_id, (url, _title) in BOOKS.items():
            download(url, raw_dir / f"{book_id}.txt")

    cleaned: list[tuple[str, str]] = []
    for book_id, (_, title) in BOOKS.items():
        raw_path = raw_dir / f"{book_id}.txt"
        if not raw_path.exists():
            print(f"[prose] skipping {book_id} (missing on tmp)")
            continue
        cleaned_text = clean_text(raw_path.read_text(encoding="utf-8", errors="replace"))
        cleaned_text = dedupe_paragraphs(cleaned_text)
        cleaned.append((book_id, title, cleaned_text))
        print(f"[prose] {book_id}: {len(cleaned_text):,} chars cleaned")

    total = sum(len(t) for *_t, t in cleaned)
    print(f"[prose] cleaned corpus: {len(cleaned)} books, {total:,} chars")

    # frozen splits: last full book is eval, second-to-last is val, rest train
    train_parts = [t for *_x, t in cleaned[:-2]]
    val_parts = [cleaned[-2][2]]
    eval_parts = [cleaned[-1][2]]

    def write_split(name: str, parts: list[str]) -> None:
        content = "\n\n".join(parts).strip() + "\n"
        path = out_dir / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        print(f"[prose] {name}: {len(content):,} chars -> {path}")
        return content

    train_text = write_split("train", train_parts)
    val_text = write_split("val", val_parts)
    eval_text = write_split("eval", eval_parts)

    train_sha = sha256_bytes(train_text.encode("utf-8"))
    val_sha = sha256_bytes(val_text.encode("utf-8"))
    eval_sha = sha256_bytes(eval_text.encode("utf-8"))

    manifest = {
        "dataset": "prose",
        "pipeline": "tools/build_prose_corpus.py",
        "source": "Project Gutenberg (US public domain)",
        "license": "public domain (US); Project Gutenberg License applies to presentation",
        "books": [
            {"id": bid, "title": title, "raw_sha256": sha256_bytes(
                (raw_dir / f"{bid}.txt").read_bytes() if (raw_dir / f"{bid}.txt").exists() else b"")}
            for bid, title in [(b, BOOKS[b][1]) for b in [c[0] for c in cleaned]]
        ],
        "cleaning_rules": "strip Gutenberg START/END boilerplate; control-char scrub; paragraph dedup; whitespace normalize",
        "splits": {
            "train": {"path": "datasets/prose/train.txt", "sha256": train_sha, "bytes": len(train_text)},
            "val": {"path": "datasets/prose/val.txt", "sha256": val_sha, "bytes": len(val_text)},
            "eval": {"path": "datasets/prose/eval.txt", "sha256": eval_sha, "bytes": len(eval_text)},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[prose] manifest -> {out_dir / 'manifest.json'}")
    print("[prose] next: train the vocab-8192 tokenizer on datasets/prose/train.txt")


if __name__ == "__main__":
    main()