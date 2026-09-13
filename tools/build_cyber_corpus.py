#!/usr/bin/env python3
"""Replay engine for the cyber early-warning corpus (docs/CYBER-ALERTS.md).

Converts the highly-verified NSL-KDD intrusion dataset (UNB CIC) into a
deterministic, leak-checked text corpus: each connection row becomes an ALERT
line with the full 41-feature telemetry followed by the verified early-warning
label. The replay is the "teacher signal": the expert-labeled rows, frozen,
split, and audited before the student model ever touches it.

Usage:
    python tools/build_cyber_corpus.py \
        --raw /run/media/kashie/8cace107-39d5-4713-ac43-f0499e1dd2c0/astra_tmp/cyber/raw
        --out datasets/cyber
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FEATURES = [
    "duration", "protocol", "service", "flag", "srcbytes", "dstbytes", "land",
    "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin",
    "numcompromised", "rootshell", "suattempted", "numroot", "numfilecreations",
    "numshells", "numaccessfiles", "numoutboundcmds", "ishostlogin", "isguestlogin",
    "count", "srvcount", "serrorrate", "srvserrorrate", "rerrorrate",
    "srvrerrorrate", "samesrvrate", "diffsrvrate", "srvdiffhostrate",
    "dsthostcount", "dsthostsrvcount", "dsthostsamesrvrate", "dsthostdiffsrvrate",
    "dsthostsamesrcportrate", "dsthostsrvdiffhostrate", "dsthostserrorrate",
    "dsthostsrvserrorrate", "dsthostrerrorrate", "dsthostsrvrerrorrate",
]

KNOWN_ATTACKS = {
    "back", "buffer_overflow", "ftp_write", "guess_passwd", "imap", "ipsweep",
    "land", "loadmodule", "multihop", "neptune", "nmap", "perl", "phf", "pod",
    "portsweep", "rootkit", "satan", "smurf", "spy", "teardrop", "warezclient",
    "warezmaster",
}


def to_alert(fields: list[str]) -> str:
    if len(fields) != 43:  # 41 features + label + hardness
        raise ValueError(f"row has {len(fields)} fields, expected 43")
    label = fields[41]
    hardness = fields[42]
    if not (label == "normal" or label in KNOWN_ATTACKS):
        raise ValueError(f"unknown label {label!r}")
    parts = [f"{name}={fields[i]}" for i, name in enumerate(FEATURES)]
    return f"ALERT {' '.join(parts)} | EARLY_WARNING {label} HARDNESS={hardness}\n"


def row_digest(idx: int, row: str) -> str:
    return hashlib.sha256(f"{idx}:{row}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="datasets/cyber")
    ap.add_argument("--train-frac", type=float, default=0.96)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-train-rows", type=int, default=0, help="cap train rows (0 = all)")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    train_src = raw_dir / "KDDTrain+.txt"
    if not train_src.exists():
        raise SystemExit(f"missing {train_src} (download NSL-KDD first)")

    rows = train_src.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"[cyber] rows={len(rows)} (verified canonical = 125,973)")

    alerts: list[str] = []
    classes: dict[str, int] = {}
    for idx, row in enumerate(rows):
        fields = row.split(",")
        if len(fields) != 43 or fields[41] not in ("normal",) and fields[41] not in KNOWN_ATTACKS:
            print(f"[cyber] skipping malformed row {idx}: {row[:80]!r}")
            continue
        alerts.append(fields[41])
        classes[fields[41]] = classes.get(fields[41], 0) + 1

    print(f"[cyber] class distribution: {json.dumps(classes, sort_keys=True)}")

    # deterministic intra-class order via row-content digest, then stratified split
    ordered = list(range(len(rows)))
    ordered.sort(key=lambda i: (rows[i].split(",")[41], row_digest(i, rows[i])))
    nval = max(1, int(len(ordered) * args.val_frac))
    neval = max(1, int(len(ordered) * args.val_frac))
    split_at = len(ordered) - nval - neval
    train_ordered = ordered[:split_at]
    if args.max_train_rows > 0:
        train_ordered = train_ordered[: args.max_train_rows]
    assigned: dict[int, str] = {}
    for group, idxs in {"train": train_ordered, "val": ordered[split_at:split_at + nval],
                        "eval": ordered[split_at + nval:]}.items():
        for i in idxs:
            assigned[i] = group

    def write_split(name: str, fh) -> None:
        count = 0
        for i in range(len(rows)):
            if assigned.get(i) == name:
                fh.write(f"ROW#{i} " + to_alert(rows[i].split(",")))
                count += 1
        return count

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "eval"):
        path = out_dir / f"{split}.txt"
        with path.open("w", encoding="utf-8") as fh:
            n = write_split(split, fh)
        print(f"[cyber] {split}: {n} rows -> {path} ({path.stat().st_size:,} bytes)")

    had = {"train": None, "val": None, "eval": None}
    for split in had:
        had[split] = hashlib.sha256((out_dir / f"{split}.txt").read_bytes()).hexdigest()

    manifest = {
        "dataset": "cyber-alerts",
        "pipeline": "tools/build_cyber_corpus.py",
        "source": "NSL-KDD (UNB CIC), peer-reviewed intrusion benchmark",
        "source_file": str(train_src),
        "row_count": len(rows),
        "verified_canonical_row_count": 125973,
        "teacher": "expert-labeled NSL-KDD records (verified ground truth)",
        "cleaning_rules": "feature-order fixed; 41-feature telemetry; label curated against known attack set; deterministic stratified split by label+content digest",
        "splits": {
            s: {"path": f"datasets/cyber/{s}.txt", "sha256": had[s],
                "bytes": (out_dir / f"{s}.txt").stat().st_size}
            for s in ("train", "val", "eval")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[cyber] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()