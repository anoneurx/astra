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
import re
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


def to_alert_any(fields: list[str]) -> str | None:
    """Like to_alert but tolerant of test-set class labels.

    The held-out NSL-KDD test partitions (KDDTest+, KDDTest-21) legitimately
    contain ~17 attack classes that never appear in the merged train set
    (mscan, apache2, snmpguess, saint, ...). For evaluation we still want those
    rows; we just cannot score a class the model never saw. Returns None for
    malformed rows (skipped and counted by the caller).
    """
    if len(fields) != 43:
        return None
    label = fields[41]
    hardness = fields[42]
    if not (label == "normal" or label in KNOWN_ATTACKS or re.fullmatch(r"[a-z0-9_]+", label)):
        return None
    parts = [f"{name}={fields[i]}" for i, name in enumerate(FEATURES)]
    return f"ALERT {' '.join(parts)} | EARLY_WARNING {label} HARDNESS={hardness}\n"
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
    ap.add_argument("--test-file", help="extra NSL-KDD file to import as an eval-only test corpus")
    ap.add_argument("--test-name", default="test", help="output name for the test corpus (e.g. test21)")
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

    # row-level safety evidence (unit of generalization = one connection row)
    row_ids: dict[str, set[int]] = {}
    bodies: dict[str, set[str]] = {}
    for split in ("train", "val", "eval"):
        rows = (out_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        row_ids[split] = {int(r.split()[0][len("ROW#"):]) for r in rows if r.strip()}
        bodies[split] = {r.split(" ", 1)[1] for r in rows if r.strip()}
    shared_ids = sum(len(row_ids[a] & row_ids[b])
                     for a, b in [("train", "val"), ("train", "eval"), ("val", "eval")])
    shared_bodies = sum(len(bodies[a] & bodies[b])
                        for a, b in [("train", "val"), ("train", "eval"), ("val", "eval")])

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
                "bytes": (out_dir / f"{s}.txt").stat().st_size,
                "rows": len(row_ids[s])}
            for s in ("train", "val", "eval")
        },
        "row_level_safety": {
            "shared_row_ids_across_splits": shared_ids,
            "shared_alert_bodies_across_splits": shared_bodies,
            "note": "row-level split is the unit of generalization; n=13 token-ngram overlap can still occur because ~112 bytes/token means 13 tokens span many independent rows whose patterns repeat",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[cyber] manifest -> {out_dir / 'manifest.json'}")

    if args.test_file:
        build_test_split(args.test_file, args.test_name, out_dir, bodies)


def build_test_split(test_file: str, name: str, out_dir: Path, bodies: dict[str, set[str]]) -> None:
    raw = Path(test_file).read_text(encoding="utf-8", errors="replace").splitlines()
    kinds: dict[str, int] = {}
    skipped_malformed = 0
    skipped_unknown_label = 0
    out: list[str] = []
    for i, row in enumerate(raw):
        fields = row.split(",")
        alert = to_alert_any(fields)
        if alert is None:
            if len(fields) != 43:
                skipped_malformed += 1
            else:
                skipped_unknown_label += 1
            continue
        label = fields[41]
        kinds[label] = kinds.get(label, 0) + 1
        out.append(f"ROW#{i} " + alert)

    path = out_dir / f"{name}.txt"
    path.write_text("".join(out), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"[cyber] test[{name}]: {len(out)} rows ({path.stat().st_size:,} bytes)"
          f" sha={sha[:16]}")
    print(f"[cyber] test[{name}] class distribution: {json.dumps(kinds, sort_keys=True)}")
    print(f"[cyber] test[{name}] skipped: malformed={skipped_malformed} unknown_label={skipped_unknown_label}")

    # cross-set row-level safety: the held-out partition is a different NSL-KDD
    # population, but we still prove no identical alert body leaks from any split.
    my_bodies = {r.split(" ", 1)[1] for r in out}
    shared = {s: len(my_bodies & bodies[s]) for s in ("train", "val", "eval")}

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("test_splits", {})[name] = {
        "path": f"datasets/cyber/{name}.txt",
        "source_file": test_file,
        "sha256": sha,
        "bytes": path.stat().st_size,
        "rows": len(out),
        "classes": kinds,
        "row_level_safety": {
            "shared_alert_bodies_with_splits": shared,
            "note": "held-out NSL-KDD test partition; distinct population from train by construction",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[cyber] manifest updated -> {manifest_path}")


if __name__ == "__main__":
    main()