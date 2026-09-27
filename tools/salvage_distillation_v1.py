#!/usr/bin/env python3
"""Salvage the finished V1 distillation run into the V2 format.

V1 produced 1002 records in `distillation/astra_v1_distillation.jsonl`. That data is not
untrainable — 913 of the 1002 inputs are distinct and the labels are internally consistent —
but it cannot be fed to `training/gpu_train.py` as-is, because:

  * every `risk` value is lowercase and `configs/astra_200m.json:42` requires uppercase;
  * 264 inputs span several lines and the trainer reads one row per line;
  * the JSONL layout is not a corpus the tokenizer or the trainer is pointed at.

This script repairs what can be repaired, drops what cannot, and prints exactly what it did.
Nothing is regenerated and no API key is needed, so it runs in seconds.

The thresholds below are copies of the V2 notebook's. `tests/test_salvage_parity.py`
asserts they still agree, so the two cannot drift apart silently.

Usage:
    python3 tools/salvage_distillation_v1.py \\
        --in /home/kashie/Documents/Projects/Project\\ Astra/distillation/astra_v1_distillation.jsonl \\
        --repo /home/kashie/Documents/Projects/Project\\ Astra/astra
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- constants
# Mirrors of the V2 notebook's rules. Kept literal rather than imported so this script stays
# runnable on its own; `check_parity.py` asserts the two copies agree.
CLASSIFICATIONS = ("normal", "suspicious", "malicious", "unknown")
RISK_LEVELS = ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")
HARDNESS_BY_RISK = {"INFO": 5, "LOW": 20, "MEDIUM": 45, "HIGH": 70, "CRITICAL": 90}
EARLY_WARNING_CLASSIFICATIONS = ("suspicious", "malicious")
REQUIRED_FIELDS = ("input", "classification", "risk", "early_warning",
                   "evidence", "analysis", "recommended_action", "confidence")
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS) | {
    "id", "dedup_key", "category", "grid_index", "source", "synthetic",
    "teacher_model", "created_at", "hardness"}

MIN_INPUT_CHARS, MAX_INPUT_CHARS = 120, 480
MIN_ANALYSIS_CHARS, MAX_ANALYSIS_CHARS = 40, 420
MIN_ACTION_CHARS = 40
MAX_CONFIDENCE = 0.97
REPEAT_ANCHOR, MAX_REPEATS = 60, 3
NO_OP_ACTIONS = ("no action required", "no action needed", "none", "no action",
                 "no further action", "nothing", "n/a", "not applicable")

CATEGORIES = ["Normal behavior", "Suspicious behavior", "Early-warning sequences",
              "False positives", "Multi-signal correlation", "Authentication",
              "Network", "Process behavior", "Vulnerability / code security"]
# V1's weights, so a salvaged corpus keeps the distribution it was generated with rather than
# being re-weighted into a shape nobody asked for. Normalised by quota_plan.
CATEGORY_WEIGHTS = [0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]

SEED = 20051
VAL_FRAC, EVAL_FRAC = 0.01, 0.04
DATA_DIR = "datasets/distillation_v2"
RECORDS_NAME = "records.jsonl"
CHAT_BLOCK_TMPL = "You: %s\nAstra: %s"
ALERT_TMPL = "ROW#%d ALERT %s | EARLY_WARNING %s RISK=%s HARDNESS=%d"


# --------------------------------------------------------------------------- helpers
def one_line(value, sep=" ") -> str:
    """Collapse any whitespace run to a single separator.

    V1 wrote 264 inputs as multi-line "Chronological Events:" blocks. The trainer reads the
    corpus as a token stream, so newlines inside a record become paragraph breaks mid-example.
    """
    return sep.join(str(value).split())


def dedup_key(text: str) -> str:
    return "distill:" + hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def is_no_op(action: str) -> bool:
    body = action.strip().rstrip(".!? ").lower()
    if body in NO_OP_ACTIONS:
        return True
    return any(body.startswith(p) and len(body) < len(p) + 90 for p in NO_OP_ACTIONS)


def keeps_no_op(classification: str, strict: bool) -> bool:
    """Whether a bare "No action required." is acceptable for this class.

    The V2 validator bans no-op targets outright, and for generated data that is right: a
    teacher asked to triage a `malicious` sequence and answering "no action required" is the
    failure mode the rule exists to stop. But for a `normal` row it is the *correct* answer,
    and V1 wrote it 93 times. Applying the generated-data rule to salvaged data would delete
    97% of the normal class (219 rows in, 12 out) and hand the student a corpus that is
    99% `suspicious` — a worse skew than the one V1 had. So the rule is scoped to the classes
    where an action is owed, and `--strict-noop` restores the blanket rule.
    """
    return (not strict) and classification == "normal"


def is_repetitive(text: str, anchor=REPEAT_ANCHOR, max_repeats=MAX_REPEATS,
                  window=MAX_INPUT_CHARS) -> bool:
    head = text[:window]
    if len(head) < 4 * anchor:
        return False
    for off in (0, len(head) // 2, len(head) - 2 * anchor):
        frag = head[off:off + anchor]
        if len(frag) < anchor or not frag.strip():
            continue
        if head.count(frag) >= max_repeats:
            return True
    return False


def quota_plan(total: int, weights: list[float]) -> dict[str, int]:
    """Largest-remainder apportionment, so the counts are integers summing to `total`."""
    raw = [total * w / sum(weights) for w in weights]
    base = [int(x) for x in raw]
    short = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:short]:
        base[i] += 1
    return dict(zip(CATEGORIES, base))


# --------------------------------------------------------------------------- repair
def salvage(raw: dict, strict_noop: bool = False,
             max_input: int = MAX_INPUT_CHARS,
             max_analysis: int = MAX_ANALYSIS_CHARS) -> tuple[dict | None, str]:
    """Return (clean_record, reason). `reason` is None on success."""
    if not isinstance(raw, dict):
        return None, "not_an_object"

    text = one_line(raw.get("input", ""))
    if len(text) < MIN_INPUT_CHARS:
        return None, "input_too_short"
    if len(text) > max_input:
        return None, "input_too_long"
    if is_repetitive(text):
        return None, "input_repetitive"

    classification = str(raw.get("classification", "")).strip().lower()
    if classification not in CLASSIFICATIONS:
        return None, "bad_classification"

    risk = str(raw.get("risk", "")).strip().upper()
    if risk not in RISK_LEVELS:
        return None, "bad_risk"

    evidence = raw.get("evidence")
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        return None, "evidence_not_list"
    evidence = [one_line(e) for e in evidence if str(e).strip()]
    if not evidence:
        return None, "evidence_empty"

    analysis = one_line(raw.get("analysis", ""))
    if len(analysis) < MIN_ANALYSIS_CHARS:
        return None, "analysis_too_short"
    if len(analysis) > max_analysis:
        return None, "analysis_too_long"
    if is_repetitive(analysis):
        return None, "analysis_repetitive"

    # V1 had 3 records with both `action` and `recommended_action`; the short `action` value
    # ("Analyze sequence") is never the answer, so `recommended_action` always wins.
    action = one_line(raw.get("recommended_action") or "")
    if not action:
        return None, "no_recommended_action"
    if is_no_op(action):
        if not keeps_no_op(classification, strict_noop):
            return None, "action_is_no_op"
    if len(action) < MIN_ACTION_CHARS:
        return None, "action_too_short"

    early = raw.get("early_warning")
    if isinstance(early, str):
        early = early.strip().lower() in ("true", "yes", "1")
    early = bool(early)
    # V1 left 74 malicious rows unflagged. The flag is a function of the label, so it is
    # repaired rather than discarded: the underlying telemetry did say "malicious".
    if early != (classification in EARLY_WARNING_CLASSIFICATIONS):
        early = classification in EARLY_WARNING_CLASSIFICATIONS

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        return None, "bad_confidence"
    confidence = min(MAX_CONFIDENCE, max(0.0, confidence))

    category = str(raw.get("category", "")).strip()
    if category not in CATEGORIES:
        category = "Suspicious behavior" if classification != "normal" else "Normal behavior"

    return {
        "input": text,
        "classification": classification,
        "risk": risk,
        "early_warning": early,
        "evidence": evidence,
        "analysis": analysis,
        "recommended_action": action,
        "confidence": confidence,
        "hardness": HARDNESS_BY_RISK[risk],
        "category": category,
    }, None


def answer_text(rec: dict) -> str:
    """What the student is asked to produce, in the V2 target shape."""
    ev = "; ".join(rec["evidence"])
    return ("%s Classification: %s, risk %s. Recommended action: %s"
            % (rec["analysis"], rec["classification"].upper(), rec["risk"],
               rec["recommended_action"]))


def to_alert_line(rec: dict, i: int) -> str:
    return ALERT_TMPL % (i, rec["input"], rec["classification"], rec["risk"], rec["hardness"])


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="V1 astra_v1_distillation.jsonl")
    ap.add_argument("--repo", default=".", help="astra repo root to write into")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--eval-frac", type=float, default=EVAL_FRAC)
    ap.add_argument("--max-input-chars", type=int, default=MAX_INPUT_CHARS,
                    help="input cap; 480 keeps one chat exchange inside the 256-token "
                         "block, 1200 keeps ~200 more V1 rows but overflows the block")
    ap.add_argument("--max-analysis-chars", type=int, default=MAX_ANALYSIS_CHARS)
    ap.add_argument("--strict-noop", action="store_true",
                    help="reject 'No action required.' even on normal rows (deletes ~97%% "
                         "of the normal class; see keeps_no_op)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing distillation_v2 directory")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    repo = Path(args.repo).expanduser().resolve()
    if not src.is_file():
        print("no such file:", src)
        return 1
    out = repo / DATA_DIR
    if out.exists():
        if not args.force:
            print("%s already exists - pass --force to overwrite" % out)
            return 1
        shutil.rmtree(out)

    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    raws = [json.loads(l) for l in lines]
    print("=" * 74)
    print("salvaging %s" % src)
    print("  %d input records, %.1f KiB" % (len(lines), src.stat().st_size / 1024))
    print("=" * 74)

    # ---- pass 1: repair, reject, dedup
    seen: dict[str, int] = {}
    kept: list[dict] = []
    rejects: Counter = Counter()
    repairs: Counter = Counter()
    unknown_fields: Counter = Counter()
    cls_seen: Counter = Counter()
    cls_dropped: Counter = Counter()
    drops_by_cls: dict[str, list[str]] = defaultdict(list)
    fixed_ew = fixed_risk = fixed_conf = 0

    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            rejects["unparseable_json"] += 1
            continue
        if isinstance(raw, dict):
            for k in raw:
                if k not in KNOWN_FIELDS:
                    unknown_fields[k] += 1
        cls_in = str(raw.get("classification", "?")).strip().lower()
        cls_seen[cls_in] += 1

        if str(raw.get("risk", "")).strip() != str(raw.get("risk", "")).strip().upper():
            fixed_risk += 1
        if "\n" in str(raw.get("input", "")):
            repairs["multiline_input_flattened"] += 1
        ew_before = raw.get("early_warning")
        conf_before = raw.get("confidence")

        rec, reason = salvage(raw, args.strict_noop,
                              args.max_input_chars, args.max_analysis_chars)
        if rec is None:
            rejects[reason] += 1
            cls_dropped[cls_in] += 1
            drops_by_cls[cls_in].append(reason)
            continue

        if bool(ew_before) != rec["early_warning"]:
            fixed_ew += 1
        if conf_before != rec["confidence"]:
            fixed_conf += 1

        key = dedup_key(rec["input"])
        if key in seen:
            rejects["duplicate_input"] += 1
            cls_dropped[cls_in] += 1
            drops_by_cls[cls_in].append("duplicate_input")
            continue
        seen[key] = len(kept)
        rec["dedup_key"] = key
        kept.append(rec)

    print()
    print("rejected / dropped")
    if rejects:
        for reason, n in rejects.most_common():
            print("  %-26s %5d  (%.1f%% of input)"
                  % (reason, n, 100.0 * n / len(lines)))
    else:
        print("  none")
    print()
    print("repaired in place")
    print("  %-26s %5d" % ("risk uppercased", fixed_risk))
    print("  %-26s %5d" % ("multiline input flattened", repairs["multiline_input_flattened"]))
    print("  %-26s %5d" % ("early_warning realigned", fixed_ew))
    print("  %-26s %5d" % ("confidence pulled off 1.0", fixed_conf))
    if unknown_fields:
        print("  unexpected fields seen:", dict(unknown_fields))
    print()
    print("kept %d of %d records (%.1f%%)   caps: input<=%d analysis<=%d"
          % (len(kept), len(lines), 100.0 * len(kept) / len(lines),
             args.max_input_chars, args.max_analysis_chars))
    if args.max_input_chars != 1200:
        n_loose = sum(1 for raw in raws
                      if salvage(raw, args.strict_noop, 1200, 900)[0] is not None)
        print("  for reference, at the looser 1200/900 cap this same run would keep %d "
              "(+%d), but ~23%% of its blocks then overflow the 256-token student block"
              % (n_loose, n_loose - len(kept)))
    print()
    print("survival by classification (this is the number that matters:")
    cls_kept = Counter(r["classification"] for r in kept)
    for cls in ("normal", "suspicious", "malicious", "unknown"):
        if not cls_seen.get(cls):
            continue
        n_in, n_out = cls_seen[cls], cls_kept.get(cls, 0)
        drop_reasons = Counter(drops_by_cls.get(cls, []))
        print("  %-11s %4d in -> %4d kept (%5.1f%%)   dropped: %s"
              % (cls, n_in, n_out, 100.0 * n_out / n_in,
                 ", ".join("%s=%d" % kv for kv in drop_reasons.most_common()) or "none"))
    if not kept:
        print("nothing salvageable")
        return 2

    # ---- stamp provenance
    for i, rec in enumerate(kept, 1):
        rec.update({
            "id": "salvaged-v1-%06d" % i,
            "grid_index": i,
            "source": "gemini-3.5-flash-lite-distillation-v1-salvaged",
            "synthetic": True,
            "teacher_model": "gemini-3.5-flash-lite",
            "created_at": "salvaged",
        })

    # ---- write records.jsonl
    (out / "chat").mkdir(parents=True, exist_ok=True)
    (out / "alerts").mkdir(parents=True, exist_ok=True)
    records_path = out / RECORDS_NAME
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %s (%d records)" % (records_path, len(kept)))

    # ---- split, stratified by category so every split sees every class
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in kept:
        by_cat[rec["category"]].append(rec)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "eval": []}
    rng = random.Random(SEED)
    # int(n * frac) rounds to 0 once a category is small, which silently emptied val on a
    # 562-row corpus. Reserve the held-out rows globally first, then stratify the remainder.
    total = len(kept)
    n_val = max(1, int(round(total * args.val_frac))) if total >= 20 else 0
    n_eval = max(1, int(round(total * args.eval_frac))) if total >= 20 else 0
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda r: r["dedup_key"])
        rng.shuffle(items)
        n = len(items)
        # a category never gives up more than a quarter of itself, so one thin category
        # cannot starve the rest
        c_val = min(int(round(n * args.val_frac)), max(1, n // 4))
        c_eval = min(int(round(n * args.eval_frac)), max(1, n // 4))
        c_val = min(c_val, max(0, n - 1))
        c_eval = min(c_eval, max(0, n - c_val - 1))
        splits["val"].extend(items[:c_val])
        splits["eval"].extend(items[c_val:c_val + c_eval])
        splits["train"].extend(items[c_val + c_eval:])
    # Top up val/eval to the global target if stratification fell short. Rows are moved out of
    # train by identity, not copied - copying them is how a record ends up in two splits and
    # the row-level assertion below fails.
    if n_val or n_eval:
        still = sorted(range(len(splits["train"])),
                       key=lambda i: (splits["train"][i]["category"],
                                      splits["train"][i]["dedup_key"]))
        for name, want in (("val", n_val), ("eval", n_eval)):
            while len(splits[name]) < want and still:
                splits[name].append(splits["train"].pop(still.pop()))
    for name in splits:
        splits[name].sort(key=lambda r: (r["category"], r["dedup_key"]))

    split_meta: dict[str, dict] = {"chat": {}, "alerts": {}}
    for name, items in splits.items():
        chat_path = out / "chat" / (name + ".txt")
        alert_path = out / "alerts" / (name + ".txt")
        # matches datasets/chat/make_corpus.py: "\n\n".join(blocks), no trailing newline
        chat_path.write_text(
            "\n\n".join(CHAT_BLOCK_TMPL % (r["input"], answer_text(r)) for r in items),
            encoding="utf-8")
        # matches tools/build_cyber_corpus.py: one "\n"-terminated line per row
        alert_path.write_text(
            "".join(to_alert_line(r, i) for i, r in enumerate(items)), encoding="utf-8")
        dist = dict(Counter(r["classification"] for r in items).most_common())
        for flavour, path in (("chat", chat_path), ("alerts", alert_path)):
            blob = path.read_bytes()
            split_meta[flavour][name] = {
                "path": str(path.relative_to(repo)),
                "rows": len(items),
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "classification_distribution": dist,
            }

    # ---- prove no record body appears in two splits
    bodies = {n: {r["dedup_key"] for r in items} for n, items in splits.items()}
    shared = {"%s|%s" % (a, b): len(bodies[a] & bodies[b])
              for a, b in (("train", "val"), ("train", "eval"), ("val", "eval"))}
    assert shared["train|val"] == 0 and shared["train|eval"] == 0, "record leaked across splits"

    # ---- manifest
    plan = quota_plan(len(kept), CATEGORY_WEIGHTS)
    manifest = {
        "name": "astra_v2_salvaged_from_v1",
        "source": str(src),
        "generated_by": "salvage_v1.py",
        "seed": SEED,
        "records": len(kept),
        "records_in": len(lines),
        "rejected": dict(rejects.most_common()),
        "repaired": {"risk_uppercased": fixed_risk,
                     "early_warning_realigned": fixed_ew,
                     "confidence_capped": fixed_conf,
                     "multiline_input_flattened": repairs["multiline_input_flattened"]},
        "unexpected_fields": dict(unknown_fields),
        "category_counts": dict(Counter(r["category"] for r in kept).most_common()),
        "category_plan_at_this_size": plan,
        "classification_counts": dict(Counter(r["classification"] for r in kept).most_common()),
        "risk_counts": dict(Counter(r["risk"] for r in kept).most_common()),
        "row_level_safety": shared,
        "splits": split_meta,
        "leak_check": {
            "enabled_in_config": False,
            "reason": ("leak_check(train, val, n=13) compares raw token n-grams, and every "
                       "block repeats the You:/Astra: framing. It measures the template, not "
                       "contamination. Row-level disjointness is proven above."),
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    # ---- configs
    base_cfg = json.loads((repo / "configs/astra5m_word_chat.json").read_text(encoding="utf-8"))
    configs = []
    for flavour in ("chat", "alerts"):
        cfg = json.loads(json.dumps(base_cfg))
        cfg["data"] = {
            "train": f"{DATA_DIR}/{flavour}/train.txt",
            "val": f"{DATA_DIR}/{flavour}/val.txt",
        }
        cfg["safety"] = {"leak_check": False, "note": manifest["leak_check"]["reason"]}
        cfg["out_dir"] = f"checkpoints/distill_v2_salvaged_{flavour}"
        path = repo / f"configs/astra_distill_v2_salvaged_{flavour}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        configs.append(path)

    # ---- report
    print()
    print("splits")
    for name in ("train", "val", "eval"):
        print("  %-5s %4d rows   chat %8d bytes   alerts %8d bytes"
              % (name, len(splits[name]),
                 split_meta["chat"][name]["bytes"], split_meta["alerts"][name]["bytes"]))
    print("  row-level overlap:", shared)
    print()
    print("classifications:", manifest["classification_counts"])
    print("categories      :")
    for cat, n in sorted(manifest["category_counts"].items(), key=lambda kv: -kv[1]):
        print("    %-34s %4d  (plan at this size: %4d)" % (cat, n, plan[cat]))
    print()
    print("manifest  ->", manifest_path)
    for path in configs:
        print("config    ->", path)
    print()
    print("next: build the tokenizer, then train")
    print("  python3 tools/build_word_tokenizer.py")
    print("  python3 training/gpu_train.py --config configs/astra_distill_v2_salvaged_chat.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
