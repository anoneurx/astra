#!/usr/bin/env python3
"""Phase-7 release checklist automation (docs/PHASE_STATUS.md, docs/RELEASES.md § 2).

One command aggregates the mandatory release gates before a version is tagged:

    python tools/release.py check                # run every gate, no git mutation
    python tools/release.py check --version 1.0.0 --skip rollback,memory-eval
    python tools/release.py tag --version 1.0.0  # gates must pass first, then tag v1.0.0

Gates (each must pass; exit 1 on any failure):
  1. tests        full test suite (pytest)
  2. e2e          Phase-7 full-stack walk (tools/e2e.py)
  3. rollback     offline auto-rollback drill (self_improve --rollback-drill)
  4. memory-eval  adopted retrieval thresholds enforced (memory_eval --enforce)
  5. registry     every registered artifact's checksum verified (no drift)
  6. docs         required docs exist, relative links resolve, no gate markers in source
  7. changelog    docs/CHANGELOG.md has the version heading (+ release notes for tag)

A gate that cannot run because its artifact is absent is a *failed* gate — it
must be provisioned, not skipped silently. Skipping is explicit (--skip).

``tag`` creates an annotated git tag at HEAD after the gates pass; it never
pushes (maintainers push, per docs/CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

REQUIRED_DOCS = [
    "docs/ARCHITECTURE.md",
    "docs/BENCHMARKS.md",
    "docs/CHANGELOG.md",
    "docs/CONTRIBUTING.md",
    "docs/EVALUATION.md",
    "docs/LEARNING.md",
    "docs/MEMORY.md",
    "docs/PHASE_STATUS.md",
    "docs/RELEASES.md",
    "docs/ROADMAP.md",
    "docs/SAFETY.md",
    "docs/VERSIONING.md",
]

SOURCE_DIRS = ["python/astra", "tools", "tests", "service"]
COMMENT_MARKER = re.compile(r"#.*\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[A-Za-z0-9.-]+)?$")

ALL_GATES = ("tests", "e2e", "rollback", "memory-eval", "registry", "docs", "changelog")


class Gate:
    """Runs one release gate; ``failures`` blocks passage, ``notes`` report on ok."""

    def __init__(self, name: str, run: callable) -> None:
        self.name = name
        self._run = run
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self) -> bool:
        return not self.failures

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def execute(self, env: dict) -> None:
        try:
            self._run(self, env)
        except subprocess.CalledProcessError as e:
            self.fail(f"command failed [{e.returncode}]")
        except Exception as e:  # noqa: BLE001 - a gate failure must be reported, not crash the run
            self.fail(f"gate raised: {e}")


def _sh(env: dict, *args: str) -> None:
    """Run a subcommand; raise CalledProcessError on non-zero exit."""
    subprocess.run(list(args), cwd=str(ROOT), env=env, check=True)


# --------------------------------------------------------------------------- gates

def gate_tests(g: Gate, env: dict) -> None:
    _sh(env, sys.executable, "-m", "pytest", "tests/", "-q")
    g.notes.append("full suite green")


def gate_e2e(g: Gate, env: dict) -> None:
    _sh(env, sys.executable, "tools/e2e.py", "--steps", "150")
    g.notes.append("full-stack walk promoted/rejected with an audit row")


def gate_rollback(g: Gate, env: dict) -> None:
    _sh(env, sys.executable, "tools/self_improve.py", "--rollback-drill")
    g.notes.append("rollback drill restored the prior active sha")


def gate_memory_eval(g: Gate, env: dict) -> None:
    _sh(env, sys.executable, "tools/memory_eval.py", "--enforce")
    g.notes.append("memory retrieval thresholds enforced")


def gate_registry(g: Gate, env: dict) -> None:
    from astra.registry import ModelRegistry

    reg_path = ROOT / "checkpoints" / "registry.json"
    if not reg_path.exists():
        g.fail("no registry at checkpoints/registry.json")
        return
    model = ModelRegistry(str(reg_path))
    names = model.list_names()
    if not names:
        g.fail("registry is empty — nothing verified")
        return
    for name in names:
        rec, ok = model.verify(name)
        if rec is None:
            g.fail(f"{name}: no entry")
        elif not ok:
            g.fail(f"{name}: checksum MISMATCH at {rec.path} ({rec.sha256[:12]})")
    g.notes.append(f"{len(names)} registered artifact(s) checksum-verified")


def gate_docs(g: Gate, env: dict) -> None:
    missing = [p for p in REQUIRED_DOCS if not (ROOT / p).exists()]
    if missing:
        g.fail(f"required docs missing: {', '.join(missing)}")

    broken: list[str] = []
    for md in sorted((ROOT / "docs").rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            candidate = target.strip().strip("`")
            if not candidate or candidate.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path = candidate.split()[0]
            if not path.endswith(".md"):
                continue
            res = (md.parent / path).resolve()
            if not res.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {candidate}")
    if broken:
        g.fail(f"{len(broken)} broken relative .md link(s), e.g. {broken[0]}")

    markers: list[str] = []
    for d in SOURCE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if (ROOT / "tools" / "release.py").resolve() == py.resolve():
                continue  # this tool legitimately spells out the marker words
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if COMMENT_MARKER.search(line):
                    markers.append(f"{py.relative_to(ROOT)}:{i}")
    if markers:
        g.fail(f"release-gate markers in source comments: {', '.join(markers[:5])}")

    g.notes.append("required docs present, relative links resolve, no gate markers")


def gate_changelog(g: Gate, env: dict, version: str | None, require_notes: bool) -> None:
    changelog = ROOT / "docs" / "CHANGELOG.md"
    if not changelog.exists():
        g.fail("docs/CHANGELOG.md missing")
        return
    if version and f"## {version}" not in changelog.read_text(encoding="utf-8"):
        g.fail(f"docs/CHANGELOG.md has no '## {version}' heading")
    if require_notes:
        notes = ROOT / "docs" / "releases" / f"v{version}.md"
        if version and not notes.exists():
            g.fail(f"release notes missing: {notes.relative_to(ROOT)}")
    g.notes.append("changelog updated")


# -------------------------------------------------------------------------- driver

def build_env() -> dict:
    env = dict(__import__("os").environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    return env


def run_check(version: str | None, skip: set[str], require_notes: bool) -> list[Gate]:
    env = build_env()
    skipped = set(skip)
    gates: list[Gate] = []

    def add(name: str, run: callable, *extra: object) -> None:
        if name not in ALL_GATES or name in skipped:
            return

        def wrapped(g: Gate, _env: dict, fn=run) -> None:
            fn(g, _env)

        gate = Gate(name, wrapped)
        gate.execute(env)
        gates.append(gate)

    add("tests", gate_tests)
    add("e2e", gate_e2e)
    add("rollback", gate_rollback)
    add("memory-eval", gate_memory_eval)
    add("registry", gate_registry)
    add("docs", gate_docs)
    if "changelog" not in skipped:
        gate = Gate("changelog", lambda g, _env: gate_changelog(g, _env, version, require_notes))
        gate.execute(env)
        gates.append(gate)
    return gates


def _print_report(gates: list[Gate], version: str | None) -> bool:
    print(f"\n=== Astra release gates {f'for v{version}' if version else ''} ===")
    all_ok = True
    for gate in gates:
        if gate.ok():
            print(f"[release {gate.name:11}] PASS  {'; '.join(gate.notes)}")
        else:
            all_ok = False
            print(f"[release {gate.name:11}] FAIL")
            for m in gate.failures:
                print(f"            - {m}")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="run every release gate; exit 1 on any failure")
    check.add_argument("--version", default=None, help="target semver; ``changelog`` gate checks its heading")
    check.add_argument("--skip", default="", help=f"comma-separated gates to skip: {', '.join(ALL_GATES)}")

    tag = sub.add_parser("tag", help="run gates then create annotated tag v<version> at HEAD")
    tag.add_argument("--version", required=True)
    tag.add_argument("--skip", default="")
    tag.add_argument("--no-check", action="store_true", help="skip gates; tag anyway (dangerous)")

    args = ap.parse_args()
    version = getattr(args, "version", None) or None
    if args.cmd == "tag" and version and not SEMVER_RE.match(version):
        raise SystemExit(f"invalid version {version!r}; want X.Y.Z[-suffix]")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(ALL_GATES)
    if unknown:
        raise SystemExit(f"unknown gates in --skip: {sorted(unknown)}")

    require_notes = args.cmd == "tag"
    ok = True
    if args.cmd == "tag" and args.no_check:
        print("[release ] --no-check: tagging without gates (explicit override)")
    else:
        gates = run_check(version, skip, require_notes=require_notes)
        ok = _print_report(gates, version)
        print(f"[release ] overall {'PASS' if ok else 'FAIL'}")

    if args.cmd == "tag":
        if not ok:
            raise SystemExit("release gates FAILED — no tag created")
        tag = ROOT / "docs" / "releases" / f"v{version}.md"
        if not tag.exists():
            raise SystemExit(f"release notes missing: {tag} (write them before tagging)")
        subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"Astra {version} release"],
                       cwd=str(ROOT), check=True)
        print(f"[release ] tagged v{version} at HEAD")
        print(f"[release ] push with: git push origin v{version}")
    else:
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()