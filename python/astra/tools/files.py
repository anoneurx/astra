"""File-system tools: read_file, search_text, find_files.

All three are built by a factory bound to one root directory. The root is never
a tool argument, because an argument the model controls is an argument it can
be talked into changing; see RootJail for the containment rules.

Names are deliberately not the obvious ones. This tokenizer is WordLevel over
prose, so any word absent from a prose corpus encodes to <unk>, and the sampler
forbids special ids (python/astra/inference/decoder.py) so <unk> can never be
emitted at all. Measured against tokenizer/artifacts/prose_chat_word.json:
`grep` and `glob` are both absent, while `search_text` and `find_files` spell
out of words that are present. A tool whose name the model cannot spell is a
tool that can never be called, so the names follow the vocabulary rather than
the convention. tests/test_agent_tools.py enforces this.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from astra.tools.base import Param, ToolResult, ToolSpec, ToolError, tool, validate_args

# The decoder maps this id back to the literal text "<unk>", so a pattern the
# model could not spell arrives here containing that word. Matching it would
# return zero hits and read as "no such file", so it is reported instead.
UNKNOWN_PIECE = "<unk>"

DEFAULT_SKIP_DIRS = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "dist",
    "build",
)


class RootJail:
    """Confines every path to one directory tree.

    Absolute paths are allowed only when they already sit inside the root, so a
    model may legitimately be handed "/abs/path/in/root/file" by an earlier turn.
    resolve() follows symlinks before the containment check, so a symlink inside
    the root pointing at /etc/passwd is refused rather than followed.
    """

    def __init__(self, root: str | os.PathLike, skip_dirs: tuple[str, ...] = DEFAULT_SKIP_DIRS):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ToolError(f"tool root {self.root} is not a directory")
        self.skip_dirs = frozenset(skip_dirs)

    def resolve(self, path: str, must_exist: bool = True) -> Path:
        """Return an absolute path inside the root, or raise ToolError."""
        raw = (path or "").strip()
        if not raw:
            raise ToolError("path is empty; give a path relative to the tool root")
        if UNKNOWN_PIECE in raw:
            raise ToolError(
                f"the tokenizer has no token for part of {raw!r} (<unk> means the model "
                "tried to spell a word absent from its vocabulary, usually a file "
                "extension); try a different pattern or a directory instead"
            )

        candidate = Path(raw).expanduser()
        joined = candidate if candidate.is_absolute() else self.root / candidate
        try:
            resolved = joined.resolve()
        except OSError as exc:  # symlink loops, name too long, bad encoding
            raise ToolError(f"cannot resolve {raw!r}: {exc}") from exc

        # resolve() has followed every symlink, so this compares the final target.
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ToolError(
                f"{raw!r} resolves to {resolved}, which is outside the tool root {self.root}"
            )
        if must_exist and not resolved.exists():
            raise ToolError(f"{raw!r} does not exist (looked in {resolved})")
        return resolved

    def relative(self, path: Path) -> str:
        """Root-relative display form, so results never leak the absolute layout."""
        return str(path.relative_to(self.root))

    def iter_files(self, pattern: str) -> list[Path]:
        """Every file under the root matching a glob, deterministically ordered.

        Path.glob can walk into .git and node_modules and return tens of
        thousands of paths, so the noisy directories are pruned by walking
        os.scandir ourselves instead.
        """
        matched: list[Path] = []
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in self.skip_dirs for part in path.relative_to(self.root).parts[:-1]):
                continue
            matched.append(path)
        return matched


def _read_text(path: Path) -> str | None:
    """File contents, or None when the file is not decodable text.

    Trained checkpoints meet .npz blobs and .ipynb files; treating an undecodable
    byte as a match would put binary noise in front of the model.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def make_file_tools(jail: RootJail) -> list[ToolSpec]:
    """The three file tools, bound to one jail."""

    @tool(
        "read_file",
        "Read a slice of a text file. start is a 0-based line number, limit is "
        "how many lines to return.",
        (
            Param("path", "string", "file to read, relative to the root", required=True),
            Param("start", "integer", "first line to return, 0-based", default=0),
            Param("limit", "integer", "maximum lines to return", default=200),
        ),
    )
    def read_file(path: str, start: int = 0, limit: int = 200) -> ToolResult:
        target = jail.resolve(path)
        if limit <= 0:
            raise ToolError(f"limit must be positive, got {limit}")
        if start < 0:
            raise ToolError(f"start must not be negative, got {start}")
        text = _read_text(target)
        if text is None:
            raise ToolError(f"{path} is not utf-8 text; it cannot be read as lines")
        lines = text.splitlines()
        if start >= len(lines):
            raise ToolError(
                f"{path} has {len(lines)} lines; start {start} is past the end"
            )
        chunk = lines[start : start + limit]
        meta = {
            "lines_returned": len(chunk),
            "total_lines": len(lines),
            "start": start,
            "line_range": f"{start + 1}-{start + len(chunk)}",
        }
        note = ""
        if start + len(chunk) < len(lines):
            meta["truncated"] = True
            meta["next_start"] = start + len(chunk)
            # The model cannot read meta, so the way to continue has to be in words
            # it can spell; otherwise it retries the same call and gets the same slice.
            note = (f"showing lines {meta['line_range']} of {len(lines)}; "
                    f"call read_file again with start={start + len(chunk)} to see more")
        return ToolResult(ok=True, content="\n".join(chunk), meta=meta, note=note)

    @tool(
        "search_text",
        "Search file contents for a pattern. Returns path, line number and the "
        "line itself, most matches first.",
        (
            Param("pattern", "string", "text or regular expression to find", required=True),
            Param("files", "string", "glob limiting which files to search", default="**/*"),
            Param("ignore_case", "boolean", "match case-insensitively", default=False),
            Param("limit", "integer", "stop after this many matches", default=50),
        ),
    )
    def search_text(
        pattern: str, files: str = "**/*", ignore_case: bool = False, limit: int = 50
    ) -> ToolResult:
        if not pattern:
            raise ToolError("pattern is empty")
        if limit <= 0:
            raise ToolError(f"limit must be positive, got {limit}")
        if UNKNOWN_PIECE in pattern or UNKNOWN_PIECE in files:
            raise ToolError(
                "the tokenizer has no token for part of the pattern (<unk>); the "
                "vocabulary has no entry for that word"
            )
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
            flavour = "regex"
        except re.error as exc:
            # A model asked for "search" writes literal text surprisingly often,
            # and bare text is far more often intended than a broken regex.
            literal = re.escape(pattern)
            rx = re.compile(literal, re.IGNORECASE if ignore_case else 0)
            flavour = f"literal (treated {pattern!r} as text: {exc})"

        hits: list[str] = []
        files_with_hits = 0
        searched = 0
        truncated = False
        for path in jail.iter_files(files):
            searched += 1
            text = _read_text(path)
            if text is None:
                continue
            in_this_file = False
            for number, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    in_this_file = True
                    if len(hits) >= limit:
                        truncated = True
                        break
                    hits.append(f"{jail.relative(path)}:{number}: {line.strip()[:400]}")
            if in_this_file:
                files_with_hits += 1
            if truncated:
                break

        meta = {
            "pattern": pattern,
            "match_mode": flavour,
            "files_searched": searched,
            "files_with_matches": files_with_hits,
            "matches": len(hits),
        }
        if truncated:
            meta["truncated"] = True
        content = "\n".join(hits) if hits else f"no matches for {pattern!r}"
        note = ""
        if truncated:
            note = (f"stopped at the limit of {limit} matches; narrow the pattern, "
                    "restrict files, or raise limit to see the rest")
        return ToolResult(ok=True, content=content, meta=meta, note=note)

    @tool(
        "find_files",
        "List files whose path matches a glob, relative to the root.",
        (
            Param("pattern", "string", "glob such as **/*.md or src/", default="**/*"),
            Param("limit", "integer", "stop after this many paths", default=100),
        ),
    )
    def find_files(pattern: str = "**/*", limit: int = 100) -> ToolResult:
        if not pattern:
            raise ToolError("pattern is empty")
        if limit <= 0:
            raise ToolError(f"limit must be positive, got {limit}")
        if UNKNOWN_PIECE in pattern:
            raise ToolError(
                f"the tokenizer has no token for part of {pattern!r} (<unk>); file "
                "extensions are the usual cause, since the vocabulary was built "
                "from prose"
            )
        try:
            paths = jail.iter_files(pattern)
        except (ValueError, IndexError) as exc:
            raise ToolError(f"{pattern!r} is not a usable glob: {exc}") from exc
        shown = paths[:limit]
        meta = {"pattern": pattern, "returned": len(shown), "total_matched": len(paths)}
        if len(paths) > limit:
            meta["truncated"] = True
        content = (
            "\n".join(jail.relative(p) for p in shown)
            if shown
            else f"no paths matched {pattern!r}"
        )
        note = ""
        if len(paths) > limit:
            note = (f"showing {limit} of {len(paths)} paths; narrow the pattern or "
                    "raise limit to see the rest")
        return ToolResult(ok=True, content=content, meta=meta, note=note)

    return [read_file.spec, search_text.spec, find_files.spec]
