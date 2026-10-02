"""Tool-call primitives.

A tool is an ordinary function plus a JSON-schema description of its
arguments. This module holds the parts that are not specific to any one tool:
the result envelope, an argument validator, and the decorator that turns a
function into a spec.

The validator exists because the arguments arrive from a language model rather
than from a programmer. A hallucinated key, or a string where an integer was
expected, has to come back as a sentence the model can read and correct rather
than as a TypeError from three frames down.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SCHEMA_TYPES = ("string", "integer", "number", "boolean")


class ToolError(Exception):
    """A tool refused or failed. The message is written to be shown to a model."""


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None

    def json_schema(self) -> dict:
        return {"type": self.type, "description": self.description}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[Param, ...]
    fn: Callable[..., dict]

    def json_schema(self) -> dict:
        """OpenAI-style function schema, so the registry can be dumped verbatim."""
        properties = {p.name: p.json_schema() for p in self.params}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }

    def signature(self) -> str:
        """One line naming each argument and its type, for the prompt block."""
        parts = []
        for p in self.params:
            bits = p.type
            if not p.required:
                bits += f"={p.default!r}" if p.default is not None else "=optional"
            parts.append(f"{p.name}: {bits}")
        return f"{self.name}({', '.join(parts)})" if parts else f"{self.name}()"


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: dict = field(default_factory=dict)
    note: str = ""

    def to_json(self, max_chars: int | None = None) -> str:
        """The model-facing envelope.

        Deliberately two keys, content and error, rather than the customary
        ok/content/meta. "ok" and "meta" are both absent from the prose-built
        vocabulary, so an envelope using them hands the model two <unk> tokens
        in the place of the part of the message that says what happened.
        Diagnostic detail belongs in note, which is appended as plain words, or
        in meta, which the loop prints for the human and never shows the model.
        """
        if self.ok:
            body: dict[str, Any] = {"content": self.content + (("\n" + self.note) if self.note else "")}
        else:
            body = {"error": self.content or self.note or "tool failed"}
        text = json.dumps(body, ensure_ascii=False, sort_keys=True)
        if max_chars is not None and len(text) > max_chars:
            text = text[: max_chars - 20] + '"... (truncated)"}'
        return text


def unknown_words(text: str, tokenizer) -> list[str]:
    """Words in text that this tokenizer cannot represent.

    The decoder maps the unknown id back to the literal string "<unk>", and the
    sampler refuses to draw special ids, so a word listed here can never appear
    in generated text. That makes it a property of the protocol rather than a
    quality-of-generation problem, and worth failing on in a test instead of
    discovering as a tool that is never called.
    """
    import re as _re

    tokens = _re.compile(r"[A-Za-z0-9']+| |\n|.")
    return sorted(
        {
            w
            for w in tokens.findall(text)
            if w not in (" ", "\n") and w.encode("utf-8") not in tokenizer.piece_to_id
        }
    )


def protocol_words(specs: tuple[ToolSpec, ...] | list[ToolSpec]) -> list[str]:
    """Every word the protocol requires a model to *write*.

    Tool names, argument names and the two markers. Type names and descriptions
    are deliberately excluded: those are prompt-side, and the model only has to
    read them. An argument key costs a <unk> on the way out of the model and the
    executor then rejects the call for an unknown argument, which is fatal; a
    type name costing a <unk> on the way in only costs the model a word of the
    signature line.
    """
    words = {"tool", "result"}
    for spec in specs:
        words.update(re.findall(r"[a-z]+", spec.name.lower()))
        for p in spec.params:
            words.update(re.findall(r"[a-z]+", p.name.lower()))
    return sorted(words)


def tool(
    name: str,
    description: str,
    params: tuple[Param, ...] = (),
) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Attach a ToolSpec to a function without changing what the function does."""

    def wrap(fn: Callable[..., dict]) -> Callable[..., dict]:
        spec = ToolSpec(name=name, description=description, params=params, fn=fn)
        fn.spec = spec  # type: ignore[attr-defined]
        return fn

    return wrap


def validate_args(spec: ToolSpec, args: dict) -> dict:
    """Coerce and check model-supplied arguments, or raise ToolError.

    Unknown keys are refused rather than ignored. Silently dropping a key the
    model believed it had set produces a call that ran against a default, and
    the model has no way to learn why the result came back different from what
    it asked for.
    """
    if not isinstance(args, dict):
        raise ToolError(f"arguments for {spec.name} must be a JSON object, got {type(args).__name__}")

    known = {p.name: p for p in spec.params}
    unknown = sorted(set(args) - set(known))
    if unknown:
        raise ToolError(
            f"{spec.name} has no argument(s) {', '.join(unknown)}; "
            f"valid arguments are {', '.join(sorted(known)) or '(none)'}"
        )

    out: dict[str, Any] = {}
    for p in spec.params:
        if p.name not in args:
            if p.required:
                raise ToolError(f"{spec.name} requires the argument {p.name} ({p.type})")
            out[p.name] = p.default
            continue
        out[p.name] = _coerce(spec, p, args[p.name])
    return out


def _coerce(spec: ToolSpec, p: Param, value: Any) -> Any:
    """Accept the loose types a model actually emits, or explain what was wanted.

    A model asked for an integer frequently answers with "200" or 200.0. Both
    are unambiguous, so they are converted rather than rejected; "many" is not.
    """
    if p.type == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        raise ToolError(f"{spec.name}.{p.name} must be a string, got {type(value).__name__}")

    if p.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ToolError(f"{spec.name}.{p.name} must be true or false, got {value!r}")

    if p.type == "integer":
        if isinstance(value, bool):
            raise ToolError(f"{spec.name}.{p.name} must be a whole number, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ToolError(f"{spec.name}.{p.name} must be a whole number, got {value!r}")

    if p.type == "number":
        if isinstance(value, bool):
            raise ToolError(f"{spec.name}.{p.name} must be a number, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ToolError(f"{spec.name}.{p.name} must be a number, got {value!r}")

    raise ToolError(f"{spec.name}.{p.name} has unsupported type {p.type}")
