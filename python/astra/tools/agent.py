"""The agent loop: generate, act on any tool call, feed the result back, repeat.

Generation is injected as a plain callable taking a prompt string and returning
the model's text. That keeps the loop testable without a checkpoint, a GPU or a
tokenizer, and it means swapping the sampler for a hosted API is a one-line
change at the call site rather than a rewrite here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from astra.tools.base import ToolError
from astra.tools.registry import ToolRegistry

TOOL_LINE = re.compile(r"^[ \t]*tool[ \t]+([a-z_][a-z0-9_]*)[ \t]*(\{.*)?$", re.MULTILINE)
RESULT_PREFIX = "result "


@dataclass
class ToolCall:
    name: str
    args: dict
    raw: str
    error: str | None = None


@dataclass
class AgentTurn:
    """What one exchange with the model cost, for logging and for tests."""

    prompt: str
    text: str
    steps: int
    calls: list[ToolCall] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    exhausted: bool = False


def extract_calls(text: str, registry: ToolRegistry) -> list[ToolCall]:
    """Every tool call in a model reply, in the order written.

    raw_decode rather than json.loads so a model that trails off after the
    closing brace still has its call honoured, and an unterminated one produces a
    message the loop can hand back instead of an exception that ends the turn.
    """
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    for match in TOOL_LINE.finditer(text or ""):
        name, raw_args = match.group(1), match.group(2)
        if not raw_args:
            calls.append(ToolCall(name=name, args={}, raw=match.group(0), error="no arguments given"))
            continue
        try:
            args, _end = decoder.raw_decode(raw_args.lstrip())
        except json.JSONDecodeError as exc:
            calls.append(
                ToolCall(
                    name=name,
                    args={},
                    raw=match.group(0),
                    error=f"arguments for {name} are not valid JSON: {exc.msg} at position {exc.pos}",
                )
            )
            continue
        if not isinstance(args, dict):
            calls.append(
                ToolCall(name=name, args={}, raw=match.group(0), error="arguments must be a JSON object")
            )
            continue
        calls.append(ToolCall(name=name, args=args, raw=match.group(0)))
    # A name the registry does not hold is still returned, so the loop can reply
    # with the list of real tools instead of silently doing nothing.
    for call in calls:
        if call.error is None and call.name not in registry:
            call.error = (
                f"there is no tool named {call.name!r}; the available tools are "
                f"{', '.join(s.name for s in registry.specs) or '(none)'}"
            )
    return calls


class AgentSession:
    """A model that can call tools, over any generate(prompt) -> text callable."""

    def __init__(
        self,
        registry: ToolRegistry,
        generate: Callable[[str], str],
        *,
        system_block: str | None = None,
        max_steps: int = 4,
        max_new: int = 256,
        max_result_chars: int = 4000,
    ):
        self.registry = registry
        self.generate = generate
        self.system_block = system_block if system_block is not None else registry.prompt_block()
        self.max_steps = max_steps
        self.max_new = max_new
        self.max_result_chars = max_result_chars
        self.history: list[str] = []

    def _render(self, turns: list[tuple[str, str]]) -> str:
        parts = [self.system_block, "", ""] if self.system_block else []
        for user, astra in turns:
            parts.append(f"You: {user}")
            parts.append(f"Astra: {astra}".rstrip())
        return "\n".join(parts).rstrip()

    def ask(self, prompt: str) -> AgentTurn:
        """Run one exchange to completion: at most max_steps tool round trips."""
        transcript: list[str] = [self.system_block, "", f"You: {prompt}", "Astra:"]
        call_log: list[ToolCall] = []
        result_log: list[str] = []
        text = ""
        exhausted = False

        for _step in range(1, self.max_steps + 1):
            text = self.generate("\n".join(transcript))
            calls = extract_calls(text, self.registry)
            if not calls:
                break
            transcript.append(text.strip())
            for call in calls:
                call_log.append(call)
                if call.error:
                    payload = json.dumps({"error": call.error})
                else:
                    payload = self.registry.call(call.name, call.args).to_json(
                        max_chars=self.max_result_chars
                    )
                result_log.append(payload)
                transcript.append(f"{RESULT_PREFIX}{payload}")
            transcript.append("Astra:")

        # The for/else runs only when the loop ran out of steps with a call still
        # pending, which is the one case where the reply the caller gets is a
        # truncated instruction rather than an answer.
        else:
            exhausted = True

        turn = AgentTurn(
            prompt=prompt,
            text=text,
            steps=len(call_log),
            calls=call_log,
            results=result_log,
            exhausted=exhausted,
        )
        self.history.append(f"You: {prompt}\nAstra: {text}")
        return turn
