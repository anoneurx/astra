"""Tool registry: name to callable, with schema export and safe dispatch."""

from __future__ import annotations

import json

from astra.tools.base import ToolError, ToolResult, ToolSpec, validate_args


class ToolRegistry:
    """Holds the tools one session may call.

    call() never raises. A tool that fails is a fact the model needs, and the
    loop's job is to hand it back as one more turn rather than to abort a
    conversation that may recover on the next attempt.
    """

    def __init__(self, specs: tuple[ToolSpec, ...] | list[ToolSpec] = ()):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ToolError(f"tool {spec.name} is already registered")
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolError(
                f"there is no tool named {name!r}; the available tools are "
                f"{', '.join(sorted(self._specs)) or '(none)'}"
            ) from None

    def json_schemas(self) -> list[dict]:
        return [spec.json_schema() for spec in self.specs]

    def prompt_block(self) -> str:
        """The tool list, in the one-line-per-tool form used in the prompt.

        Kept as prose rather than a JSON blob because the model has to continue
        this text, and a nested schema is harder to imitate than a signature and
        a sentence. The authoritative schemas are still available from
        json_schemas() for a caller that wants to send them to a real API.
        """
        lines = [
            "You can look at files in the project. To use a tool, write one line "
            "starting with tool, then the tool name, then its arguments as JSON:",
            "",
        ]
        for spec in self.specs:
            lines.append(f"tool {spec.signature()} - {spec.description}")
        lines += [
            "",
            "Only one tool call per reply, and arguments must be valid JSON. "
            f"You will get a result line beginning with result. "
            f"The tools are: {', '.join(sorted(self._specs))}.",
        ]
        return "\n".join(lines)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        try:
            spec = self.get(name)
            kwargs = validate_args(spec, args if args is not None else {})
            out = spec.fn(**kwargs)
        except ToolError as exc:
            return ToolResult(ok=False, content=str(exc))
        except TypeError as exc:
            # A signature that no longer matches the schema. Surfacing the
            # TypeError keeps the bug visible instead of dressing it as a
            # refusal the model would try to work around.
            return ToolResult(ok=False, content=f"tool {name} is misconfigured: {exc}")
        if not isinstance(out, ToolResult):
            return ToolResult(
                ok=True,
                content=out if isinstance(out, str) else json.dumps(out, default=str),
            )
        return out
