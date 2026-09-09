"""Memory → working-context injection (docs/MEMORY.md § 9).

Assembles retrieved memories into a ``<|memory|>`` block bounded by a token
budget so retrieval can never outsize the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astra.memory.retrieval import RetrievalHit

MEMORY_OPEN = "<|memory|>"
MEMORY_CLOSE = "<|/memory|>"
ITEM_OPEN = "<|mem|>"
ITEM_CLOSE = "<|/mem|>"


@dataclass
class MemoryBlock:
    text: str
    tokens: list[Any]
    included: list[RetrievalHit] = field(default_factory=list)
    dropped: list[RetrievalHit] = field(default_factory=list)

    def prepend(self, prompt: str) -> str:
        if not self.text:
            return prompt
        return f"{self.text}\n{prompt}"


def build_memory_block(
    hits: list[RetrievalHit],
    tokenizer: Any,
    budget_tokens: int = 256,
    kind_labels: bool = False,
) -> MemoryBlock:
    """Build the ``<|memory|>`` zone, top-ranked records fit the token budget.

    Records that don't fit are reported on the block as ``dropped``; when the
    top record alone exceeds the budget it is still included (never empty).
    """
    if not hits:
        return MemoryBlock(text="", tokens=[])

    count = 0
    total = 0
    for hit in hits:
        if hit.tokens == 0:
            hit.tokens = len(tokenizer.encode(hit.record.content))
        if count > 0 and total + hit.tokens > budget_tokens:
            break
        total += hit.tokens
        count += 1

    parts = [MEMORY_OPEN]
    for h in hits[:count]:
        label = f" kind={h.record.kind}" if kind_labels else ""
        parts.append(f"{ITEM_OPEN} id={h.record.id}{label}\n{h.record.content}\n{ITEM_CLOSE}")
    parts.append(MEMORY_CLOSE)
    text = "\n".join(parts)
    tokens = tokenizer.encode(text) if tokenizer else []
    return MemoryBlock(text=text, tokens=tokens, included=hits[:count], dropped=hits[count:])