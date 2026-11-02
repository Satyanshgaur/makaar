"""Engine interface: every recognizer (specialist or general model) implements this."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

TASKS = ("text", "formula", "table", "handwriting")


@dataclass
class Region:
    image: bytes  # encoded crop (PNG/JPEG)
    bbox: tuple[float, float, float, float]
    type: str


@dataclass
class Recognition:
    text: str
    token_probs: list[float] = field(default_factory=list)
    engine: str = ""

    @property
    def confidence(self) -> float:
        """Mean token probability; 1.0 when the engine reports none."""
        return sum(self.token_probs) / len(self.token_probs) if self.token_probs else 1.0


class Engine(Protocol):
    name: str

    def recognize(self, region: Region, task: str, draft: str | None = None) -> Recognition: ...
