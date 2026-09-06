from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Evidence:
    support: int = 0
    conflict: int = 0

    def observe(self, is_support: bool) -> None:
        if is_support:
            self.support += 1
        else:
            self.conflict += 1

    @property
    def total(self) -> int:
        return self.support + self.conflict


class EvidenceMemory:
    """
    Shared objective evidence.

    The memory stores observations only.
    It does not decide whether a hypothesis is RIGHT, WRONG,
    UNKNOWN, or CONFLICT.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], Evidence] = defaultdict(Evidence)

    def observe(
        self,
        question: int,
        candidate: int,
        is_support: bool,
    ) -> None:
        self._data[(question, candidate)].observe(is_support)

    def get(self, question: int, candidate: int) -> Evidence:
        return self._data[(question, candidate)]

    def snapshot(self, question: int, candidate: int) -> Evidence:
        evidence = self.get(question, candidate)
        return Evidence(
            support=evidence.support,
            conflict=evidence.conflict,
        )
