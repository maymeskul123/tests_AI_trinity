from __future__ import annotations

from abc import ABC, abstractmethod

from ..constants import State
from ..evidence import Evidence, EvidenceMemory


class BaseAgent(ABC):
    name = "BASE"

    def __init__(self, evidence: EvidenceMemory) -> None:
        self.evidence = evidence

    def observe(
        self,
        question: int,
        candidate: int,
        is_support: bool,
    ) -> None:
        self.evidence.observe(
            question,
            candidate,
            is_support,
        )

    def evidence_for(
        self,
        question: int,
        candidate: int,
    ) -> Evidence:
        return self.evidence.get(question, candidate)

    @abstractmethod
    def state(
        self,
        question: int,
        candidate: int,
    ) -> State:
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        question: int,
        candidates: tuple[int, ...],
    ) -> int | None:
        raise NotImplementedError
