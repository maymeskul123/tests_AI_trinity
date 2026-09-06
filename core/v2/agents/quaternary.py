from __future__ import annotations

from .base import BaseAgent
from ..constants import State


class QuaternaryAgent(BaseAgent):
    name = "QUATERNARY"

    def state(
        self,
        question: int,
        candidate: int,
    ) -> State:
        evidence = self.evidence_for(question, candidate)

        if evidence.total == 0:
            return State.UNKNOWN

        if evidence.support > 0 and evidence.conflict > 0:
            return State.CONFLICT

        if evidence.support > 0:
            return State.RIGHT

        return State.WRONG

    def decide(
        self,
        question: int,
        candidates: tuple[int, ...],
    ) -> int | None:
        right = [
            candidate
            for candidate in candidates
            if self.state(question, candidate) == State.RIGHT
        ]

        if len(right) != 1:
            return None

        return right[0]
