from __future__ import annotations

import random
from collections import defaultdict

from .constants import UNKNOWN, WRONG, RIGHT


class BinaryAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.memory: dict[tuple[int, int], bool] = {}

    def predict(
        self,
        x: int,
        candidates: tuple[int, ...],
    ) -> int:

        known = [
            c
            for c in candidates
            if self.memory.get((x, c)) is True
        ]

        if known:
            return known[0]

        unknown = [
            c
            for c in candidates
            if (x, c) not in self.memory
        ]

        if unknown:
            return self.rng.choice(unknown)

        return self.rng.choice(list(candidates))

    def learn(
        self,
        x: int,
        prediction: int,
        feedback: int,
    ) -> None:

        if feedback not in (WRONG, UNKNOWN, RIGHT):
            raise ValueError(f"Invalid feedback: {feedback}")

        self.memory[(x, prediction)] = (
            feedback == RIGHT
        )


class TernaryAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.memory = defaultdict(int)

    def state(self, x: int, candidate: int) -> int:
        return self.memory[(x, candidate)]

    def predict(
        self,
        x: int,
        candidates: tuple[int, ...],
    ) -> int:

        positives = [
            c
            for c in candidates
            if self.state(x, c) == RIGHT
        ]

        if positives:
            return positives[0]

        unknown = [
            c
            for c in candidates
            if self.state(x, c) == UNKNOWN
        ]

        if unknown:
            return self.rng.choice(unknown)

        return self.rng.choice(list(candidates))

    def learn(
        self,
        x: int,
        prediction: int,
        feedback: int,
    ) -> None:

        if feedback not in (WRONG, UNKNOWN, RIGHT):
            raise ValueError(f"Invalid feedback: {feedback}")

        self.memory[(x, prediction)] = feedback