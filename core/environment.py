from dataclasses import dataclass
from typing import Callable

from .constants import WRONG, RIGHT


@dataclass(frozen=True)
class Task:
    x: int
    candidates: tuple[int, ...]
    answer: int


class HiddenRuleEnvironment:
    def __init__(self, rule: Callable[[int], int], candidates=range(10)):
        self.rule = rule
        self.candidates = tuple(candidates)

    def make_task(self, x: int) -> Task:
        answer = self.rule(x)

        if answer not in self.candidates:
            raise ValueError(
                f"Rule returned {answer}, but it is not in candidates"
            )

        return Task(
            x=x,
            candidates=self.candidates,
            answer=answer,
        )

    def evaluate(self, task: Task, prediction: int) -> int:
        if prediction == task.answer:
            return RIGHT

        return WRONG