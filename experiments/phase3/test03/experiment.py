from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.v2.agents import BinaryAgent, TernaryAgent
from core.v2.agents.base import BaseAgent
from core.v2.constants import State
from core.v2.evidence import EvidenceMemory


class QuaternaryPersistentAgent(BaseAgent):
    name = "QUATERNARY_PERSISTENT"

    def state(self, question: int, candidate: int) -> State:
        evidence = self.evidence_for(question, candidate)

        if evidence.total == 0:
            return State.UNKNOWN

        if evidence.support > 0 and evidence.conflict > 0:
            return State.CONFLICT

        if evidence.support > 0:
            return State.RIGHT

        return State.WRONG

    def decide(self, question: int, candidates: tuple[int, ...]) -> int | None:
        right = [
            c for c in candidates
            if self.state(question, c) == State.RIGHT
        ]
        return right[0] if len(right) == 1 else None


class QuaternaryDominantAgent(BaseAgent):
    name = "QUATERNARY_DOMINANT"

    def state(self, question: int, candidate: int) -> State:
        evidence = self.evidence_for(question, candidate)

        if evidence.total == 0:
            return State.UNKNOWN

        if evidence.support > evidence.conflict:
            return State.RIGHT

        if evidence.conflict > evidence.support:
            return State.WRONG

        return State.CONFLICT

    def decide(self, question: int, candidates: tuple[int, ...]) -> int | None:
        right = [
            c for c in candidates
            if self.state(question, c) == State.RIGHT
        ]
        return right[0] if len(right) == 1 else None


SCENARIOS = {
    "CLEAN": [True, True, True, True, True],
    "SINGLE_CONFLICT": [True, True, True, True, False],
    "STRONG_SUPPORT_AFTER_CONFLICT": [
        True, True, True, True, False, True, True, True
    ],
    "STRONG_CONFLICT_AFTER_SUPPORT": [
        True, True, True, True, False, False, False, False
    ],
    "BALANCED_CONFLICT": [
        True, True, True, False, False, False
    ],
    "CONFLICT_THEN_RECOVERY": [
        True, True, False, False, True, True, True, True
    ],
    "ALTERNATING": [
        True, False, True, False, True, False, True, False
    ],
}


AGENTS = (
    BinaryAgent,
    TernaryAgent,
    QuaternaryPersistentAgent,
    QuaternaryDominantAgent,
)


def run_scenario(name: str, observations: list[bool]) -> list[dict]:
    question = 0
    candidate = 1
    candidates = (0, 1)
    results = []

    for agent_cls in AGENTS:
        memory = EvidenceMemory()
        agent = agent_cls(memory)

        for step, is_support in enumerate(observations, 1):
            agent.observe(question, candidate, is_support)

            evidence = agent.evidence_for(question, candidate)
            state = agent.state(question, candidate)

            results.append({
                "test": "phase3_test03",
                "scenario": name,
                "agent": agent_cls.name,
                "step": step,
                "observation": "SUPPORT" if is_support else "CONFLICT",
                "support": evidence.support,
                "conflict": evidence.conflict,
                "state": state.name,
            })

    return results


def main() -> None:
    results = []

    for name, observations in SCENARIOS.items():
        results.extend(run_scenario(name, observations))

    output = ROOT / "results" / "phase3" / "test03" / "results.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("TEST03: completed")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Agents: {len(AGENTS)}")
    print(f"Total observations: {sum(len(x) for x in SCENARIOS.values())}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
