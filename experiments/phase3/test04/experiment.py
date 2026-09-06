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


AGENTS = (
    BinaryAgent,
    TernaryAgent,
    QuaternaryPersistentAgent,
    QuaternaryDominantAgent,
)


# Each tuple is:
# (candidate, support, conflict)
#
# Candidate 0 is the true answer.
SCENARIOS = {
    "CLEAN": [
        (0, 5, 0),
        (1, 0, 5),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "WEAK_CONFLICT_TRUE": [
        (0, 5, 1),
        (1, 0, 5),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "STRONG_CONFLICT_TRUE": [
        (0, 7, 3),
        (1, 0, 5),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "BALANCED_TRUE": [
        (0, 5, 5),
        (1, 0, 5),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "TRUE_RECOVERY": [
        (0, 7, 2),
        (1, 0, 5),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "FALSE_CANDIDATE_WITH_CONFLICT": [
        (0, 5, 0),
        (1, 7, 3),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "TWO_STRONG_CANDIDATES": [
        (0, 7, 1),
        (1, 6, 0),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],

    "TRUE_MORE_EVIDENCE_BUT_CONFLICT": [
        (0, 10, 2),
        (1, 6, 0),
        (2, 0, 5),
        (3, 0, 5),
        (4, 0, 5),
    ],
}


def build_memory(observations: list[tuple[int, int, int]]) -> EvidenceMemory:
    memory = EvidenceMemory()

    for candidate, support, conflict in observations:
        for _ in range(support):
            memory.observe(0, candidate, True)

        for _ in range(conflict):
            memory.observe(0, candidate, False)

    return memory


def run_scenario(name: str, observations: list[tuple[int, int, int]]) -> list[dict]:
    candidates = (0, 1, 2, 3, 4)
    true_candidate = 0

    rows = []

    for agent_cls in AGENTS:
        memory = build_memory(observations)
        agent = agent_cls(memory)

        decision = agent.decide(0, candidates)

        states = {
            candidate: agent.state(0, candidate)
            for candidate in candidates
        }

        rows.append({
            "test": "phase3_test04",
            "scenario": name,
            "agent": agent_cls.name,
            "true_candidate": true_candidate,
            "decision": "" if decision is None else decision,
            "correct": int(decision == true_candidate),
            "decision_made": int(decision is not None),
            "unknown_count": sum(
                state == State.UNKNOWN
                for state in states.values()
            ),
            "conflict_count": sum(
                state == State.CONFLICT
                for state in states.values()
            ),
            "right_count": sum(
                state == State.RIGHT
                for state in states.values()
            ),
            "wrong_count": sum(
                state == State.WRONG
                for state in states.values()
            ),
            "true_support": memory.get(0, true_candidate).support,
            "true_conflict": memory.get(0, true_candidate).conflict,
            "true_state": states[true_candidate].name,
        })

    return rows


def main() -> None:
    results = []

    for name, observations in SCENARIOS.items():
        results.extend(run_scenario(name, observations))

    output = ROOT / "results" / "phase3" / "test04" / "results.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("TEST04: completed")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Agents: {len(AGENTS)}")
    print(f"Results: {output}")
    print()
    print("----- RESULTS -----")
    print()

    for row in results:
        print(
            f"{row['scenario']:35} "
            f"{row['agent']:25} "
            f"decision={str(row['decision']):5} "
            f"correct={row['correct']} "
            f"decision_made={row['decision_made']} "
            f"true_state={row['true_state']:8} "
            f"conflicts={row['conflict_count']}"
        )


if __name__ == "__main__":
    main()
