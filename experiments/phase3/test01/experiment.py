from __future__ import annotations

import csv
import os

from core.v2.agents import (
    BinaryAgent,
    TernaryAgent,
    QuaternaryAgent,
    DynamicAgent,
)
from core.v2.constants import State
from core.v2.evidence import EvidenceMemory


RESULT_PATH = "results/phase3/test01/results.csv"

QUESTION = 0
CANDIDATE = 0

AGENTS = (
    BinaryAgent,
    TernaryAgent,
    QuaternaryAgent,
    DynamicAgent,
)


SCENARIOS = {
    "EMPTY": [],
    "SUPPORT": [True],
    "CONFLICT": [False],
    "SUPPORT_CONFLICT": [True, False],
    "RECOVERY": [True, False, True],
    "DOUBLE_CONFLICT": [True, False, False],
    "MAJORITY_SUPPORT": [True, False, True, True],
}


EXPECTED = {
    "EMPTY": {
        "BINARY": State.UNKNOWN,
        "TERNARY": State.UNKNOWN,
        "QUATERNARY": State.UNKNOWN,
        "DYNAMIC": State.UNKNOWN,
    },
    "SUPPORT": {
        "BINARY": State.RIGHT,
        "TERNARY": State.RIGHT,
        "QUATERNARY": State.RIGHT,
        "DYNAMIC": State.RIGHT,
    },
    "CONFLICT": {
        "BINARY": State.WRONG,
        "TERNARY": State.WRONG,
        "QUATERNARY": State.WRONG,
        "DYNAMIC": State.WRONG,
    },
    "SUPPORT_CONFLICT": {
        "BINARY": State.WRONG,
        "TERNARY": State.UNKNOWN,
        "QUATERNARY": State.CONFLICT,
        "DYNAMIC": State.CONFLICT,
    },
    "RECOVERY": {
        "BINARY": State.RIGHT,
        "TERNARY": State.RIGHT,
        "QUATERNARY": State.CONFLICT,
        "DYNAMIC": State.CONFLICT,
    },
    "DOUBLE_CONFLICT": {
        "BINARY": State.WRONG,
        "TERNARY": State.WRONG,
        "QUATERNARY": State.CONFLICT,
        "DYNAMIC": State.CONFLICT,
    },
    "MAJORITY_SUPPORT": {
        "BINARY": State.RIGHT,
        "TERNARY": State.RIGHT,
        "QUATERNARY": State.CONFLICT,
        "DYNAMIC": State.CONFLICT,
    },
}


def run() -> None:
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    rows = []

    for scenario, observations in SCENARIOS.items():
        for AgentClass in AGENTS:
            memory = EvidenceMemory()
            agent = AgentClass(memory)

            trajectory = []

            for is_support in observations:
                agent.observe(
                    QUESTION,
                    CANDIDATE,
                    is_support,
                )

                trajectory.append(
                    agent.state(
                        QUESTION,
                        CANDIDATE,
                    ).name
                )

            evidence = memory.get(
                QUESTION,
                CANDIDATE,
            )

            final_state = agent.state(
                QUESTION,
                CANDIDATE,
            )

            expected = EXPECTED[scenario][agent.name]
            passed = final_state == expected

            rows.append(
                {
                    "scenario": scenario,
                    "agent": agent.name,
                    "support": evidence.support,
                    "conflict": evidence.conflict,
                    "final_state": final_state.name,
                    "expected_state": expected.name,
                    "trajectory": "->".join(trajectory),
                    "passed": int(passed),
                }
            )

    with open(
        RESULT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(row["passed"] for row in rows)

    print(f"TEST01: {passed}/{total} checks passed")

    if passed != total:
        print("FAILED")
        for row in rows:
            if row["passed"] == "0":
                print(
                    row["scenario"],
                    row["agent"],
                    row["final_state"],
                    "expected",
                    row["expected_state"],
                )
        raise SystemExit(1)

    print("PASSED")


if __name__ == "__main__":
    run()
