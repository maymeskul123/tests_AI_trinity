from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from core.v2.evidence import EvidenceMemory


@dataclass(frozen=True)
class Evidence:
    support: int
    conflict: int

    @property
    def total(self) -> int:
        return self.support + self.conflict

    @property
    def net(self) -> int:
        return self.support - self.conflict

    @property
    def consistency(self) -> float:
        if self.total == 0:
            return 0.0
        return self.support / self.total


@dataclass(frozen=True)
class Scenario:
    name: str
    true_candidate: int
    evidence: dict[int, Evidence]


CANDIDATES = (0, 1, 2)


# ---------------------------------------------------------------------------
# TEST DESIGN
# ---------------------------------------------------------------------------
#
# Candidate 0 is always the ground-truth candidate.
#
# All policies receive exactly the same evidence.
#
# Policies:
#
# 1. PERSISTENT
#    Select a unique candidate that has support and ZERO conflict.
#    This represents a strict "clean evidence only" policy.
#
# 2. NET
#    Select the candidate with the largest support-conflict score.
#    Abstain on ties or if the best score is not positive.
#
# 3. CONSISTENCY
#    Select the candidate with the highest support/(support+conflict).
#    Abstain on ties or when there is no evidence.
#
# 4. SUPPORT
#    Select the candidate with the largest absolute support count.
#    Abstain on ties.
#
# The test does NOT claim that any policy is universally correct.
# It measures the trade-offs produced by each decision criterion.
# ---------------------------------------------------------------------------


SCENARIOS = [
    Scenario(
        "CLEAN_TRUE_STRONGER",
        0,
        {
            0: Evidence(10, 0),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CLEAN_TRUE_WEAKER",
        0,
        {
            0: Evidence(6, 0),
            1: Evidence(10, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CONFLICT_TRUE_NET_STRONGER",
        0,
        {
            0: Evidence(10, 2),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CONFLICT_TRUE_NET_WEAKER",
        0,
        {
            0: Evidence(7, 3),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CONFLICT_TRUE_NET_EQUAL",
        0,
        {
            0: Evidence(8, 2),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "LARGE_CONFLICT_TRUE",
        0,
        {
            0: Evidence(20, 5),
            1: Evidence(8, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "VERY_LARGE_CONFLICT_TRUE",
        0,
        {
            0: Evidence(100, 20),
            1: Evidence(40, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CONFLICT_TRUE_HIGHER_CONSISTENCY_FALSE_CLEAN",
        0,
        {
            0: Evidence(12, 6),
            1: Evidence(8, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "CONFLICT_TRUE_LOW_CONSISTENCY_FALSE_CLEAN",
        0,
        {
            0: Evidence(20, 10),
            1: Evidence(12, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "TRUE_BALANCED_CONFLICT_FALSE_CLEAN",
        0,
        {
            0: Evidence(5, 5),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "TRUE_SLIGHTLY_POSITIVE",
        0,
        {
            0: Evidence(6, 5),
            1: Evidence(6, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "WEAK_TRUE_CLEAN_FALSE",
        0,
        {
            0: Evidence(4, 1),
            1: Evidence(3, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "WEAK_TRUE_HIGH_CONFLICT",
        0,
        {
            0: Evidence(4, 3),
            1: Evidence(3, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "TWO_CLEAN_POSITIVE",
        0,
        {
            0: Evidence(10, 0),
            1: Evidence(9, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "TRUE_CONFLICT_FALSE_CLEAN",
        0,
        {
            0: Evidence(10, 2),
            1: Evidence(9, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "TRUE_STRONG_CONFLICT_FALSE_CLEAN",
        0,
        {
            0: Evidence(20, 5),
            1: Evidence(15, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "SAME_RATIO_SMALL",
        0,
        {
            0: Evidence(10, 2),
            1: Evidence(8, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "SAME_RATIO_LARGE",
        0,
        {
            0: Evidence(20, 4),
            1: Evidence(16, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "RECOVERY_TRUE",
        0,
        {
            0: Evidence(12, 3),
            1: Evidence(8, 0),
            2: Evidence(0, 5),
        },
    ),
    Scenario(
        "NO_EVIDENCE",
        0,
        {
            0: Evidence(0, 0),
            1: Evidence(0, 0),
            2: Evidence(0, 0),
        },
    ),
]


def unique_best(values: dict[int, float]) -> int | None:
    best = max(values.values())
    winners = [candidate for candidate, value in values.items() if value == best]

    if len(winners) != 1:
        return None

    return winners[0]


def persistent_policy(evidence: dict[int, Evidence]) -> int | None:
    eligible = [
        candidate
        for candidate, value in evidence.items()
        if value.support > 0 and value.conflict == 0
    ]

    if len(eligible) == 1:
        return eligible[0]

    return None


def net_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {
        candidate: value.net
        for candidate, value in evidence.items()
    }

    best = unique_best(scores)

    if best is None:
        return None

    if scores[best] <= 0:
        return None

    return best


def consistency_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {
        candidate: value.consistency
        for candidate, value in evidence.items()
        if value.total > 0
    }

    if not scores:
        return None

    return unique_best(scores)


def support_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {
        candidate: value.support
        for candidate, value in evidence.items()
    }

    return unique_best(scores)


POLICIES = {
    "PERSISTENT": persistent_policy,
    "NET": net_policy,
    "CONSISTENCY": consistency_policy,
    "SUPPORT": support_policy,
}


def state_name(value: Evidence) -> str:
    if value.total == 0:
        return "UNKNOWN"

    if value.support > 0 and value.conflict > 0:
        return "CONFLICT"

    if value.support > 0:
        return "RIGHT"

    return "WRONG"


def run_scenario(scenario: Scenario, policy_name: str, policy) -> dict:
    # Use the same EvidenceMemory abstraction as the Phase 3 core.
    memory = EvidenceMemory()

    for candidate, value in scenario.evidence.items():
        for _ in range(value.support):
            memory.observe(
                scenario.true_candidate,
                candidate,
                True,
            )

        for _ in range(value.conflict):
            memory.observe(
                scenario.true_candidate,
                candidate,
                False,
            )

    # Reconstruct the exact evidence supplied to the policy.
    evidence = {
        candidate: Evidence(
            support=memory.get(
                scenario.true_candidate,
                candidate,
            ).support,
            conflict=memory.get(
                scenario.true_candidate,
                candidate,
            ).conflict,
        )
        for candidate in CANDIDATES
    }

    decision = policy(evidence)

    true_evidence = evidence[scenario.true_candidate]

    correct = decision == scenario.true_candidate
    decision_made = decision is not None
    true_has_conflict = true_evidence.conflict > 0

    if decision is None:
        decision_text = "ABSTAIN"
    else:
        decision_text = str(decision)

    return {
        "scenario": scenario.name,
        "policy": policy_name,
        "true_candidate": scenario.true_candidate,
        "support_true": true_evidence.support,
        "conflict_true": true_evidence.conflict,
        "true_state": state_name(true_evidence),
        "true_net": true_evidence.net,
        "true_consistency": round(true_evidence.consistency, 6),
        "decision": decision_text,
        "correct": int(correct),
        "decision_made": int(decision_made),
        "true_has_conflict": int(true_has_conflict),
    }


def main() -> None:
    output = Path("results/phase3/test05/results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for scenario in SCENARIOS:
        for policy_name, policy in POLICIES.items():
            rows.append(
                run_scenario(
                    scenario,
                    policy_name,
                    policy,
                )
            )

    fieldnames = [
        "scenario",
        "policy",
        "true_candidate",
        "support_true",
        "conflict_true",
        "true_state",
        "true_net",
        "true_consistency",
        "decision",
        "correct",
        "decision_made",
        "true_has_conflict",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("TEST05: Decision Policy Comparison")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Policies: {len(POLICIES)}")
    print(f"Rows: {len(rows)}")
    print()

    print(
        f"{'POLICY':<14}"
        f"{'DECISIONS':>10}"
        f"{'CORRECT':>10}"
        f"{'ACCURACY':>12}"
        f"{'COVERAGE':>11}"
        f"{'OVERALL':>11}"
    )

    for policy_name in POLICIES:
        policy_rows = [
            row
            for row in rows
            if row["policy"] == policy_name
        ]

        decisions = sum(row["decision_made"] for row in policy_rows)
        correct = sum(row["correct"] for row in policy_rows)

        accuracy = (
            correct / decisions * 100
            if decisions
            else 0.0
        )

        coverage = decisions / len(SCENARIOS) * 100

        overall = correct / len(SCENARIOS) * 100

        print(
            f"{policy_name:<14}"
            f"{decisions:>4}/{len(SCENARIOS):<5}"
            f"{correct:>4}/{decisions:<5}"
            f"{accuracy:>10.1f}%"
            f"{coverage:>10.1f}%"
            f"{overall:>10.1f}%"
        )

    print()
    print("Conflict scenarios:")

    conflict_scenarios = [
        scenario
        for scenario in SCENARIOS
        if scenario.evidence[scenario.true_candidate].conflict > 0
    ]

    print(f"  {len(conflict_scenarios)}/{len(SCENARIOS)}")

    for policy_name in POLICIES:
        policy_rows = [
            row
            for row in rows
            if row["policy"] == policy_name
            and row["true_has_conflict"] == 1
        ]

        decisions = sum(row["decision_made"] for row in policy_rows)
        correct = sum(row["correct"] for row in policy_rows)

        print(
            f"  {policy_name:<14}"
            f"decisions={decisions}/{len(policy_rows)}, "
            f"correct={correct}/{decisions if decisions else 0}"
        )

    print()
    print(f"Results written to: {output}")

    if len(SCENARIOS) != 20:
        raise AssertionError(
            f"Expected 20 scenarios, got {len(SCENARIOS)}"
        )

    if len(rows) != 80:
        raise AssertionError(
            f"Expected 80 result rows, got {len(rows)}"
        )

    print()
    print("TEST05: PASSED")


if __name__ == "__main__":
    main()
