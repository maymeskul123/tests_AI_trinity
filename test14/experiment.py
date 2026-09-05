from __future__ import annotations

import csv
import math
import random
from pathlib import Path

# ============================================================
# TEST 14
# FAIR STATE REPRESENTATION
#
# Binary:
#   0 = eliminated
#   1 = possible
#
# Ternary:
#  -1 = eliminated
#   0 = unknown
#  +1 = confirmed
#
# Quaternary:
#  -1 = eliminated
#   0 = unknown
#  +1 = confirmed
#   2 = conflict
#
# IMPORTANT:
# No agent receives an implicit "unseen" state.
# ============================================================

BINARY = "binary"
TERNARY = "ternary"
QUATERNARY = "quaternary"

AGENTS = (BINARY, TERNARY, QUATERNARY)

N_CANDIDATES = 8
EPISODES = 5000

QUERY_BUDGETS = (1, 2, 3, 4, 5, 6, 8, 10)

UNKNOWN_RATES = (0.0, 0.2, 0.4)
CONFLICT_RATES = (0.0, 0.1, 0.2)

RIGHT = 1
WRONG = -1
UNKNOWN = 0
CONFLICT = 2

REWARD_CORRECT = 1.0
PENALTY_WRONG = -5.0
COST_QUERY = -0.1
PENALTY_ABSTAIN = -0.2

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "test14"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "result.csv"


# ============================================================
# ENVIRONMENT
# ============================================================

def hidden_answer(task_id: int) -> int:
    return (task_id * 5 + 3) % N_CANDIDATES


class Environment:

    def __init__(
        self,
        unknown_rate: float,
        conflict_rate: float,
        rng: random.Random,
    ):
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng

    def query(
        self,
        task_id: int,
        candidate: int,
    ) -> int:

        answer = hidden_answer(task_id)

        # Unknown observation.
        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        correct = candidate == answer

        # Contradictory observation.
        if self.rng.random() < self.conflict_rate:
            return WRONG if correct else RIGHT

        return RIGHT if correct else WRONG


# ============================================================
# BASE
# ============================================================

class BaseAgent:

    def __init__(self, rng: random.Random):
        self.rng = rng

    def observe(self, candidate: int, feedback: int):
        raise NotImplementedError

    def state(self, candidate: int):
        raise NotImplementedError

    def choose_query(self) -> int:
        raise NotImplementedError

    def answer(self):
        raise NotImplementedError

    def memory_states(self):
        return [
            self.state(c)
            for c in range(N_CANDIDATES)
        ]


# ============================================================
# BINARY
#
# EXACTLY TWO EXPLICIT STATES:
#
# 0 = eliminated
# 1 = possible
#
# There is no "unseen" state.
# Initially every candidate is POSSIBLE.
#
# A WRONG observation eliminates.
# A RIGHT observation confirms by eliminating all others.
# UNKNOWN does not change state.
#
# This is intentionally simple and fair.
# ============================================================

class BinaryAgent(BaseAgent):

    def __init__(self, rng):
        super().__init__(rng)

        # Every candidate starts POSSIBLE.
        self.states = [1] * N_CANDIDATES

    def observe(self, candidate, feedback):

        if feedback == WRONG:
            self.states[candidate] = 0

        elif feedback == RIGHT:
            # Candidate is the answer.
            # All others become eliminated.
            for c in range(N_CANDIDATES):
                self.states[c] = 1 if c == candidate else 0

        elif feedback == UNKNOWN:
            # No state change.
            pass

    def state(self, candidate):
        return self.states[candidate]

    def choose_query(self):

        possible = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == 1
        ]

        if not possible:
            return self.rng.randrange(N_CANDIDATES)

        return self.rng.choice(possible)

    def answer(self):

        possible = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == 1
        ]

        # Only answer if exactly one candidate remains.
        if len(possible) == 1:
            return possible[0]

        return None


# ============================================================
# TERNARY
#
# -1 = eliminated
#  0 = unknown
# +1 = confirmed
#
# Initially ALL candidates are UNKNOWN.
#
# UNKNOWN observation preserves UNKNOWN.
# WRONG -> ELIMINATED.
# RIGHT -> CONFIRMED and all others ELIMINATED.
# ============================================================

class TernaryAgent(BaseAgent):

    def __init__(self, rng):
        super().__init__(rng)

        self.states = [UNKNOWN] * N_CANDIDATES

    def observe(self, candidate, feedback):

        if feedback == WRONG:
            self.states[candidate] = WRONG

        elif feedback == RIGHT:

            self.states[candidate] = RIGHT

            for c in range(N_CANDIDATES):
                if c != candidate:
                    self.states[c] = WRONG

        elif feedback == UNKNOWN:
            # Explicit UNKNOWN remains UNKNOWN.
            pass

    def state(self, candidate):
        return self.states[candidate]

    def choose_query(self):

        # Prefer UNKNOWN candidates because they contain unresolved
        # information. This is where the third state becomes active.
        unknowns = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == UNKNOWN
        ]

        if unknowns:
            return self.rng.choice(unknowns)

        possible = [
            c for c in range(N_CANDIDATES)
            if self.states[c] != WRONG
        ]

        if possible:
            return self.rng.choice(possible)

        return self.rng.randrange(N_CANDIDATES)

    def answer(self):

        confirmed = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


# ============================================================
# QUATERNARY
#
# -1 = eliminated
#  0 = unknown
# +1 = confirmed
#  2 = conflict
#
# Conflict is explicitly represented.
# ============================================================

class QuaternaryAgent(BaseAgent):

    def __init__(self, rng):
        super().__init__(rng)

        self.states = [UNKNOWN] * N_CANDIDATES

    def observe(self, candidate, feedback):

        current = self.states[candidate]

        if feedback == UNKNOWN:
            return

        if feedback == RIGHT:

            if current == WRONG:
                self.states[candidate] = CONFLICT
            elif current == CONFLICT:
                self.states[candidate] = CONFLICT
            else:
                self.states[candidate] = RIGHT

        elif feedback == WRONG:

            if current == RIGHT:
                self.states[candidate] = CONFLICT
            elif current == CONFLICT:
                self.states[candidate] = CONFLICT
            else:
                self.states[candidate] = WRONG

        # If a candidate is confirmed without conflict,
        # eliminate all others.
        if self.states[candidate] == RIGHT:

            for c in range(N_CANDIDATES):
                if c != candidate and self.states[c] != CONFLICT:
                    self.states[c] = WRONG

    def state(self, candidate):
        return self.states[candidate]

    def choose_query(self):

        # Highest priority: conflicts need verification.
        conflicts = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == CONFLICT
        ]

        if conflicts:
            return self.rng.choice(conflicts)

        unknowns = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == UNKNOWN
        ]

        if unknowns:
            return self.rng.choice(unknowns)

        possible = [
            c for c in range(N_CANDIDATES)
            if self.states[c] != WRONG
        ]

        if possible:
            return self.rng.choice(possible)

        return self.rng.randrange(N_CANDIDATES)

    def answer(self):

        confirmed = [
            c for c in range(N_CANDIDATES)
            if self.states[c] == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


# ============================================================
# FACTORY
# ============================================================

def make_agent(name, rng):

    if name == BINARY:
        return BinaryAgent(rng)

    if name == TERNARY:
        return TernaryAgent(rng)

    if name == QUATERNARY:
        return QuaternaryAgent(rng)

    raise ValueError(name)


# ============================================================
# ENTROPY
# ============================================================

def entropy(n):

    if n <= 1:
        return 0.0

    return math.log2(n)


# ============================================================
# EPISODE
# ============================================================

def run_episode(
    agent_name,
    unknown_rate,
    conflict_rate,
    budget,
    seed,
):

    rng = random.Random(seed)

    env = Environment(
        unknown_rate,
        conflict_rate,
        rng,
    )

    agent = make_agent(
        agent_name,
        rng,
    )

    task_id = rng.randrange(100000)

    answer = hidden_answer(task_id)

    queries = 0
    reward = 0.0

    information_gain = 0.0

    false_confidence = 0
    abstained = 0
    contradictions = 0
    recoveries = 0

    while queries < budget:

        prediction = agent.answer()

        # ----------------------------------------------------
        # If agent believes it has enough information.
        # ----------------------------------------------------

        if prediction is not None:

            if prediction == answer:

                reward += REWARD_CORRECT

                return {
                    "solved": 1,
                    "correct": 1,
                    "queries": queries,
                    "reward": reward,
                    "information_gain": information_gain,
                    "false_confidence": 0,
                    "abstained": 0,
                    "contradictions": contradictions,
                    "recoveries": recoveries,
                }

            reward += PENALTY_WRONG

            return {
                "solved": 1,
                "correct": 0,
                "queries": queries,
                "reward": reward,
                "information_gain": information_gain,
                "false_confidence": 1,
                "abstained": 0,
                "contradictions": contradictions,
                "recoveries": recoveries,
            }

        # ----------------------------------------------------
        # Information before query.
        # ----------------------------------------------------

        before_possible = sum(
            1
            for c in range(N_CANDIDATES)
            if agent.state(c) != WRONG
        )

        H_before = entropy(before_possible)

        # ----------------------------------------------------
        # Choose query.
        # ----------------------------------------------------

        candidate = agent.choose_query()

        old_state = agent.state(candidate)

        # ----------------------------------------------------
        # Environment.
        # ----------------------------------------------------

        feedback = env.query(
            task_id,
            candidate,
        )

        queries += 1

        reward += COST_QUERY

        # ----------------------------------------------------
        # Learn.
        # ----------------------------------------------------

        agent.observe(
            candidate,
            feedback,
        )

        new_state = agent.state(candidate)

        # ----------------------------------------------------
        # Detect contradiction.
        # ----------------------------------------------------

        if (
            old_state == RIGHT
            and new_state == CONFLICT
        ):
            contradictions += 1

        if (
            old_state == CONFLICT
            and new_state == RIGHT
        ):
            recoveries += 1

        # ----------------------------------------------------
        # Information after query.
        # ----------------------------------------------------

        after_possible = sum(
            1
            for c in range(N_CANDIDATES)
            if agent.state(c) != WRONG
        )

        H_after = entropy(after_possible)

        information_gain += max(
            0.0,
            H_before - H_after,
        )

    # ========================================================
    # Budget exhausted.
    # ========================================================

    prediction = agent.answer()

    if prediction is None:

        reward += PENALTY_ABSTAIN

        return {
            "solved": 0,
            "correct": 0,
            "queries": queries,
            "reward": reward,
            "information_gain": information_gain,
            "false_confidence": 0,
            "abstained": 1,
            "contradictions": contradictions,
            "recoveries": recoveries,
        }

    if prediction == answer:

        reward += REWARD_CORRECT

        return {
            "solved": 1,
            "correct": 1,
            "queries": queries,
            "reward": reward,
            "information_gain": information_gain,
            "false_confidence": 0,
            "abstained": 0,
            "contradictions": contradictions,
            "recoveries": recoveries,
        }

    reward += PENALTY_WRONG

    false_confidence = 1

    return {
        "solved": 1,
        "correct": 0,
        "queries": queries,
        "reward": reward,
        "information_gain": information_gain,
        "false_confidence": false_confidence,
        "abstained": 0,
        "contradictions": contradictions,
        "recoveries": recoveries,
    }


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(rows):

    n = len(rows)

    total_queries = sum(
        r["queries"]
        for r in rows
    )

    total_reward = sum(
        r["reward"]
        for r in rows
    )

    total_information = sum(
        r["information_gain"]
        for r in rows
    )

    return {

        "solved_rate":
            sum(r["solved"] for r in rows) / n,

        "accuracy":
            sum(r["correct"] for r in rows) / n,

        "avg_queries":
            total_queries / n,

        "avg_reward":
            total_reward / n,

        "reward_per_query":
            total_reward / max(1, total_queries),

        "information_gain":
            total_information / n,

        "information_gain_per_query":
            total_information / max(1, total_queries),

        "false_confidence_rate":
            sum(
                r["false_confidence"]
                for r in rows
            ) / n,

        "abstain_rate":
            sum(
                r["abstained"]
                for r in rows
            ) / n,

        "contradiction_rate":
            sum(
                r["contradictions"]
                for r in rows
            ) / n,

        "recovery_rate":
            sum(
                r["recoveries"]
                for r in rows
            ) / n,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    fieldnames = [
        "agent",
        "unknown_rate",
        "conflict_rate",
        "query_budget",
        "solved_rate",
        "accuracy",
        "avg_queries",
        "avg_reward",
        "reward_per_query",
        "information_gain",
        "information_gain_per_query",
        "false_confidence_rate",
        "abstain_rate",
        "contradiction_rate",
        "recovery_rate",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for unknown_rate in UNKNOWN_RATES:

            for conflict_rate in CONFLICT_RATES:

                print()
                print(
                    f"UNKNOWN={unknown_rate:.1f} "
                    f"CONFLICT={conflict_rate:.1f}"
                )

                print("-" * 90)

                for budget in QUERY_BUDGETS:

                    for agent_index, agent_name in enumerate(AGENTS):

                        rows = []

                        for episode in range(EPISODES):

                            seed = (
                                episode
                                + int(unknown_rate * 1000) * 1_000_000
                                + int(conflict_rate * 1000) * 10_000
                                + budget * 1_000
                                + agent_index * 100_000_000
                            )

                            rows.append(
                                run_episode(
                                    agent_name,
                                    unknown_rate,
                                    conflict_rate,
                                    budget,
                                    seed,
                                )
                            )

                        result = aggregate(rows)

                        row = {
                            "agent": agent_name,
                            "unknown_rate": unknown_rate,
                            "conflict_rate": conflict_rate,
                            "query_budget": budget,
                            **result,
                        }

                        writer.writerow(row)

                        print(
                            f"{agent_name:11s} "
                            f"B={budget:2d} "
                            f"acc={result['accuracy']:.4f} "
                            f"reward={result['avg_reward']:.4f} "
                            f"IG/Q={result['information_gain_per_query']:.4f} "
                            f"false={result['false_confidence_rate']:.4f}"
                        )

    print()
    print("=" * 90)
    print("TEST 14 COMPLETE")
    print("=" * 90)
    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
