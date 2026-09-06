from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

N_CANDIDATES = 10
SEEDS = 100
MAX_QUERIES = 30

UNKNOWN_RATES = [0.0, 0.1, 0.2, 0.4]
CONFLICT_RATES = [0.0, 0.1, 0.2]

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "test10"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2


class Environment:
    def __init__(
        self,
        target: int,
        rng: random.Random,
        unknown_rate: float,
        conflict_rate: float,
    ):
        self.target = target
        self.rng = rng
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate

    def query(self, candidate: int) -> int:

        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        feedback = RIGHT if candidate == self.target else WRONG

        if self.rng.random() < self.conflict_rate:
            return WRONG if feedback == RIGHT else RIGHT

        return feedback


class BinaryAgent:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        # Binary:
        # False = WRONG
        # True  = RIGHT
        self.memory = {
            c: False
            for c in range(N_CANDIDATES)
        }

    def choose_candidate(self):

        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c]
        ]

        if right:
            return right[0]

        candidates = [
            c for c in range(N_CANDIDATES)
            if not self.memory[c]
        ]

        return self.rng.choice(candidates)

    def learn(self, candidate, feedback):

        if feedback == RIGHT:
            self.memory[candidate] = True

        elif feedback == WRONG:
            self.memory[candidate] = False

        # UNKNOWN cannot be represented.


class TernaryAgent:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        # -1 WRONG
        #  0 UNKNOWN
        # +1 RIGHT
        self.memory = {
            c: UNKNOWN
            for c in range(N_CANDIDATES)
        }

    def choose_candidate(self):

        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == RIGHT
        ]

        if right:
            return right[0]

        unknown = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == UNKNOWN
        ]

        if unknown:
            return self.rng.choice(unknown)

        wrong = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == WRONG
        ]

        if wrong:
            return self.rng.choice(wrong)

        return self.rng.randrange(N_CANDIDATES)

    def learn(self, candidate, feedback):

        if feedback in (WRONG, UNKNOWN, RIGHT):
            self.memory[candidate] = feedback


class QuaternaryAgent:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        # -1 WRONG
        #  0 UNKNOWN
        # +1 RIGHT
        #  2 CONFLICT
        self.memory = {
            c: UNKNOWN
            for c in range(N_CANDIDATES)
        }

    def choose_candidate(self):

        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == RIGHT
        ]

        if right:
            return right[0]

        conflict = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == CONFLICT
        ]

        if conflict:
            return self.rng.choice(conflict)

        unknown = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == UNKNOWN
        ]

        if unknown:
            return self.rng.choice(unknown)

        return self.rng.randrange(N_CANDIDATES)

    def learn(self, candidate, feedback):

        previous = self.memory[candidate]

        if (
            (previous == RIGHT and feedback == WRONG)
            or
            (previous == WRONG and feedback == RIGHT)
        ):
            self.memory[candidate] = CONFLICT

        else:
            self.memory[candidate] = feedback

    def solved(self):

        return any(
            self.memory[c] == RIGHT
            for c in range(N_CANDIDATES)
        )


def run_episode(
    agent_class,
    seed,
    unknown_rate,
    conflict_rate,
):

    rng = random.Random(seed)

    target = rng.randrange(N_CANDIDATES)

    env = Environment(
        target=target,
        rng=rng,
        unknown_rate=unknown_rate,
        conflict_rate=conflict_rate,
    )

    agent = agent_class(seed + 10000)

    for queries in range(1, MAX_QUERIES + 1):

        candidate = agent.choose_candidate()

        feedback = env.query(candidate)

        agent.learn(candidate, feedback)

        # Check whether agent has found target.
        if isinstance(agent, BinaryAgent):

            if agent.memory.get(candidate) is True:

                correct = candidate == target

                return {
                    "solved": True,
                    "correct": correct,
                    "queries": queries,
                }

        else:

            if agent.memory.get(candidate) == RIGHT:

                correct = candidate == target

                return {
                    "solved": True,
                    "correct": correct,
                    "queries": queries,
                }

    return {
        "solved": False,
        "correct": False,
        "queries": MAX_QUERIES,
    }


def run_experiment():

    agents = {
        "binary": BinaryAgent,
        "ternary": TernaryAgent,
        "quaternary": QuaternaryAgent,
    }

    rows = []

    for unknown_rate in UNKNOWN_RATES:

        for conflict_rate in CONFLICT_RATES:

            print()
            print("=" * 75)
            print(
                f"UNKNOWN={unknown_rate:.1f} "
                f"CONFLICT={conflict_rate:.1f}"
            )
            print("=" * 75)

            for name, agent_class in agents.items():

                solved_values = []
                correct_values = []
                query_values = []

                for seed in range(SEEDS):

                    result = run_episode(
                        agent_class,
                        seed,
                        unknown_rate,
                        conflict_rate,
                    )

                    solved_values.append(
                        int(result["solved"])
                    )

                    correct_values.append(
                        int(result["correct"])
                    )

                    if result["solved"]:
                        query_values.append(
                            result["queries"]
                        )

                solved_rate = mean(solved_values)
                accuracy = mean(correct_values)

                avg_queries = (
                    mean(query_values)
                    if query_values
                    else MAX_QUERIES
                )

                print(
                    f"{name:12s} "
                    f"solved={solved_rate:.3f} "
                    f"accuracy={accuracy:.3f} "
                    f"queries={avg_queries:.2f}"
                )

                rows.append({
                    "agent": name,
                    "unknown_rate": unknown_rate,
                    "conflict_rate": conflict_rate,
                    "solved_rate": solved_rate,
                    "accuracy": accuracy,
                    "avg_queries": avg_queries,
                })

    csv_path = RESULT_DIR / "result.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "unknown_rate",
                "conflict_rate",
                "solved_rate",
                "accuracy",
                "avg_queries",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 75)
    print(f"CSV saved: {csv_path}")
    print("=" * 75)


def contradiction_test():

    print()
    print("=" * 75)
    print("CONTRADICTION TEST")
    print("=" * 75)

    candidate = 5

    binary = BinaryAgent(1)

    binary.learn(candidate, RIGHT)
    print("Binary   RIGHT ->", binary.memory[candidate])

    binary.learn(candidate, WRONG)
    print("Binary   WRONG ->", binary.memory[candidate])

    ternary = TernaryAgent(1)

    ternary.learn(candidate, RIGHT)
    print("Ternary  RIGHT ->", ternary.memory[candidate])

    ternary.learn(candidate, WRONG)
    print("Ternary  WRONG ->", ternary.memory[candidate])

    quaternary = QuaternaryAgent(1)

    quaternary.learn(candidate, RIGHT)
    print("Quaternary RIGHT ->", quaternary.memory[candidate])

    quaternary.learn(candidate, WRONG)
    print("Quaternary WRONG ->", quaternary.memory[candidate])


if __name__ == "__main__":
    run_experiment()
    contradiction_test()
