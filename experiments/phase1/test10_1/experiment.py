from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean

N_CANDIDATES = 10
N_TASKS = 1000
SEEDS = 100
MAX_QUERIES = 40

UNKNOWN_RATES = [0.0, 0.1, 0.2, 0.4]
CONFLICT_RATES = [0.0, 0.1, 0.2]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "test10_1"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

        # No information.
        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        # True observation.
        true_feedback = (
            RIGHT if candidate == self.target
            else WRONG
        )

        # Generate contradictory evidence.
        if self.rng.random() < self.conflict_rate:
            return (
                WRONG
                if true_feedback == RIGHT
                else RIGHT
            )

        return true_feedback


class BinaryAgent:
    """
    Binary epistemic memory.

    FALSE = WRONG
    TRUE  = RIGHT

    There is no explicit UNKNOWN or CONFLICT state.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        self.memory = {
            c: False
            for c in range(N_CANDIDATES)
        }

        self.queries = 0
        self.false_confidence = 0

    def choose_candidate(self) -> int:

        # If we believe we found the answer,
        # stop querying.
        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c]
        ]

        if right:
            return right[0]

        # Binary cannot distinguish:
        # UNKNOWN from WRONG.
        candidates = [
            c for c in range(N_CANDIDATES)
            if not self.memory[c]
        ]

        return self.rng.choice(candidates)

    def learn(self, candidate: int, feedback: int):

        if feedback == RIGHT:
            self.memory[candidate] = True

        elif feedback == WRONG:
            self.memory[candidate] = False

        elif feedback == UNKNOWN:
            # Cannot represent UNKNOWN.
            pass


class TernaryAgent:
    """
    WRONG / UNKNOWN / RIGHT
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        self.memory = {
            c: UNKNOWN
            for c in range(N_CANDIDATES)
        }

        self.queries = 0

    def choose_candidate(self) -> int:

        # Confirmed answer.
        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == RIGHT
        ]

        if right:
            return right[0]

        # UNKNOWN candidates are explicitly known
        # to be unresolved.
        unknown = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == UNKNOWN
        ]

        if unknown:
            return self.rng.choice(unknown)

        # Everything else is WRONG.
        wrong = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == WRONG
        ]

        if wrong:
            return self.rng.choice(wrong)

        return self.rng.randrange(N_CANDIDATES)

    def learn(self, candidate: int, feedback: int):

        previous = self.memory[candidate]

        # If contradictory evidence arrives,
        # collapse to UNKNOWN.
        if (
            (previous == RIGHT and feedback == WRONG)
            or
            (previous == WRONG and feedback == RIGHT)
        ):
            self.memory[candidate] = UNKNOWN

        elif feedback in (
            WRONG,
            UNKNOWN,
            RIGHT,
        ):
            self.memory[candidate] = feedback


class QuaternaryAgent:
    """
    WRONG / UNKNOWN / RIGHT / CONFLICT
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        self.memory = {
            c: UNKNOWN
            for c in range(N_CANDIDATES)
        }

        self.queries = 0

    def choose_candidate(self) -> int:

        # Confirmed answer.
        right = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == RIGHT
        ]

        if right:
            return right[0]

        # CONFLICT has priority:
        # explicitly investigate it.
        conflict = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == CONFLICT
        ]

        if conflict:
            return self.rng.choice(conflict)

        # Then investigate UNKNOWN.
        unknown = [
            c for c in range(N_CANDIDATES)
            if self.memory[c] == UNKNOWN
        ]

        if unknown:
            return self.rng.choice(unknown)

        return self.rng.randrange(N_CANDIDATES)

    def learn(self, candidate: int, feedback: int):

        previous = self.memory[candidate]

        # RIGHT <-> WRONG creates persistent CONFLICT.
        if (
            (previous == RIGHT and feedback == WRONG)
            or
            (previous == WRONG and feedback == RIGHT)
        ):
            self.memory[candidate] = CONFLICT

        elif previous == CONFLICT:
            # Keep conflict until there is repeated
            # confirming evidence.
            if feedback == RIGHT:
                self.memory[candidate] = RIGHT

            elif feedback == WRONG:
                self.memory[candidate] = WRONG

            else:
                self.memory[candidate] = CONFLICT

        elif feedback in (
            WRONG,
            UNKNOWN,
            RIGHT,
        ):
            self.memory[candidate] = feedback


def evaluate(agent, target: int):

    # Determine whether the agent currently claims
    # a specific answer.
    for candidate in range(N_CANDIDATES):

        if isinstance(agent, BinaryAgent):
            claimed = agent.memory[candidate] is True
        else:
            claimed = agent.memory[candidate] == RIGHT

        if claimed:
            return True, candidate == target

    return False, False


def run_episode(
    agent_class,
    seed: int,
    unknown_rate: float,
    conflict_rate: float,
):

    rng = random.Random(seed)

    target = rng.randrange(N_CANDIDATES)

    env = Environment(
        target=target,
        rng=rng,
        unknown_rate=unknown_rate,
        conflict_rate=conflict_rate,
    )

    agent = agent_class(seed + 100000)

    contradiction_events = 0
    recoveries = 0

    for query_number in range(
        1,
        MAX_QUERIES + 1,
    ):

        candidate = agent.choose_candidate()

        previous = None

        if candidate in agent.memory:
            previous = agent.memory[candidate]

        feedback = env.query(candidate)

        # Detect a contradiction before learning.
        if (
            previous in (RIGHT, WRONG)
            and feedback in (RIGHT, WRONG)
            and (
                (previous == RIGHT and feedback == WRONG)
                or
                (previous == WRONG and feedback == RIGHT)
            )
        ):
            contradiction_events += 1

        agent.learn(candidate, feedback)

        solved, correct = evaluate(
            agent,
            target,
        )

        if solved:

            if correct and contradiction_events > 0:
                recoveries += 1

            return {
                "solved": True,
                "correct": correct,
                "queries": query_number,
                "contradictions": contradiction_events,
                "recoveries": recoveries,
            }

    return {
        "solved": False,
        "correct": False,
        "queries": MAX_QUERIES,
        "contradictions": contradiction_events,
        "recoveries": recoveries,
    }


def run():

    agents = {
        "binary": BinaryAgent,
        "ternary": TernaryAgent,
        "quaternary": QuaternaryAgent,
    }

    rows = []

    total_experiments = (
        len(UNKNOWN_RATES)
        * len(CONFLICT_RATES)
        * len(agents)
    )

    current = 0

    for unknown_rate in UNKNOWN_RATES:

        for conflict_rate in CONFLICT_RATES:

            print()
            print("=" * 90)
            print(
                f"UNKNOWN={unknown_rate:.1f}  "
                f"CONFLICT={conflict_rate:.1f}"
            )
            print("=" * 90)

            for name, agent_class in agents.items():

                current += 1

                results = []

                for seed in range(SEEDS):

                    for task in range(N_TASKS // SEEDS):

                        result = run_episode(
                            agent_class=agent_class,
                            seed=seed * 10000 + task,
                            unknown_rate=unknown_rate,
                            conflict_rate=conflict_rate,
                        )

                        results.append(result)

                solved = mean(
                    r["solved"]
                    for r in results
                )

                accuracy = mean(
                    r["correct"]
                    for r in results
                )

                queries = mean(
                    r["queries"]
                    for r in results
                )

                contradiction_rate = mean(
                    r["contradictions"] > 0
                    for r in results
                )

                recovery_rate = mean(
                    r["recoveries"] > 0
                    for r in results
                )

                print(
                    f"{name:12s} "
                    f"solved={solved:.3f} "
                    f"accuracy={accuracy:.3f} "
                    f"queries={queries:.2f} "
                    f"contradictions={contradiction_rate:.3f} "
                    f"recovery={recovery_rate:.3f}"
                )

                rows.append({
                    "agent": name,
                    "unknown_rate": unknown_rate,
                    "conflict_rate": conflict_rate,
                    "solved_rate": solved,
                    "accuracy": accuracy,
                    "avg_queries": queries,
                    "contradiction_rate": contradiction_rate,
                    "recovery_rate": recovery_rate,
                })

                print(
                    f"progress: {current}/{total_experiments}"
                )

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
                "contradiction_rate",
                "recovery_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 90)
    print("DONE")
    print(f"CSV: {csv_path}")
    print("=" * 90)


def contradiction_demo():

    print()
    print("=" * 90)
    print("CONTRADICTION DEMO")
    print("=" * 90)

    candidate = 5

    for Agent in (
        BinaryAgent,
        TernaryAgent,
        QuaternaryAgent,
    ):

        agent = Agent(seed=42)

        agent.learn(candidate, RIGHT)

        state1 = agent.memory[candidate]

        agent.learn(candidate, WRONG)

        state2 = agent.memory[candidate]

        agent.learn(candidate, RIGHT)

        state3 = agent.memory[candidate]

        print(
            f"{Agent.__name__:16s}: "
            f"RIGHT -> {state1} -> "
            f"WRONG -> {state2} -> "
            f"RIGHT -> {state3}"
        )


if __name__ == "__main__":
    run()
    contradiction_demo()
