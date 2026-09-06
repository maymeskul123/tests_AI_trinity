from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from collections import defaultdict

# ============================================================
# TEST 11.1
# FAIR INFORMATION-STATE EXPERIMENT
#
# Binary:
#   WRONG / RIGHT
#
# Ternary:
#   WRONG / UNKNOWN / RIGHT
#
# Quaternary:
#   WRONG / UNKNOWN / RIGHT / CONFLICT
#
# UNKNOWN does NOT mean WRONG.
# CONFLICT means: "information is contradictory; verify again".
# ============================================================

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

AGENTS = ("binary", "ternary", "quaternary")

N_TASKS = 100
N_CANDIDATES = 10
EPISODES = 3000
MAX_QUERIES = 30

UNKNOWN_RATES = (0.0, 0.1, 0.2, 0.4)
CONFLICT_RATES = (0.0, 0.05, 0.1, 0.2)

# Reward model
REWARD_CORRECT = 1.0
PENALTY_WRONG = -5.0
COST_QUERY = -0.1
PENALTY_ABSTAIN = -0.2
PENALTY_FALSE_CONFIDENCE = -10.0

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "test11_1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULT_DIR / "result.csv"


def entropy(states):
    """
    Shannon entropy over remaining possible candidates.
    """
    if not states:
        return 0.0

    n = len(states)
    p = 1.0 / n
    return -n * p * math.log2(p)


def hidden_answer(task_id: int) -> int:
    """
    Deterministic hidden answer.
    """
    return (task_id * 7 + 3) % N_CANDIDATES


class Environment:
    def __init__(self, unknown_rate, conflict_rate, rng):
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng

    def query(self, task_id, candidate):
        answer = hidden_answer(task_id)

        # Unknown observation
        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        correct = candidate == answer

        # Conflict/noisy observation
        if self.rng.random() < self.conflict_rate:
            return WRONG if correct else RIGHT

        return RIGHT if correct else WRONG


class BaseAgent:
    def __init__(self, rng):
        self.rng = rng
        self.history = defaultdict(list)
        self.queries = 0

    def observe(self, task_id, candidate, feedback):
        self.history[(task_id, candidate)].append(feedback)

    def state(self, task_id, candidate):
        return UNKNOWN

    def score(self, task_id, candidate):
        return 0.0

    def choose_candidate(self, task_id):
        scores = {
            c: self.score(task_id, c)
            for c in range(N_CANDIDATES)
        }

        best = max(scores.values())
        candidates = [
            c for c, s in scores.items()
            if s == best
        ]

        return self.rng.choice(candidates)

    def confidence(self, task_id, candidate):
        return False

    def answer(self, task_id):
        candidates = [
            c for c in range(N_CANDIDATES)
            if self.confidence(task_id, c)
        ]

        if not candidates:
            return None

        # strongest candidate
        scores = {
            c: self.score(task_id, c)
            for c in candidates
        }

        best = max(scores.values())
        best_candidates = [
            c for c, s in scores.items()
            if s == best
        ]

        return self.rng.choice(best_candidates)


class BinaryAgent(BaseAgent):
    """
    Two information states.

    UNKNOWN observations do not create a third state.
    """

    def state(self, task_id, candidate):
        history = self.history[(task_id, candidate)]

        if RIGHT in history:
            return RIGHT

        if WRONG in history:
            return WRONG

        return UNKNOWN

    def score(self, task_id, candidate):
        history = self.history[(task_id, candidate)]

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

        return rights - wrongs

    def confidence(self, task_id, candidate):
        history = self.history[(task_id, candidate)]

        return RIGHT in history and history.count(RIGHT) > history.count(WRONG)


class TernaryAgent(BaseAgent):
    """
    WRONG / UNKNOWN / RIGHT

    UNKNOWN keeps the hypothesis alive.
    """

    def state(self, task_id, candidate):
        history = self.history[(task_id, candidate)]

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

        if rights > wrongs:
            return RIGHT

        if wrongs > rights:
            return WRONG

        return UNKNOWN

    def score(self, task_id, candidate):
        state = self.state(task_id, candidate)

        if state == RIGHT:
            return 2.0

        if state == UNKNOWN:
            # UNKNOWN remains a live hypothesis.
            return 0.0

        return -2.0

    def confidence(self, task_id, candidate):
        return self.state(task_id, candidate) == RIGHT


class QuaternaryAgent(BaseAgent):
    """
    WRONG / UNKNOWN / RIGHT / CONFLICT

    Conflict is explicitly represented.

    If evidence contains both RIGHT and WRONG with equal
    strength, the state becomes CONFLICT.

    CONFLICT does not mean rejection.
    It means "verify this candidate".
    """

    def state(self, task_id, candidate):
        history = self.history[(task_id, candidate)]

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

        if rights == 0 and wrongs == 0:
            return UNKNOWN

        if rights == wrongs and rights > 0:
            return CONFLICT

        if rights > wrongs:
            return RIGHT

        return WRONG

    def score(self, task_id, candidate):
        state = self.state(task_id, candidate)

        if state == RIGHT:
            return 2.0

        if state == CONFLICT:
            # Conflict is interesting but not trustworthy.
            return 0.5

        if state == UNKNOWN:
            return 0.0

        return -2.0

    def confidence(self, task_id, candidate):
        return self.state(task_id, candidate) == RIGHT


def make_agent(name, rng):
    if name == "binary":
        return BinaryAgent(rng)

    if name == "ternary":
        return TernaryAgent(rng)

    if name == "quaternary":
        return QuaternaryAgent(rng)

    raise ValueError(name)


def run_episode(agent_name, unknown_rate, conflict_rate, seed):
    rng = random.Random(seed)

    env = Environment(
        unknown_rate=unknown_rate,
        conflict_rate=conflict_rate,
        rng=rng,
    )

    agent = make_agent(agent_name, rng)

    task_id = rng.randrange(N_TASKS)
    true_answer = hidden_answer(task_id)

    possible_before = set(range(N_CANDIDATES))

    total_reward = 0.0
    information_gain = 0.0
    contradiction_detected = False
    recovery = False
    queries = 0

    while queries < MAX_QUERIES:

        # ----------------------------------------------------
        # Current knowledge
        # ----------------------------------------------------
        possible = set()

        for candidate in range(N_CANDIDATES):
            state = agent.state(task_id, candidate)

            if state != WRONG:
                possible.add(candidate)

        if not possible:
            possible = set(range(N_CANDIDATES))

        # ----------------------------------------------------
        # Try to answer before another query
        # ----------------------------------------------------
        answer = agent.answer(task_id)

        if answer is not None:

            if answer == true_answer:
                total_reward += REWARD_CORRECT
                break

            total_reward += PENALTY_FALSE_CONFIDENCE
            break

        # ----------------------------------------------------
        # Select next query
        # ----------------------------------------------------
        candidate = agent.choose_candidate(task_id)

        before_entropy = entropy(possible)

        feedback = env.query(task_id, candidate)

        queries += 1
        total_reward += COST_QUERY

        # ----------------------------------------------------
        # Observe
        # ----------------------------------------------------
        old_state = agent.state(task_id, candidate)

        agent.observe(task_id, candidate, feedback)

        new_state = agent.state(task_id, candidate)

        # Detect contradiction
        history = agent.history[(task_id, candidate)]

        if (
            RIGHT in history
            and WRONG in history
        ):
            contradiction_detected = True

        # Recovery:
        # candidate was conflict/uncertain and later became RIGHT
        if old_state == CONFLICT and new_state == RIGHT:
            recovery = True

        # ----------------------------------------------------
        # Information gain
        # ----------------------------------------------------
        possible_after = set()

        for c in range(N_CANDIDATES):
            state = agent.state(task_id, c)

            if state != WRONG:
                possible_after.add(c)

        if not possible_after:
            possible_after = set(range(N_CANDIDATES))

        after_entropy = entropy(possible_after)

        gain = max(0.0, before_entropy - after_entropy)
        information_gain += gain

    else:
        # max queries exhausted
        answer = agent.answer(task_id)

        if answer is None:
            total_reward += PENALTY_ABSTAIN
        elif answer == true_answer:
            total_reward += REWARD_CORRECT
        else:
            total_reward += PENALTY_FALSE_CONFIDENCE

    final_answer = agent.answer(task_id)

    if final_answer is None:
        solved = False
        correct = False
        abstained = True
        false_confidence = False

    else:
        solved = True
        correct = final_answer == true_answer
        abstained = False
        false_confidence = not correct

    return {
        "solved": solved,
        "correct": correct,
        "queries": queries,
        "reward": total_reward,
        "information_gain": information_gain,
        "false_confidence": false_confidence,
        "abstained": abstained,
        "contradiction": contradiction_detected,
        "recovery": recovery,
    }


def aggregate(rows):
    n = len(rows)

    return {
        "solved_rate": sum(r["solved"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "avg_queries": sum(r["queries"] for r in rows) / n,
        "avg_reward": sum(r["reward"] for r in rows) / n,
        "reward_per_query": (
            sum(r["reward"] for r in rows)
            / max(1, sum(r["queries"] for r in rows))
        ),
        "information_gain": (
            sum(r["information_gain"] for r in rows) / n
        ),
        "information_gain_per_query": (
            sum(r["information_gain"] for r in rows)
            / max(1, sum(r["queries"] for r in rows))
        ),
        "false_confidence_rate": (
            sum(r["false_confidence"] for r in rows) / n
        ),
        "abstain_rate": (
            sum(r["abstained"] for r in rows) / n
        ),
        "contradiction_rate": (
            sum(r["contradiction"] for r in rows) / n
        ),
        "recovery_rate": (
            sum(r["recovery"] for r in rows) / n
        ),
    }


def main():

    fieldnames = [
        "agent",
        "unknown_rate",
        "conflict_rate",
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

    all_rows = []

    print("=" * 90)
    print("TEST 11.1 — FAIR INFORMATION-STATE EXPERIMENT")
    print("=" * 90)

    for unknown_rate in UNKNOWN_RATES:
        for conflict_rate in CONFLICT_RATES:

            print()
            print(
                f"UNKNOWN={unknown_rate:.2f} "
                f"CONFLICT={conflict_rate:.2f}"
            )

            for agent_name in AGENTS:

                rows = []

                for episode in range(EPISODES):

                    seed = (
                        episode * 1000003
                        + int(unknown_rate * 1000) * 1009
                        + int(conflict_rate * 1000) * 9176
                        + hash(agent_name) % 100000
                    )

                    result = run_episode(
                        agent_name,
                        unknown_rate,
                        conflict_rate,
                        seed,
                    )

                    rows.append(result)

                stats = aggregate(rows)

                row = {
                    "agent": agent_name,
                    "unknown_rate": unknown_rate,
                    "conflict_rate": conflict_rate,
                    **stats,
                }

                all_rows.append(row)

                print(
                    f"{agent_name:11s} "
                    f"acc={stats['accuracy']:.4f} "
                    f"queries={stats['avg_queries']:.3f} "
                    f"reward={stats['avg_reward']:.3f} "
                    f"R/Q={stats['reward_per_query']:.4f} "
                    f"IG={stats['information_gain']:.3f} "
                    f"IG/Q={stats['information_gain_per_query']:.4f} "
                    f"false={stats['false_confidence_rate']:.4f} "
                    f"abstain={stats['abstain_rate']:.4f}"
                )

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 90)
    print(f"CSV saved: {CSV_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
