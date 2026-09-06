from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from collections import defaultdict

# ============================================================
# TEST 12
# ACTIVE INFORMATION LEARNING
#
# Binary            : WRONG / RIGHT
# Binary+Confidence : WRONG / RIGHT + confidence score
# Ternary           : WRONG / UNKNOWN / RIGHT
# Quaternary        : WRONG / UNKNOWN / RIGHT / CONFLICT
#
# The agent actively chooses WHICH candidate to query next.
# ============================================================

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

AGENTS = (
    "binary",
    "binary_confidence",
    "ternary",
    "quaternary",
)

N_CANDIDATES = 10
EPISODES = 5000
MAX_QUERIES = 30

UNKNOWN_RATES = (0.0, 0.1, 0.2, 0.4)
CONFLICT_RATES = (0.0, 0.05, 0.1, 0.2)

REWARD_CORRECT = 1.0
PENALTY_WRONG = -5.0
COST_QUERY = -0.1
PENALTY_ABSTAIN = -0.2

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "test12"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULT_DIR / "result.csv"


def entropy(n):
    if n <= 1:
        return 0.0
    p = 1.0 / n
    return -n * p * math.log2(p)


def hidden_answer(task_id):
    return (task_id * 7 + 3) % N_CANDIDATES


class Environment:
    def __init__(self, unknown_rate, conflict_rate, rng):
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng

    def query(self, task_id, candidate):
        answer = hidden_answer(task_id)

        # Unknown information
        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        correct = candidate == answer

        # Contradictory/noisy information
        if self.rng.random() < self.conflict_rate:
            return WRONG if correct else RIGHT

        return RIGHT if correct else WRONG


class BaseAgent:
    def __init__(self, rng):
        self.rng = rng
        self.history = defaultdict(list)

    def observe(self, candidate, feedback):
        self.history[candidate].append(feedback)

    def state(self, candidate):
        return UNKNOWN

    def score(self, candidate):
        return 0.0

    def confidence(self, candidate):
        return False

    def possible_candidates(self):
        result = []

        for c in range(N_CANDIDATES):
            if self.state(c) != WRONG:
                result.append(c)

        return result

    def choose_query(self):
        """
        Active information seeking.

        Prefer unresolved candidates.
        Candidates with UNKNOWN / CONFLICT are deliberately
        revisited because another observation may resolve them.
        """

        candidates = self.possible_candidates()

        if not candidates:
            candidates = list(range(N_CANDIDATES))

        # Explore unresolved candidates first.
        unresolved = [
            c for c in candidates
            if self.state(c) in (UNKNOWN, CONFLICT)
        ]

        if unresolved:
            return self.rng.choice(unresolved)

        # Otherwise query the weakest candidate.
        scores = {
            c: self.score(c)
            for c in candidates
        }

        minimum = min(scores.values())

        weakest = [
            c for c, s in scores.items()
            if s == minimum
        ]

        return self.rng.choice(weakest)

    def answer(self):
        confident = [
            c for c in range(N_CANDIDATES)
            if self.confidence(c)
        ]

        if not confident:
            return None

        scores = {
            c: self.score(c)
            for c in confident
        }

        best = max(scores.values())

        candidates = [
            c for c, s in scores.items()
            if s == best
        ]

        return self.rng.choice(candidates)


class BinaryAgent(BaseAgent):

    def state(self, candidate):
        history = self.history[candidate]

        if RIGHT in history:
            return RIGHT

        if WRONG in history:
            return WRONG

        return UNKNOWN

    def score(self, candidate):
        h = self.history[candidate]
        return h.count(RIGHT) - h.count(WRONG)

    def confidence(self, candidate):
        h = self.history[candidate]

        return (
            RIGHT in h
            and h.count(RIGHT) > h.count(WRONG)
        )


class BinaryConfidenceAgent(BinaryAgent):

    def confidence_value(self, candidate):
        h = self.history[candidate]

        if not h:
            return 0.0

        rights = h.count(RIGHT)
        wrongs = h.count(WRONG)

        total = rights + wrongs

        if total == 0:
            return 0.0

        return rights / total

    def confidence(self, candidate):
        return self.confidence_value(candidate) >= 0.7

    def choose_query(self):
        candidates = list(range(N_CANDIDATES))

        # Explicit uncertainty signal, but still binary memory.
        values = {
            c: abs(self.confidence_value(c) - 0.5)
            for c in candidates
        }

        minimum = min(values.values())

        best = [
            c for c, v in values.items()
            if v == minimum
        ]

        return self.rng.choice(best)


class TernaryAgent(BaseAgent):

    def state(self, candidate):
        h = self.history[candidate]

        rights = h.count(RIGHT)
        wrongs = h.count(WRONG)

        if rights > wrongs:
            return RIGHT

        if wrongs > rights:
            return WRONG

        return UNKNOWN

    def score(self, candidate):
        state = self.state(candidate)

        if state == RIGHT:
            return 2.0

        if state == UNKNOWN:
            return 0.0

        return -2.0

    def confidence(self, candidate):
        return self.state(candidate) == RIGHT


class QuaternaryAgent(BaseAgent):

    def state(self, candidate):
        h = self.history[candidate]

        rights = h.count(RIGHT)
        wrongs = h.count(WRONG)

        if rights == 0 and wrongs == 0:
            return UNKNOWN

        if rights == wrongs:
            return CONFLICT

        if rights > wrongs:
            return RIGHT

        return WRONG

    def score(self, candidate):
        state = self.state(candidate)

        if state == RIGHT:
            return 2.0

        if state == CONFLICT:
            return 0.5

        if state == UNKNOWN:
            return 0.0

        return -2.0

    def confidence(self, candidate):
        return self.state(candidate) == RIGHT


def make_agent(name, rng):
    if name == "binary":
        return BinaryAgent(rng)

    if name == "binary_confidence":
        return BinaryConfidenceAgent(rng)

    if name == "ternary":
        return TernaryAgent(rng)

    if name == "quaternary":
        return QuaternaryAgent(rng)

    raise ValueError(name)


def run_episode(agent_name, unknown_rate, conflict_rate, seed):

    rng = random.Random(seed)

    env = Environment(
        unknown_rate,
        conflict_rate,
        rng,
    )

    agent = make_agent(agent_name, rng)

    task_id = rng.randrange(1000)
    answer = hidden_answer(task_id)

    queries = 0
    reward = 0.0

    total_information_gain = 0.0

    contradictions = 0
    recoveries = 0

    while queries < MAX_QUERIES:

        # ----------------------------------------------------
        # Try to answer
        # ----------------------------------------------------
        prediction = agent.answer()

        if prediction is not None:

            if prediction == answer:
                reward += REWARD_CORRECT
                return {
                    "solved": 1,
                    "correct": 1,
                    "queries": queries,
                    "reward": reward,
                    "information_gain": total_information_gain,
                    "false_confidence": 0,
                    "abstain": 0,
                    "contradiction": contradictions > 0,
                    "recovery": recoveries > 0,
                }

            reward += PENALTY_WRONG

            return {
                "solved": 1,
                "correct": 0,
                "queries": queries,
                "reward": reward,
                "information_gain": total_information_gain,
                "false_confidence": 1,
                "abstain": 0,
                "contradiction": contradictions > 0,
                "recovery": recoveries > 0,
            }

        # ----------------------------------------------------
        # Determine current uncertainty
        # ----------------------------------------------------
        possible_before = set(agent.possible_candidates())

        H_before = entropy(len(possible_before))

        # ----------------------------------------------------
        # ACTIVE QUERY
        # ----------------------------------------------------
        candidate = agent.choose_query()

        old_state = agent.state(candidate)

        feedback = env.query(task_id, candidate)

        queries += 1
        reward += COST_QUERY

        agent.observe(candidate, feedback)

        new_state = agent.state(candidate)

        # ----------------------------------------------------
        # Contradiction / recovery
        # ----------------------------------------------------
        h = agent.history[candidate]

        if RIGHT in h and WRONG in h:
            contradictions += 1

        if (
            old_state == CONFLICT
            and new_state == RIGHT
        ):
            recoveries += 1

        # ----------------------------------------------------
        # Information gain
        # ----------------------------------------------------
        possible_after = set(agent.possible_candidates())

        H_after = entropy(len(possible_after))

        total_information_gain += max(
            0.0,
            H_before - H_after
        )

    # --------------------------------------------------------
    # Query limit reached
    # --------------------------------------------------------
    prediction = agent.answer()

    if prediction is None:

        reward += PENALTY_ABSTAIN

        return {
            "solved": 0,
            "correct": 0,
            "queries": queries,
            "reward": reward,
            "information_gain": total_information_gain,
            "false_confidence": 0,
            "abstain": 1,
            "contradiction": contradictions > 0,
            "recovery": recoveries > 0,
        }

    if prediction == answer:
        reward += REWARD_CORRECT
        correct = 1
        false_confidence = 0
    else:
        reward += PENALTY_WRONG
        correct = 0
        false_confidence = 1

    return {
        "solved": 1,
        "correct": correct,
        "queries": queries,
        "reward": reward,
        "information_gain": total_information_gain,
        "false_confidence": false_confidence,
        "abstain": 0,
        "contradiction": contradictions > 0,
        "recovery": recoveries > 0,
    }


def aggregate(rows):

    n = len(rows)

    total_queries = sum(r["queries"] for r in rows)

    total_reward = sum(r["reward"] for r in rows)

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
            sum(r["false_confidence"] for r in rows) / n,

        "abstain_rate":
            sum(r["abstain"] for r in rows) / n,

        "contradiction_rate":
            sum(r["contradiction"] for r in rows) / n,

        "recovery_rate":
            sum(r["recovery"] for r in rows) / n,
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

    results = []

    print("=" * 110)
    print("TEST 12 — ACTIVE INFORMATION LEARNING")
    print("=" * 110)

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
                        + int(unknown_rate * 10000) * 1009
                        + int(conflict_rate * 10000) * 9176
                        + sum(
                            ord(c)
                            for c in agent_name
                        )
                    )

                    rows.append(
                        run_episode(
                            agent_name,
                            unknown_rate,
                            conflict_rate,
                            seed,
                        )
                    )

                stats = aggregate(rows)

                result = {
                    "agent": agent_name,
                    "unknown_rate": unknown_rate,
                    "conflict_rate": conflict_rate,
                    **stats,
                }

                results.append(result)

                print(
                    f"{agent_name:19s} "
                    f"acc={stats['accuracy']:.4f} "
                    f"q={stats['avg_queries']:.3f} "
                    f"reward={stats['avg_reward']:.3f} "
                    f"R/Q={stats['reward_per_query']:.4f} "
                    f"IG={stats['information_gain']:.3f} "
                    f"IG/Q={stats['information_gain_per_query']:.4f} "
                    f"false={stats['false_confidence_rate']:.4f} "
                    f"abstain={stats['abstain_rate']:.4f}"
                )

    with CSV_PATH.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 110)
    print(f"CSV saved: {CSV_PATH}")
    print("=" * 110)


if __name__ == "__main__":
    main()
