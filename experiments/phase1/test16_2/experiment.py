from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from collections import Counter

# ============================================================
# TEST 15.2
#
# Goal:
#   Test whether an explicit ternary UNKNOWN state improves
#   active learning when the agent has limited internal state.
#
# Binary:
#   0 = eliminated
#   1 = possible
#
# Ternary:
#  -1 = WRONG
#   0 = UNKNOWN
#  +1 = RIGHT
#
# Quaternary:
#  -1 = WRONG
#   0 = UNKNOWN
#  +1 = RIGHT
#   +2 = CONFLICT
#
# IMPORTANT:
#   Agents do NOT keep raw history.
#   Their decision policy operates on their compressed states.
#
# 16 hypotheses
# 8 questions
# repeated questions allowed
# unknown + contradictory observations
# ============================================================

N_H = 32
N_Q = 8
EPISODES = 5000

BUDGETS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

UNKNOWN_RATES = [0.0, 0.25, 0.5, 0.75]
CONFLICT_RATES = [0.0, 0.1, 0.2, 0.3]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2


# ------------------------------------------------------------
# Hypotheses
# ------------------------------------------------------------

HYPOTHESES = tuple(range(N_H))


# Each question partitions the 16 hypotheses.
#
# q0-q3: ordinary binary partitions
# q4-q7: deliberately redundant / differently balanced questions
#
# This prevents the problem from degenerating into "8 perfect
# independent bits identify the answer immediately".
# ------------------------------------------------------------

def build_questions():
    qs = []

    # Balanced binary partitions
    qs.append(tuple((h >> 0) & 1 for h in HYPOTHESES))
    qs.append(tuple((h >> 1) & 1 for h in HYPOTHESES))
    qs.append(tuple((h >> 2) & 1 for h in HYPOTHESES))
    qs.append(tuple((h >> 3) & 1 for h in HYPOTHESES))

    # Redundant / mixed partitions
    qs.append(tuple(((h >> 0) ^ (h >> 1)) & 1 for h in HYPOTHESES))
    qs.append(tuple(((h >> 1) ^ (h >> 2)) & 1 for h in HYPOTHESES))
    qs.append(tuple(((h >> 0) ^ (h >> 3)) & 1 for h in HYPOTHESES))

    # Slightly unbalanced partition
    qs.append(tuple(1 if h in {0, 1, 2, 3, 4, 5, 6} else 0
                    for h in HYPOTHESES))

    return tuple(qs)


QUESTIONS = build_questions()


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

@dataclass
class Observation:
    question: int
    value: int
    true_value: int


class Environment:
    def __init__(
        self,
        target: int,
        unknown_rate: float,
        conflict_rate: float,
        rng: random.Random,
    ):
        self.target = target
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng

    def ask(self, q: int) -> Observation:
        true_value = QUESTIONS[q][self.target]

        # Unknown observation
        if self.rng.random() < self.unknown_rate:
            return Observation(q, UNKNOWN, true_value)

        # Contradictory observation
        if self.rng.random() < self.conflict_rate:
            return Observation(q, 1 - true_value, true_value)

        return Observation(q, true_value, true_value)


# ------------------------------------------------------------
# Bayesian utility
# ------------------------------------------------------------

def entropy(probs):
    return -sum(
        p * math.log2(p)
        for p in probs
        if p > 0
    )


def posterior_after(
    prior: list[float],
    q: int,
    observation: int,
    unknown_rate: float,
    conflict_rate: float,
):
    """
    Correct likelihood model.

    UNKNOWN:
        probability = unknown_rate

    Normal answer:
        (1-unknown_rate) *
        ((1-conflict_rate) if matching else conflict_rate)
    """

    likelihoods = []

    for h in HYPOTHESES:
        true_value = QUESTIONS[q][h]

        if observation == UNKNOWN:
            likelihood = unknown_rate

        else:
            if observation == true_value:
                likelihood = (
                    (1 - unknown_rate) *
                    (1 - conflict_rate)
                )
            else:
                likelihood = (
                    (1 - unknown_rate) *
                    conflict_rate
                )

        likelihoods.append(likelihood)

    weighted = [
        prior[h] * likelihoods[h]
        for h in HYPOTHESES
    ]

    total = sum(weighted)

    if total <= 0:
        return prior[:]

    return [x / total for x in weighted]


def expected_information_gain(
    prior: list[float],
    q: int,
    unknown_rate: float,
    conflict_rate: float,
):
    before = entropy(prior)

    # Possible observations
    outcomes = [0, 1]
    if unknown_rate > 0:
        outcomes.append(UNKNOWN)

    expected_after = 0.0

    for obs in outcomes:
        posterior = posterior_after(
            prior,
            q,
            obs,
            unknown_rate,
            conflict_rate,
        )

        # probability of this observation
        p_obs = 0.0

        for h in HYPOTHESES:
            true_value = QUESTIONS[q][h]

            if obs == UNKNOWN:
                likelihood = unknown_rate

            elif obs == true_value:
                likelihood = (
                    (1 - unknown_rate) *
                    (1 - conflict_rate)
                )

            else:
                likelihood = (
                    (1 - unknown_rate) *
                    conflict_rate
                )

            p_obs += prior[h] * likelihood

        expected_after += p_obs * entropy(posterior)

    return max(0.0, before - expected_after)


# ------------------------------------------------------------
# Base agent
# ------------------------------------------------------------

class AgentBase:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = None
        self.correct = 0
        self.wrong = 0
        self.unknown = 0
        self.conflicts = 0

    def answer(self):
        raise NotImplementedError

    def choose_question(self, unknown_rate, conflict_rate):
        raise NotImplementedError

    def observe(self, q, value):
        raise NotImplementedError


# ------------------------------------------------------------
# BINARY
#
# Only two states:
#
#   0 = eliminated
#   1 = possible
#
# There is NO UNKNOWN state.
#
# This is deliberately strict.
# ------------------------------------------------------------

class BinaryAgent(AgentBase):

    def reset(self):
        self.states = [1] * N_H
        self.correct = 0
        self.wrong = 0
        self.unknown = 0
        self.conflicts = 0

    def possible(self, h):
        return self.states[h] == 1

    def choose_question(self, unknown_rate, conflict_rate):
        possible = [
            h for h in HYPOTHESES
            if self.possible(h)
        ]

        if len(possible) <= 1:
            return None

        # Binary does not know which hypotheses are "unknown".
        # It therefore optimizes using only currently possible ones.
        prior = [
            1.0 / len(possible) if h in possible else 0.0
            for h in HYPOTHESES
        ]

        scores = [
            expected_information_gain(
                prior,
                q,
                unknown_rate,
                conflict_rate,
            )
            for q in range(N_Q)
        ]

        best = max(scores)
        candidates = [
            q for q, score in enumerate(scores)
            if abs(score - best) < 1e-12
        ]

        return self.rng.choice(candidates)

    def observe(self, q, value):
        if value == UNKNOWN:
            # Binary cannot represent UNKNOWN.
            # It therefore receives no usable state update.
            self.unknown += 1
            return

        for h in HYPOTHESES:
            if self.states[h] == 1:
                if QUESTIONS[q][h] != value:
                    self.states[h] = 0

    def answer(self):
        possible = [
            h for h in HYPOTHESES
            if self.possible(h)
        ]

        if len(possible) == 1:
            return possible[0]

        return None


# ------------------------------------------------------------
# TERNARY
#
# Explicit:
#
#   -1 WRONG
#    0 UNKNOWN
#   +1 RIGHT
#
# UNKNOWN is preserved.
# ------------------------------------------------------------

class TernaryAgent(AgentBase):

    def reset(self):
        self.states = [UNKNOWN] * N_H
        self.correct = 0
        self.wrong = 0
        self.unknown = 0
        self.conflicts = 0

    def choose_question(self, unknown_rate, conflict_rate):
        active = [
            h for h in HYPOTHESES
            if self.states[h] != WRONG
        ]

        if len(active) <= 1:
            return None

        # Build posterior from current ternary states.
        confirmed = [
            h for h in active
            if self.states[h] == RIGHT
        ]

        if len(confirmed) == 1:
            return None

        prior = [
            1.0 / len(active) if h in active else 0.0
            for h in HYPOTHESES
        ]

        scores = [
            expected_information_gain(
                prior,
                q,
                unknown_rate,
                conflict_rate,
            )
            for q in range(N_Q)
        ]

        # Explicit UNKNOWN gets priority:
        # questions that distinguish unknown candidates
        # are preferred.
        unknown_candidates = {
            h for h in active
            if self.states[h] == UNKNOWN
        }

        adjusted = []

        for q, score in enumerate(scores):
            split = sum(
                1 for h in unknown_candidates
                if QUESTIONS[q][h] == 1
            )

            adjusted.append(
                score + 0.01 * min(split, len(unknown_candidates) - split)
            )

        best = max(adjusted)

        candidates = [
            q for q, score in enumerate(adjusted)
            if abs(score - best) < 1e-12
        ]

        return self.rng.choice(candidates)

    def observe(self, q, value):
        if value == UNKNOWN:
            self.unknown += 1
            return

        for h in HYPOTHESES:
            if self.states[h] == WRONG:
                continue

            expected = QUESTIONS[q][h]

            if expected == value:
                self.states[h] = RIGHT
            else:
                self.states[h] = WRONG

    def answer(self):
        confirmed = [
            h for h in HYPOTHESES
            if self.states[h] == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


# ------------------------------------------------------------
# QUATERNARY
#
#   -1 WRONG
#    0 UNKNOWN
#   +1 RIGHT
#   +2 CONFLICT
# ------------------------------------------------------------

class QuaternaryAgent(AgentBase):

    def reset(self):
        self.states = [UNKNOWN] * N_H
        self.correct = 0
        self.wrong = 0
        self.unknown = 0
        self.conflicts = 0

    def choose_question(self, unknown_rate, conflict_rate):
        active = [
            h for h in HYPOTHESES
            if self.states[h] != WRONG
        ]

        if len(active) <= 1:
            return None

        confirmed = [
            h for h in active
            if self.states[h] == RIGHT
        ]

        if len(confirmed) == 1:
            return None

        # Conflict gets highest verification priority.
        conflict_candidates = [
            h for h in active
            if self.states[h] == CONFLICT
        ]

        if conflict_candidates:
            # Prefer questions that distinguish conflicts.
            scores = []

            for q in range(N_Q):
                score = sum(
                    1 for h in conflict_candidates
                    if QUESTIONS[q][h] == 1
                )

                scores.append(score)

            best = max(scores)
            candidates = [
                q for q, score in enumerate(scores)
                if score == best
            ]

            return self.rng.choice(candidates)

        prior = [
            1.0 / len(active) if h in active else 0.0
            for h in HYPOTHESES
        ]

        scores = [
            expected_information_gain(
                prior,
                q,
                unknown_rate,
                conflict_rate,
            )
            for q in range(N_Q)
        ]

        best = max(scores)

        candidates = [
            q for q, score in enumerate(scores)
            if abs(score - best) < 1e-12
        ]

        return self.rng.choice(candidates)

    def observe(self, q, value):
        if value == UNKNOWN:
            self.unknown += 1
            return

        for h in HYPOTHESES:
            state = self.states[h]

            if state == WRONG:
                continue

            expected = QUESTIONS[q][h]

            if expected == value:
                if state == WRONG:
                    self.states[h] = CONFLICT
                elif state == CONFLICT:
                    self.states[h] = RIGHT
                else:
                    self.states[h] = RIGHT
            else:
                if state == RIGHT:
                    self.states[h] = CONFLICT
                elif state == CONFLICT:
                    self.states[h] = WRONG
                else:
                    self.states[h] = WRONG

    def answer(self):
        confirmed = [
            h for h in HYPOTHESES
            if self.states[h] == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

def run_episode(
    AgentClass,
    seed,
    budget,
    unknown_rate,
    conflict_rate,
):
    rng = random.Random(seed)

    target = rng.randrange(N_H)

    env = Environment(
        target=target,
        unknown_rate=unknown_rate,
        conflict_rate=conflict_rate,
        rng=rng,
    )

    agent = AgentClass(seed)

    queries = 0
    information = 0.0
    false_confidence = 0
    abstain = 0

    prior = [1.0 / N_H] * N_H

    for step in range(budget):

        # Stop if agent is already confident.
        answer = agent.answer()

        if answer is not None:
            break

        q = agent.choose_question(
            unknown_rate,
            conflict_rate,
        )

        if q is None:
            break

        information += expected_information_gain(
            prior,
            q,
            unknown_rate,
            conflict_rate,
        )

        obs = env.ask(q)

        queries += 1

        if obs.value == UNKNOWN:
            agent.observe(q, UNKNOWN)
        else:
            prior = posterior_after(
                prior,
                q,
                obs.value,
                unknown_rate,
                conflict_rate,
            )

            agent.observe(q, obs.value)

    answer = agent.answer()

    if answer is None:
        abstain = 1
        correct = 0
    else:
        correct = int(answer == target)
        if not correct:
            false_confidence = 1

    # Simple reward:
    # correct = +1
    # wrong = -5
    # abstain = -0.2
    # query = -0.05
    reward = (
        (1.0 if correct else -5.0 if answer is not None else -0.2)
        - 0.05 * queries
    )

    return {
        "correct": correct,
        "queries": queries,
        "information": information,
        "false_confidence": false_confidence,
        "abstain": abstain,
        "reward": reward,
    }


def aggregate(rows):
    n = len(rows)

    return {
        key: sum(r[key] for r in rows) / n
        for key in rows[0]
    }


def main():
    out = []

    for unknown_rate in UNKNOWN_RATES:
        for conflict_rate in CONFLICT_RATES:

            print()
            print(
                f"UNKNOWN={unknown_rate:.1f} "
                f"CONFLICT={conflict_rate:.1f}"
            )

            for budget in BUDGETS:

                results = {}

                for name, AgentClass in [
                    ("BINARY", BinaryAgent),
                    ("TERNARY", TernaryAgent),
                    ("QUATERNARY", QuaternaryAgent),
                ]:

                    rows = []

                    for episode in range(EPISODES):
                        seed = (
                            episode
                            + budget * 100000
                            + int(unknown_rate * 10000)
                            + int(conflict_rate * 1000)
                        )

                        rows.append(
                            run_episode(
                                AgentClass,
                                seed,
                                budget,
                                unknown_rate,
                                conflict_rate,
                            )
                        )

                    results[name] = aggregate(rows)

                    r = results[name]

                    print(
                        f"{name:10s} "
                        f"b={budget:2d} "
                        f"acc={r['correct']:.4f} "
                        f"q={r['queries']:.3f} "
                        f"info={r['information']:.4f} "
                        f"false={r['false_confidence']:.4f} "
                        f"abst={r['abstain']:.4f} "
                        f"reward={r['reward']:.4f}"
                    )

                    out.append({
                        "unknown_rate": unknown_rate,
                        "conflict_rate": conflict_rate,
                        "budget": budget,
                        "agent": name,
                        **r,
                    })

    path = "results/test16_2/results.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=out[0].keys(),
        )
        writer.writeheader()
        writer.writerows(out)

    print()
    print("=" * 70)
    print("SAVED:", path)
    print("=" * 70)


if __name__ == "__main__":
    main()
