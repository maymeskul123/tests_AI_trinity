from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# TEST 15 FIXED
#
# Active learning with:
#   - multiple hypotheses
#   - multiple questions
#   - explicit epistemic states
#
# Binary:
#   0 = eliminated
#   1 = possible
#
# Ternary:
#  -1 = wrong/eliminated
#   0 = unknown
#  +1 = confirmed
#
# Quaternary:
#  -1 = wrong/eliminated
#   0 = unknown
#  +1 = confirmed
#   +2 = conflict
#
# IMPORTANT:
# All agents use the SAME observation history.
# State is derived from the complete history.
# ============================================================

N_HYPOTHESES = 8
N_QUESTIONS = 3
EPISODES = 5000

BUDGETS = [1, 2, 3, 4, 5, 6, 8, 10]
UNKNOWN_RATES = [0.0, 0.2, 0.4]
CONFLICT_RATES = [0.0, 0.1, 0.2]

REWARD_CORRECT = 1.0
PENALTY_WRONG = -5.0
QUERY_COST = -0.10
ABSTAIN_COST = -0.20


# ============================================================
# Hypothesis space
#
# Each hypothesis specifies answers to Q0,Q1,Q2.
# ============================================================

HYPOTHESES = {
    0: (0, 0, 0),
    1: (0, 1, 1),
    2: (1, 0, 1),
    3: (1, 1, 0),
    4: (0, 0, 1),
    5: (0, 1, 0),
    6: (1, 0, 0),
    7: (1, 1, 1),
}


@dataclass(frozen=True)
class Observation:
    question: int
    answer: int | None


# ============================================================
# Environment
# ============================================================

class Environment:

    def __init__(
        self,
        rng: random.Random,
        unknown_rate: float,
        conflict_rate: float,
    ):
        self.rng = rng
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.target = rng.randrange(N_HYPOTHESES)

    def ask(self, question: int) -> int | None:

        true_answer = HYPOTHESES[self.target][question]

        # UNKNOWN feedback
        if self.rng.random() < self.unknown_rate:
            return None

        # CONFLICT / noisy feedback
        if self.rng.random() < self.conflict_rate:
            return 1 - true_answer

        return true_answer


# ============================================================
# Common history
# ============================================================

class HistoryAgent:

    def __init__(self):
        self.observations: list[Observation] = []

    def add_observation(
        self,
        question: int,
        answer: int | None,
    ):
        self.observations.append(
            Observation(question, answer)
        )


# ============================================================
# Hypothesis compatibility
#
# A hypothesis is compatible with ALL known observations.
#
# UNKNOWN gives no information.
# ============================================================

def compatible(
    hypothesis: int,
    observations: list[Observation],
) -> bool:

    for obs in observations:

        if obs.answer is None:
            continue

        expected = HYPOTHESES[hypothesis][obs.question]

        if expected != obs.answer:
            return False

    return True


def possible_hypotheses(
    observations: list[Observation],
) -> list[int]:

    return [
        h
        for h in range(N_HYPOTHESES)
        if compatible(h, observations)
    ]


# ============================================================
# Bayesian posterior
#
# Used for measuring information, not for agent state.
# ============================================================

def likelihood(
    hypothesis: int,
    observations: list[Observation],
    unknown_rate: float,
    conflict_rate: float,
) -> float:

    p = 1.0

    for obs in observations:

        predicted = HYPOTHESES[hypothesis][obs.question]

        if obs.answer is None:

            p *= max(
                unknown_rate,
                1e-9,
            )

        elif obs.answer == predicted:

            p *= (
                max(1.0 - unknown_rate, 1e-9)
                *
                max(1.0 - conflict_rate, 1e-9)
            )

        else:

            p *= (
                max(1.0 - unknown_rate, 1e-9)
                *
                max(conflict_rate, 1e-9)
            )

    return p


def posterior(
    observations: list[Observation],
    unknown_rate: float,
    conflict_rate: float,
) -> list[float]:

    values = [
        likelihood(
            h,
            observations,
            unknown_rate,
            conflict_rate,
        )
        for h in range(N_HYPOTHESES)
    ]

    total = sum(values)

    if total <= 0:
        return [
            1.0 / N_HYPOTHESES
            for _ in range(N_HYPOTHESES)
        ]

    return [
        v / total
        for v in values
    ]


def entropy(probs: list[float]) -> float:

    return -sum(
        p * math.log2(p)
        for p in probs
        if p > 0
    )


def information_gain(
    observations: list[Observation],
    question: int,
    unknown_rate: float,
    conflict_rate: float,
) -> float:

    before = posterior(
        observations,
        unknown_rate,
        conflict_rate,
    )

    h_before = entropy(before)

    # Expected entropy after answer 0/1.
    expected = 0.0

    for answer in (0, 1):

        extended = observations + [
            Observation(question, answer)
        ]

        after = posterior(
            extended,
            unknown_rate,
            conflict_rate,
        )

        # Estimate probability of this answer
        p_answer = 0.0

        for h, p in enumerate(before):

            predicted = HYPOTHESES[h][question]

            if conflict_rate == 0:
                if predicted == answer:
                    p_answer += p
            else:
                if predicted == answer:
                    p_answer += (
                        p
                        * (1.0 - conflict_rate)
                    )
                else:
                    p_answer += (
                        p
                        * conflict_rate
                    )

        expected += p_answer * entropy(after)

    # UNKNOWN branch
    if unknown_rate > 0:

        expected += (
            unknown_rate
            * h_before
        )

    return max(
        0.0,
        h_before - expected,
    )


# ============================================================
# Question selection
#
# All agents operate on the same hypothesis history.
# ============================================================

def choose_best_question(
    observations: list[Observation],
    unknown_rate: float,
    conflict_rate: float,
    candidate_questions: list[int] | None = None,
) -> int:

    if candidate_questions is None:
        candidate_questions = list(
            range(N_QUESTIONS)
        )

    scores = {
        q: information_gain(
            observations,
            q,
            unknown_rate,
            conflict_rate,
        )
        for q in candidate_questions
    }

    return max(
        candidate_questions,
        key=lambda q: scores[q],
    )


# ============================================================
# BINARY AGENT
#
# IMPORTANT:
# Binary has NO explicit UNKNOWN state.
#
# 0 = eliminated
# 1 = possible
#
# Its complete history is still available separately.
# ============================================================

class BinaryAgent(HistoryAgent):

    def state(self, hypothesis: int) -> int:

        return int(
            compatible(
                hypothesis,
                self.observations,
            )
        )

    def choose_question(
        self,
        unknown_rate: float,
        conflict_rate: float,
    ) -> int:

        return choose_best_question(
            self.observations,
            unknown_rate,
            conflict_rate,
        )

    def answer(self) -> int | None:

        candidates = possible_hypotheses(
            self.observations
        )

        if len(candidates) == 1:
            return candidates[0]

        return None


# ============================================================
# TERNARY AGENT
#
# -1 = wrong
#  0 = unknown
# +1 = confirmed
#
# State is derived from COMPLETE history.
# ============================================================

class TernaryAgent(HistoryAgent):

    WRONG = -1
    UNKNOWN = 0
    RIGHT = 1

    def state(self, hypothesis: int) -> int:

        # No information about hypothesis.
        known = [
            obs
            for obs in self.observations
            if obs.answer is not None
        ]

        if not known:
            return self.UNKNOWN

        # Any contradiction eliminates it.
        if not compatible(
            hypothesis,
            self.observations,
        ):
            return self.WRONG

        # If hypothesis is the only compatible
        # hypothesis, it is confirmed.
        candidates = possible_hypotheses(
            self.observations
        )

        if (
            len(candidates) == 1
            and candidates[0] == hypothesis
        ):
            return self.RIGHT

        return self.UNKNOWN

    def choose_question(
        self,
        unknown_rate: float,
        conflict_rate: float,
    ) -> int:

        return choose_best_question(
            self.observations,
            unknown_rate,
            conflict_rate,
        )

    def answer(self) -> int | None:

        candidates = possible_hypotheses(
            self.observations
        )

        if len(candidates) == 1:
            return candidates[0]

        return None


# ============================================================
# QUATERNARY AGENT
#
# -1 = wrong
#  0 = unknown
# +1 = confirmed
# +2 = conflict
#
# Conflict means that the same hypothesis has received
# contradictory evidence across the history.
# ============================================================

class QuaternaryAgent(HistoryAgent):

    WRONG = -1
    UNKNOWN = 0
    RIGHT = 1
    CONFLICT = 2

    def state(self, hypothesis: int) -> int:

        known = [
            obs
            for obs in self.observations
            if obs.answer is not None
        ]

        if not known:
            return self.UNKNOWN

        # Detect contradictory evidence specifically
        # for this hypothesis.
        positive = False
        negative = False

        for obs in known:

            expected = HYPOTHESES[hypothesis][obs.question]

            if obs.answer == expected:
                positive = True
            else:
                negative = True

        if positive and negative:
            return self.CONFLICT

        if negative:
            return self.WRONG

        candidates = possible_hypotheses(
            self.observations
        )

        if (
            len(candidates) == 1
            and candidates[0] == hypothesis
        ):
            return self.RIGHT

        return self.UNKNOWN

    def choose_question(
        self,
        unknown_rate: float,
        conflict_rate: float,
    ) -> int:

        # First try to resolve conflicts.
        conflict_hypotheses = [
            h
            for h in range(N_HYPOTHESES)
            if self.state(h) == self.CONFLICT
        ]

        if conflict_hypotheses:

            scores = []

            for q in range(N_QUESTIONS):

                score = 0.0

                for h in conflict_hypotheses:

                    expected = HYPOTHESES[h][q]

                    score += (
                        1.0
                        if expected == 0
                        else 0.5
                    )

                score += information_gain(
                    self.observations,
                    q,
                    unknown_rate,
                    conflict_rate,
                )

                scores.append(score)

            return max(
                range(N_QUESTIONS),
                key=lambda q: scores[q],
            )

        return choose_best_question(
            self.observations,
            unknown_rate,
            conflict_rate,
        )

    def answer(self) -> int | None:

        # Never answer while any candidate is in conflict.
        if any(
            self.state(h) == self.CONFLICT
            for h in range(N_HYPOTHESES)
        ):
            return None

        candidates = possible_hypotheses(
            self.observations
        )

        if len(candidates) == 1:
            return candidates[0]

        return None


# ============================================================
# Episode
# ============================================================

def run_episode(
    agent_cls,
    seed: int,
    budget: int,
    unknown_rate: float,
    conflict_rate: float,
):

    rng = random.Random(seed)

    env = Environment(
        rng,
        unknown_rate,
        conflict_rate,
    )

    agent = agent_cls()

    reward = 0.0
    queries = 0
    total_information = 0.0
    abstain = 0

    for _ in range(budget):

        prediction = agent.answer()

        if prediction is not None:
            break

        question = agent.choose_question(
            unknown_rate,
            conflict_rate,
        )

        ig = information_gain(
            agent.observations,
            question,
            unknown_rate,
            conflict_rate,
        )

        total_information += ig

        answer = env.ask(question)

        agent.add_observation(
            question,
            answer,
        )

        queries += 1
        reward += QUERY_COST

    prediction = agent.answer()

    if prediction is None:

        abstain = 1
        reward += ABSTAIN_COST

        probs = posterior(
            agent.observations,
            unknown_rate,
            conflict_rate,
        )

        prediction = max(
            range(N_HYPOTHESES),
            key=lambda h: probs[h],
        )

    correct = int(
        prediction == env.target
    )

    if correct:
        reward += REWARD_CORRECT
    else:
        reward += PENALTY_WRONG

    false_confidence = int(
        prediction != env.target
    )

    return {
        "correct": correct,
        "reward": reward,
        "queries": queries,
        "information": total_information,
        "abstain": abstain,
        "false_confidence": false_confidence,
    }


# ============================================================
# Run condition
# ============================================================

def run_condition(
    agent_cls,
    unknown_rate: float,
    conflict_rate: float,
    budget: int,
):

    results = []

    for seed in range(EPISODES):

        results.append(
            run_episode(
                agent_cls,
                seed,
                budget,
                unknown_rate,
                conflict_rate,
            )
        )

    n = len(results)

    return {
        "accuracy": sum(
            r["correct"]
            for r in results
        ) / n,

        "reward": sum(
            r["reward"]
            for r in results
        ) / n,

        "queries": sum(
            r["queries"]
            for r in results
        ) / n,

        "information": sum(
            r["information"]
            for r in results
        ) / n,

        "abstain": sum(
            r["abstain"]
            for r in results
        ) / n,

        "false_confidence": sum(
            r["false_confidence"]
            for r in results
        ) / n,
    }


# ============================================================
# Main
# ============================================================

def main():

    output_dir = Path(
        "results/test15"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "results.csv"
    )

    agents = {
        "BINARY": BinaryAgent,
        "TERNARY": TernaryAgent,
        "QUATERNARY": QuaternaryAgent,
    }

    fieldnames = [
        "agent",
        "unknown_rate",
        "conflict_rate",
        "budget",
        "accuracy",
        "reward",
        "queries",
        "information",
        "abstain",
        "false_confidence",
    ]

    rows = []

    print("=" * 100)
    print("TEST 15 FIXED")
    print("Active learning with complete history")
    print("=" * 100)

    for unknown_rate in UNKNOWN_RATES:

        for conflict_rate in CONFLICT_RATES:

            print(
                f"\nUNKNOWN={unknown_rate:.1f} "
                f"CONFLICT={conflict_rate:.1f}"
            )

            for budget in BUDGETS:

                print(
                    f"\n  BUDGET={budget}"
                )

                for name, cls in agents.items():

                    result = run_condition(
                        cls,
                        unknown_rate,
                        conflict_rate,
                        budget,
                    )

                    row = {
                        "agent": name,
                        "unknown_rate": unknown_rate,
                        "conflict_rate": conflict_rate,
                        "budget": budget,
                        "accuracy": result[
                            "accuracy"
                        ],
                        "reward": result[
                            "reward"
                        ],
                        "queries": result[
                            "queries"
                        ],
                        "information": result[
                            "information"
                        ],
                        "abstain": result[
                            "abstain"
                        ],
                        "false_confidence": result[
                            "false_confidence"
                        ],
                    }

                    rows.append(row)

                    print(
                        f"    {name:10s} "
                        f"acc={result['accuracy']:.4f} "
                        f"reward={result['reward']:.4f} "
                        f"queries={result['queries']:.3f} "
                        f"IG={result['information']:.4f} "
                        f"abstain={result['abstain']:.4f} "
                        f"false={result['false_confidence']:.4f}"
                    )

    # ========================================================
    # ONLY CSV
    # ========================================================

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"Saved: {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()

