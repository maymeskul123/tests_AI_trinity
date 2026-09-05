from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# TEST 15.1
# Explicit state-driven active learning
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
#   2 = CONFLICT
#
# IMPORTANT:
# Unlike Test 15, the internal state representation itself
# participates in choosing the next question.
# ============================================================

SEED = 42
EPISODES = 5000

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

N_HYPOTHESES = len(HYPOTHESES)
QUESTIONS = tuple(range(3))

BUDGETS = [1, 2, 3, 4, 5, 6, 8, 10]
UNKNOWN_RATES = [0.0, 0.2, 0.4]
CONFLICT_RATES = [0.0, 0.1, 0.2]

RIGHT = 1
WRONG = -1
UNKNOWN = 0
CONFLICT = 2

QUERY_COST = 0.10
ABSTAIN_COST = 0.20
WRONG_COST = 5.0


@dataclass
class Result:
    agent: str
    unknown_rate: float
    conflict_rate: float
    budget: int
    accuracy: float
    reward: float
    queries: float
    information: float
    abstain: float
    false_confidence: float


# ============================================================
# Environment
# ============================================================

def observe(
    true_h: int,
    question: int,
    unknown_rate: float,
    conflict_rate: float,
    rng: random.Random,
) -> int:

    true_answer = HYPOTHESES[true_h][question]

    if rng.random() < unknown_rate:
        return UNKNOWN

    answer = true_answer

    if rng.random() < conflict_rate:
        answer = 1 - answer

    return answer


# ============================================================
# Information theory
# ============================================================

def entropy(probs):
    return -sum(
        p * math.log2(p)
        for p in probs
        if p > 0
    )


def posterior_from_history(history):
    """
    Exact Bayesian posterior under the actual observation model.

    history = [(question, observed_answer), ...]

    We assume:
      - uniform prior
      - UNKNOWN probability = unknown_rate
      - binary answer flip probability = conflict_rate

    The rates are supplied globally by information_gain().
    """
    raise RuntimeError("Use posterior_from_history_with_noise")


def likelihood(observation, expected, unknown_rate, conflict_rate):
    """
    P(observation | expected answer)
    """

    if observation == UNKNOWN:
        return unknown_rate

    if observation == expected:
        return (1.0 - unknown_rate) * (1.0 - conflict_rate)

    return (1.0 - unknown_rate) * conflict_rate


def posterior_from_history_with_noise(
    history,
    unknown_rate,
    conflict_rate,
):
    weights = []

    for h in HYPOTHESES:
        p = 1.0

        for q, obs in history:
            expected = HYPOTHESES[h][q]
            p *= likelihood(
                obs,
                expected,
                unknown_rate,
                conflict_rate,
            )

        weights.append(p)

    total = sum(weights)

    if total == 0:
        return [1.0 / N_HYPOTHESES] * N_HYPOTHESES

    return [w / total for w in weights]


def expected_information_gain(
    history,
    question,
    unknown_rate,
    conflict_rate,
):
    """
    Proper expected mutual information:

        IG = H(H) - E_o[H(H | o)]

    This fixes the Test 15 information metric.
    """

    prior = posterior_from_history_with_noise(
        history,
        unknown_rate,
        conflict_rate,
    )

    prior_entropy = entropy(prior)

    outcomes = (0, 1, UNKNOWN)

    expected_posterior_entropy = 0.0

    for outcome in outcomes:

        p_outcome = 0.0

        for h, p_h in enumerate(prior):
            expected = HYPOTHESES[h][question]

            p_obs = likelihood(
                outcome,
                expected,
                unknown_rate,
                conflict_rate,
            )

            p_outcome += p_h * p_obs

        if p_outcome == 0:
            continue

        posterior_weights = []

        for h, p_h in enumerate(prior):
            expected = HYPOTHESES[h][question]

            p_obs = likelihood(
                outcome,
                expected,
                unknown_rate,
                conflict_rate,
            )

            posterior_weights.append(
                p_h * p_obs
            )

        z = sum(posterior_weights)

        posterior = [
            x / z
            for x in posterior_weights
        ]

        expected_posterior_entropy += (
            p_outcome * entropy(posterior)
        )

    return max(
        0.0,
        prior_entropy - expected_posterior_entropy
    )


# ============================================================
# State-driven agents
# ============================================================

class BaseAgent:

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.history = []

    def observe(self, question, answer):
        self.history.append((question, answer))

    def reset(self):
        self.history.clear()

    def answer(self):
        raise NotImplementedError

    def choose_question(
        self,
        unknown_rate,
        conflict_rate,
        asked,
    ):
        raise NotImplementedError


class BinaryAgent(BaseAgent):

    name = "BINARY"

    def states(self):
        """
        Binary representation:
            possible / eliminated
        """

        possible = []

        for h in HYPOTHESES:
            compatible = True

            for q, obs in self.history:
                if obs == UNKNOWN:
                    continue

                if HYPOTHESES[h][q] != obs:
                    compatible = False
                    break

            if compatible:
                possible.append(h)

        return {
            h: 1 if h in possible else 0
            for h in HYPOTHESES
        }

    def choose_question(
        self,
        unknown_rate,
        conflict_rate,
        asked,
    ):

        states = self.states()

        possible = [
            h for h, s in states.items()
            if s == 1
        ]

        # Binary policy:
        # if uncertainty remains, maximize split of POSSIBLE states.
        best = None

        for q in QUESTIONS:

            if q in asked:
                continue

            zeros = sum(
                HYPOTHESES[h][q] == 0
                for h in possible
            )

            ones = sum(
                HYPOTHESES[h][q] == 1
                for h in possible
            )

            if zeros + ones == 0:
                score = 0
            else:
                score = min(zeros, ones)

            candidate = (score, -q, q)

            if best is None or candidate > best:
                best = candidate

        if best is None:
            return None

        return best[-1]

    def answer(self):

        states = self.states()

        possible = [
            h for h, s in states.items()
            if s == 1
        ]

        if len(possible) == 1:
            return possible[0]

        return None


class TernaryAgent(BaseAgent):

    name = "TERNARY"

    def states(self):

        result = {}

        compatible = []

        for h in HYPOTHESES:

            ok = True
            has_evidence = False

            for q, obs in self.history:

                if obs == UNKNOWN:
                    continue

                has_evidence = True

                if HYPOTHESES[h][q] != obs:
                    ok = False
                    break

            if ok:
                compatible.append(h)

        for h in HYPOTHESES:

            if h not in compatible:
                result[h] = WRONG

            elif len(compatible) == 1:
                result[h] = RIGHT

            else:
                result[h] = UNKNOWN

        return result

    def choose_question(
        self,
        unknown_rate,
        conflict_rate,
        asked,
    ):

        states = self.states()

        unknown_hypotheses = [
            h
            for h, state in states.items()
            if state == UNKNOWN
        ]

        # Ternary policy:
        # prioritize questions that split UNKNOWN hypotheses.
        candidate_questions = []

        for q in QUESTIONS:

            if q in asked:
                continue

            zeros = sum(
                HYPOTHESES[h][q] == 0
                for h in unknown_hypotheses
            )

            ones = sum(
                HYPOTHESES[h][q] == 1
                for h in unknown_hypotheses
            )

            # Prefer balanced resolution of UNKNOWN.
            split = min(zeros, ones)

            # Also include proper expected information.
            ig = expected_information_gain(
                self.history,
                q,
                unknown_rate,
                conflict_rate,
            )

            candidate_questions.append(
                (
                    split,
                    ig,
                    -q,
                    q,
                )
            )

        if not candidate_questions:
            return None

        return max(candidate_questions)[-1]

    def answer(self):

        states = self.states()

        confirmed = [
            h
            for h, state in states.items()
            if state == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


class QuaternaryAgent(BaseAgent):

    name = "QUATERNARY"

    def states(self):

        result = {}

        for h in HYPOTHESES:

            has_right = False
            has_wrong = False

            for q, obs in self.history:

                if obs == UNKNOWN:
                    continue

                if HYPOTHESES[h][q] == obs:
                    has_right = True
                else:
                    has_wrong = True

            if has_right and has_wrong:
                result[h] = CONFLICT

            elif has_wrong:
                result[h] = WRONG

            elif has_right:
                result[h] = RIGHT

            else:
                result[h] = UNKNOWN

        return result

    def choose_question(
        self,
        unknown_rate,
        conflict_rate,
        asked,
    ):

        states = self.states()

        # Quaternary policy:
        # first resolve conflicts,
        # then resolve unknowns.

        conflict_h = [
            h
            for h, state in states.items()
            if state == CONFLICT
        ]

        unknown_h = [
            h
            for h, state in states.items()
            if state == UNKNOWN
        ]

        target = (
            conflict_h
            if conflict_h
            else unknown_h
        )

        candidates = []

        for q in QUESTIONS:

            if q in asked:
                continue

            split = min(
                sum(
                    HYPOTHESES[h][q] == 0
                    for h in target
                ),
                sum(
                    HYPOTHESES[h][q] == 1
                    for h in target
                ),
            )

            ig = expected_information_gain(
                self.history,
                q,
                unknown_rate,
                conflict_rate,
            )

            candidates.append(
                (
                    split,
                    ig,
                    -q,
                    q,
                )
            )

        if not candidates:
            return None

        return max(candidates)[-1]

    def answer(self):

        states = self.states()

        confirmed = [
            h
            for h, state in states.items()
            if state == RIGHT
        ]

        if len(confirmed) == 1:
            return confirmed[0]

        return None


# ============================================================
# Episode
# ============================================================

def run_episode(
    agent_class,
    budget,
    unknown_rate,
    conflict_rate,
    seed,
):

    agent = agent_class(seed)

    true_h = random.Random(seed + 100000).randrange(
        N_HYPOTHESES
    )

    asked = set()
    queries = 0

    for _ in range(budget):

        prediction = agent.answer()

        if prediction is not None:
            break

        q = agent.choose_question(
            unknown_rate,
            conflict_rate,
            asked,
        )

        if q is None:
            break

        asked.add(q)
        queries += 1

        obs = observe(
            true_h,
            q,
            unknown_rate,
            conflict_rate,
            agent.rng,
        )

        agent.observe(q, obs)

    prediction = agent.answer()

    correct = (
        prediction == true_h
        if prediction is not None
        else False
    )

    abstain = prediction is None
    false_confidence = (
        prediction is not None
        and prediction != true_h
    )

    reward = 0.0

    reward -= queries * QUERY_COST

    if abstain:
        reward -= ABSTAIN_COST
    elif correct:
        reward += 1.0
    else:
        reward -= WRONG_COST

    information = 0.0

    for q in QUESTIONS:

        if q in asked:
            continue

        # Information actually available before this question.
        pass

    # Total posterior information gained from the history.
    posterior = posterior_from_history_with_noise(
        agent.history,
        unknown_rate,
        conflict_rate,
    )

    information = (
        math.log2(N_HYPOTHESES)
        - entropy(posterior)
    )

    return {
        "accuracy": 1.0 if correct else 0.0,
        "reward": reward,
        "queries": queries,
        "information": information,
        "abstain": 1.0 if abstain else 0.0,
        "false_confidence": 1.0 if false_confidence else 0.0,
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment():

    out = Path("results/test15_1/results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    agents = [
        BinaryAgent,
        TernaryAgent,
        QuaternaryAgent,
    ]

    for unknown_rate in UNKNOWN_RATES:

        for conflict_rate in CONFLICT_RATES:

            for budget in BUDGETS:

                for agent_class in agents:

                    metrics = {
                        "accuracy": [],
                        "reward": [],
                        "queries": [],
                        "information": [],
                        "abstain": [],
                        "false_confidence": [],
                    }

                    for episode in range(EPISODES):

                        result = run_episode(
                            agent_class,
                            budget,
                            unknown_rate,
                            conflict_rate,
                            SEED
                            + episode
                            + budget * 10000
                            + int(unknown_rate * 1000) * 100000
                            + int(conflict_rate * 1000) * 1000000,
                        )

                        for key in metrics:
                            metrics[key].append(
                                result[key]
                            )

                    avg = {
                        key: sum(values) / len(values)
                        for key, values in metrics.items()
                    }

                    rows.append({
                        "agent": agent_class.name,
                        "unknown_rate": unknown_rate,
                        "conflict_rate": conflict_rate,
                        "budget": budget,
                        "accuracy": round(
                            avg["accuracy"], 6
                        ),
                        "reward": round(
                            avg["reward"], 6
                        ),
                        "queries": round(
                            avg["queries"], 6
                        ),
                        "information": round(
                            avg["information"], 6
                        ),
                        "abstain": round(
                            avg["abstain"], 6
                        ),
                        "false_confidence": round(
                            avg["false_confidence"], 6
                        ),
                    })

    with out.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("TEST 15.1 COMPLETE")
    print("=" * 72)
    print()
    print(f"Results: {out}")
    print()

    # Compact comparison
    for unknown_rate in UNKNOWN_RATES:
        for conflict_rate in CONFLICT_RATES:

            print(
                f"UNKNOWN={unknown_rate:.1f} "
                f"CONFLICT={conflict_rate:.1f}"
            )

            for budget in BUDGETS:

                subset = [
                    r for r in rows
                    if r["unknown_rate"] == unknown_rate
                    and r["conflict_rate"] == conflict_rate
                    and r["budget"] == budget
                ]

                parts = []

                for r in subset:
                    parts.append(
                        f"{r['agent'][0]}:"
                        f"{r['accuracy']:.3f}"
                        f"/{r['reward']:.3f}"
                        f"/q{r['queries']:.2f}"
                    )

                print(
                    f"  BUDGET {budget:2d}: "
                    + "  ".join(parts)
                )

            print()


if __name__ == "__main__":
    run_experiment()
