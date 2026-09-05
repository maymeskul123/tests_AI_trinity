from __future__ import annotations
import csv, math, random
from pathlib import Path
from collections import defaultdict

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

AGENTS = ("binary", "ternary", "quaternary")

N_CANDIDATES = 10
EPISODES = 5000
QUERY_BUDGETS = (1, 2, 3, 4, 5, 6, 8, 10)

UNKNOWN_RATES = (0.0, 0.2, 0.4)
CONFLICT_RATES = (0.0, 0.1, 0.2)

REWARD_CORRECT = 1.0
PENALTY_WRONG = -5.0
COST_QUERY = -0.1
PENALTY_ABSTAIN = -0.2

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "test13"
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

        if self.rng.random() < self.unknown_rate:
            return UNKNOWN

        correct = candidate == answer

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
        raise NotImplementedError

    def score(self, candidate):
        raise NotImplementedError

    def confidence(self, candidate):
        raise NotImplementedError

    def possible_candidates(self):
        return [
            c for c in range(N_CANDIDATES)
            if self.state(c) != WRONG
        ]

    def choose_query(self):
        possible = self.possible_candidates()

        if not possible:
            return self.rng.randrange(N_CANDIDATES)

        unresolved = [
            c for c in possible
            if self.state(c) in (UNKNOWN, CONFLICT)
        ]

        if unresolved:
            return self.rng.choice(unresolved)

        scores = {c: self.score(c) for c in possible}
        minimum = min(scores.values())

        weakest = [
            c for c, score in scores.items()
            if score == minimum
        ]

        return self.rng.choice(weakest)

    def answer(self):
        confident = [
            c for c in range(N_CANDIDATES)
            if self.confidence(c)
        ]

        if not confident:
            return None

        scores = {c: self.score(c) for c in confident}
        best = max(scores.values())

        candidates = [
            c for c, score in scores.items()
            if score == best
        ]

        return self.rng.choice(candidates)


class BinaryAgent(BaseAgent):
    """
    Binary epistemic memory:
    only RIGHT/WRONG evidence is stored.
    Missing history is not a stored third state.
    """

    def state(self, candidate):
        history = self.history[candidate]

        if not history:
            return UNKNOWN

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

        if rights > wrongs:
            return RIGHT

        return WRONG

    def score(self, candidate):
        history = self.history[candidate]
        return history.count(RIGHT) - history.count(WRONG)

    def confidence(self, candidate):
        history = self.history[candidate]

        if not history:
            return False

        return (
            history.count(RIGHT)
            > history.count(WRONG)
        )


class TernaryAgent(BaseAgent):
    """
    Explicit:
    -1 WRONG
     0 UNKNOWN
    +1 RIGHT
    """

    def state(self, candidate):
        history = self.history[candidate]

        if not history:
            return UNKNOWN

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

        if rights > wrongs:
            return RIGHT

        if wrongs > rights:
            return WRONG

        return UNKNOWN

    def score(self, candidate):
        state = self.state(candidate)

        if state == RIGHT:
            return 1.0

        if state == WRONG:
            return -1.0

        return 0.0

    def confidence(self, candidate):
        return self.state(candidate) == RIGHT


class QuaternaryAgent(BaseAgent):
    """
    -1 WRONG
     0 UNKNOWN
    +1 RIGHT
     2 CONFLICT
    """

    def state(self, candidate):
        history = self.history[candidate]

        if not history:
            return UNKNOWN

        rights = history.count(RIGHT)
        wrongs = history.count(WRONG)

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
            return 1.0

        if state in (UNKNOWN, CONFLICT):
            return 0.0

        return -1.0

    def confidence(self, candidate):
        return self.state(candidate) == RIGHT


def make_agent(name, rng):
    if name == "binary":
        return BinaryAgent(rng)

    if name == "ternary":
        return TernaryAgent(rng)

    if name == "quaternary":
        return QuaternaryAgent(rng)

    raise ValueError(name)


def run_episode(
    agent_name,
    unknown_rate,
    conflict_rate,
    max_queries,
    seed,
):
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
    information_gain = 0.0

    false_confidence = 0
    abstained = 0
    contradictions = 0
    recoveries = 0

    while queries < max_queries:

        prediction = agent.answer()

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

        possible_before = set(
            agent.possible_candidates()
        )

        H_before = entropy(len(possible_before))

        candidate = agent.choose_query()

        old_state = agent.state(candidate)

        feedback = env.query(
            task_id,
            candidate,
        )

        queries += 1
        reward += COST_QUERY

        agent.observe(candidate, feedback)

        new_state = agent.state(candidate)

        history = agent.history[candidate]

        if (
            RIGHT in history
            and WRONG in history
        ):
            contradictions += 1

        if (
            old_state == CONFLICT
            and new_state == RIGHT
        ):
            recoveries += 1

        possible_after = set(
            agent.possible_candidates()
        )

        H_after = entropy(len(possible_after))

        information_gain += max(
            0.0,
            H_before - H_after,
        )

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
        "information_gain": information_gain,
        "false_confidence": false_confidence,
        "abstained": 0,
        "contradictions": contradictions,
        "recoveries": recoveries,
    }


def aggregate(rows):
    n = len(rows)

    total_queries = sum(
        r["queries"] for r in rows
    )

    total_information = sum(
        r["information_gain"] for r in rows
    )

    total_reward = sum(
        r["reward"] for r in rows
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

                for budget in QUERY_BUDGETS:

                    for agent_index, agent_name in enumerate(AGENTS):

                        rows = []

                        for episode in range(EPISODES):

                            seed = (
                                episode
                                + int(unknown_rate * 1000) * 1000000
                                + int(conflict_rate * 1000) * 10000
                                + budget * 1000
                                + agent_index * 100000000
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
                            f"{agent_name:12s} "
                            f"unknown={unknown_rate:.1f} "
                            f"conflict={conflict_rate:.1f} "
                            f"budget={budget:2d} "
                            f"acc={result['accuracy']:.4f} "
                            f"queries={result['avg_queries']:.3f} "
                            f"reward={result['avg_reward']:.4f} "
                            f"IG/Q={result['information_gain_per_query']:.4f}"
                        )

    print()
    print("=" * 80)
    print("TEST 13 COMPLETE")
    print("=" * 80)
    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
