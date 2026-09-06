import csv
import random
from collections import defaultdict

from core.environment import Environment


# ============================================================
# TEST04
# Binary vs Ternary vs Quaternary under contradiction
#
# Two complementary benchmarks:
#
# 1. NOISY
#    Ordinary noisy environment.
#    Measures decision quality, false confidence,
#    abstention and target survival.
#
# 2. RECOVERY
#    Deterministic contradiction trajectories.
#    The agent is NOT allowed to stop after an early decision.
#    We explicitly observe state transitions:
#
#       RIGHT -> WRONG
#       RIGHT -> UNKNOWN -> RIGHT
#       RIGHT -> CONFLICT -> RIGHT
#
# This makes recovery a real state-transition measurement
# instead of the old "target_ever_conflict and final correct"
# proxy.
# ============================================================


N_H = 16
N_Q = 8

EPISODES = 2000
RECOVERY_EPISODES = 2000

BUDGETS = [5, 7, 10]
CONFLICT_RATES = [0.0, 0.2, 0.4]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

REWARD_CORRECT = 1.0
REWARD_FALSE = -100.0
REWARD_ABSTAIN = -0.2
QUERY_COST = 0.05

HYPOTHESES = tuple(range(N_H))


# ============================================================
# Questions
# ============================================================

def build_questions():
    questions = []

    # Four independent bits uniquely identify 16 hypotheses.
    for bit in range(4):
        questions.append(
            lambda h, bit=bit: (h >> bit) & 1
        )

    # Three parity questions.
    questions.append(
        lambda h: ((h >> 0) & 1) ^ ((h >> 1) & 1)
    )
    questions.append(
        lambda h: ((h >> 1) & 1) ^ ((h >> 2) & 1)
    )
    questions.append(
        lambda h: ((h >> 0) & 1) ^ ((h >> 3) & 1)
    )

    # Unbalanced membership question.
    questions.append(
        lambda h: 1 if h in {0, 1, 2, 3, 4, 5, 6} else 0
    )

    return questions


QUESTIONS = build_questions()


# ============================================================
# Base agent
# ============================================================

class BaseAgent:
    name = "BASE"

    def __init__(self, seed, budget):
        self.rng = random.Random(seed)
        self.budget = budget
        self.queries = 0

        self.support = [0] * N_H
        self.conflicts = [0] * N_H

        self.question_history = []

    def score(self, h):
        return self.support[h] - self.conflicts[h]

    def state(self, h):
        raise NotImplementedError

    def choose_question(self, conflict_rate=0.0):
        remaining = [
            q for q in range(N_Q)
            if q not in self.question_history
        ]

        if not remaining:
            return None

        # Prefer unseen questions.
        q = self.rng.choice(remaining)
        self.question_history.append(q)
        return q

    def observe(self, question, value):
        self.queries += 1

        truth = QUESTIONS[question]

        for h in HYPOTHESES:
            expected = truth(h)

            if value == expected:
                self.support[h] += 1
            else:
                self.conflicts[h] += 1

    def answer(self):
        raise NotImplementedError


# ============================================================
# Binary
# ============================================================

class BinaryAgent(BaseAgent):
    name = "BINARY"

    def state(self, h):
        if self.conflicts[h] > 0:
            return WRONG

        if self.support[h] > 0:
            return RIGHT

        return UNKNOWN

    def answer(self):
        right = [
            h for h in HYPOTHESES
            if self.state(h) == RIGHT
        ]

        if len(right) == 1:
            return right[0]

        return None


# ============================================================
# Ternary
# ============================================================

class TernaryAgent(BaseAgent):
    name = "TERNARY"

    MARGIN = 2

    def state(self, h):
        support = self.support[h]
        conflicts = self.conflicts[h]

        if support == 0 and conflicts == 0:
            return UNKNOWN

        if support - conflicts >= self.MARGIN:
            return RIGHT

        if conflicts - support >= self.MARGIN:
            return WRONG

        return UNKNOWN

    def answer(self):
        right = [
            h for h in HYPOTHESES
            if self.state(h) == RIGHT
        ]

        if len(right) != 1:
            return None

        candidate = right[0]
        candidate_score = self.score(candidate)

        alternatives = [
            self.score(h)
            for h in HYPOTHESES
            if h != candidate
            and self.state(h) != WRONG
        ]

        if not alternatives:
            return candidate

        if candidate_score > max(alternatives):
            return candidate

        return None


# ============================================================
# Quaternary
# ============================================================

class QuaternaryThresholdAgent(BaseAgent):
    name = "THRESHOLD"

    FIXED_THRESHOLD = 3

    def threshold(self):
        return self.FIXED_THRESHOLD

    def state(self, h):
        support = self.support[h]
        conflicts = self.conflicts[h]

        if support == 0 and conflicts == 0:
            return UNKNOWN

        threshold = self.threshold()

        if conflicts >= threshold and conflicts > support:
            return WRONG

        if support >= 2 and conflicts == 0:
            return RIGHT

        if support > 0 and conflicts > 0:
            return CONFLICT

        return UNKNOWN

    def answer(self):
        right = [
            h for h in HYPOTHESES
            if self.state(h) == RIGHT
        ]

        if len(right) != 1:
            return None

        candidate = right[0]
        candidate_score = self.score(candidate)

        alternatives = [
            self.score(h)
            for h in HYPOTHESES
            if h != candidate
            and self.state(h) != WRONG
        ]

        if not alternatives:
            return candidate

        if candidate_score > max(alternatives):
            return candidate

        return None


# ============================================================
# Dynamic quaternary
# ============================================================

class DynamicThresholdAgent(QuaternaryThresholdAgent):
    name = "DYNAMIC"

    def threshold(self):
        remaining = self.budget - self.queries

        if self.budget <= 0:
            return 1

        ratio = remaining / self.budget

        if ratio > 0.66:
            return 3

        if ratio > 0.33:
            return 2

        return 1


AGENTS = (
    BinaryAgent,
    TernaryAgent,
    QuaternaryThresholdAgent,
    DynamicThresholdAgent,
)


# ============================================================
# State names
# ============================================================

STATE_NAMES = {
    WRONG: "WRONG",
    UNKNOWN: "UNKNOWN",
    RIGHT: "RIGHT",
    CONFLICT: "CONFLICT",
}


def state_name(state):
    return STATE_NAMES[state]


# ============================================================
# Contradiction detection
# ============================================================

def contradiction_seen(agent, target):
    return (
        agent.support[target] > 0
        and agent.conflicts[target] > 0
    )


# ============================================================
# NOISY benchmark
# ============================================================

def run_noisy_episode(seed, budget, conflict_rate):
    target_rng = random.Random(seed)
    target = target_rng.randrange(N_H)

    results = []

    for agent_index, AgentClass in enumerate(AGENTS):
        env = Environment(
            target=target,
            conflict_rate=conflict_rate,
            seed=seed + 100000,
            max_budget=budget,
        )

        agent = AgentClass(
            seed=seed + agent_index,
            budget=budget,
        )

        decision = None

        for _ in range(budget):
            q = agent.choose_question(conflict_rate)

            if q is None:
                break

            obs = env.ask(q)
            agent.observe(obs.question, obs.value)

            decision = agent.answer()

            if decision is not None:
                break

        made_decision = decision is not None
        correct = made_decision and decision == target
        false_confidence = made_decision and decision != target
        abstain = not made_decision

        reward = -QUERY_COST * agent.queries

        if correct:
            reward += REWARD_CORRECT
        elif false_confidence:
            reward += REWARD_FALSE
        else:
            reward += REWARD_ABSTAIN

        survived = agent.state(target) != WRONG
        contradiction = contradiction_seen(agent, target)

        results.append({
            "test_type": "NOISY",
            "scenario": "RANDOM_NOISE",
            "agent": agent.name,
            "correct": int(correct),
            "queries": agent.queries,
            "false_confidence": int(false_confidence),
            "abstain": int(abstain),
            "decision": int(made_decision),
            "reward": reward,
            "contradiction_seen": int(contradiction),
            "recovered": 0,
            "ended_wrong": int(agent.state(target) == WRONG),
            "ended_unknown": int(agent.state(target) == UNKNOWN),
            "ended_conflict": int(agent.state(target) == CONFLICT),
            "target_survived": int(survived),
            "final_state": state_name(agent.state(target)),
        })

    return results


# ============================================================
# RECOVERY benchmark
# ============================================================

RECOVERY_SCENARIOS = {
    # Truth, Truth, False, Truth, Truth
    #
    # Binary:
    #   RIGHT -> WRONG
    #
    # Ternary:
    #   RIGHT -> UNKNOWN -> RIGHT
    #
    # Quaternary:
    #   RIGHT -> CONFLICT -> RIGHT
    "SINGLE_CONTRADICTION": [1, 1, 0, 1, 1],

    # Truth, False, Truth, Truth, Truth
    "EARLY_CONTRADICTION": [1, 0, 1, 1, 1],

    # Truth, False, False, Truth, Truth, Truth
    "DOUBLE_CONTRADICTION": [1, 0, 0, 1, 1, 1],

    # Truth followed by enough contradictory evidence
    # to force a wrong conclusion.
    "PERSISTENT_CONTRADICTION": [1, 0, 0, 0, 0],

    # Contradiction followed by enough correct evidence
    # to demonstrate genuine recovery.
    "RECOVERY_AFTER_DAMAGE": [1, 0, 0, 1, 1, 1, 1],
}


def make_recovery_agent(agent_class, seed, budget):
    return agent_class(
        seed=seed,
        budget=budget,
    )


def run_recovery_episode(seed, scenario_name, sequence):
    target_rng = random.Random(seed)

    target = target_rng.randrange(N_H)

    # Use question 0 because its truth value is simply bit 0.
    question = 0
    truth = QUESTIONS[question](target)

    results = []

    for agent_index, AgentClass in enumerate(AGENTS):
        agent = make_recovery_agent(
            AgentClass,
            seed + agent_index,
            budget=len(sequence),
        )

        trajectory = []
        entered_contradiction = False
        recovered = False

        previously_contradictory = False
        previously_wrong = False

        for expected_value in sequence:
            # Convert symbolic 1/0 sequence into observations
            # relative to the actual truth value.
            #
            # 1 = truthful observation
            # 0 = contradictory observation.
            if expected_value == 1:
                observed_value = truth
            else:
                observed_value = 1 - truth

            agent.observe(question, observed_value)

            current_state = agent.state(target)
            trajectory.append(state_name(current_state))

            contradiction = contradiction_seen(agent, target)

            if contradiction:
                entered_contradiction = True

            if (
                previously_contradictory
                and current_state == RIGHT
            ):
                recovered = True

            previously_contradictory = contradiction
            previously_wrong = current_state == WRONG

        final_state = agent.state(target)

        results.append({
            "test_type": "RECOVERY",
            "scenario": scenario_name,
            "agent": agent.name,
            "correct": int(final_state == RIGHT),
            "queries": len(sequence),
            "false_confidence": 0,
            "abstain": int(final_state in (UNKNOWN, CONFLICT)),
            "decision": int(final_state == RIGHT),
            "reward": 0.0,
            "contradiction_seen": int(entered_contradiction),
            "recovered": int(recovered),
            "ended_wrong": int(final_state == WRONG),
            "ended_unknown": int(final_state == UNKNOWN),
            "ended_conflict": int(final_state == CONFLICT),
            "target_survived": int(final_state != WRONG),
            "final_state": state_name(final_state),
            "trajectory": " -> ".join(trajectory),
        })

    return results


# ============================================================
# Aggregation
# ============================================================

def aggregate(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["test_type"],
            row["scenario"],
            row["agent"],
        )
        groups[key].append(row)

    aggregated = []

    for (test_type, scenario, agent), items in sorted(groups.items()):
        n = len(items)

        correct = sum(x["correct"] for x in items)
        decisions = sum(x["decision"] for x in items)
        false_confidence = sum(
            x["false_confidence"]
            for x in items
        )
        abstain = sum(x["abstain"] for x in items)

        contradiction_cases = sum(
            x["contradiction_seen"]
            for x in items
        )

        recovered = sum(
            x["recovered"]
            for x in items
        )

        aggregated.append({
            "test_type": test_type,
            "scenario": scenario,
            "agent": agent,
            "episodes": n,
            "correct": correct / n,
            "decision_rate": decisions / n,
            "conditional_accuracy": (
                correct / decisions
                if decisions > 0
                else 0.0
            ),
            "false_confidence_rate": false_confidence / n,
            "abstain_rate": abstain / n,
            "mean_queries": (
                sum(x["queries"] for x in items) / n
            ),
            "mean_reward": (
                sum(x["reward"] for x in items) / n
            ),
            "contradiction_rate": (
                contradiction_cases / n
            ),
            "recovery_rate": (
                recovered / contradiction_cases
                if contradiction_cases > 0
                else 0.0
            ),
            "ended_wrong_rate": (
                sum(x["ended_wrong"] for x in items) / n
            ),
            "ended_unknown_rate": (
                sum(x["ended_unknown"] for x in items) / n
            ),
            "ended_conflict_rate": (
                sum(x["ended_conflict"] for x in items) / n
            ),
            "target_survival": (
                sum(x["target_survived"] for x in items) / n
            ),
        })

    return aggregated


# ============================================================
# Main
# ============================================================

def main():
    all_rows = []

    # --------------------------------------------------------
    # Main noisy benchmark
    # --------------------------------------------------------

    for conflict_rate in CONFLICT_RATES:
        for budget in BUDGETS:
            for episode in range(EPISODES):
                seed = (
                    1000000
                    + int(conflict_rate * 1000) * 100000
                    + budget * 10000
                    + episode
                )

                rows = run_noisy_episode(
                    seed=seed,
                    budget=budget,
                    conflict_rate=conflict_rate,
                )

                for row in rows:
                    row["budget"] = budget
                    row["conflict_rate"] = conflict_rate
                    all_rows.append(row)

    # --------------------------------------------------------
    # Explicit deterministic recovery benchmark
    # --------------------------------------------------------

    for scenario_index, (scenario_name, sequence) in enumerate(
        RECOVERY_SCENARIOS.items()
    ):
        for episode in range(RECOVERY_EPISODES):
            seed = (
                2000000
                + scenario_index * 100000
                + episode
            )

            rows = run_recovery_episode(
                seed=seed,
                scenario_name=scenario_name,
                sequence=sequence,
            )

            for row in rows:
                row["budget"] = len(sequence)
                row["conflict_rate"] = 0.0
                all_rows.append(row)

    # --------------------------------------------------------
    # Write raw episode results
    # --------------------------------------------------------

    output_path = "results/phase2/test04/results.csv"

    fieldnames = [
        "test_type",
        "scenario",
        "conflict_rate",
        "budget",
        "agent",
        "correct",
        "queries",
        "false_confidence",
        "abstain",
        "decision",
        "reward",
        "decision_rate",
        "conditional_accuracy",
        "false_confidence_rate",
        "abstain_rate",
        "mean_queries",
        "mean_reward",
        "contradiction_seen",
        "contradiction_rate",
        "recovered",
        "recovery_rate",
        "ended_wrong",
        "ended_wrong_rate",
        "ended_unknown",
        "ended_unknown_rate",
        "ended_conflict",
        "ended_conflict_rate",
        "target_survived",
        "target_survival",
        "final_state",
        "trajectory",
    ]

    # Convert raw rows to the unified CSV schema.
    output_rows = []

    for row in all_rows:
        output_rows.append({
            "test_type": row["test_type"],
            "scenario": row["scenario"],
            "conflict_rate": row["conflict_rate"],
            "budget": row["budget"],
            "agent": row["agent"],
            "correct": row["correct"],
            "queries": row["queries"],
            "false_confidence": row["false_confidence"],
            "abstain": row["abstain"],
            "decision": row["decision"],
            "reward": row["reward"],
            "decision_rate": "",
            "conditional_accuracy": "",
            "false_confidence_rate": "",
            "abstain_rate": "",
            "mean_queries": "",
            "mean_reward": "",
            "contradiction_seen": row["contradiction_seen"],
            "contradiction_rate": "",
            "recovered": row["recovered"],
            "recovery_rate": "",
            "ended_wrong": row["ended_wrong"],
            "ended_wrong_rate": "",
            "ended_unknown": row["ended_unknown"],
            "ended_unknown_rate": "",
            "ended_conflict": row["ended_conflict"],
            "ended_conflict_rate": "",
            "target_survived": row["target_survived"],
            "target_survival": "",
            "final_state": row["final_state"],
            "trajectory": row.get("trajectory", ""),
        })

    # Add aggregated rows.
    aggregated = aggregate(all_rows)

    for row in aggregated:
        output_rows.append({
            "test_type": row["test_type"] + "_AGGREGATE",
            "scenario": row["scenario"],
            "conflict_rate": "",
            "budget": "",
            "agent": row["agent"],
            "correct": row["correct"],
            "queries": "",
            "false_confidence": "",
            "abstain": "",
            "decision": "",
            "reward": "",
            "decision_rate": row["decision_rate"],
            "conditional_accuracy": row["conditional_accuracy"],
            "false_confidence_rate": row["false_confidence_rate"],
            "abstain_rate": row["abstain_rate"],
            "mean_queries": row["mean_queries"],
            "mean_reward": row["mean_reward"],
            "contradiction_seen": "",
            "contradiction_rate": row["contradiction_rate"],
            "recovered": "",
            "recovery_rate": row["recovery_rate"],
            "ended_wrong": "",
            "ended_wrong_rate": row["ended_wrong_rate"],
            "ended_unknown": "",
            "ended_unknown_rate": row["ended_unknown_rate"],
            "ended_conflict": "",
            "ended_conflict_rate": row["ended_conflict_rate"],
            "target_survived": "",
            "target_survival": row["target_survival"],
            "final_state": "",
            "trajectory": "",
        })

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("TEST04 completed.")
    print(f"Results: {output_path}")
    print()
    print("Benchmarks:")
    print("  NOISY    - random contradictory observations")
    print("  RECOVERY - deterministic state-transition test")
    print()
    print("Agents:")
    print("  BINARY")
    print("  TERNARY")
    print("  THRESHOLD")
    print("  DYNAMIC")


if __name__ == "__main__":
    main()
