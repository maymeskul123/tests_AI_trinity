from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.v2.agents import (
    BinaryAgent,
    TernaryAgent,
    QuaternaryAgent,
    DynamicAgent,
)
from core.v2.evidence import EvidenceMemory


SEED = 42
N_TASKS = 1000
N_CANDIDATES = 5

NOISE_LEVELS = (0.0, 0.10, 0.20, 0.30, 0.40)
ROUNDS = (1, 2, 4, 8)


def generate_observations(
    rng: random.Random,
    true_answer: int,
    candidates: tuple[int, ...],
    rounds: int,
    noise: float,
):
    """
    Generate one shared observation stream.

    Every candidate is observed once per round.
    The same observations are supplied to every agent.
    """

    observations = []

    for _ in range(rounds):
        for candidate in candidates:
            is_support = candidate == true_answer

            if rng.random() < noise:
                is_support = not is_support

            observations.append((candidate, is_support))

    return observations


def evaluate_agent(
    agent_cls,
    observations,
    question: int,
    candidates: tuple[int, ...],
    true_answer: int,
):
    memory = EvidenceMemory()
    agent = agent_cls(memory)

    for candidate, is_support in observations:
        agent.observe(question, candidate, is_support)

    prediction = agent.decide(question, candidates)

    correct = prediction == true_answer
    unknown = prediction is None

    conflict_count = sum(
        1
        for candidate in candidates
        if agent.state(question, candidate).name == "CONFLICT"
    )

    return correct, unknown, conflict_count


def main() -> None:
    rng = random.Random(SEED)

    agent_classes = (
        BinaryAgent,
        TernaryAgent,
        QuaternaryAgent,
        DynamicAgent,
    )

    results = []

    for noise in NOISE_LEVELS:
        for rounds in ROUNDS:
            observations_per_task = rounds * N_CANDIDATES

            stats = {
                agent_cls.name: {
                    "correct": 0,
                    "unknown": 0,
                    "conflict_sum": 0,
                }
                for agent_cls in agent_classes
            }

            for task_id in range(N_TASKS):
                candidates = tuple(range(N_CANDIDATES))

                true_answer = rng.choice(candidates)

                observations = generate_observations(
                    rng=rng,
                    true_answer=true_answer,
                    candidates=candidates,
                    rounds=rounds,
                    noise=noise,
                )

                for agent_cls in agent_classes:
                    correct, unknown, conflict_count = evaluate_agent(
                        agent_cls=agent_cls,
                        observations=observations,
                        question=task_id,
                        candidates=candidates,
                        true_answer=true_answer,
                    )

                    stats[agent_cls.name]["correct"] += int(correct)
                    stats[agent_cls.name]["unknown"] += int(unknown)
                    stats[agent_cls.name]["conflict_sum"] += conflict_count

            for agent_cls in agent_classes:
                name = agent_cls.name
                correct = stats[name]["correct"]
                unknown = stats[name]["unknown"]
                conflict_sum = stats[name]["conflict_sum"]

                results.append(
                    {
                        "test": "phase3_test02",
                        "seed": SEED,
                        "tasks": N_TASKS,
                        "candidates": N_CANDIDATES,
                        "noise": noise,
                        "rounds": rounds,
                        "queries_per_task": observations_per_task,
                        "agent": name,
                        "correct": correct,
                        "accuracy": correct / N_TASKS,
                        "unknown": unknown,
                        "unknown_rate": unknown / N_TASKS,
                        "conflict_sum": conflict_sum,
                        "avg_conflicts_per_task": conflict_sum / N_TASKS,
                    }
                )

    output = ROOT / "results" / "phase3" / "test02" / "results.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"TEST02: completed")
    print(f"Tasks: {N_TASKS}")
    print(f"Candidates: {N_CANDIDATES}")
    print(f"Noise levels: {', '.join(f'{x:.0%}' for x in NOISE_LEVELS)}")
    print(f"Rounds: {', '.join(map(str, ROUNDS))}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
