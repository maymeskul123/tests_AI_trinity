import sys
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agents import BinaryAgent, TernaryAgent
from core.environment import HiddenRuleEnvironment
from core.constants import RIGHT


RULE = lambda x: (3 * x + 1) % 10

TRAIN_X = [0, 1, 2, 3, 4]
TEST_X = [5, 6, 7, 8, 9]

EPISODES = 1000
SEEDS = 100


def run_training(agent, env, seed):
    rng = random.Random(seed)
    history = []

    for episode in range(EPISODES):
        x = rng.choice(TRAIN_X)

        task = env.make_task(x)

        prediction = agent.predict(
            x,
            task.candidates,
        )

        feedback = env.evaluate(
            task,
            prediction,
        )

        agent.learn(
            x,
            prediction,
            feedback,
        )

        history.append(
            int(feedback == RIGHT)
        )

    return history


def evaluate(agent, env, inputs):
    correct = 0

    for x in inputs:
        task = env.make_task(x)

        prediction = agent.predict(
            x,
            task.candidates,
        )

        if prediction == task.answer:
            correct += 1

    return correct / len(inputs)


def episodes_to_threshold(
    history,
    threshold=0.90,
    window=50,
):
    for i in range(window, len(history)):

        accuracy = (
            sum(history[i - window:i])
            / window
        )

        if accuracy >= threshold:
            return i

    return None


def run_single(agent_cls, seed):
    env = HiddenRuleEnvironment(RULE)

    agent = agent_cls(seed=seed)

    history = run_training(
        agent,
        env,
        seed,
    )

    train_accuracy = (
        sum(history[-100:])
        / 100
    )

    test_accuracy = evaluate(
        agent,
        env,
        TEST_X,
    )

    speed = episodes_to_threshold(
        history
    )

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "episodes_to_90": speed,
        "memory": len(agent.memory),
    }


def main():

    print("=" * 65)
    print("TEST 2 — BINARY vs TERNARY SELF-LEARNING")
    print("=" * 65)

    print(f"Training inputs : {TRAIN_X}")
    print(f"Test inputs     : {TEST_X}")
    print(f"Episodes        : {EPISODES}")
    print(f"Seeds           : {SEEDS}")

    results = {}

    for name, cls in [
        ("BINARY", BinaryAgent),
        ("TERNARY", TernaryAgent),
    ]:

        rows = []

        for seed in range(SEEDS):
            rows.append(
                run_single(
                    cls,
                    seed,
                )
            )

        results[name] = rows

        avg_train = (
            sum(
                r["train_accuracy"]
                for r in rows
            )
            / SEEDS
        )

        avg_test = (
            sum(
                r["test_accuracy"]
                for r in rows
            )
            / SEEDS
        )

        speeds = [
            r["episodes_to_90"]
            for r in rows
            if r["episodes_to_90"] is not None
        ]

        avg_speed = (
            sum(speeds) / len(speeds)
            if speeds
            else None
        )

        avg_memory = (
            sum(
                r["memory"]
                for r in rows
            )
            / SEEDS
        )

        print("\n" + "-" * 65)
        print(name)

        print(
            f"Train accuracy : "
            f"{avg_train:.3f}"
        )

        print(
            f"Held-out test  : "
            f"{avg_test:.3f}"
        )

        print(
            f"To 90%         : "
            f"{avg_speed}"
        )

        print(
            f"Memory         : "
            f"{avg_memory:.1f}"
        )

    print("\n")
    print("=" * 65)
    print("GENERALIZATION")
    print("=" * 65)

    for name, cls in [
        ("BINARY", BinaryAgent),
        ("TERNARY", TernaryAgent),
    ]:

        env = HiddenRuleEnvironment(RULE)

        agent = cls(seed=123)

        run_training(
            agent,
            env,
            123,
        )

        print(f"\n{name}")

        correct = 0

        for x in TEST_X:

            task = env.make_task(x)

            prediction = agent.predict(
                x,
                task.candidates,
            )

            expected = task.answer

            if prediction == expected:
                correct += 1
                status = "OK"
            else:
                status = "ERROR"

            print(
                f"x={x} "
                f"expected={expected} "
                f"predicted={prediction} "
                f"{status}"
            )

        print(
            "Generalization accuracy: "
            f"{correct / len(TEST_X):.3f}"
        )

    output = (
        ROOT
        / "results"
        / "test2"
        / "summary.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "agent",
            "seed",
            "train_accuracy",
            "test_accuracy",
            "episodes_to_90",
            "memory",
        ])

        for name, rows in results.items():

            for seed, row in enumerate(rows):

                writer.writerow([
                    name,
                    seed,
                    row["train_accuracy"],
                    row["test_accuracy"],
                    row["episodes_to_90"],
                    row["memory"],
                ])

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
