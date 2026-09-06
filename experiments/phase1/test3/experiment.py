import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import UNKNOWN, WRONG, RIGHT


TRUE_A = 3
TRUE_B = 1


def hidden_rule(x):
    return (TRUE_A * x + TRUE_B) % 10


HYPOTHESES = [
    (a, b)
    for a in range(10)
    for b in range(10)
]


class BinaryRuleLearner:

    def __init__(self):
        self.memory = {
            h: True
            for h in HYPOTHESES
        }

    def observe(self, x, y):

        for h in self.memory:

            if not self.memory[h]:
                continue

            a, b = h
            prediction = (a * x + b) % 10

            if prediction != y:
                self.memory[h] = False

    def predict(self, x):

        valid = [
            h
            for h, state in self.memory.items()
            if state
        ]

        if not valid:
            return None

        a, b = valid[0]

        return (a * x + b) % 10

    def remaining(self):

        return sum(
            1
            for state in self.memory.values()
            if state
        )

    def knows_rule(self):
        return self.remaining() == 1


class TernaryRuleLearner:

    def __init__(self):

        self.memory = {
            h: UNKNOWN
            for h in HYPOTHESES
        }

    def observe(self, x, y):

        for h in self.memory:

            # Не изменяем уже опровергнутые гипотезы.
            if self.memory[h] == WRONG:
                continue

            a, b = h
            prediction = (a * x + b) % 10

            if prediction == y:
                self.memory[h] = RIGHT
            else:
                self.memory[h] = WRONG

    def predict(self, x):

        valid = [
            h
            for h, state in self.memory.items()
            if state != WRONG
        ]

        if not valid:
            return None

        confirmed = [
            h
            for h in valid
            if self.memory[h] == RIGHT
        ]

        if confirmed:
            valid = confirmed

        a, b = valid[0]

        return (a * x + b) % 10

    def remaining(self):

        return sum(
            1
            for state in self.memory.values()
            if state != WRONG
        )

    def knows_rule(self):
        return self.remaining() == 1


def run(agent_class):

    agent = agent_class()

    history = []

    for x in range(10):

        y = hidden_rule(x)

        agent.observe(x, y)

        history.append({
            "x": x,
            "y": y,
            "remaining": agent.remaining(),
        })

        if agent.knows_rule():
            break

    test_inputs = [10, 11, 12, 13, 14]

    correct = 0

    predictions = []

    for x in test_inputs:

        prediction = agent.predict(x)
        expected = hidden_rule(x)

        if prediction == expected:
            correct += 1

        predictions.append(
            (x, expected, prediction)
        )

    return {
        "history": history,
        "accuracy": correct / len(test_inputs),
        "observations": len(history),
        "remaining": agent.remaining(),
        "predictions": predictions,
    }


def main():

    print("=" * 70)
    print("TEST 3 — TRUE RULE INDUCTION")
    print("=" * 70)

    print()
    print("Hidden rule:")
    print("f(x) = (3*x + 1) mod 10")

    print()
    print("Hypothesis space:")
    print("f(x) = (a*x + b) mod 10")
    print("a,b ∈ {0,...,9}")
    print()
    print(f"Initial hypotheses: {len(HYPOTHESES)}")

    results = []

    for name, agent_class in [
        ("BINARY", BinaryRuleLearner),
        ("TERNARY", TernaryRuleLearner),
    ]:

        result = run(agent_class)

        results.append({
            "agent": name,
            "accuracy": result["accuracy"],
            "observations": result["observations"],
            "remaining": result["remaining"],
        })

        print()
        print("-" * 70)
        print(name)

        print()
        print("Hypothesis reduction:")

        for row in result["history"]:

            print(
                f"x={row['x']} "
                f"y={row['y']} "
                f"remaining={row['remaining']}"
            )

        print()
        print(
            f"Observations needed : "
            f"{result['observations']}"
        )

        print(
            f"Remaining hypotheses: "
            f"{result['remaining']}"
        )

        print(
            f"Generalization      : "
            f"{result['accuracy']:.3f}"
        )

        print()
        print("Predictions:")

        for x, expected, predicted in result["predictions"]:

            status = (
                "OK"
                if expected == predicted
                else "ERROR"
            )

            print(
                f"x={x} "
                f"expected={expected} "
                f"predicted={predicted} "
                f"{status}"
            )

    output = (
        ROOT
        / "results"
        / "test3"
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

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "accuracy",
                "observations",
                "remaining",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 70)
    print(f"Saved: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
