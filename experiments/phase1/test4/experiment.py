import sys
import random
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# TEST 4 — BINARY vs TERNARY NEURAL NETWORK
# ============================================================

BINARY_VALUES = (-1, 1)
TERNARY_VALUES = (-1, 0, 1)


def sign(x):
    return 1 if x >= 0 else -1


def activation(x):
    return 1 if x >= 0 else -1


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def make_dataset():
    """
    3 binary input features.
    Last element is bias = 1.
    """

    data = []

    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):

                # A mixed Boolean task.
                #
                # The target depends mainly on a and b,
                # while c is sometimes irrelevant.
                #
                # This lets us investigate whether ternary
                # weights can suppress useless connections.

                y = 1 if (a + b + a * b) >= 0 else -1

                x = (a, b, c, 1)

                data.append((x, y))

    return data


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

class DiscreteNetwork:

    def __init__(
        self,
        weight_values,
        seed=0,
        hidden_size=4,
    ):
        self.rng = random.Random(seed)

        self.weight_values = tuple(weight_values)
        self.hidden_size = hidden_size

        # input dimension = 4
        # 3 features + bias
        #
        # hidden weights:
        # hidden_size x 4
        #
        # output weights:
        # hidden_size + bias

        self.hidden = [
            [
                self.rng.choice(self.weight_values)
                for _ in range(4)
            ]
            for _ in range(hidden_size)
        ]

        self.output = [
            self.rng.choice(self.weight_values)
            for _ in range(hidden_size + 1)
        ]

    def forward(self, x):

        hidden_values = []

        for weights in self.hidden:

            total = sum(
                w * xi
                for w, xi in zip(weights, x)
            )

            hidden_values.append(
                activation(total)
            )

        hidden_with_bias = tuple(hidden_values) + (1,)

        total = sum(
            w * h
            for w, h in zip(
                self.output,
                hidden_with_bias,
            )
        )

        return activation(total)

    def accuracy(self, data):

        correct = 0

        for x, y in data:

            if self.forward(x) == y:
                correct += 1

        return correct / len(data)

    def error(self, data):

        return sum(
            self.forward(x) != y
            for x, y in data
        )

    def nonzero_weights(self):

        weights = []

        for row in self.hidden:
            weights.extend(row)

        weights.extend(self.output)

        return sum(
            w != 0
            for w in weights
        )

    def total_weights(self):

        return (
            self.hidden_size * 4
            + self.hidden_size
            + 1
        )

    def train(
        self,
        data,
        max_steps=10000,
    ):

        current_error = self.error(data)

        for step in range(max_steps):

            # Pick one weight randomly.

            layer = self.rng.choice(
                ("hidden", "output")
            )

            if layer == "hidden":

                neuron = self.rng.randrange(
                    self.hidden_size
                )

                index = self.rng.randrange(4)

                old = self.hidden[neuron][index]

                candidates = [
                    v
                    for v in self.weight_values
                    if v != old
                ]

                new = self.rng.choice(candidates)

                self.hidden[neuron][index] = new

            else:

                index = self.rng.randrange(
                    self.hidden_size + 1
                )

                old = self.output[index]

                candidates = [
                    v
                    for v in self.weight_values
                    if v != old
                ]

                new = self.rng.choice(candidates)

                self.output[index] = new

            new_error = self.error(data)

            # Keep modification only if it is
            # at least as good as the previous state.

            if new_error <= current_error:

                current_error = new_error

            else:

                if layer == "hidden":
                    self.hidden[neuron][index] = old
                else:
                    self.output[index] = old

            if current_error == 0:

                return step + 1

        return max_steps


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

def run_experiment(agent_class, values, seeds=100):

    data = make_dataset()

    results = []

    for seed in range(seeds):

        network = agent_class(
            values,
            seed=seed,
        )

        initial_accuracy = network.accuracy(data)

        steps = network.train(
            data,
            max_steps=10000,
        )

        final_accuracy = network.accuracy(data)

        results.append({
            "seed": seed,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "steps": steps,
            "nonzero": network.nonzero_weights(),
            "total_weights": network.total_weights(),
        })

    return results


def summarize(results):

    solved = [
        r for r in results
        if r["final_accuracy"] == 1.0
    ]

    return {
        "success_rate": (
            len(solved) / len(results)
        ),

        "avg_steps_solved": (
            sum(r["steps"] for r in solved)
            / len(solved)
            if solved
            else 0
        ),

        "avg_nonzero": (
            sum(r["nonzero"] for r in results)
            / len(results)
        ),

        "avg_final_accuracy": (
            sum(r["final_accuracy"] for r in results)
            / len(results)
        ),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TEST 4 — BINARY vs TERNARY NEURAL NETWORK")
    print("=" * 70)

    print()
    print("Architecture:")
    print("3 inputs + bias -> 4 hidden neurons -> 1 output")

    print()
    print("BINARY weights : {-1, +1}")
    print("TERNARY weights: {-1, 0, +1}")

    print()
    print("Training: discrete coordinate search")
    print("Seeds: 100")
    print("Maximum steps: 10000")

    data = make_dataset()

    print()
    print("Dataset:")
    for x, y in data:
        print(
            f"x={x[:3]} -> y={y}"
        )

    all_results = []

    for name, values in [
        ("BINARY", BINARY_VALUES),
        ("TERNARY", TERNARY_VALUES),
    ]:

        results = run_experiment(
            DiscreteNetwork,
            values,
            seeds=100,
        )

        summary = summarize(results)

        all_results.append({
            "agent": name,
            **summary,
        })

        print()
        print("-" * 70)
        print(name)
        print()

        print(
            f"Success rate      : "
            f"{summary['success_rate']:.3f}"
        )

        print(
            f"Final accuracy    : "
            f"{summary['avg_final_accuracy']:.3f}"
        )

        print(
            f"Avg steps solved  : "
            f"{summary['avg_steps_solved']:.2f}"
        )

        print(
            f"Avg nonzero       : "
            f"{summary['avg_nonzero']:.2f}"
        )

        print(
            f"Total weights     : "
            f"{results[0]['total_weights']}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = (
        ROOT
        / "results"
        / "test4"
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
                "success_rate",
                "avg_final_accuracy",
                "avg_steps_solved",
                "avg_nonzero",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("=" * 70)
    print(f"Saved: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
