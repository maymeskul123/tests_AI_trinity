import sys
import random
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# TEST 5
# Binary vs Ternary vs Float
# Robustness to noisy training data
# ============================================================


BINARY_VALUES = (-1, 1)
TERNARY_VALUES = (-1, 0, 1)


def activation(x):
    return 1 if x >= 0 else -1


# ------------------------------------------------------------
# TRUE FUNCTION
# ------------------------------------------------------------

def true_rule(x):
    """
    Ground-truth function.

    The target depends on x1 and x2.
    x3 is deliberately irrelevant.

        y = sign(x1 + x2)

    When x1 + x2 == 0 -> -1
    """

    x1, x2, x3 = x[:3]

    return 1 if (x1 + x2) > 0 else -1


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

def make_dataset():
    data = []

    for x1 in (-1, 1):
        for x2 in (-1, 1):
            for x3 in (-1, 1):

                x = (x1, x2, x3, 1)
                y = true_rule(x)

                data.append((x, y))

    return data


# ------------------------------------------------------------
# NOISE
# ------------------------------------------------------------

def add_noise(data, noise_rate, seed):
    rng = random.Random(seed)

    noisy = []

    for x, y in data:

        if rng.random() < noise_rate:
            y = -y

        noisy.append((x, y))

    return noisy


# ------------------------------------------------------------
# BINARY / TERNARY NETWORK
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
# FLOAT NETWORK
# ------------------------------------------------------------

class FloatNetwork:

    def __init__(
        self,
        seed=0,
        hidden_size=4,
    ):
        self.rng = random.Random(seed)

        self.hidden_size = hidden_size

        self.hidden = [
            [
                self.rng.uniform(-1, 1)
                for _ in range(4)
            ]
            for _ in range(hidden_size)
        ]

        self.output = [
            self.rng.uniform(-1, 1)
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
            abs(w) > 1e-9
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

            layer = self.rng.choice(
                ("hidden", "output")
            )

            if layer == "hidden":

                neuron = self.rng.randrange(
                    self.hidden_size
                )

                index = self.rng.randrange(4)

                old = self.hidden[neuron][index]

                new = old + self.rng.uniform(
                    -0.5,
                    0.5,
                )

                self.hidden[neuron][index] = new

            else:

                index = self.rng.randrange(
                    self.hidden_size + 1
                )

                old = self.output[index]

                new = old + self.rng.uniform(
                    -0.5,
                    0.5,
                )

                self.output[index] = new

            new_error = self.error(data)

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
# EXPERIMENT
# ------------------------------------------------------------

def evaluate_model(
    name,
    network_factory,
    train_data,
    clean_data,
    seed,
):

    network = network_factory(seed)

    initial_accuracy = network.accuracy(clean_data)

    steps = network.train(
        train_data,
        max_steps=10000,
    )

    train_accuracy = network.accuracy(train_data)

    clean_accuracy = network.accuracy(clean_data)

    return {
        "agent": name,
        "seed": seed,
        "initial_accuracy": initial_accuracy,
        "train_accuracy": train_accuracy,
        "clean_accuracy": clean_accuracy,
        "steps": steps,
        "nonzero": network.nonzero_weights(),
        "total_weights": network.total_weights(),
    }


def summarize(rows):

    return {
        "success_rate": sum(
            r["clean_accuracy"] == 1.0
            for r in rows
        ) / len(rows),

        "avg_train_accuracy": sum(
            r["train_accuracy"]
            for r in rows
        ) / len(rows),

        "avg_clean_accuracy": sum(
            r["clean_accuracy"]
            for r in rows
        ) / len(rows),

        "avg_steps": sum(
            r["steps"]
            for r in rows
        ) / len(rows),

        "avg_nonzero": sum(
            r["nonzero"]
            for r in rows
        ) / len(rows),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TEST 5 — BINARY vs TERNARY vs FLOAT")
    print("ROBUSTNESS TO NOISY TRAINING DATA")
    print("=" * 70)

    clean_data = make_dataset()

    noise_levels = (
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
    )

    seeds = 100

    print()
    print("Architecture:")
    print("3 inputs + bias -> 4 hidden neurons -> 1 output")

    print()
    print("Binary : {-1, +1}")
    print("Ternary: {-1, 0, +1}")
    print("Float  : continuous weights")

    print()
    print(f"Seeds: {seeds}")
    print("Maximum steps: 10000")

    all_rows = []

    for noise in noise_levels:

        print()
        print("=" * 70)
        print(f"NOISE LEVEL: {noise:.0%}")
        print("=" * 70)

        noisy_data = add_noise(
            clean_data,
            noise,
            seed=12345,
        )

        models = [
            (
                "BINARY",
                lambda seed: DiscreteNetwork(
                    BINARY_VALUES,
                    seed=seed,
                ),
            ),
            (
                "TERNARY",
                lambda seed: DiscreteNetwork(
                    TERNARY_VALUES,
                    seed=seed,
                ),
            ),
            (
                "FLOAT",
                lambda seed: FloatNetwork(
                    seed=seed,
                ),
            ),
        ]

        for name, factory in models:

            rows = []

            for seed in range(seeds):

                result = evaluate_model(
                    name,
                    factory,
                    noisy_data,
                    clean_data,
                    seed,
                )

                result["noise"] = noise

                rows.append(result)
                all_rows.append(result)

            summary = summarize(rows)

            print()
            print("-" * 70)
            print(name)

            print(
                f"Train accuracy : "
                f"{summary['avg_train_accuracy']:.3f}"
            )

            print(
                f"Clean accuracy : "
                f"{summary['avg_clean_accuracy']:.3f}"
            )

            print(
                f"Success rate   : "
                f"{summary['success_rate']:.3f}"
            )

            print(
                f"Avg steps      : "
                f"{summary['avg_steps']:.2f}"
            )

            print(
                f"Avg nonzero    : "
                f"{summary['avg_nonzero']:.2f}"
            )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    output = (
        ROOT
        / "results"
        / "test5"
        / "results.csv"
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
                "noise",
                "seed",
                "initial_accuracy",
                "train_accuracy",
                "clean_accuracy",
                "steps",
                "nonzero",
                "total_weights",
            ],
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_output = (
        ROOT
        / "results"
        / "test5"
        / "summary.csv"
    )

    with summary_output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "noise",
            "agent",
            "train_accuracy",
            "clean_accuracy",
            "success_rate",
            "avg_steps",
            "avg_nonzero",
        ])

        for noise in noise_levels:

            for name in (
                "BINARY",
                "TERNARY",
                "FLOAT",
            ):

                rows = [
                    r
                    for r in all_rows
                    if r["noise"] == noise
                    and r["agent"] == name
                ]

                s = summarize(rows)

                writer.writerow([
                    noise,
                    name,
                    s["avg_train_accuracy"],
                    s["avg_clean_accuracy"],
                    s["success_rate"],
                    s["avg_steps"],
                    s["avg_nonzero"],
                ])

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(output)
    print(summary_output)


if __name__ == "__main__":
    main()
