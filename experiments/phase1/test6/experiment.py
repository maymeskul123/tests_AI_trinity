import os

# ВАЖНО:
# Каждый multiprocessing-процесс должен использовать только 1 поток BLAS/OpenMP.
# Иначе 8 процессов могут попытаться создать десятки потоков каждый.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

N_FEATURES = 10
N_HIDDEN = 8

N_TRAIN = 300
N_TEST = 300

SEEDS = list(range(10))

NOISE_LEVELS = [0.0, 0.20, 0.40]

MAX_STEPS = 3000

# Ryzen 7 7700 = 8 physical cores / 16 threads.
# Для такого discrete search начинаем с 8 процессов.
N_PROCESSES = min(8, os.cpu_count() or 1)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "test6"

TRAIN_SEED_BASE = 100_000
NOISE_SEED_BASE = 500_000


# ============================================================
# TARGET FUNCTION
# ============================================================

def target_function(X):
    """
    Реальная зависимость использует только x0, x1, x2:

        y = sign(x0 + x1 - x2)

    x3..x9 являются чистым шумом.
    """

    s = X[:, 0] + X[:, 1] - X[:, 2]

    y = np.where(s >= 0.0, 1.0, -1.0)

    return y


# ============================================================
# DATA
# ============================================================

def make_dataset(seed, n_samples):
    rng = np.random.default_rng(seed)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, N_FEATURES),
    )

    y = target_function(X)

    return X, y


def add_label_noise(y, noise_level, seed):
    """
    Инвертируем часть label.
    """

    if noise_level <= 0:
        return y.copy()

    rng = np.random.default_rng(seed)

    noisy = y.copy()

    mask = rng.random(len(y)) < noise_level

    noisy[mask] *= -1.0

    return noisy


# ============================================================
# MODEL
# ============================================================

class DiscreteNetwork:
    """
    Маленькая нейросеть:

        10 inputs
             |
          8 hidden
             |
          1 output

    + bias.

    Binary:
        weights ∈ {-1, +1}

    Ternary:
        weights ∈ {-1, 0, +1}

    Float:
        обычные float weights.
    """

    def __init__(self, mode, seed):

        self.mode = mode

        self.rng = np.random.default_rng(seed)

        # input -> hidden
        self.W1 = self._init_weights(
            N_FEATURES,
            N_HIDDEN,
        )

        self.b1 = self._init_weights(
            1,
            N_HIDDEN,
        )

        # hidden -> output
        self.W2 = self._init_weights(
            N_HIDDEN,
            1,
        )

        self.b2 = self._init_weights(
            1,
            1,
        )

    def _init_weights(self, rows, cols):

        if self.mode == "binary":

            return self.rng.choice(
                [-1.0, 1.0],
                size=(rows, cols),
            )

        if self.mode == "ternary":

            return self.rng.choice(
                [-1.0, 0.0, 1.0],
                size=(rows, cols),
            )

        if self.mode == "float":

            return self.rng.normal(
                0.0,
                1.0,
                size=(rows, cols),
            )

        raise ValueError(self.mode)

    def forward(self, X):

        h = X @ self.W1 + self.b1

        # sign activation
        h = np.where(
            h >= 0.0,
            1.0,
            -1.0,
        )

        y = h @ self.W2 + self.b2

        y = np.where(
            y >= 0.0,
            1.0,
            -1.0,
        )

        return y.ravel()

    def accuracy(self, X, y):

        pred = self.forward(X)

        return float(
            np.mean(pred == y)
        )

    def error(self, X, y):

        return 1.0 - self.accuracy(X, y)

    def nonzero(self):

        return int(
            np.count_nonzero(self.W1)
            + np.count_nonzero(self.b1)
            + np.count_nonzero(self.W2)
            + np.count_nonzero(self.b2)
        )

    def total_weights(self):

        return (
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
        )

    def feature_usage(self):

        """
        Для каждого input feature считаем,
        сколько его connections являются ненулевыми.
        """

        return [
            int(np.count_nonzero(self.W1[i]))
            for i in range(N_FEATURES)
        ]

    def random_mutation(self):
        """
        Меняем один weight.

        Сначала выбираем индекс массива Python-способом,
        а не через numpy.choice(), потому что матрицы
        имеют разные размеры.
        """

        arrays = [
            self.W1,
            self.b1,
            self.W2,
            self.b2,
        ]

        # Выбираем сам массив по индексу.
        array_index = int(
            self.rng.integers(0, len(arrays))
        )

        arr = arrays[array_index]

        # Выбираем координату внутри выбранного массива.
        index = tuple(
            int(
                self.rng.integers(0, size)
            )
            for size in arr.shape
        )

        old = arr[index]

        if self.mode == "binary":

            new = -old

        elif self.mode == "ternary":

            alternatives = [
                -1.0,
                0.0,
                1.0,
            ]

            alternatives.remove(float(old))

            new = self.rng.choice(
                alternatives
            )

        elif self.mode == "float":

            new = old + self.rng.normal(
                0.0,
                0.25,
            )

        else:

            raise ValueError(
                f"Unknown mode: {self.mode}"
            )

        arr[index] = new

        return arr, index, old

# ============================================================
# TRAINING
# ============================================================

def train_model(
    mode,
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
):
    """
    Discrete hill-climbing.

    Принимаем мутацию только если она улучшает
    train accuracy.

    Если улучшения нет — откатываем.
    """

    model = DiscreteNetwork(
        mode,
        seed,
    )

    best_train = model.accuracy(
        X_train,
        y_train,
    )

    best_test = model.accuracy(
        X_test,
        y_test,
    )

    steps = 0

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        steps = step

        arr, index, old = model.random_mutation()

        train_acc = model.accuracy(
            X_train,
            y_train,
        )

        if train_acc >= best_train:

            best_train = train_acc

            best_test = model.accuracy(
                X_test,
                y_test,
            )

        else:

            # rollback
            arr[index] = old

        if best_train >= 1.0:

            break

    final_train = model.accuracy(
        X_train,
        y_train,
    )

    final_test = model.accuracy(
        X_test,
        y_test,
    )

    return {
        "mode": mode,
        "seed": seed,
        "train_accuracy": final_train,
        "test_accuracy": final_test,
        "generalization_gap": final_train - final_test,
        "perfect_test": int(final_test >= 1.0),
        "steps": steps,
        "nonzero": model.nonzero(),
        "total_weights": model.total_weights(),
        "feature_usage": model.feature_usage(),
    }


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(args):

    (
        noise_level,
        seed,
    ) = args

    # --------------------------------------------------------
    # Generate train data
    # --------------------------------------------------------

    X_train, y_train = make_dataset(
        TRAIN_SEED_BASE + seed,
        N_TRAIN,
    )

    y_train_noisy = add_label_noise(
        y_train,
        noise_level,
        NOISE_SEED_BASE + seed,
    )

    # --------------------------------------------------------
    # Generate clean test data
    # --------------------------------------------------------

    X_test, y_test = make_dataset(
        900_000 + seed,
        N_TEST,
    )

    results = []

    for mode in (
        "binary",
        "ternary",
        "float",
    ):

        result = train_model(
            mode,
            X_train,
            y_train_noisy,
            X_test,
            y_test,
            seed,
        )

        result["noise"] = noise_level

        results.append(result)

    return results


# ============================================================
# CSV
# ============================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        RESULTS_DIR
        / "results.csv"
    )

    rows = []

    for r in results:

        row = {
            "noise": r["noise"],
            "mode": r["mode"],
            "seed": r["seed"],
            "train_accuracy": r["train_accuracy"],
            "test_accuracy": r["test_accuracy"],
            "generalization_gap": r["generalization_gap"],
            "perfect_test": r["perfect_test"],
            "steps": r["steps"],
            "nonzero": r["nonzero"],
            "total_weights": r["total_weights"],
        }

        for i, value in enumerate(
            r["feature_usage"]
        ):

            row[f"x{i}_usage"] = value

        rows.append(row)

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    return csv_path


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 78)
    print("TEST 6 — MULTIPROCESSING RESULTS")
    print("=" * 78)

    for noise in NOISE_LEVELS:

        print()
        print(
            f"NOISE {noise * 100:.0f}%"
        )

        print(
            "-" * 78
        )

        for mode in (
            "binary",
            "ternary",
            "float",
        ):

            subset = [
                r
                for r in results
                if r["noise"] == noise
                and r["mode"] == mode
            ]

            if not subset:
                continue

            avg_train = np.mean([
                r["train_accuracy"]
                for r in subset
            ])

            avg_test = np.mean([
                r["test_accuracy"]
                for r in subset
            ])

            avg_gap = np.mean([
                r["generalization_gap"]
                for r in subset
            ])

            perfect = np.mean([
                r["perfect_test"]
                for r in subset
            ])

            avg_steps = np.mean([
                r["steps"]
                for r in subset
            ])

            avg_nonzero = np.mean([
                r["nonzero"]
                for r in subset
            ])

            print(
                f"{mode.upper():7}"
                f" train={avg_train:.3f}"
                f" test={avg_test:.3f}"
                f" gap={avg_gap:.3f}"
                f" perfect={perfect:.3f}"
                f" steps={avg_steps:.1f}"
                f" nonzero={avg_nonzero:.2f}"
            )

    # --------------------------------------------------------
    # Feature usage
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FEATURE USAGE")
    print("=" * 78)

    print(
        "Expected relevant features: x0, x1, x2"
    )
    print(
        "Expected irrelevant features: x3..x9"
    )

    for mode in (
        "binary",
        "ternary",
        "float",
    ):

        subset = [
            r
            for r in results
            if r["mode"] == mode
        ]

        usage = np.mean(
            [
                r["feature_usage"]
                for r in subset
            ],
            axis=0,
        )

        print()
        print(mode.upper())

        for i, value in enumerate(
            usage
        ):

            print(
                f"x{i}: {value:.2f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("TEST 6 — PARALLEL TERNARY GENERALIZATION")
    print("=" * 78)

    print()
    print(
        f"CPU threads available : {os.cpu_count()}"
    )

    print(
        f"Processes              : {N_PROCESSES}"
    )

    print(
        f"Train samples          : {N_TRAIN}"
    )

    print(
        f"Test samples           : {N_TEST}"
    )

    print(
        f"Seeds                  : {len(SEEDS)}"
    )

    print(
        f"Noise levels           : {NOISE_LEVELS}"
    )

    print(
        f"Max steps/model       : {MAX_STEPS}"
    )

    print()

    jobs = [
        (noise, seed)
        for noise in NOISE_LEVELS
        for seed in SEEDS
    ]

    total_jobs = len(jobs)

    print(
        f"Parallel jobs          : {total_jobs}"
    )

    print()
    print("Starting...")
    print()

    start = time.perf_counter()

    results = []

    completed = 0

    with ProcessPoolExecutor(
        max_workers=N_PROCESSES
    ) as executor:

        futures = [
            executor.submit(
                run_experiment,
                job,
            )
            for job in jobs
        ]

        for future in as_completed(
            futures
        ):

            batch = future.result()

            results.extend(batch)

            completed += 1

            elapsed = (
                time.perf_counter()
                - start
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                total_jobs - completed
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"[{completed:2d}/{total_jobs}] "
                f"{completed / total_jobs * 100:5.1f}% "
                f"| "
                f"{elapsed:6.1f}s "
                f"| "
                f"ETA {eta:6.1f}s"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    results.sort(
        key=lambda r: (
            r["noise"],
            r["mode"],
            r["seed"],
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    csv_path = save_results(
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)

    print(
        f"Total time : {elapsed:.2f} sec"
    )

    print(
        f"Jobs/sec   : {total_jobs / elapsed:.2f}"
    )

    print(
        f"Results    : {csv_path}"
    )

    print()


if __name__ == "__main__":
    main()
