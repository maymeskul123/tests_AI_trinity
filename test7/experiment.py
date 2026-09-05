import os

# Не позволяем каждому worker создавать дополнительные BLAS/OpenMP threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

N_FEATURES = 50
N_HIDDEN = 32

N_TRAIN = 5000
N_TEST = 5000

# Для первого прогона достаточно 100 seeds.
SEEDS = list(range(100))

# Штраф за ненулевые веса.
#
# Чем больше lambda:
#   тем сильнее модель заинтересована
#   в разреженности.
#
LAMBDA_VALUES = [
    0.000,
    0.001,
    0.003,
    0.005,
    0.010,
]

# Одинаковый computational budget
MAX_STEPS = 5000

# Ryzen 7 7700
N_PROCESSES = min(8, os.cpu_count() or 1)

RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "test7"
)

# Только 3 первых признака реально важны.
RELEVANT_FEATURES = (0, 1, 2)


# ============================================================
# TARGET
# ============================================================

def target_function(X):
    """
    Реальная зависимость:

        y = sign(x0 + x1 - x2)

    x3..x49 — полностью бесполезные признаки.
    """

    s = (
        X[:, 0]
        + X[:, 1]
        - X[:, 2]
    )

    return np.where(
        s >= 0,
        1.0,
        -1.0,
    )


# ============================================================
# DATA
# ============================================================

def make_dataset(seed, n):
    rng = np.random.default_rng(seed)

    X = rng.normal(
        0.0,
        1.0,
        size=(n, N_FEATURES),
    )

    y = target_function(X)

    return X, y


# ============================================================
# NETWORK
# ============================================================

class DiscreteNetwork:

    def __init__(self, mode, seed):

        self.mode = mode

        self.rng = np.random.default_rng(seed)

        # 50 -> 32
        self.W1 = self.init_weights(
            N_FEATURES,
            N_HIDDEN,
        )

        self.b1 = self.init_weights(
            1,
            N_HIDDEN,
        )

        # 32 -> 1
        self.W2 = self.init_weights(
            N_HIDDEN,
            1,
        )

        self.b2 = self.init_weights(
            1,
            1,
        )

    def init_weights(self, rows, cols):

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

        raise ValueError(
            f"Unknown mode: {self.mode}"
        )

    def forward(self, X):

        hidden = (
            X @ self.W1
            + self.b1
        )

        hidden = np.where(
            hidden >= 0,
            1.0,
            -1.0,
        )

        output = (
            hidden @ self.W2
            + self.b2
        )

        return np.where(
            output >= 0,
            1.0,
            -1.0,
        ).ravel()

    def accuracy(self, X, y):

        pred = self.forward(X)

        return float(
            np.mean(pred == y)
        )

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

        return np.count_nonzero(
            self.W1,
            axis=1,
        ).astype(int)

    def mutate(self):

        arrays = [
            self.W1,
            self.b1,
            self.W2,
            self.b2,
        ]

        array_id = int(
            self.rng.integers(
                0,
                len(arrays),
            )
        )

        arr = arrays[array_id]

        index = tuple(
            int(
                self.rng.integers(
                    0,
                    size,
                )
            )
            for size in arr.shape
        )

        old = float(arr[index])

        if self.mode == "binary":

            new = -old

        else:

            alternatives = [
                -1.0,
                0.0,
                1.0,
            ]

            alternatives.remove(old)

            new = float(
                self.rng.choice(
                    alternatives
                )
            )

        arr[index] = new

        return arr, index, old


# ============================================================
# OBJECTIVE
# ============================================================

def objective(
    accuracy,
    nonzero,
    total_weights,
    lam,
):
    """
    Максимизируем:

        accuracy - lambda * sparsity

    Нормируем sparsity на количество весов,
    чтобы lambda имела понятный масштаб.
    """

    sparsity_penalty = (
        lam
        * nonzero
        / total_weights
    )

    return accuracy - sparsity_penalty


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    mode,
    lam,
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
):

    model = DiscreteNetwork(
        mode,
        seed,
    )

    total_weights = (
        model.total_weights()
    )

    train_acc = model.accuracy(
        X_train,
        y_train,
    )

    nonzero = model.nonzero()

    best_score = objective(
        train_acc,
        nonzero,
        total_weights,
        lam,
    )

    accepted = 0

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        arr, index, old = (
            model.mutate()
        )

        new_train_acc = (
            model.accuracy(
                X_train,
                y_train,
            )
        )

        new_nonzero = (
            model.nonzero()
        )

        new_score = objective(
            new_train_acc,
            new_nonzero,
            total_weights,
            lam,
        )

        if new_score >= best_score:

            best_score = new_score
            train_acc = new_train_acc
            nonzero = new_nonzero

            accepted += 1

        else:

            arr[index] = old

    final_train = model.accuracy(
        X_train,
        y_train,
    )

    final_test = model.accuracy(
        X_test,
        y_test,
    )

    usage = model.feature_usage()

    relevant_usage = float(
        np.mean(
            usage[
                list(RELEVANT_FEATURES)
            ]
        )
    )

    irrelevant_usage = float(
        np.mean(
            usage[
                3:
            ]
        )
    )

    return {
        "mode": mode,
        "lambda": lam,
        "seed": seed,
        "train_accuracy": final_train,
        "test_accuracy": final_test,
        "gap": final_train - final_test,
        "nonzero": model.nonzero(),
        "total_weights": total_weights,
        "sparsity": (
            1.0
            - model.nonzero()
            / total_weights
        ),
        "relevant_usage": relevant_usage,
        "irrelevant_usage": irrelevant_usage,
        "accepted": accepted,
        "steps": MAX_STEPS,
        "feature_usage": usage.tolist(),
    }


# ============================================================
# ONE JOB
# ============================================================

def run_job(args):

    lam, seed = args

    X_train, y_train = make_dataset(
        100000 + seed,
        N_TRAIN,
    )

    X_test, y_test = make_dataset(
        900000 + seed,
        N_TEST,
    )

    results = []

    for mode in (
        "binary",
        "ternary",
    ):

        result = train_model(
            mode,
            lam,
            X_train,
            y_train,
            X_test,
            y_test,
            seed,
        )

        results.append(result)

    return results


# ============================================================
# SAVE
# ============================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULTS_DIR
        / "results.csv"
    )

    rows = []

    for r in results:

        row = {
            "lambda": r["lambda"],
            "mode": r["mode"],
            "seed": r["seed"],
            "train_accuracy":
                r["train_accuracy"],
            "test_accuracy":
                r["test_accuracy"],
            "gap":
                r["gap"],
            "nonzero":
                r["nonzero"],
            "total_weights":
                r["total_weights"],
            "sparsity":
                r["sparsity"],
            "relevant_usage":
                r["relevant_usage"],
            "irrelevant_usage":
                r["irrelevant_usage"],
            "accepted":
                r["accepted"],
            "steps":
                r["steps"],
        }

        for i, value in enumerate(
            r["feature_usage"]
        ):
            row[
                f"x{i}_usage"
            ] = value

        rows.append(row)

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 90)
    print("TEST 7 — TERNARY SPARSITY")
    print("=" * 90)

    for lam in LAMBDA_VALUES:

        print()
        print(
            f"LAMBDA = {lam}"
        )

        print("-" * 90)

        for mode in (
            "binary",
            "ternary",
        ):

            subset = [
                r
                for r in results
                if r["lambda"] == lam
                and r["mode"] == mode
            ]

            if not subset:
                continue

            train = np.mean([
                r["train_accuracy"]
                for r in subset
            ])

            test = np.mean([
                r["test_accuracy"]
                for r in subset
            ])

            gap = np.mean([
                r["gap"]
                for r in subset
            ])

            nonzero = np.mean([
                r["nonzero"]
                for r in subset
            ])

            sparsity = np.mean([
                r["sparsity"]
                for r in subset
            ])

            useful = np.mean([
                r["relevant_usage"]
                for r in subset
            ])

            noise = np.mean([
                r["irrelevant_usage"]
                for r in subset
            ])

            accepted = np.mean([
                r["accepted"]
                for r in subset
            ])

            print(
                f"{mode.upper():8}"
                f" train={train:.4f}"
                f" test={test:.4f}"
                f" gap={gap:.4f}"
                f" nonzero={nonzero:.1f}"
                f" sparse={sparsity:.3f}"
                f" useful={useful:.2f}"
                f" noise={noise:.2f}"
                f" accepted={accepted:.1f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "TEST 7 — PARALLEL TERNARY SPARSITY"
    )
    print("=" * 90)

    print()
    print(
        f"CPU threads       : {os.cpu_count()}"
    )

    print(
        f"Processes          : {N_PROCESSES}"
    )

    print(
        f"Features           : {N_FEATURES}"
    )

    print(
        f"Hidden             : {N_HIDDEN}"
    )

    print(
        f"Train samples      : {N_TRAIN}"
    )

    print(
        f"Test samples       : {N_TEST}"
    )

    print(
        f"Seeds              : {len(SEEDS)}"
    )

    print(
        f"Lambda values      : {LAMBDA_VALUES}"
    )

    print(
        f"Steps/model        : {MAX_STEPS}"
    )

    total_jobs = (
        len(SEEDS)
        * len(LAMBDA_VALUES)
    )

    print(
        f"Parallel jobs      : {total_jobs}"
    )

    print()

    jobs = [
        (lam, seed)
        for lam in LAMBDA_VALUES
        for seed in SEEDS
    ]

    start = time.perf_counter()

    results = []

    completed = 0

    with ProcessPoolExecutor(
        max_workers=N_PROCESSES
    ) as executor:

        futures = [
            executor.submit(
                run_job,
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
            )

            remaining = (
                total_jobs
                - completed
            )

            eta = (
                remaining / rate
            )

            print(
                f"[{completed:3d}/{total_jobs}] "
                f"{completed / total_jobs * 100:6.2f}% "
                f"| {elapsed:7.2f}s "
                f"| ETA {eta:7.2f}s"
            )

    results.sort(
        key=lambda r: (
            r["lambda"],
            r["mode"],
            r["seed"],
        )
    )

    path = save_results(
        results
    )

    print_summary(
        results
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        f"Total time : {elapsed:.2f} sec"
    )

    print(
        f"Jobs/sec   : {total_jobs / elapsed:.2f}"
    )

    print(
        f"CSV        : {path}"
    )

    print()


if __name__ == "__main__":
    main()
