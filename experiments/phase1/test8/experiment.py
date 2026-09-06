import os

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

N_VALUES = 20

N_EPISODES = 2000

SEEDS = list(range(100))

# Сколько наблюдений агент получает
OBSERVATIONS = [
    1,
    2,
    3,
    5,
    8,
    12,
    20,
    30,
]

# Вероятность того, что наблюдение будет UNKNOWN.
UNKNOWN_RATES = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
]

# Вероятность неправильного/противоречивого наблюдения.
NOISE_RATES = [
    0.0,
    0.1,
    0.2,
]

N_PROCESSES = min(8, os.cpu_count() or 1)

RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "test8"
)


# ============================================================
# HIDDEN WORLD
# ============================================================

def hidden_rule(x):
    return (3 * x + 1) % 10


# ============================================================
# BINARY AGENT
# ============================================================

class BinaryAgent:

    """
    Binary memory:

        0 = FALSE
        1 = TRUE

    Важное ограничение:
    отсутствие информации нельзя представить отдельно.
    """

    def __init__(self):

        self.memory = {}

    def state(self, x, candidate):

        return self.memory.get(
            (x, candidate),
            False,
        )

    def learn(
        self,
        x,
        candidate,
        feedback,
    ):

        if feedback == 1:

            self.memory[
                (x, candidate)
            ] = True

        elif feedback == -1:

            self.memory[
                (x, candidate)
            ] = False

        # feedback == 0:
        # Binary не имеет состояния UNKNOWN.
        # Поэтому информация просто теряется.

    def predict(
        self,
        x,
        candidates,
    ):

        positives = [
            c
            for c in candidates
            if self.state(x, c)
        ]

        if positives:
            return positives[0]

        # Binary вынужден сделать выбор
        # даже когда информации нет.
        return candidates[0]


# ============================================================
# TERNARY AGENT
# ============================================================

class TernaryAgent:

    """
    -1 = WRONG
     0 = UNKNOWN
    +1 = RIGHT
    """

    WRONG = -1
    UNKNOWN = 0
    RIGHT = 1

    def __init__(self):

        self.memory = {}

    def state(self, x, candidate):

        return self.memory.get(
            (x, candidate),
            self.UNKNOWN,
        )

    def learn(
        self,
        x,
        candidate,
        feedback,
    ):

        self.memory[
            (x, candidate)
        ] = feedback

    def predict(
        self,
        x,
        candidates,
    ):

        # Сначала подтверждённые ответы.
        positives = [
            c
            for c in candidates
            if self.state(x, c)
            == self.RIGHT
        ]

        if positives:
            return positives[0]

        # Затем неизвестные.
        unknown = [
            c
            for c in candidates
            if self.state(x, c)
            == self.UNKNOWN
        ]

        if unknown:
            return unknown[0]

        # Всё известно как WRONG.
        return candidates[0]


# ============================================================
# EPISODE
# ============================================================

def run_episode(
    mode,
    n_observations,
    unknown_rate,
    noise_rate,
    seed,
):

    rng = np.random.default_rng(seed)

    if mode == "binary":

        agent = BinaryAgent()

    else:

        agent = TernaryAgent()

    # Истинное значение x
    x = int(
        rng.integers(
            0,
            N_VALUES,
        )
    )

    answer = hidden_rule(x)

    candidates = tuple(
        range(10)
    )

    # --------------------------------------------------------
    # Generate observations
    # --------------------------------------------------------

    for _ in range(
        n_observations
    ):

        candidate = int(
            rng.integers(
                0,
                10,
            )
        )

        # UNKNOWN
        if rng.random() < unknown_rate:

            feedback = 0

        else:

            correct = (
                candidate == answer
            )

            if correct:

                feedback = 1

            else:

                feedback = -1

            # Иногда специально искажаем
            # информацию.
            if (
                rng.random()
                < noise_rate
            ):

                feedback = -feedback

        agent.learn(
            x,
            candidate,
            feedback,
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = agent.predict(
        x,
        candidates,
    )

    correct = (
        prediction == answer
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if mode == "ternary":

        state = agent.state(
            x,
            prediction,
        )

        confident = (
            state
            == TernaryAgent.RIGHT
        )

        unknown_prediction = (
            state
            == TernaryAgent.UNKNOWN
        )

    else:

        # Binary всегда считает свой выбор
        # определённым.
        confident = True

        unknown_prediction = False

    # --------------------------------------------------------
    # State statistics
    # --------------------------------------------------------

    if mode == "ternary":

        states = [
            agent.state(
                x,
                c,
            )
            for c in candidates
        ]

        unknown_count = states.count(
            TernaryAgent.UNKNOWN
        )

        wrong_count = states.count(
            TernaryAgent.WRONG
        )

        right_count = states.count(
            TernaryAgent.RIGHT
        )

    else:

        states = [
            agent.state(
                x,
                c,
            )
            for c in candidates
        ]

        unknown_count = 0

        wrong_count = states.count(
            False
        )

        right_count = states.count(
            True
        )

    return {
        "correct": int(correct),
        "confident": int(confident),
        "unknown_prediction":
            int(unknown_prediction),
        "unknown_count":
            unknown_count,
        "wrong_count":
            wrong_count,
        "right_count":
            right_count,
    }


# ============================================================
# JOB
# ============================================================

def run_job(args):

    (
        mode,
        observations,
        unknown_rate,
        noise_rate,
        seed,
    ) = args

    results = []

    for episode in range(
        N_EPISODES
    ):

        episode_seed = (
            seed * 1000000
            + episode
        )

        result = run_episode(
            mode,
            observations,
            unknown_rate,
            noise_rate,
            episode_seed,
        )

        results.append(result)

    return {
        "mode": mode,
        "observations": observations,
        "unknown_rate":
            unknown_rate,
        "noise_rate":
            noise_rate,
        "seed": seed,

        "accuracy":
            np.mean([
                r["correct"]
                for r in results
            ]),

        "confident_accuracy":
            (
                np.mean([
                    r["correct"]
                    for r in results
                    if r["confident"]
                ])
                if any(
                    r["confident"]
                    for r in results
                )
                else 0.0
            ),

        "confidence":
            np.mean([
                r["confident"]
                for r in results
            ]),

        "unknown_predictions":
            np.mean([
                r["unknown_prediction"]
                for r in results
            ]),

        "unknown_count":
            np.mean([
                r["unknown_count"]
                for r in results
            ]),

        "wrong_count":
            np.mean([
                r["wrong_count"]
                for r in results
            ]),

        "right_count":
            np.mean([
                r["right_count"]
                for r in results
            ]),
    }


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

    fields = [
        "mode",
        "observations",
        "unknown_rate",
        "noise_rate",
        "seed",
        "accuracy",
        "confident_accuracy",
        "confidence",
        "unknown_predictions",
        "unknown_count",
        "wrong_count",
        "right_count",
    ]

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

        for row in results:

            writer.writerow(row)

    return path


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 110)
    print("TEST 8 — INFORMATIONAL TERNARY STATE")
    print("=" * 110)

    for unknown_rate in UNKNOWN_RATES:

        for noise_rate in NOISE_RATES:

            print()
            print(
                f"UNKNOWN={unknown_rate:.1f} "
                f"NOISE={noise_rate:.1f}"
            )

            print("-" * 110)

            print(
                f"{'OBS':>4} "
                f"{'MODEL':>8} "
                f"{'ACC':>8} "
                f"{'CONF':>8} "
                f"{'CONF_ACC':>10} "
                f"{'UNKNOWN':>10} "
                f"{'RIGHT':>8}"
            )

            for observations in OBSERVATIONS:

                for mode in (
                    "binary",
                    "ternary",
                ):

                    subset = [
                        r
                        for r in results
                        if (
                            r["unknown_rate"]
                            == unknown_rate
                            and r["noise_rate"]
                            == noise_rate
                            and r["observations"]
                            == observations
                            and r["mode"]
                            == mode
                        )
                    ]

                    if not subset:
                        continue

                    acc = np.mean([
                        r["accuracy"]
                        for r in subset
                    ])

                    conf = np.mean([
                        r["confidence"]
                        for r in subset
                    ])

                    conf_acc = np.mean([
                        r[
                            "confident_accuracy"
                        ]
                        for r in subset
                    ])

                    unknown = np.mean([
                        r["unknown_count"]
                        for r in subset
                    ])

                    right = np.mean([
                        r["right_count"]
                        for r in subset
                    ])

                    print(
                        f"{observations:4d} "
                        f"{mode:>8} "
                        f"{acc:8.4f} "
                        f"{conf:8.4f} "
                        f"{conf_acc:10.4f} "
                        f"{unknown:10.2f} "
                        f"{right:8.2f}"
                    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print(
        "TEST 8 — INFORMATIONAL TERNARY STATE"
    )
    print("=" * 110)

    print()
    print(
        f"CPU threads      : {os.cpu_count()}"
    )

    print(
        f"Processes        : {N_PROCESSES}"
    )

    print(
        f"Episodes/job     : {N_EPISODES}"
    )

    print(
        f"Seeds            : {len(SEEDS)}"
    )

    print(
        f"Observations     : {OBSERVATIONS}"
    )

    print(
        f"Unknown rates    : {UNKNOWN_RATES}"
    )

    print(
        f"Noise rates      : {NOISE_RATES}"
    )

    jobs = []

    for observations in OBSERVATIONS:

        for unknown_rate in UNKNOWN_RATES:

            for noise_rate in NOISE_RATES:

                for seed in SEEDS:

                    for mode in (
                        "binary",
                        "ternary",
                    ):

                        jobs.append(
                            (
                                mode,
                                observations,
                                unknown_rate,
                                noise_rate,
                                seed,
                            )
                        )

    total_jobs = len(jobs)

    print()
    print(
        f"Parallel jobs    : {total_jobs}"
    )

    print()

    start = time.perf_counter()

    results = []

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

        completed = 0

        for future in as_completed(
            futures
        ):

            results.append(
                future.result()
            )

            completed += 1

            if (
                completed == 1
                or completed % 100 == 0
                or completed == total_jobs
            ):

                elapsed = (
                    time.perf_counter()
                    - start
                )

                rate = (
                    completed / elapsed
                )

                eta = (
                    (total_jobs - completed)
                    / rate
                )

                print(
                    f"[{completed:5d}/{total_jobs}] "
                    f"{completed / total_jobs * 100:6.2f}% "
                    f"| {elapsed:8.1f}s "
                    f"| ETA {eta:8.1f}s"
                )

    results.sort(
        key=lambda r: (
            r["unknown_rate"],
            r["noise_rate"],
            r["observations"],
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
    print("=" * 110)
    print("DONE")
    print("=" * 110)

    print(
        f"Total time : {elapsed:.2f} sec"
    )

    print(
        f"CSV        : {path}"
    )


if __name__ == "__main__":
    main()
