from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# TEST 9
#
# Fair comparison:
#   Binary  = WRONG / RIGHT
#   Ternary = WRONG / UNKNOWN / RIGHT
#
# Important:
#   - fixed task set x=0..9
#   - same observations for both agents
#   - no hidden confidence counters
#   - UNKNOWN is explicitly tested
#   - contradictions are explicitly tested
#   - results saved to CSV
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_CANDIDATES = 10

# Fixed tasks.
# This is important: training and testing use the SAME x values.
TASKS = tuple(range(10))

# Number of observations PER TASK.
OBSERVATIONS = [1, 2, 3, 5, 8, 12, 20, 30]

# Probability that the observation contains no information.
UNKNOWN_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]

# Probability of an incorrect / noisy feedback.
NOISE_RATES = [0.0, 0.1, 0.2]

# Number of independent runs.
SEEDS = 50

# Output
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "test9"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "result.csv"


# ============================================================
# STATES
# ============================================================

WRONG = -1
UNKNOWN = 0
RIGHT = 1

ABSTAIN = -1


# ============================================================
# ENVIRONMENT
# ============================================================

def hidden_rule(x: int) -> int:
    """
    Hidden rule that the agents are trying to learn.

        f(x) = (3*x + 1) % 10
    """
    return (3 * x + 1) % 10


def correct_candidate(x: int) -> int:
    return hidden_rule(x)


# ============================================================
# BINARY AGENT
# ============================================================

class BinaryAgent:
    """
    Binary information state:

        0 = FALSE
        1 = TRUE

    There is NO explicit UNKNOWN state.

    If an observation is UNKNOWN, Binary simply cannot encode
    that state and therefore does not change its memory.

    If the same candidate receives contradictory information,
    the latest information replaces the previous information.
    """

    def __init__(self):
        self.memory: dict[tuple[int, int], bool] = {}

    def state(self, x: int, candidate: int) -> bool:
        return self.memory.get((x, candidate), False)

    def learn(
        self,
        x: int,
        candidate: int,
        feedback: int,
    ) -> None:

        if feedback == UNKNOWN:
            # Binary has no representation for UNKNOWN.
            # Therefore no information is stored.
            return

        if feedback == RIGHT:
            self.memory[(x, candidate)] = True
            return

        if feedback == WRONG:
            self.memory[(x, candidate)] = False
            return

        raise ValueError(f"Invalid feedback: {feedback}")

    def predict(
        self,
        x: int,
        candidates: tuple[int, ...],
    ) -> tuple[int, bool]:

        # Find candidate believed to be correct.
        for candidate in candidates:
            if self.state(x, candidate):
                return candidate, True

        # No known answer.
        return ABSTAIN, False


# ============================================================
# TERNARY AGENT
# ============================================================

class TernaryAgent:
    """
    Ternary information state:

        -1 = WRONG
         0 = UNKNOWN
        +1 = RIGHT

    Unlike Binary, UNKNOWN is explicitly represented.
    """

    def __init__(self):
        self.memory: dict[tuple[int, int], int] = {}

    def state(self, x: int, candidate: int) -> int:
        return self.memory.get((x, candidate), UNKNOWN)

    def learn(
        self,
        x: int,
        candidate: int,
        feedback: int,
    ) -> None:

        if feedback not in (WRONG, UNKNOWN, RIGHT):
            raise ValueError(f"Invalid feedback: {feedback}")

        self.memory[(x, candidate)] = feedback

    def predict(
        self,
        x: int,
        candidates: tuple[int, ...],
    ) -> tuple[int, bool]:

        # First priority: known correct candidate.
        for candidate in candidates:
            if self.state(x, candidate) == RIGHT:
                return candidate, True

        # No known answer.
        return ABSTAIN, False


# ============================================================
# OBSERVATION GENERATION
# ============================================================

@dataclass(frozen=True)
class Observation:
    x: int
    candidate: int
    feedback: int


def generate_observation(
    rng: random.Random,
    x: int,
    unknown_rate: float,
    noise_rate: float,
) -> Observation:

    answer = correct_candidate(x)

    # Choose a candidate.
    candidate = rng.randrange(N_CANDIDATES)

    # UNKNOWN observation.
    if rng.random() < unknown_rate:
        return Observation(
            x=x,
            candidate=candidate,
            feedback=UNKNOWN,
        )

    # Candidate is actually correct.
    if candidate == answer:

        # Noise can turn a correct observation into WRONG.
        if rng.random() < noise_rate:
            feedback = WRONG
        else:
            feedback = RIGHT

    # Candidate is wrong.
    else:

        # Noise can turn WRONG into RIGHT.
        if rng.random() < noise_rate:
            feedback = RIGHT
        else:
            feedback = WRONG

    return Observation(
        x=x,
        candidate=candidate,
        feedback=feedback,
    )


# ============================================================
# CONTRADICTORY OBSERVATIONS
# ============================================================

def generate_contradictory_observation_sequence(
    x: int,
) -> list[Observation]:

    """
    Deliberately create:

        RIGHT
        WRONG

    for the same (x, candidate).

    Binary:
        TRUE -> FALSE

    Ternary:
        RIGHT -> WRONG

    We additionally test the UNKNOWN state separately below.
    """

    answer = correct_candidate(x)

    return [
        Observation(
            x=x,
            candidate=answer,
            feedback=RIGHT,
        ),
        Observation(
            x=x,
            candidate=answer,
            feedback=WRONG,
        ),
    ]


# ============================================================
# UNKNOWN SEQUENCE
# ============================================================

def generate_unknown_sequence(
    x: int,
) -> list[Observation]:

    """
    Deliberately create:

        UNKNOWN
        UNKNOWN
        UNKNOWN

    for the correct candidate.
    """

    answer = correct_candidate(x)

    return [
        Observation(
            x=x,
            candidate=answer,
            feedback=UNKNOWN,
        ),
        Observation(
            x=x,
            candidate=answer,
            feedback=UNKNOWN,
        ),
        Observation(
            x=x,
            candidate=answer,
            feedback=UNKNOWN,
        ),
    ]


# ============================================================
# TRAINING
# ============================================================

def train_agent(
    agent,
    observations_per_task: int,
    unknown_rate: float,
    noise_rate: float,
    seed: int,
) -> None:

    rng = random.Random(seed)

    # Use fixed tasks.
    #
    # Every task receives exactly the requested number
    # of observations.
    for _ in range(observations_per_task):

        # Randomize task order so that agents do not get
        # any advantage from deterministic ordering.
        tasks = list(TASKS)
        rng.shuffle(tasks)

        for x in tasks:

            observation = generate_observation(
                rng=rng,
                x=x,
                unknown_rate=unknown_rate,
                noise_rate=noise_rate,
            )

            agent.learn(
                observation.x,
                observation.candidate,
                observation.feedback,
            )


# ============================================================
# EVALUATION
# ============================================================

@dataclass
class Metrics:
    total: int = 0
    correct: int = 0

    confident: int = 0
    confident_correct: int = 0
    false_confident: int = 0

    abstained: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def confidence_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.confident / self.total

    @property
    def confident_accuracy(self) -> float:
        if self.confident == 0:
            return 0.0
        return self.confident_correct / self.confident

    @property
    def false_confidence_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.false_confident / self.total

    @property
    def abstain_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.abstained / self.total


def evaluate_agent(agent) -> Metrics:

    metrics = Metrics()

    candidates = tuple(range(N_CANDIDATES))

    for x in TASKS:

        answer = correct_candidate(x)

        prediction, confident = agent.predict(
            x,
            candidates,
        )

        metrics.total += 1

        if prediction == ABSTAIN:

            metrics.abstained += 1

            continue

        metrics.confident += 1

        if prediction == answer:

            metrics.correct += 1
            metrics.confident_correct += 1

        else:

            metrics.false_confident += 1

    return metrics


# ============================================================
# CONTRADICTION TEST
# ============================================================

def evaluate_contradiction_handling() -> dict:

    binary = BinaryAgent()
    ternary = TernaryAgent()

    x = 7
    answer = correct_candidate(x)

    sequence = generate_contradictory_observation_sequence(x)

    for observation in sequence:

        binary.learn(
            observation.x,
            observation.candidate,
            observation.feedback,
        )

        ternary.learn(
            observation.x,
            observation.candidate,
            observation.feedback,
        )

    binary_state = binary.state(x, answer)
    ternary_state = ternary.state(x, answer)

    return {
        "x": x,
        "answer": answer,
        "binary_state": int(binary_state),
        "ternary_state": ternary_state,
        "binary_prediction": binary.predict(
            x,
            tuple(range(N_CANDIDATES)),
        )[0],
        "ternary_prediction": ternary.predict(
            x,
            tuple(range(N_CANDIDATES)),
        )[0],
    }


# ============================================================
# UNKNOWN TEST
# ============================================================

def evaluate_unknown_handling() -> dict:

    binary = BinaryAgent()
    ternary = TernaryAgent()

    x = 7
    answer = correct_candidate(x)

    sequence = generate_unknown_sequence(x)

    for observation in sequence:

        binary.learn(
            observation.x,
            observation.candidate,
            observation.feedback,
        )

        ternary.learn(
            observation.x,
            observation.candidate,
            observation.feedback,
        )

    binary_state = binary.state(x, answer)
    ternary_state = ternary.state(x, answer)

    binary_prediction, binary_confident = binary.predict(
        x,
        tuple(range(N_CANDIDATES)),
    )

    ternary_prediction, ternary_confident = ternary.predict(
        x,
        tuple(range(N_CANDIDATES)),
    )

    return {
        "x": x,
        "answer": answer,
        "binary_state": int(binary_state),
        "ternary_state": ternary_state,
        "binary_prediction": binary_prediction,
        "ternary_prediction": ternary_prediction,
        "binary_confident": int(binary_confident),
        "ternary_confident": int(ternary_confident),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    rows = []

    print("=" * 78)
    print("TEST 9 — FAIR BINARY vs TERNARY INFORMATION STATE")
    print("=" * 78)

    print()
    print("Tasks:", TASKS)
    print("Rule : f(x) = (3*x + 1) % 10")
    print()
    print("Binary : WRONG / RIGHT")
    print("Ternary: WRONG / UNKNOWN / RIGHT")
    print()

    # --------------------------------------------------------
    # Main matrix
    # --------------------------------------------------------

    for noise_rate in NOISE_RATES:

        print()
        print("-" * 78)
        print(f"NOISE = {noise_rate:.0%}")
        print("-" * 78)

        for unknown_rate in UNKNOWN_RATES:

            for observations in OBSERVATIONS:

                binary_metrics_all = []
                ternary_metrics_all = []

                for run_seed in range(SEEDS):

                    seed = SEED + run_seed

                    # ----------------------------
                    # Binary
                    # ----------------------------

                    binary = BinaryAgent()

                    train_agent(
                        agent=binary,
                        observations_per_task=observations,
                        unknown_rate=unknown_rate,
                        noise_rate=noise_rate,
                        seed=seed,
                    )

                    binary_metrics = evaluate_agent(binary)

                    binary_metrics_all.append(binary_metrics)

                    # ----------------------------
                    # Ternary
                    # ----------------------------

                    ternary = TernaryAgent()

                    train_agent(
                        agent=ternary,
                        observations_per_task=observations,
                        unknown_rate=unknown_rate,
                        noise_rate=noise_rate,
                        seed=seed,
                    )

                    ternary_metrics = evaluate_agent(ternary)

                    ternary_metrics_all.append(ternary_metrics)

                # ------------------------------------------------
                # Aggregate
                # ------------------------------------------------

                def avg(values):
                    return sum(values) / len(values)

                binary_acc = avg(
                    [m.accuracy for m in binary_metrics_all]
                )

                ternary_acc = avg(
                    [m.accuracy for m in ternary_metrics_all]
                )

                binary_conf = avg(
                    [m.confidence_rate for m in binary_metrics_all]
                )

                ternary_conf = avg(
                    [m.confidence_rate for m in ternary_metrics_all]
                )

                binary_conf_acc = avg(
                    [m.confident_accuracy for m in binary_metrics_all]
                )

                ternary_conf_acc = avg(
                    [m.confident_accuracy for m in ternary_metrics_all]
                )

                binary_false_conf = avg(
                    [m.false_confidence_rate for m in binary_metrics_all]
                )

                ternary_false_conf = avg(
                    [m.false_confidence_rate for m in ternary_metrics_all]
                )

                binary_abstain = avg(
                    [m.abstain_rate for m in binary_metrics_all]
                )

                ternary_abstain = avg(
                    [m.abstain_rate for m in ternary_metrics_all]
                )

                row = {
                    "noise": noise_rate,
                    "unknown": unknown_rate,
                    "observations": observations,

                    "binary_acc": binary_acc,
                    "ternary_acc": ternary_acc,
                    "acc_diff": ternary_acc - binary_acc,

                    "binary_conf": binary_conf,
                    "ternary_conf": ternary_conf,

                    "binary_conf_acc": binary_conf_acc,
                    "ternary_conf_acc": ternary_conf_acc,

                    "binary_false_conf": binary_false_conf,
                    "ternary_false_conf": ternary_false_conf,
                    "false_conf_diff":
                        ternary_false_conf - binary_false_conf,

                    "binary_abstain": binary_abstain,
                    "ternary_abstain": ternary_abstain,
                    "abstain_diff":
                        ternary_abstain - binary_abstain,
                }

                rows.append(row)

                print(
                    f"UNKNOWN={unknown_rate:.0%} "
                    f"OBS={observations:2d} | "
                    f"B={binary_acc:.3f} "
                    f"T={ternary_acc:.3f} | "
                    f"Δ={ternary_acc - binary_acc:+.3f} | "
                    f""
                    f"conf B={binary_conf:.3f} "
                    f"T={ternary_conf:.3f} | "
                    f""
                    f"false B={binary_false_conf:.3f} "
                    f"T={ternary_false_conf:.3f}"
                )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = list(rows[0].keys())

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
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("CSV SAVED")
    print("=" * 78)
    print(CSV_PATH)
    print(f"Rows: {len(rows)}")

    # --------------------------------------------------------
    # Special tests
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("SPECIAL TEST — UNKNOWN")
    print("=" * 78)

    unknown_result = evaluate_unknown_handling()

    print(
        f"x={unknown_result['x']}, "
        f"answer={unknown_result['answer']}"
    )

    print(
        "Binary:"
        f" state={unknown_result['binary_state']}, "
        f"prediction={unknown_result['binary_prediction']}, "
        f"confident={unknown_result['binary_confident']}"
    )

    print(
        "Ternary:"
        f" state={unknown_result['ternary_state']}, "
        f"prediction={unknown_result['ternary_prediction']}, "
        f"confident={unknown_result['ternary_confident']}"
    )

    print()
    print("=" * 78)
    print("SPECIAL TEST — CONTRADICTION")
    print("=" * 78)

    contradiction_result = evaluate_contradiction_handling()

    print(
        f"x={contradiction_result['x']}, "
        f"answer={contradiction_result['answer']}"
    )

    print(
        "Binary:"
        f" state={contradiction_result['binary_state']}, "
        f"prediction={contradiction_result['binary_prediction']}"
    )

    print(
        "Ternary:"
        f" state={contradiction_result['ternary_state']}, "
        f"prediction={contradiction_result['ternary_prediction']}"
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_experiment()