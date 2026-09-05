import random
import csv
from dataclasses import dataclass

# ============================================================
# TEST 11
# Экономика информации:
#
# Binary:
#   WRONG / RIGHT
#
# Ternary:
#   WRONG / UNKNOWN / RIGHT
#
# Quaternary:
#   WRONG / UNKNOWN / RIGHT / CONFLICT
#
# Агент получает стоимость:
#   query            = -0.1
#   wrong answer     = -5
#   false confidence = -10
#   abstain          = -0.2
#   correct answer   = +1
#
# Задача:
# Агент должен определить скрытый правильный объект,
# одновременно решая:
#
#   1. Когда запрашивать новую информацию?
#   2. Когда доверять текущим данным?
#   3. Когда обнаруживать конфликт?
#   4. Когда отвечать?
#   5. Когда воздерживаться?
#
# ============================================================


SEED = 42
EPISODES = 5000

NUM_CANDIDATES = 10
MAX_QUERIES = 30

UNKNOWN_RATES = [0.0, 0.1, 0.2, 0.4]
CONFLICT_RATES = [0.0, 0.05, 0.1, 0.2]

QUERY_COST = 0.10
CORRECT_REWARD = 1.0
WRONG_PENALTY = 5.0
FALSE_CONFIDENCE_PENALTY = 10.0
ABSTAIN_PENALTY = 0.20


BINARY = "binary"
TERNARY = "ternary"
QUATERNARY = "quaternary"


@dataclass
class Result:
    agent: str
    unknown_rate: float
    conflict_rate: float

    solved_rate: float = 0.0
    accuracy: float = 0.0

    avg_queries: float = 0.0

    avg_reward: float = 0.0

    false_confidence_rate: float = 0.0
    abstain_rate: float = 0.0

    contradiction_rate: float = 0.0
    recovery_rate: float = 0.0


# ============================================================
# ENVIRONMENT
# ============================================================

class Environment:

    def __init__(
        self,
        num_candidates,
        unknown_rate,
        conflict_rate,
        rng,
    ):
        self.num_candidates = num_candidates
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng

        self.target = rng.randrange(num_candidates)

        # История наблюдений:
        #
        # candidate -> list:
        #
        # +1 = evidence TRUE
        # -1 = evidence FALSE
        #  0 = UNKNOWN
        #
        self.history = {
            i: []
            for i in range(num_candidates)
        }

    def query(self, candidate):

        r = self.rng.random()

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        if r < self.unknown_rate:

            value = 0

        else:

            # Базовая истина

            if candidate == self.target:
                value = 1
            else:
                value = -1

            # ------------------------------------------------
            # CONFLICT / corrupted observation
            # ------------------------------------------------

            if self.rng.random() < self.conflict_rate:

                value = -value

        self.history[candidate].append(value)

        return value


# ============================================================
# BASE AGENT
# ============================================================

class BaseAgent:

    def __init__(self, name, num_candidates):

        self.name = name
        self.num_candidates = num_candidates

        self.observations = {
            i: []
            for i in range(num_candidates)
        }

        self.total_queries = 0

        self.contradictions = 0
        self.recoveries = 0

        self.had_contradiction = set()

    def observe(self, candidate, value):

        self.observations[candidate].append(value)

        self.total_queries += 1

    def score_candidate(self, candidate):

        raise NotImplementedError

    def choose_candidate(self):

        scores = [
            self.score_candidate(i)
            for i in range(self.num_candidates)
        ]

        max_score = max(scores)

        best = [
            i
            for i, s in enumerate(scores)
            if s == max_score
        ]

        return random.choice(best)

    def confidence(self):

        scores = [
            self.score_candidate(i)
            for i in range(self.num_candidates)
        ]

        scores_sorted = sorted(scores, reverse=True)

        best = scores_sorted[0]

        second = (
            scores_sorted[1]
            if len(scores_sorted) > 1
            else -999
        )

        return best - second

    def answer(self):

        candidate = self.choose_candidate()

        confidence = self.confidence()

        return candidate, confidence


# ============================================================
# BINARY AGENT
#
# UNKNOWN не существует как отдельное состояние.
#
# UNKNOWN интерпретируется как отсутствие доказательства.
# ============================================================

class BinaryAgent(BaseAgent):

    def __init__(self, num_candidates):
        super().__init__(BINARY, num_candidates)

    def score_candidate(self, candidate):

        values = self.observations[candidate]

        if not values:
            return 0

        right = sum(
            1
            for v in values
            if v == 1
        )

        wrong = sum(
            1
            for v in values
            if v == -1
        )

        # Binary:
        #
        # RIGHT vs WRONG
        #
        # UNKNOWN просто игнорируется.

        return right - wrong


# ============================================================
# TERNARY AGENT
#
# -1 WRONG
#  0 UNKNOWN
# +1 RIGHT
#
# UNKNOWN является полноценным состоянием:
# он уменьшает уверенность, но не означает WRONG.
# ============================================================

class TernaryAgent(BaseAgent):

    def __init__(self, num_candidates):
        super().__init__(TERNARY, num_candidates)

    def score_candidate(self, candidate):

        values = self.observations[candidate]

        if not values:
            return 0

        right = sum(
            1
            for v in values
            if v == 1
        )

        wrong = sum(
            1
            for v in values
            if v == -1
        )

        unknown = sum(
            1
            for v in values
            if v == 0
        )

        # UNKNOWN снижает силу уверенности.
        #
        # Это ключевое отличие:
        #
        # Binary:
        #   evidence отсутствует
        #
        # Ternary:
        #   evidence = UNKNOWN
        #
        # UNKNOWN является информацией о том,
        # что система столкнулась с неопределённостью.

        evidence = right - wrong

        uncertainty_penalty = unknown * 0.15

        if evidence > 0:
            return evidence - uncertainty_penalty

        elif evidence < 0:
            return evidence + uncertainty_penalty

        return 0


# ============================================================
# QUATERNARY AGENT
#
# WRONG
# UNKNOWN
# RIGHT
# CONFLICT
#
# CONFLICT возникает когда есть:
#
# RIGHT + WRONG evidence
#
# В отличие от ternary агент сохраняет конфликт
# как отдельное состояние и может целенаправленно
# перепроверять кандидата.
# ============================================================

class QuaternaryAgent(BaseAgent):

    def __init__(self, num_candidates):
        super().__init__(QUATERNARY, num_candidates)

        self.conflict_candidates = set()

    def observe(self, candidate, value):

        super().observe(candidate, value)

        values = self.observations[candidate]

        has_right = 1 in values
        has_wrong = -1 in values

        if has_right and has_wrong:

            if candidate not in self.conflict_candidates:

                self.contradictions += 1

                self.conflict_candidates.add(candidate)

                self.had_contradiction.add(candidate)

    def score_candidate(self, candidate):

        values = self.observations[candidate]

        if not values:
            return 0

        right = sum(
            1
            for v in values
            if v == 1
        )

        wrong = sum(
            1
            for v in values
            if v == -1
        )

        unknown = sum(
            1
            for v in values
            if v == 0
        )

        evidence = right - wrong

        # CONFLICT не уничтожает информацию.
        #
        # Агент сохраняет:
        #
        # RIGHT evidence
        # WRONG evidence
        #
        # и использует степень конфликта.

        conflict = min(right, wrong)

        conflict_penalty = conflict * 0.35

        uncertainty_penalty = unknown * 0.10

        if evidence > 0:
            return evidence - conflict_penalty - uncertainty_penalty

        elif evidence < 0:
            return evidence + conflict_penalty + uncertainty_penalty

        return -conflict_penalty


# ============================================================
# QUERY POLICY
# ============================================================

def choose_query(agent):

    # --------------------------------------------------------
    # QUATERNARY:
    #
    # Если обнаружен конфликт —
    # сначала перепроверяем конфликтующие кандидаты.
    # --------------------------------------------------------

    if isinstance(agent, QuaternaryAgent):

        conflicts = list(agent.conflict_candidates)

        if conflicts:

            candidate = min(
                conflicts,
                key=lambda c: abs(agent.score_candidate(c))
            )

            return candidate

    # --------------------------------------------------------
    # TERNARY:
    #
    # Предпочитаем кандидатов с высокой неопределённостью.
    # --------------------------------------------------------

    if isinstance(agent, TernaryAgent):

        unknown_counts = {}

        for c in range(agent.num_candidates):

            unknown_counts[c] = sum(
                1
                for v in agent.observations[c]
                if v == 0
            )

        max_unknown = max(unknown_counts.values())

        if max_unknown > 0:

            candidates = [
                c
                for c, count in unknown_counts.items()
                if count == max_unknown
            ]

            return random.choice(candidates)

    # --------------------------------------------------------
    # GENERAL:
    #
    # Проверяем кандидатов с наибольшим потенциалом.
    # --------------------------------------------------------

    scores = [
        agent.score_candidate(c)
        for c in range(agent.num_candidates)
    ]

    max_score = max(scores)

    candidates = [
        c
        for c, score in enumerate(scores)
        if score == max_score
    ]

    # Если всё неизвестно —
    # исследуем наименее проверенного.

    if max_score == 0:

        counts = [
            len(agent.observations[c])
            for c in range(agent.num_candidates)
        ]

        min_count = min(counts)

        candidates = [
            c
            for c, count in enumerate(counts)
            if count == min_count
        ]

    return random.choice(candidates)


# ============================================================
# RUN EPISODE
# ============================================================

def run_episode(
    agent_type,
    unknown_rate,
    conflict_rate,
    rng,
):

    env = Environment(
        NUM_CANDIDATES,
        unknown_rate,
        conflict_rate,
        rng,
    )

    if agent_type == BINARY:
        agent = BinaryAgent(NUM_CANDIDATES)

    elif agent_type == TERNARY:
        agent = TernaryAgent(NUM_CANDIDATES)

    elif agent_type == QUATERNARY:
        agent = QuaternaryAgent(NUM_CANDIDATES)

    else:
        raise ValueError(agent_type)

    reward = 0.0

    abstained = False

    # ========================================================
    # ACTIVE INFORMATION SEARCH
    # ========================================================

    for step in range(MAX_QUERIES):

        candidate = choose_query(agent)

        value = env.query(candidate)

        agent.observe(candidate, value)

        reward -= QUERY_COST

        # ----------------------------------------------------
        # Проверяем уверенность.
        # ----------------------------------------------------

        answer, confidence = agent.answer()

        # Для принятия решения нужна
        # достаточная разница между лучшей
        # и второй гипотезой.

        threshold = 2.0

        # Чем больше UNKNOWN,
        # тем выше требуется уверенность.

        if isinstance(agent, TernaryAgent):

            total_unknown = sum(
                sum(
                    1
                    for v in agent.observations[c]
                    if v == 0
                )
                for c in range(NUM_CANDIDATES)
            )

            threshold += min(
                total_unknown * 0.05,
                1.0,
            )

        if isinstance(agent, QuaternaryAgent):

            threshold += (
                len(agent.conflict_candidates) * 0.25
            )

        # ----------------------------------------------------
        # Агент может закончить раньше.
        # ----------------------------------------------------

        if confidence >= threshold:

            break

    # ========================================================
    # FINAL DECISION
    # ========================================================

    answer, confidence = agent.answer()

    # --------------------------------------------------------
    # Abstain policy
    # --------------------------------------------------------

    abstain_threshold = 0.75

    if confidence < abstain_threshold:

        abstained = True

        reward -= ABSTAIN_PENALTY

        correct = False
        false_confidence = False

    else:

        correct = (
            answer == env.target
        )

        if correct:

            reward += CORRECT_REWARD

            false_confidence = False

        else:

            # Высокая уверенность в неправильном ответе
            # получает дополнительный штраф.

            if confidence >= 2.0:

                reward -= FALSE_CONFIDENCE_PENALTY

                false_confidence = True

            else:

                reward -= WRONG_PENALTY

                false_confidence = False

    # ========================================================
    # QUATERNARY RECOVERY
    # ========================================================

    recovery = False

    if isinstance(agent, QuaternaryAgent):

        if agent.contradictions > 0:

            # Если после конфликта
            # агент всё-таки дал правильный ответ,
            # считаем это recovery.

            if correct:

                recovery = True

                agent.recoveries += 1

    return {
        "correct": correct,
        "queries": agent.total_queries,
        "reward": reward,
        "abstained": abstained,
        "false_confidence": false_confidence,
        "contradictions": agent.contradictions,
        "recovery": recovery,
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment():

    rng = random.Random(SEED)

    results = []

    agent_types = [
        BINARY,
        TERNARY,
        QUATERNARY,
    ]

    total_tests = (
        len(UNKNOWN_RATES)
        * len(CONFLICT_RATES)
        * len(agent_types)
    )

    test_number = 0

    for unknown_rate in UNKNOWN_RATES:

        for conflict_rate in CONFLICT_RATES:

            for agent_type in agent_types:

                test_number += 1

                print(
                    f"[{test_number}/{total_tests}] "
                    f"{agent_type} "
                    f"unknown={unknown_rate} "
                    f"conflict={conflict_rate}"
                )

                solved = 0

                total_queries = 0

                total_reward = 0.0

                false_confidence = 0

                abstains = 0

                contradictions = 0

                recoveries = 0

                for _ in range(EPISODES):

                    episode_rng = random.Random(
                        rng.randrange(10**12)
                    )

                    r = run_episode(
                        agent_type,
                        unknown_rate,
                        conflict_rate,
                        episode_rng,
                    )

                    if r["correct"]:
                        solved += 1

                    total_queries += r["queries"]

                    total_reward += r["reward"]

                    if r["false_confidence"]:
                        false_confidence += 1

                    if r["abstained"]:
                        abstains += 1

                    contradictions += r["contradictions"]

                    if r["recovery"]:
                        recoveries += 1

                solved_rate = solved / EPISODES

                # Accuracy среди всех эпизодов,
                # abstain считается неуспехом.

                accuracy = solved_rate

                contradiction_rate = (
                    contradictions / EPISODES
                )

                recovery_rate = (
                    recoveries / EPISODES
                )

                result = Result(
                    agent=agent_type,
                    unknown_rate=unknown_rate,
                    conflict_rate=conflict_rate,
                    solved_rate=solved_rate,
                    accuracy=accuracy,
                    avg_queries=(
                        total_queries / EPISODES
                    ),
                    avg_reward=(
                        total_reward / EPISODES
                    ),
                    false_confidence_rate=(
                        false_confidence / EPISODES
                    ),
                    abstain_rate=(
                        abstains / EPISODES
                    ),
                    contradiction_rate=(
                        contradiction_rate
                    ),
                    recovery_rate=(
                        recovery_rate
                    ),
                )

                results.append(result)

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    filename = "results/test11/result.csv"

    fields = [
        "agent",
        "unknown_rate",
        "conflict_rate",
        "solved_rate",
        "accuracy",
        "avg_queries",
        "avg_reward",
        "false_confidence_rate",
        "abstain_rate",
        "contradiction_rate",
        "recovery_rate",
    ]

    with open(
        filename,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for r in results:

            writer.writerow({
                "agent": r.agent,
                "unknown_rate": r.unknown_rate,
                "conflict_rate": r.conflict_rate,
                "solved_rate": r.solved_rate,
                "accuracy": r.accuracy,
                "avg_queries": r.avg_queries,
                "avg_reward": r.avg_reward,
                "false_confidence_rate": (
                    r.false_confidence_rate
                ),
                "abstain_rate": r.abstain_rate,
                "contradiction_rate": (
                    r.contradiction_rate
                ),
                "recovery_rate": r.recovery_rate,
            })

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for r in results:

        print(
            f"{r.agent:10} "
            f"unknown={r.unknown_rate:.1f} "
            f"conflict={r.conflict_rate:.2f} | "
            f"acc={r.accuracy:.3f} | "
            f"queries={r.avg_queries:.2f} | "
            f"reward={r.avg_reward:.3f} | "
            f"false_conf={r.false_confidence_rate:.3f} | "
            f"abstain={r.abstain_rate:.3f} | "
            f"contradictions={r.contradiction_rate:.3f} | "
            f"recovery={r.recovery_rate:.3f}"
        )

    print()
    print(f"Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TEST 11 — ECONOMY OF INFORMATION")
    print("=" * 70)
    print()

    results = run_experiment()

    save_results(results)
