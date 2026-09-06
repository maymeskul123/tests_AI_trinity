from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST29 — АНСАМБЛЕВЫЙ АГЕНТ (TERNARY + QUATERNARY_THRESHOLD + BINARY)
#
# Три агента работают параллельно, каждый собирает свою информацию.
# Решение принимается голосованием (консенсус).
# ============================================================

N_H = 16
N_Q = 8
EPISODES = 3000
BUDGETS = [5, 7, 10]
CONFLICT_RATES = [0.0, 0.2, 0.4]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2
CONFLICT_THRESHOLD = 3

REWARD_CORRECT = 1.0
REWARD_FALSE = -100.0
REWARD_ABSTAIN = -0.2
QUERY_COST = 0.05

HYPOTHESES = tuple(range(N_H))

def build_questions():
    qs = []
    for i in range(4):
        qs.append(tuple((h >> i) & 1 for h in HYPOTHESES))
    qs.append(tuple(((h >> 0) ^ (h >> 1)) & 1 for h in HYPOTHESES))
    qs.append(tuple(((h >> 1) ^ (h >> 2)) & 1 for h in HYPOTHESES))
    qs.append(tuple(((h >> 0) ^ (h >> 3)) & 1 for h in HYPOTHESES))
    qs.append(tuple(1 if h in {0,1,2,3,4,5,6} else 0 for h in HYPOTHESES))
    return tuple(qs)

QUESTIONS = build_questions()

@dataclass
class Observation:
    question: int
    value: int
    true_value: int

class Environment:
    def __init__(self, target, conflict_rate, rng):
        self.target = target
        self.conflict_rate = conflict_rate
        self.rng = rng

    def ask(self, q):
        true_value = QUESTIONS[q][self.target]
        if self.rng.random() < self.conflict_rate:
            return Observation(q, 1 - true_value, true_value)
        return Observation(q, true_value, true_value)

def entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)

def posterior_after(prior, q, obs_value, conflict_rate):
    likelihoods = []
    for h in HYPOTHESES:
        true_value = QUESTIONS[q][h]
        if obs_value == true_value:
            likelihood = 1 - conflict_rate
        else:
            likelihood = conflict_rate
        likelihoods.append(likelihood)
    weighted = [prior[h] * likelihoods[h] for h in HYPOTHESES]
    total = sum(weighted)
    if total <= 0:
        return prior[:]
    return [x / total for x in weighted]

def expected_information_gain(prior, q, conflict_rate):
    before = entropy(prior)
    after = 0.0
    for obs in [0, 1]:
        post = posterior_after(prior, q, obs, conflict_rate)
        p_obs = sum(prior[h] * ((1 - conflict_rate) if obs == QUESTIONS[q][h] else conflict_rate) for h in HYPOTHESES)
        after += p_obs * entropy(post)
    return max(0.0, before - after)

# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ АГЕНТЫ (для ансамбля)
# ------------------------------------------------------------

class BinaryInternal:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = [1] * N_H

    def possible(self, h):
        return self.states[h] == 1

    def choose_question(self, conflict_rate):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) <= 1:
            return None
        prior = [1.0/len(possible) if h in possible else 0.0 for h in HYPOTHESES]
        best_gain = -1
        best_q = None
        for q in range(N_Q):
            gain = expected_information_gain(prior, q, conflict_rate)
            if gain > best_gain:
                best_gain = gain
                best_q = q
        return best_q

    def observe(self, q, value):
        for h in HYPOTHESES:
            if self.states[h] == 1 and QUESTIONS[q][h] != value:
                self.states[h] = 0

    def answer(self):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) == 1:
            return possible[0]
        return None

class TernaryInternal:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = [UNKNOWN] * N_H

    def choose_question(self, conflict_rate):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None
        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        unknown_candidates = {h for h in active if self.states[h] == UNKNOWN}
        best_gain = -1
        best_q = None
        for q in range(N_Q):
            gain = expected_information_gain(prior, q, conflict_rate)
            split = sum(1 for h in unknown_candidates if QUESTIONS[q][h] == 1)
            bonus = 0.01 * min(split, len(unknown_candidates)-split) if unknown_candidates else 0
            total = gain + bonus
            if total > best_gain:
                best_gain = total
                best_q = q
        return best_q

    def observe(self, q, value):
        for h in HYPOTHESES:
            if self.states[h] == WRONG:
                continue
            expected = QUESTIONS[q][h]
            if expected == value:
                self.states[h] = RIGHT
            else:
                self.states[h] = WRONG

    def answer(self):
        confirmed = [h for h in HYPOTHESES if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return confirmed[0]
        return None

class ThresholdInternal:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = [UNKNOWN] * N_H
        self.counter = [0] * N_H

    def choose_question(self, conflict_rate):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None

        conflict_candidates = [h for h in active if self.states[h] == CONFLICT]
        if conflict_candidates:
            best_score = -1
            best_q = None
            for q in range(N_Q):
                split = sum(1 for h in conflict_candidates if QUESTIONS[q][h] == 1)
                score = split / len(conflict_candidates)
                if score > best_score:
                    best_score = score
                    best_q = q
            if best_q is not None:
                return best_q

        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        best_gain = -1
        best_q = None
        for q in range(N_Q):
            gain = expected_information_gain(prior, q, conflict_rate)
            if gain > best_gain:
                best_gain = gain
                best_q = q
        return best_q

    def observe(self, q, value):
        for h in HYPOTHESES:
            state = self.states[h]
            if state == WRONG:
                continue
            expected = QUESTIONS[q][h]
            if expected == value:
                self.counter[h] = 0
                self.states[h] = RIGHT
            else:
                if state == RIGHT:
                    self.states[h] = CONFLICT
                    self.counter[h] = 1
                elif state == CONFLICT:
                    self.counter[h] += 1
                    if self.counter[h] >= CONFLICT_THRESHOLD:
                        self.states[h] = WRONG
                else:
                    self.states[h] = WRONG

    def answer(self):
        confirmed = [h for h in HYPOTHESES if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return confirmed[0]
        return None

# ------------------------------------------------------------
# АНСАМБЛЕВЫЙ АГЕНТ
# ------------------------------------------------------------
class EnsembleAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        # Создаём трёх внутренних агентов с разными сидами
        self.binary = BinaryInternal(seed + 1)
        self.ternary = TernaryInternal(seed + 2)
        self.threshold = ThresholdInternal(seed + 3)

    def reset(self):
        self.binary.reset()
        self.ternary.reset()
        self.threshold.reset()

    def choose_question(self, conflict_rate):
        # Выбираем вопрос от агента с наибольшей неопределённостью
        # (чем больше активных гипотез, тем выше неопределённость)
        def active_count(agent):
            return sum(1 for h in HYPOTHESES if agent.states[h] != WRONG and agent.states[h] != 0)
        
        counts = [active_count(self.binary), active_count(self.ternary), active_count(self.threshold)]
        # Предпочитаем агента с наибольшим числом активных гипотез (он "менее уверен")
        # Чередуем, чтобы все агенты получали информацию
        # Выбираем случайно с весами, обратными неопределённости? Нет, выбираем агента с максимальной энтропией.
        agents = [self.binary, self.ternary, self.threshold]
        # Просто чередуем: используем random.choice
        agent = self.rng.choice(agents)
        q = agent.choose_question(conflict_rate)
        if q is not None:
            return q
        # Fallback: ищем у любого агента
        for a in agents:
            q = a.choose_question(conflict_rate)
            if q is not None:
                return q
        return None

    def observe(self, q, value):
        # Все агенты наблюдают один и тот же ответ
        self.binary.observe(q, value)
        self.ternary.observe(q, value)
        self.threshold.observe(q, value)

    def answer(self):
        answers = []
        for agent in [self.binary, self.ternary, self.threshold]:
            a = agent.answer()
            if a is not None:
                answers.append(a)
        # Если все три агента дали одинаковый ответ — возвращаем его
        if len(answers) == 3 and answers[0] == answers[1] == answers[2]:
            return answers[0]
        # Если два агента дали одинаковый ответ — возвращаем его
        if len(answers) >= 2:
            from collections import Counter
            cnt = Counter(answers)
            most_common = cnt.most_common(1)[0]
            if most_common[1] >= 2:
                return most_common[0]
        # Если нет консенсуса или недостаточно ответов — воздерживаемся
        return None

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------
def run_episode(AgentClass, seed, budget, conflict_rate):
    rng = random.Random(seed)
    target = rng.randrange(N_H)
    env = Environment(target, conflict_rate, rng)
    agent = AgentClass(seed)
    prior = [1.0/N_H] * N_H
    queries = 0
    false_confidence = 0
    abstain = 0

    for step in range(budget):
        answer = agent.answer()
        if answer is not None:
            break
        q = agent.choose_question(conflict_rate)
        if q is None:
            break
        obs = env.ask(q)
        queries += 1
        prior = posterior_after(prior, q, obs.value, conflict_rate)
        agent.observe(q, obs.value)

    final_answer = agent.answer()
    if final_answer is None:
        abstain = 1
        correct = 0
    else:
        correct = int(final_answer == target)
        if not correct:
            false_confidence = 1

    reward = (REWARD_CORRECT if correct else
              REWARD_FALSE if final_answer is not None else
              REWARD_ABSTAIN) - QUERY_COST * queries

    return {
        "correct": correct,
        "queries": queries,
        "false_confidence": false_confidence,
        "abstain": abstain,
        "reward": reward,
    }

def aggregate(rows):
    n = len(rows)
    return {key: sum(r[key] for r in rows)/n for key in rows[0]}

def main():
    out = []
    pool = mp.Pool(mp.cpu_count())

    for conflict_rate in CONFLICT_RATES:
        print(f"\n=== conflict_rate = {conflict_rate:.1f} ===")
        for budget in BUDGETS:
            for name, AgentClass in [("BINARY", BinaryInternal),   # используем внутренние классы для сравнения
                                     ("TERNARY", TernaryInternal),
                                     ("QUATERNARY_THRESHOLD", ThresholdInternal),
                                     ("ENSEMBLE", EnsembleAgent)]:
                # Для внутренних классов нужна обёртка, чтобы они соответствовали интерфейсу
                # Но мы просто используем их напрямую, так как они уже имеют нужные методы
                # Однако в run_episode мы ожидаем AgentClass(seed) — работает
                seeds = [ep + budget*100000 + int(conflict_rate*1000) for ep in range(EPISODES)]
                func = partial(run_episode, AgentClass, budget=budget, conflict_rate=conflict_rate)
                results = pool.map(func, seeds)
                res = aggregate(results)
                print(f"{name:25s} b={budget:2d} acc={res['correct']:.4f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} "
                      f"reward={res['reward']:.4f}")
                out.append({
                    "conflict_rate": conflict_rate,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    pool.close()
    pool.join()

    path = "results/test29/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
