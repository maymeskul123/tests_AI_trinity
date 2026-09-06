from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST32 — МНОГОАГЕНТНАЯ СИСТЕМА С ОБМЕНОМ ИНФОРМАЦИЕЙ
# Два Threshold-агента обмениваются RIGHT и WRONG гипотезами
# ============================================================

N_H = 16
N_Q = 8
EPISODES = 2000
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

# ---------- БАЙЕСОВСКИЕ УТИЛИТЫ ----------
def entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)

def posterior_after(prior, q, obs_value, conflict_rate):
    likelihoods = []
    for h in HYPOTHESES:
        true_value = QUESTIONS[q][h]
        if obs_value == UNKNOWN:
            likelihood = 1.0
        else:
            if obs_value == true_value:
                likelihood = 1 - conflict_rate
            else:
                likelihood = conflict_rate
        likelihoods.append(likelihood)
    weighted = [prior[i] * likelihoods[i] for i in range(N_H)]
    total = sum(weighted)
    if total <= 0:
        return prior[:]
    return [x / total for x in weighted]

def expected_information_gain(prior, q, conflict_rate):
    before = entropy(prior)
    after = 0.0
    for obs in [0, 1]:
        post = posterior_after(prior, q, obs, conflict_rate)
        p_obs = 0.0
        for h in HYPOTHESES:
            true_value = QUESTIONS[q][h]
            if obs == true_value:
                lik = 1 - conflict_rate
            else:
                lik = conflict_rate
            p_obs += prior[h] * lik
        after += p_obs * entropy(post)
    return max(0.0, before - after)

# ---------- ВСПОМОГАТЕЛЬНЫЙ THRESHOLD-АГЕНТ (для внутреннего использования) ----------
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
        if value == UNKNOWN:
            return
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

# ---------- МНОГОАГЕНТНЫЙ С ОБМЕНОМ ----------
class CooperativeAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        # Два агента с разными счётчиками (разные сиды)
        self.agent1 = ThresholdInternal(seed + 1)
        self.agent2 = ThresholdInternal(seed + 2)
        self.turn = 0  # чередуем выбор вопроса

    def reset(self):
        self.agent1.reset()
        self.agent2.reset()
        self.turn = 0

    def choose_question(self, conflict_rate):
        # Чередуем агентов для выбора вопроса
        if self.turn % 2 == 0:
            q = self.agent1.choose_question(conflict_rate)
        else:
            q = self.agent2.choose_question(conflict_rate)
        self.turn += 1
        if q is not None:
            return q
        # Fallback: если один не смог, пробуем другого
        if self.turn % 2 == 1:
            q = self.agent2.choose_question(conflict_rate)
        else:
            q = self.agent1.choose_question(conflict_rate)
        return q

    def observe(self, q, value):
        # Оба агента получают одно и то же наблюдение
        self.agent1.observe(q, value)
        self.agent2.observe(q, value)
        # ОБМЕН ИНФОРМАЦИЕЙ
        self.exchange()

    def exchange(self):
        # Обмен RIGHT: если у одного RIGHT, другой переводит в RIGHT (если не WRONG)
        for h in HYPOTHESES:
            if self.agent1.states[h] == RIGHT and self.agent2.states[h] != WRONG:
                self.agent2.states[h] = RIGHT
                self.agent2.counter[h] = 0
            if self.agent2.states[h] == RIGHT and self.agent1.states[h] != WRONG:
                self.agent1.states[h] = RIGHT
                self.agent1.counter[h] = 0
            # Обмен WRONG: если у одного WRONG, другой тоже переводит в WRONG (если не RIGHT)
            if self.agent1.states[h] == WRONG and self.agent2.states[h] != RIGHT:
                self.agent2.states[h] = WRONG
                self.agent2.counter[h] = 0
            if self.agent2.states[h] == WRONG and self.agent1.states[h] != RIGHT:
                self.agent1.states[h] = WRONG
                self.agent1.counter[h] = 0

    def answer(self):
        # Ответ даётся, если оба агента согласны на одну гипотезу RIGHT
        ans1 = self.agent1.answer()
        ans2 = self.agent2.answer()
        if ans1 is not None and ans1 == ans2:
            return ans1
        # Если один уверен, а другой ещё нет, можно попробовать взять ответ первого,
        # но для безопасности требуем согласия.
        return None

# ---------- ОДИНОЧНЫЕ АГЕНТЫ (для сравнения) ----------
class BinaryAgent:
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
        if value == UNKNOWN:
            return
        for h in HYPOTHESES:
            if self.states[h] == 1 and QUESTIONS[q][h] != value:
                self.states[h] = 0

    def answer(self):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) == 1:
            return possible[0]
        return None

class TernaryAgent:
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
        if value == UNKNOWN:
            return
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

class QuaternaryThresholdAgent:
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
        if value == UNKNOWN:
            return
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

# ---------- ЭКСПЕРИМЕНТ ----------
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
            for name, AgentClass in [("BINARY", BinaryAgent),
                                     ("TERNARY", TernaryAgent),
                                     ("THRESHOLD", QuaternaryThresholdAgent),
                                     ("COOPERATIVE", CooperativeAgent)]:
                seeds = [ep + budget*100000 + int(conflict_rate*1000) for ep in range(EPISODES)]
                func = partial(run_episode, AgentClass, budget=budget, conflict_rate=conflict_rate)
                results = pool.map(func, seeds)
                res = aggregate(results)
                print(f"{name:12s} b={budget:2d} acc={res['correct']:.4f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} reward={res['reward']:.4f}")
                out.append({
                    "conflict_rate": conflict_rate,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    pool.close()
    pool.join()

    path = "results/test32/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
