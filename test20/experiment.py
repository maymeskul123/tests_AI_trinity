from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST 20 — ИСПРАВЛЕННЫЙ QUATERNARY (ЧИСТАЯ ЛОГИКА)
# С многопроцессорностью для ускорения.
# ============================================================

N_H = 16
N_Q = 8
EPISODES = 3000
BUDGETS = [3, 5, 7, 10]
CONFLICT_RATES_A = [0.0, 0.4]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2

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
    source: int
    question: int
    value: int
    true_value: int

class Source:
    def __init__(self, idx, conflict_rate, rng):
        self.idx = idx
        self.conflict_rate = conflict_rate
        self.rng = rng

    def ask(self, target, q):
        true_value = QUESTIONS[q][target]
        if self.rng.random() < self.conflict_rate:
            return Observation(self.idx, q, 1 - true_value, true_value)
        return Observation(self.idx, q, true_value, true_value)

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

class AgentBase:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = None

    def answer(self):
        raise NotImplementedError

    def choose_question_and_source(self, sources):
        raise NotImplementedError

    def observe(self, source_idx, q, value):
        raise NotImplementedError

# ------------------------------------------------------------
# BINARY
# ------------------------------------------------------------
class BinaryAgent(AgentBase):
    def reset(self):
        self.states = [1] * N_H

    def possible(self, h):
        return self.states[h] == 1

    def choose_question_and_source(self, sources):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) <= 1:
            return None, None
        prior = [1.0/len(possible) if h in possible else 0.0 for h in HYPOTHESES]
        best_gain = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.conflict_rate)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (idx, q)
        return best_pair

    def observe(self, source_idx, q, value):
        for h in HYPOTHESES:
            if self.states[h] == 1 and QUESTIONS[q][h] != value:
                self.states[h] = 0

    def answer(self):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) == 1:
            return possible[0]
        return None

# ------------------------------------------------------------
# TERNARY
# ------------------------------------------------------------
class TernaryAgent(AgentBase):
    def reset(self):
        self.states = [UNKNOWN] * N_H

    def choose_question_and_source(self, sources):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None, None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None, None
        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        unknown_candidates = {h for h in active if self.states[h] == UNKNOWN}
        best_gain = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.conflict_rate)
                split = sum(1 for h in unknown_candidates if QUESTIONS[q][h] == 1)
                bonus = 0.01 * min(split, len(unknown_candidates)-split) if unknown_candidates else 0
                total = gain + bonus
                if total > best_gain:
                    best_gain = total
                    best_pair = (idx, q)
        return best_pair

    def observe(self, source_idx, q, value):
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

# ------------------------------------------------------------
# QUATERNARY (ИСПРАВЛЕННАЯ ЛОГИКА)
# ------------------------------------------------------------
class QuaternaryAgent(AgentBase):
    def reset(self):
        self.states = [UNKNOWN] * N_H

    def choose_question_and_source(self, sources):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None, None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None, None

        conflict_candidates = [h for h in active if self.states[h] == CONFLICT]
        if conflict_candidates:
            best_source_idx = min(enumerate(sources), key=lambda x: x[1].conflict_rate)[0]
            best_score = -1
            best_q = None
            for q in range(N_Q):
                split = sum(1 for h in conflict_candidates if QUESTIONS[q][h] == 1)
                score = split / len(conflict_candidates)
                if score > best_score:
                    best_score = score
                    best_q = q
            if best_q is not None:
                return best_source_idx, best_q
            else:
                return best_source_idx, 0

        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        best_gain = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.conflict_rate)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (idx, q)
        return best_pair

    def observe(self, source_idx, q, value):
        for h in HYPOTHESES:
            state = self.states[h]
            if state == WRONG:
                continue
            expected = QUESTIONS[q][h]
            if expected == value:
                self.states[h] = RIGHT
            else:
                if state == RIGHT:
                    self.states[h] = CONFLICT
                elif state == CONFLICT:
                    self.states[h] = WRONG
                else:
                    self.states[h] = WRONG

    def answer(self):
        confirmed = [h for h in HYPOTHESES if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return confirmed[0]
        return None

# ------------------------------------------------------------
# Функция для одного эпизода (для multiprocessing)
# ------------------------------------------------------------
def run_single_episode(AgentClass, seed, budget, conflict_A):
    rng = random.Random(seed)
    target = rng.randrange(N_H)

    source_A = Source(0, conflict_rate=conflict_A, rng=rng)
    source_B = Source(1, conflict_rate=0.0, rng=rng)
    sources = [source_A, source_B]

    agent = AgentClass(seed)
    prior = [1.0/N_H] * N_H
    queries = 0
    source_usage = [0, 0]
    false_confidence = 0
    abstain = 0

    for step in range(budget):
        answer = agent.answer()
        if answer is not None:
            break

        src_idx, q = agent.choose_question_and_source(sources)
        if src_idx is None or q is None:
            break

        obs = sources[src_idx].ask(target, q)
        queries += 1
        source_usage[src_idx] += 1

        prior = posterior_after(prior, q, obs.value, sources[src_idx].conflict_rate)
        agent.observe(src_idx, q, obs.value)

    final_answer = agent.answer()
    if final_answer is None:
        abstain = 1
        correct = 0
    else:
        correct = int(final_answer == target)
        if not correct:
            false_confidence = 1

    reward = 1.0 if correct else (-5.0 if final_answer is not None else -0.2)

    return {
        "correct": correct,
        "queries": queries,
        "false_confidence": false_confidence,
        "abstain": abstain,
        "reward": reward,
        "usage_A": source_usage[0],
        "usage_B": source_usage[1],
    }

def aggregate(rows):
    n = len(rows)
    return {key: sum(r[key] for r in rows)/n for key in rows[0]}

# ------------------------------------------------------------
# main с multiprocessing
# ------------------------------------------------------------
def main():
    out = []
    pool = mp.Pool(mp.cpu_count())

    for conflict_A in CONFLICT_RATES_A:
        print(f"\n=== conflict_A = {conflict_A:.1f} (Источник B идеальный) ===")
        for budget in BUDGETS:
            for name, AgentClass in [("BINARY", BinaryAgent),
                                     ("TERNARY", TernaryAgent),
                                     ("QUATERNARY", QuaternaryAgent)]:
                # Подготавливаем аргументы для параллельного запуска
                seeds = [ep + budget*100000 + int(conflict_A*1000) for ep in range(EPISODES)]
                # Частичная функция с фиксированными аргументами
                func = partial(run_single_episode, AgentClass, budget=budget, conflict_A=conflict_A)
                # Параллельный запуск
                results = pool.map(func, seeds)
                res = aggregate(results)
                print(f"{name:10s} b={budget:2d} acc={res['correct']:.4f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} "
                      f"srcA={res['usage_A']:.2f} srcB={res['usage_B']:.2f}")
                out.append({
                    "conflict_A": conflict_A,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    pool.close()
    pool.join()

    path = "results/test20/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
