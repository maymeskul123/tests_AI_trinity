from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass

# ============================================================
# TEST 18 — ВЫБОР МЕЖДУ ИСТОЧНИКАМИ (ДЕШЁВЫЙ ШУМНЫЙ vs ДОРОГОЙ ТОЧНЫЙ)
#
# Гипотеза: Quaternary использует CONFLICT для переключения
# на дорогой точный источник, что повышает точность без
# увеличения ложной уверенности.
# ============================================================

N_H = 16
N_Q = 8
EPISODES = 3000

BUDGETS = [3, 5, 7, 10]
UNKNOWN_RATES = [0.0]  # неизвестность не используется, только конфликты
CONFLICT_RATES_A = [0.0, 0.2, 0.4]  # conflict_rate для источника A
COST_A = 0.01          # дешёвый источник
COST_B = 0.2           # дорогой источник (в 20 раз дороже)

STOP_THRESHOLD = 0.1   # порог для остановки

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
    def __init__(self, idx, unknown_rate, conflict_rate, cost, rng):
        self.idx = idx
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.cost = cost
        self.rng = rng

    def ask(self, target, q):
        true_value = QUESTIONS[q][target]
        if self.rng.random() < self.unknown_rate:
            return Observation(self.idx, q, UNKNOWN, true_value)
        if self.rng.random() < self.conflict_rate:
            return Observation(self.idx, q, 1 - true_value, true_value)
        return Observation(self.idx, q, true_value, true_value)

def entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)

def posterior_after(prior, q, obs_value, unknown_rate, conflict_rate):
    likelihoods = []
    for h in HYPOTHESES:
        true_value = QUESTIONS[q][h]
        if obs_value == UNKNOWN:
            likelihood = unknown_rate
        else:
            if obs_value == true_value:
                likelihood = (1 - unknown_rate) * (1 - conflict_rate)
            else:
                likelihood = (1 - unknown_rate) * conflict_rate
        likelihoods.append(likelihood)
    weighted = [prior[h] * likelihoods[h] for h in HYPOTHESES]
    total = sum(weighted)
    if total <= 0:
        return prior[:]
    return [x / total for x in weighted]

def expected_information_gain(prior, q, unknown_rate, conflict_rate):
    before = entropy(prior)
    outcomes = [0, 1]
    if unknown_rate > 0:
        outcomes.append(UNKNOWN)
    expected_after = 0.0
    for obs in outcomes:
        post = posterior_after(prior, q, obs, unknown_rate, conflict_rate)
        p_obs = 0.0
        for h in HYPOTHESES:
            true_value = QUESTIONS[q][h]
            if obs == UNKNOWN:
                lik = unknown_rate
            elif obs == true_value:
                lik = (1 - unknown_rate) * (1 - conflict_rate)
            else:
                lik = (1 - unknown_rate) * conflict_rate
            p_obs += prior[h] * lik
        expected_after += p_obs * entropy(post)
    return max(0.0, before - expected_after)

class AgentBase:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.states = None

    def answer(self):
        raise NotImplementedError

    def choose_question_and_source(self, sources, cost_b):
        # sources: list of Source objects
        raise NotImplementedError

    def observe(self, source_idx, q, value):
        raise NotImplementedError

class BinaryAgent(AgentBase):
    def reset(self):
        self.states = [1] * N_H

    def possible(self, h):
        return self.states[h] == 1

    def choose_question_and_source(self, sources, cost_b):
        possible = [h for h in HYPOTHESES if self.possible(h)]
        if len(possible) <= 1:
            return None, None
        prior = [1.0/len(possible) if h in possible else 0.0 for h in HYPOTHESES]
        best_score = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            # Оцениваем информативность с учётом стоимости
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                # Учитываем стоимость: делим gain на cost (чем дешевле, тем лучше)
                score = gain / s.cost
                if score > best_score:
                    best_score = score
                    best_pair = (idx, q)
        if best_pair[0] is not None:
            # Проверяем порог остановки (используем максимальный gain по всем источникам)
            max_gain = max(expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                           for s in sources for q in range(N_Q))
            if max_gain / min(s.cost for s in sources) < STOP_THRESHOLD:
                return None, None
        return best_pair

    def observe(self, source_idx, q, value):
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

class TernaryAgent(AgentBase):
    def reset(self):
        self.states = [UNKNOWN] * N_H

    def choose_question_and_source(self, sources, cost_b):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None, None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None, None
        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        unknown_candidates = {h for h in active if self.states[h] == UNKNOWN}
        best_score = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                # Бонус за разделение UNKNOWN
                split = sum(1 for h in unknown_candidates if QUESTIONS[q][h] == 1)
                bonus = 0.01 * min(split, len(unknown_candidates)-split) if unknown_candidates else 0
                total = gain + bonus
                score = total / s.cost
                if score > best_score:
                    best_score = score
                    best_pair = (idx, q)
        if best_pair[0] is not None:
            max_gain = max(expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                           for s in sources for q in range(N_Q))
            if max_gain / min(s.cost for s in sources) < STOP_THRESHOLD:
                return None, None
        return best_pair

    def observe(self, source_idx, q, value):
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

class QuaternaryAgent(AgentBase):
    def reset(self):
        self.states = [UNKNOWN] * N_H

    def choose_question_and_source(self, sources, cost_b):
        active = [h for h in HYPOTHESES if self.states[h] != WRONG]
        if len(active) <= 1:
            return None, None
        confirmed = [h for h in active if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return None, None

        conflict_candidates = [h for h in active if self.states[h] == CONFLICT]
        if conflict_candidates:
            # Если есть конфликты, пытаемся их разрешить с помощью более точного источника
            # Сортируем источники по conflict_rate (чем меньше, тем лучше)
            sorted_sources = sorted(enumerate(sources), key=lambda x: x[1].conflict_rate)
            for idx, s in sorted_sources:
                # Выбираем вопрос, который лучше всего разделяет конфликтующие гипотезы
                best_score = -1
                best_q = None
                for q in range(N_Q):
                    split = sum(1 for h in conflict_candidates if QUESTIONS[q][h] == 1)
                    score = split / len(conflict_candidates)
                    if score > best_score:
                        best_score = score
                        best_q = q
                if best_q is not None:
                    # Проверяем, стоит ли задавать вопрос (информативность / стоимость)
                    gain = expected_information_gain(
                        [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES],
                        best_q, s.unknown_rate, s.conflict_rate
                    )
                    if gain / s.cost >= STOP_THRESHOLD:
                        return idx, best_q
            # Если не нашли подходящего вопроса, останавливаемся
            return None, None

        # Нет конфликтов – стандартный выбор с учётом стоимости
        prior = [1.0/len(active) if h in active else 0.0 for h in HYPOTHESES]
        best_score = -1
        best_pair = (None, None)
        for idx, s in enumerate(sources):
            for q in range(N_Q):
                gain = expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                score = gain / s.cost
                if score > best_score:
                    best_score = score
                    best_pair = (idx, q)
        if best_pair[0] is not None:
            max_gain = max(expected_information_gain(prior, q, s.unknown_rate, s.conflict_rate)
                           for s in sources for q in range(N_Q))
            if max_gain / min(s.cost for s in sources) < STOP_THRESHOLD:
                return None, None
        return best_pair

    def observe(self, source_idx, q, value):
        if value == UNKNOWN:
            return
        for h in HYPOTHESES:
            state = self.states[h]
            if state == WRONG:
                continue
            expected = QUESTIONS[q][h]
            if expected == value:
                if state == WRONG:
                    self.states[h] = CONFLICT
                elif state == CONFLICT:
                    self.states[h] = RIGHT
                else:
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

def run_episode(AgentClass, seed, budget, conflict_rate_A, cost_A, cost_B):
    rng = random.Random(seed)
    target = rng.randrange(N_H)

    # Источник A: дешёвый, с conflict_rate_A
    source_A = Source(0, unknown_rate=0.0, conflict_rate=conflict_rate_A, cost=cost_A, rng=rng)
    # Источник B: дорогой, идеальный
    source_B = Source(1, unknown_rate=0.0, conflict_rate=0.0, cost=cost_B, rng=rng)
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

        src_idx, q = agent.choose_question_and_source(sources, cost_B)
        if src_idx is None or q is None:
            break

        obs = sources[src_idx].ask(target, q)
        queries += 1
        source_usage[src_idx] += 1

        # Обновляем prior
        prior = posterior_after(prior, q, obs.value, sources[src_idx].unknown_rate, sources[src_idx].conflict_rate)
        agent.observe(src_idx, q, obs.value)

    final_answer = agent.answer()
    if final_answer is None:
        abstain = 1
        correct = 0
    else:
        correct = int(final_answer == target)
        if not correct:
            false_confidence = 1

    # Награда: точность минус суммарная стоимость запросов
    total_cost = source_usage[0]*cost_A + source_usage[1]*cost_B
    reward = (1.0 if correct else -5.0 if final_answer is not None else -0.2) - total_cost

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

def main():
    out = []
    for conflict_A in CONFLICT_RATES_A:
        print(f"\n=== conflict_rate_A = {conflict_A:.1f} (cost_A={COST_A}, cost_B={COST_B}) ===")
        for budget in BUDGETS:
            for name, AgentClass in [("BINARY", BinaryAgent),
                                     ("TERNARY", TernaryAgent),
                                     ("QUATERNARY", QuaternaryAgent)]:
                rows = []
                for ep in range(EPISODES):
                    seed = ep + budget*100000 + int(conflict_A*1000)
                    rows.append(run_episode(AgentClass, seed, budget, conflict_A, COST_A, COST_B))
                res = aggregate(rows)
                print(f"{name:10s} b={budget:2d} acc={res['correct']:.4f} q={res['queries']:.3f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} reward={res['reward']:.4f} "
                      f"srcA={res['usage_A']:.2f} srcB={res['usage_B']:.2f}")
                out.append({
                    "conflict_A": conflict_A,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    path = "results/test18/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
