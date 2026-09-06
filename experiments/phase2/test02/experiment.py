from __future__ import annotations

import csv
import math
import random
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST02 — ГИБРИДНАЯ НЕЙРО-СИМВОЛЬНАЯ СИСТЕМА (Phase 2)
# Нейросеть оценивает вероятности, затем преобразует в состояния
# и применяет Threshold-логику.
# ============================================================

N_H = 16
N_Q = 8
EPISODES = 1500
BUDGETS = [5, 7, 10]
CONFLICT_RATES = [0.0, 0.2, 0.4]

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2
FIXED_THRESHOLD = 3

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

# ---------- ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ ----------
def generate_training_data(n_samples=10000, conflict_rate=0.0, seed=42):
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        h = rng.randrange(N_H)
        true_answers = [QUESTIONS[q][h] for q in range(N_Q)]
        noisy_answers = []
        for ans in true_answers:
            if rng.random() < conflict_rate:
                noisy_answers.append(1 - ans)
            else:
                noisy_answers.append(ans)
        data.append((noisy_answers, h))
    return data

# ---------- ПРОСТАЯ НЕЙРОСЕТЬ ----------
class SimpleNN:
    def __init__(self, input_dim=N_Q, hidden_dim=32, output_dim=N_H):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def train(self, data, epochs=20, lr=0.01):
        for epoch in range(epochs):
            rng = random.Random(epoch + 42)
            indices = list(range(len(data)))
            rng.shuffle(indices)
            total_loss = 0.0
            for idx in indices:
                x, target = data[idx]
                x = np.array(x, dtype=np.float32)
                h = np.maximum(0, x @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                loss = -np.log(probs[target] + 1e-8)
                total_loss += loss
                dlogits = probs.copy()
                dlogits[target] -= 1.0
                dW2 = np.outer(h, dlogits)
                db2 = dlogits
                dh = dlogits @ self.W2.T
                dh[h <= 0] = 0
                dW1 = np.outer(x, dh)
                db1 = dh
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: loss = {total_loss / len(data):.4f}")

# ---------- АГЕНТЫ ----------
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
                    if self.counter[h] >= FIXED_THRESHOLD:
                        self.states[h] = WRONG
                else:
                    self.states[h] = WRONG

    def answer(self):
        confirmed = [h for h in HYPOTHESES if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return confirmed[0]
        return None

# ---------- ГИБРИДНЫЙ АГЕНТ ----------
class HybridAgent:
    def __init__(self, seed, nn):
        self.rng = random.Random(seed)
        self.nn = nn
        self.reset()

    def reset(self):
        self.states = [UNKNOWN] * N_H
        self.counter = [0] * N_H
        self.observations = []  # список (q, value)

    def choose_question(self, conflict_rate):
        # Строим входной вектор для нейросети
        input_vec = [0.5] * N_Q
        for q, value in self.observations:
            input_vec[q] = value
        probs = self.nn.forward(np.array(input_vec, dtype=np.float32))
        # Выбираем вопрос с максимальной энтропией (или случайно, упрощаем)
        # Для простоты используем стандартную логику Threshold для выбора вопроса
        # (можно улучшить)
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
        self.observations.append((q, value))
        if value == UNKNOWN:
            return
        # Обновляем состояния по Threshold-логике
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
                    if self.counter[h] >= FIXED_THRESHOLD:
                        self.states[h] = WRONG
                else:
                    self.states[h] = WRONG

    def answer(self):
        # Сначала пробуем нейросеть
        if len(self.observations) >= 3:
            input_vec = [0.5] * N_Q
            for q, value in self.observations:
                input_vec[q] = value
            probs = self.nn.forward(np.array(input_vec, dtype=np.float32))
            max_prob = np.max(probs)
            if max_prob > 0.6:
                sorted_probs = np.sort(probs)[::-1]
                if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) > 0.1:
                    return np.argmax(probs)
        # Иначе дискретная логика
        confirmed = [h for h in HYPOTHESES if self.states[h] == RIGHT]
        if len(confirmed) == 1:
            return confirmed[0]
        return None

# ---------- ЭКСПЕРИМЕНТ ----------
def run_episode(AgentClass, seed, budget, conflict_rate, nn=None):
    rng = random.Random(seed)
    target = rng.randrange(N_H)
    env = Environment(target, conflict_rate, rng)
    if AgentClass == HybridAgent:
        agent = HybridAgent(seed, nn)
    else:
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
    # Обучаем нейросети для каждого уровня шума
    nns = {}
    for cr in CONFLICT_RATES:
        print(f"Training NN for conflict_rate={cr}...")
        data = generate_training_data(n_samples=8000, conflict_rate=cr, seed=42)
        nn = SimpleNN()
        nn.train(data, epochs=15, lr=0.01)
        nns[cr] = nn

    out = []
    pool = mp.Pool(mp.cpu_count())

    for conflict_rate in CONFLICT_RATES:
        print(f"\n=== conflict_rate = {conflict_rate:.1f} ===")
        nn = nns[conflict_rate]
        for budget in BUDGETS:
            for name, AgentClass in [("BINARY", BinaryAgent),
                                     ("TERNARY", TernaryAgent),
                                     ("THRESHOLD", QuaternaryThresholdAgent),
                                     ("HYBRID", HybridAgent)]:
                seeds = [ep + budget*100000 + int(conflict_rate*1000) for ep in range(EPISODES)]
                if name == "HYBRID":
                    func = partial(run_episode, HybridAgent, budget=budget, conflict_rate=conflict_rate, nn=nn)
                else:
                    func = partial(run_episode, AgentClass, budget=budget, conflict_rate=conflict_rate, nn=None)
                results = pool.map(func, seeds)
                res = aggregate(results)
                print(f"{name:10s} b={budget:2d} acc={res['correct']:.4f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} reward={res['reward']:.4f}")
                out.append({
                    "conflict_rate": conflict_rate,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    pool.close()
    pool.join()

    path = "results/phase2/test02/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
