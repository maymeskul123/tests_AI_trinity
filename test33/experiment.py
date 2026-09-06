from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST33 — КЛАССИФИКАЦИЯ С ШУМНЫМИ МЕТКАМИ
# Генерируем датасет, агент запрашивает метки и классифицирует
# ============================================================

N_OBJECTS = 100
N_FEATURES = 10
BUDGETS = [30, 50, 70]      # число запросов
CONFLICT_RATES = [0.0, 0.2, 0.4]
UNKNOWN_RATE = 0.2           # фиксированная вероятность UNKNOWN
EPISODES = 500               # число эпизодов (для скорости)

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2
CONFLICT_THRESHOLD = 3

REWARD_CORRECT = 1.0
REWARD_FALSE = -100.0
REWARD_ABSTAIN = -0.2
QUERY_COST = 0.05

# ---------- ГЕНЕРАЦИЯ ДАННЫХ ----------
def generate_dataset(seed):
    rng = random.Random(seed)
    # Признаки: бинарные, случайные
    features = [[rng.randint(0, 1) for _ in range(N_FEATURES)] for _ in range(N_OBJECTS)]
    # Истинный класс: XOR от первых 4 признаков (усложним)
    def true_class(f):
        # XOR от первых 4 признаков
        xor = f[0] ^ f[1] ^ f[2] ^ f[3]
        # Добавим немного зависимости от других признаков для разнообразия
        if f[4] == 1:
            xor = 1 - xor
        return xor
    labels = [true_class(f) for f in features]
    return features, labels

# ---------- СРЕДА ----------
class ClassificationEnvironment:
    def __init__(self, features, labels, unknown_rate, conflict_rate, rng):
        self.features = features
        self.labels = labels
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.rng = rng
        self.n_objects = len(labels)

    def ask(self, obj_idx):
        true_label = self.labels[obj_idx]
        if self.rng.random() < self.unknown_rate:
            return UNKNOWN
        if self.rng.random() < self.conflict_rate:
            return 1 - true_label
        return true_label

# ---------- АГЕНТЫ ----------
class BinaryAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # Для каждого объекта храним: 1 = возможно, 0 = исключено
        self.states = [[1, 1] for _ in range(N_OBJECTS)]  # [state_class0, state_class1]
        # Также храним уверенность: для каждого объекта, какая гипотеза активна
        # Упрощаем: если оба возможны, то не уверены; если один, то классифицируем

    def choose_object(self, conflict_rate):
        # Выбираем объект с максимальной неопределённостью
        # Неопределённость = число активных классов (чем больше, тем лучше)
        best_score = -1
        best_obj = None
        for i in range(N_OBJECTS):
            active = sum(self.states[i])
            if active > 1:
                # Хотим уменьшить неопределённость, поэтому выбираем с max active
                if active > best_score:
                    best_score = active
                    best_obj = i
        if best_obj is None:
            # Если все объекты уже классифицированы, берём первый с неопределённостью
            for i in range(N_OBJECTS):
                if sum(self.states[i]) > 1:
                    return i
            return None
        return best_obj

    def observe(self, obj_idx, value):
        # Обновляем состояния для данного объекта
        # Если value == UNKNOWN, ничего не делаем
        if value == UNKNOWN:
            return
        # value — это полученная метка (0 или 1)
        # Исключаем противоположный класс
        if value == 0:
            self.states[obj_idx][1] = 0  # класс 1 исключён
        else:  # value == 1
            self.states[obj_idx][0] = 0  # класс 0 исключён

    def classify(self):
        # Возвращает словарь: obj_idx -> класс (0/1) или None
        result = {}
        for i in range(N_OBJECTS):
            if self.states[i][0] == 1 and self.states[i][1] == 0:
                result[i] = 0
            elif self.states[i][0] == 0 and self.states[i][1] == 1:
                result[i] = 1
            else:
                result[i] = None
        return result

class TernaryAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # Для каждого объекта: состояние для класса 0 и класса 1
        # Состояния: WRONG, UNKNOWN, RIGHT
        self.states0 = [UNKNOWN] * N_OBJECTS
        self.states1 = [UNKNOWN] * N_OBJECTS

    def choose_object(self, conflict_rate):
        # Выбираем объект с наибольшей неопределённостью
        # Считаем неопределённость как число классов в состоянии UNKNOWN или CONFLICT
        best_score = -1
        best_obj = None
        for i in range(N_OBJECTS):
            score = 0
            if self.states0[i] != WRONG and self.states0[i] != RIGHT:
                score += 1
            if self.states1[i] != WRONG and self.states1[i] != RIGHT:
                score += 1
            # Предпочитаем объекты с конфликтами (CONFLICT)
            if self.states0[i] == CONFLICT or self.states1[i] == CONFLICT:
                score += 2
            if score > best_score:
                best_score = score
                best_obj = i
        if best_obj is None:
            # Fallback: первый не полностью определённый
            for i in range(N_OBJECTS):
                if self.states0[i] != RIGHT and self.states1[i] != RIGHT:
                    return i
            return None
        return best_obj

    def observe(self, obj_idx, value):
        if value == UNKNOWN:
            return
        # Обновляем состояния: если получили value = 0, то для класса 0 это RIGHT, для класса 1 WRONG (если не было RIGHT)
        if value == 0:
            if self.states0[obj_idx] != RIGHT:
                self.states0[obj_idx] = RIGHT
            if self.states1[obj_idx] != RIGHT:
                self.states1[obj_idx] = WRONG
        else:  # value == 1
            if self.states1[obj_idx] != RIGHT:
                self.states1[obj_idx] = RIGHT
            if self.states0[obj_idx] != RIGHT:
                self.states0[obj_idx] = WRONG

    def classify(self):
        result = {}
        for i in range(N_OBJECTS):
            if self.states0[i] == RIGHT and self.states1[i] == WRONG:
                result[i] = 0
            elif self.states0[i] == WRONG and self.states1[i] == RIGHT:
                result[i] = 1
            else:
                result[i] = None
        return result

class QuaternaryThresholdAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # Для каждого класса: состояние и счётчик несовпадений
        self.states0 = [UNKNOWN] * N_OBJECTS
        self.states1 = [UNKNOWN] * N_OBJECTS
        self.counter0 = [0] * N_OBJECTS
        self.counter1 = [0] * N_OBJECTS

    def choose_object(self, conflict_rate):
        # Выбираем объект с максимальной неопределённостью, приоритет для CONFLICT
        best_score = -1
        best_obj = None
        for i in range(N_OBJECTS):
            score = 0
            if self.states0[i] != WRONG and self.states0[i] != RIGHT:
                score += 1
            if self.states1[i] != WRONG and self.states1[i] != RIGHT:
                score += 1
            if self.states0[i] == CONFLICT or self.states1[i] == CONFLICT:
                score += 3  # высокий приоритет
            if score > best_score:
                best_score = score
                best_obj = i
        if best_obj is None:
            for i in range(N_OBJECTS):
                if self.states0[i] != RIGHT and self.states1[i] != RIGHT:
                    return i
            return None
        return best_obj

    def observe(self, obj_idx, value):
        if value == UNKNOWN:
            return
        # Обновляем состояния для класса 0 и 1 с использованием Threshold логики
        self._update_class(obj_idx, 0, value, self.states0, self.counter0)
        self._update_class(obj_idx, 1, value, self.states1, self.counter1)

    def _update_class(self, obj_idx, cls, value, states, counter):
        # cls — это класс, для которого обновляем (0 или 1)
        # value — полученная метка
        # Если cls == value, то это подтверждение
        expected = cls
        state = states[obj_idx]
        if state == WRONG:
            return
        if expected == value:
            # Совпало
            counter[obj_idx] = 0
            states[obj_idx] = RIGHT
        else:
            # Не совпало
            if state == RIGHT:
                states[obj_idx] = CONFLICT
                counter[obj_idx] = 1
            elif state == CONFLICT:
                counter[obj_idx] += 1
                if counter[obj_idx] >= CONFLICT_THRESHOLD:
                    states[obj_idx] = WRONG
            else:  # UNKNOWN
                states[obj_idx] = WRONG

    def classify(self):
        result = {}
        for i in range(N_OBJECTS):
            if self.states0[i] == RIGHT and self.states1[i] == WRONG:
                result[i] = 0
            elif self.states0[i] == WRONG and self.states1[i] == RIGHT:
                result[i] = 1
            else:
                result[i] = None
        return result

# ---------- ЭКСПЕРИМЕНТ ----------
def run_episode(AgentClass, seed, budget, conflict_rate):
    rng = random.Random(seed)
    features, true_labels = generate_dataset(seed + 1000)
    env = ClassificationEnvironment(features, true_labels, UNKNOWN_RATE, conflict_rate, rng)
    agent = AgentClass(seed)
    queries = 0
    false_confidence = 0
    abstain = 0

    for step in range(budget):
        # Сначала проверяем, можем ли классифицировать всё?
        # Но мы будем продолжать запрашивать, пока есть бюджет
        obj = agent.choose_object(conflict_rate)
        if obj is None:
            break
        obs = env.ask(obj)
        queries += 1
        agent.observe(obj, obs)

    # После завершения бюджета, классифицируем все объекты
    preds = agent.classify()
    correct = 0
    total_classified = 0
    false_pos = 0  # для false_confidence
    for i in range(N_OBJECTS):
        if preds[i] is not None:
            total_classified += 1
            if preds[i] == true_labels[i]:
                correct += 1
            else:
                false_pos += 1

    if total_classified == 0:
        abstain = 1
        acc = 0
        false_conf = 0
    else:
        abstain = 0
        acc = correct / total_classified
        false_conf = false_pos / N_OBJECTS  # доля ошибочно классифицированных объектов

    # Награда: accuracy * 1.0 - штраф за ошибки
    reward = (acc) - (false_conf * 100) - QUERY_COST * queries
    # Ограничим, чтобы не уходило в бесконечность
    reward = max(-200, min(2, reward))

    return {
        "correct": acc,               # accuracy среди классифицированных
        "queries": queries,
        "false_confidence": false_conf,
        "abstain": 1 if total_classified == 0 else 0,
        "reward": reward,
        "classified": total_classified,
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
                                     ("THRESHOLD", QuaternaryThresholdAgent)]:
                seeds = [ep + budget*100000 + int(conflict_rate*1000) for ep in range(EPISODES)]
                func = partial(run_episode, AgentClass, budget=budget, conflict_rate=conflict_rate)
                results = pool.map(func, seeds)
                res = aggregate(results)
                print(f"{name:10s} b={budget:2d} acc={res['correct']:.4f} "
                      f"false={res['false_confidence']:.4f} abst={res['abstain']:.4f} "
                      f"classified={res['classified']:.1f} reward={res['reward']:.4f}")
                out.append({
                    "conflict_rate": conflict_rate,
                    "budget": budget,
                    "agent": name,
                    **res
                })

    pool.close()
    pool.join()

    path = "results/test33/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
