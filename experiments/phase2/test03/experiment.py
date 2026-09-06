from __future__ import annotations

import csv
import math
import random
import numpy as np
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

try:
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not installed, using synthetic data")

N_SAMPLES = 1000   # увеличено, чтобы точно были данные
BUDGETS = [50, 100, 150]
CONFLICT_RATES = [0.0, 0.2, 0.4]
UNKNOWN_RATE = 0.1

WRONG = -1
UNKNOWN = 0
RIGHT = 1
CONFLICT = 2
FIXED_THRESHOLD = 3

REWARD_CORRECT = 1.0
REWARD_FALSE = -100.0
REWARD_ABSTAIN = -0.2
QUERY_COST = 0.05

def get_mnist_data(n_samples=N_SAMPLES, seed=42):
    if not SKLEARN_AVAILABLE:
        rng = random.Random(seed)
        X = [[rng.random() for _ in range(10)] for _ in range(n_samples)]
        y = [1 if sum(x) > 5 else 0 for x in X]
        return np.array(X), np.array(y)
    
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='pandas')
    # Приводим к int
    y = y.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    print(f"Found {len(y)} samples of digits 0 and 1")
    if len(y) == 0:
        raise RuntimeError("No samples with digits 0 or 1 found. Check MNIST data.")
    # Берём подвыборку
    rng = np.random.RandomState(seed)
    if len(y) > n_samples:
        indices = rng.choice(len(y), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    # Стандартизация и PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"MNIST loaded: {len(X_reduced)} samples, {X_reduced.shape[1]} features")
    return X_reduced, y

@dataclass
class Observation:
    question: int
    value: int
    true_value: int

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

def entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

def posterior_update(p, obs_value, unknown_rate, conflict_rate):
    if obs_value == UNKNOWN:
        return p
    if obs_value == 1:
        lik1 = (1 - unknown_rate) * (1 - conflict_rate)
        lik0 = (1 - unknown_rate) * conflict_rate
    else:  # obs_value == 0
        lik1 = (1 - unknown_rate) * conflict_rate
        lik0 = (1 - unknown_rate) * (1 - conflict_rate)
    p1 = p * lik1
    p0 = (1 - p) * lik0
    total = p1 + p0
    if total == 0:
        return p
    return p1 / total

def expected_information_gain(p, unknown_rate, conflict_rate):
    current_entropy = entropy(p)
    if current_entropy == 0:
        return 0.0
    p_obs1 = (1 - unknown_rate) * ((1 - conflict_rate) * p + conflict_rate * (1 - p))
    p_obs0 = (1 - unknown_rate) * (conflict_rate * p + (1 - conflict_rate) * (1 - p))
    p_unknown = unknown_rate
    post_unknown = p
    post_1 = posterior_update(p, 1, unknown_rate, conflict_rate)
    post_0 = posterior_update(p, 0, unknown_rate, conflict_rate)
    expected_after = p_unknown * entropy(post_unknown) + p_obs1 * entropy(post_1) + p_obs0 * entropy(post_0)
    gain = current_entropy - expected_after
    return max(0.0, gain)

class BaseBayesianAgent:
    def __init__(self, seed, n_objects, unknown_rate, conflict_rate):
        self.rng = random.Random(seed)
        self.n_objects = n_objects
        self.unknown_rate = unknown_rate
        self.conflict_rate = conflict_rate
        self.reset()

    def reset(self):
        self.posterior = [0.5] * self.n_objects
        self._reset_states()

    def _reset_states(self):
        raise NotImplementedError

    def choose_object(self):
        best_gain = -1
        best_obj = None
        for i in range(self.n_objects):
            p = self.posterior[i]
            gain = expected_information_gain(p, self.unknown_rate, self.conflict_rate)
            if gain > best_gain:
                best_gain = gain
                best_obj = i
        if best_gain <= 0:
            return None
        return best_obj

    def observe(self, obj_idx, value):
        p = self.posterior[obj_idx]
        p_new = posterior_update(p, value, self.unknown_rate, self.conflict_rate)
        self.posterior[obj_idx] = p_new
        self._update_states(obj_idx, value)

    def _update_states(self, obj_idx, value):
        raise NotImplementedError

    def classify(self):
        raise NotImplementedError

class BinaryAgent(BaseBayesianAgent):
    def _reset_states(self):
        self.states = [[1, 1] for _ in range(self.n_objects)]

    def _update_states(self, obj_idx, value):
        if value == UNKNOWN:
            return
        if value == 0:
            self.states[obj_idx][1] = 0
        else:
            self.states[obj_idx][0] = 0

    def classify(self):
        result = {}
        for i in range(self.n_objects):
            if self.states[i][0] == 1 and self.states[i][1] == 0:
                result[i] = 0
            elif self.states[i][0] == 0 and self.states[i][1] == 1:
                result[i] = 1
            else:
                result[i] = None
        return result

class TernaryAgent(BaseBayesianAgent):
    def _reset_states(self):
        self.states0 = [UNKNOWN] * self.n_objects
        self.states1 = [UNKNOWN] * self.n_objects

    def _update_states(self, obj_idx, value):
        if value == UNKNOWN:
            return
        if value == 0:
            if self.states0[obj_idx] != RIGHT:
                self.states0[obj_idx] = RIGHT
            if self.states1[obj_idx] != RIGHT:
                self.states1[obj_idx] = WRONG
        else:
            if self.states1[obj_idx] != RIGHT:
                self.states1[obj_idx] = RIGHT
            if self.states0[obj_idx] != RIGHT:
                self.states0[obj_idx] = WRONG

    def classify(self):
        result = {}
        for i in range(self.n_objects):
            if self.states0[i] == RIGHT and self.states1[i] == WRONG:
                result[i] = 0
            elif self.states0[i] == WRONG and self.states1[i] == RIGHT:
                result[i] = 1
            else:
                result[i] = None
        return result

class QuaternaryThresholdAgent(BaseBayesianAgent):
    def _reset_states(self):
        self.states0 = [UNKNOWN] * self.n_objects
        self.states1 = [UNKNOWN] * self.n_objects
        self.counter0 = [0] * self.n_objects
        self.counter1 = [0] * self.n_objects

    def _update_states(self, obj_idx, value):
        if value == UNKNOWN:
            return
        self._update_class(obj_idx, 0, value, self.states0, self.counter0)
        self._update_class(obj_idx, 1, value, self.states1, self.counter1)

    def _update_class(self, obj_idx, cls, value, states, counter):
        expected = cls
        state = states[obj_idx]
        if state == WRONG:
            return
        if expected == value:
            counter[obj_idx] = 0
            states[obj_idx] = RIGHT
        else:
            if state == RIGHT:
                states[obj_idx] = CONFLICT
                counter[obj_idx] = 1
            elif state == CONFLICT:
                counter[obj_idx] += 1
                if counter[obj_idx] >= FIXED_THRESHOLD:
                    states[obj_idx] = WRONG
            else:
                states[obj_idx] = WRONG

    def classify(self):
        result = {}
        for i in range(self.n_objects):
            if self.states0[i] == RIGHT and self.states1[i] == WRONG:
                result[i] = 0
            elif self.states0[i] == WRONG and self.states1[i] == RIGHT:
                result[i] = 1
            else:
                result[i] = None
        return result

def run_episode(AgentClass, seed, budget, conflict_rate, features, labels):
    rng = random.Random(seed)
    n_objects = len(labels)
    env = ClassificationEnvironment(features, labels, UNKNOWN_RATE, conflict_rate, rng)
    agent = AgentClass(seed, n_objects, UNKNOWN_RATE, conflict_rate)
    queries = 0

    for step in range(budget):
        obj = agent.choose_object()
        if obj is None:
            break
        obs = env.ask(obj)
        queries += 1
        agent.observe(obj, obs)

    preds = agent.classify()
    correct = 0
    total_classified = 0
    false_pos = 0
    for i in range(n_objects):
        if preds[i] is not None:
            total_classified += 1
            if preds[i] == labels[i]:
                correct += 1
            else:
                false_pos += 1

    if total_classified == 0:
        abstain = 1
        acc = 0.0
        false_conf = 0.0
    else:
        abstain = 0
        acc = correct / total_classified
        false_conf = false_pos / n_objects

    reward = acc - false_conf * 100 - QUERY_COST * queries
    reward = max(-200, min(2, reward))

    return {
        "correct": acc,
        "queries": queries,
        "false_confidence": false_conf,
        "abstain": abstain,
        "reward": reward,
        "classified": total_classified,
    }

def aggregate(rows):
    n = len(rows)
    return {k: sum(r[k] for r in rows)/n for k in rows[0]}

def main():
    features, labels = get_mnist_data(N_SAMPLES)
    n_objects = len(labels)
    out = []
    pool = mp.Pool(mp.cpu_count())

    for conflict_rate in CONFLICT_RATES:
        print(f"\n=== conflict_rate = {conflict_rate:.1f} ===")
        for budget in BUDGETS:
            for name, AgentClass in [("BINARY", BinaryAgent),
                                     ("TERNARY", TernaryAgent),
                                     ("THRESHOLD", QuaternaryThresholdAgent)]:
                seeds = [ep + budget*100000 + int(conflict_rate*1000) for ep in range(300)]
                func = partial(run_episode, AgentClass, budget=budget, conflict_rate=conflict_rate,
                               features=features, labels=labels)
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

    path = "results/phase2/test03/results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print("\n" + "="*70)
    print("SAVED:", path)
    print("="*70)

if __name__ == "__main__":
    main()
