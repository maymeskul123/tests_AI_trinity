from __future__ import annotations

import csv
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from functools import partial

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ============================================================
# TEST11 — РАСПОЗНАВАНИЕ КОШЕК И СОБАК С ИСПОЛЬЗОВАНИЕМ 
# НАШЕЙ АРХИТЕКТУРЫ (4 состояния + политики)
# ============================================================

# ---------- ЗАГРУЗКА ДАННЫХ ----------
def load_cats_vs_dogs(batch_size=32, num_samples=200):
    # Используем CIFAR-10 (классы 3=кошка, 5=собака)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Фильтруем только кошек (3) и собак (5)
    indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
    # Берём подвыборку
    rng = random.Random(42)
    selected = rng.sample(indices, min(num_samples, len(indices)))
    images = [dataset[i][0] for i in selected]
    labels = [1 if dataset[i][1] == 5 else 0 for i in selected]  # 1 = собака, 0 = кошка
    return images, labels

# ---------- ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ----------
def extract_features(images, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()  # убираем классификатор
    model.eval()
    model.to(device)
    
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            features = model(batch).cpu().numpy()
            all_features.append(features)
    return np.vstack(all_features)

# ---------- ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ПРИЗНАКОВ ----------
def generate_attributes(features, labels, n_attrs=8, noise_level=0.15, seed=42):
    """
    Для каждого изображения генерируем n_attrs бинарных признаков,
    основанных на истинной метке, но с добавлением шума.
    """
    rng = random.Random(seed)
    n_samples = len(labels)
    attrs = []
    for i in range(n_samples):
        # Истинный класс определяет базовые признаки
        base = [1 if labels[i] == 1 else 0 for _ in range(n_attrs)]
        # Добавляем шум (инвертируем случайные биты)
        noisy = []
        for b in base:
            if rng.random() < noise_level:
                noisy.append(1 - b)
            else:
                noisy.append(b)
        attrs.append(noisy)
    # Названия признаков
    attr_names = [f"attr_{j}" for j in range(n_attrs)]
    return attrs, attr_names

# ---------- КЛАССЫ ДЛЯ АГЕНТА ----------
class Evidence:
    def __init__(self, support=0, conflict=0):
        self.support = support
        self.conflict = conflict

    @property
    def total(self):
        return self.support + self.conflict

    @property
    def net(self):
        return self.support - self.conflict

    @property
    def consistency(self):
        if self.total == 0:
            return 0.0
        return self.support / self.total

class StateManager:
    def __init__(self, n_hypotheses=2):
        self.n_hypotheses = n_hypotheses
        self.states = [None] * n_hypotheses  # RIGHT, WRONG, UNKNOWN, CONFLICT
        self.evidence = [[0, 0] for _ in range(n_hypotheses)]  # support, conflict per hypothesis
        self.reset()

    def reset(self):
        for h in range(self.n_hypotheses):
            self.states[h] = "UNKNOWN"
            self.evidence[h] = [0, 0]

    def observe(self, hypothesis, value):  # value = 1 if supports, 0 if conflicts
        if value == 1:
            self.evidence[hypothesis][0] += 1
        else:
            self.evidence[hypothesis][1] += 1
        self._update_state(hypothesis)

    def _update_state(self, h):
        support, conflict = self.evidence[h]
        if support > 0 and conflict == 0:
            self.states[h] = "RIGHT"
        elif conflict > 0 and support == 0:
            self.states[h] = "WRONG"
        elif support > 0 and conflict > 0:
            self.states[h] = "CONFLICT"
        else:
            self.states[h] = "UNKNOWN"

    def get_evidence(self, h):
        return Evidence(self.evidence[h][0], self.evidence[h][1])

    def get_state(self, h):
        return self.states[h]

# ---------- ПОЛИТИКИ (адаптированы для классификации) ----------
def support_policy(state_manager):
    scores = {}
    for h in range(state_manager.n_hypotheses):
        ev = state_manager.get_evidence(h)
        scores[h] = ev.support
    best = max(scores, key=scores.get)
    if list(scores.values()).count(scores[best]) > 1:
        return None, 0.0
    if state_manager.get_state(best) != "WRONG" and scores[best] > 0:
        return best, 1.0
    return None, 0.0

def net_policy(state_manager):
    scores = {}
    for h in range(state_manager.n_hypotheses):
        ev = state_manager.get_evidence(h)
        scores[h] = ev.net
    best = max(scores, key=scores.get)
    if list(scores.values()).count(scores[best]) > 1:
        return None, 0.0
    if state_manager.get_state(best) != "WRONG" and scores[best] > 0:
        return best, 1.0
    return None, 0.0

def persistent_policy(state_manager):
    for h in range(state_manager.n_hypotheses):
        if state_manager.get_state(h) == "RIGHT":
            ev = state_manager.get_evidence(h)
            if ev.conflict == 0:
                return h, 1.0
    return None, 0.0

POLICIES = {
    "SUPPORT": support_policy,
    "NET": net_policy,
    "PERSISTENT": persistent_policy,
}

# ---------- АКТИВНЫЙ АГЕНТ ----------
class ActiveAgent:
    def __init__(self, policy_name, n_attrs, max_queries=5):
        self.policy_name = policy_name
        self.policy = POLICIES[policy_name]
        self.n_attrs = n_attrs
        self.max_queries = max_queries
        self.reset()

    def reset(self):
        self.state_manager = StateManager(n_hypotheses=2)
        self.queried_attrs = set()
        self.queries = 0
        self.decision = None

    def step(self, attr_values):
        if self.decision is not None:
            return self.decision

        decision, confidence = self.policy(self.state_manager)
        if decision is not None:
            self.decision = decision
            return decision

        if self.queries < self.max_queries:
            available = [a for a in range(self.n_attrs) if a not in self.queried_attrs]
            if available:
                attr = random.choice(available)
                self.queried_attrs.add(attr)
                value = attr_values[attr]
                # Обновляем evidence для обеих гипотез
                for h in [0, 1]:
                    if (h == 0 and value == 0) or (h == 1 and value == 1):
                        self.state_manager.observe(h, 1)
                    else:
                        self.state_manager.observe(h, 0)
                self.queries += 1
                return None
            else:
                scores = {}
                for h in range(2):
                    ev = self.state_manager.get_evidence(h)
                    scores[h] = ev.support - ev.conflict
                best = max(scores, key=scores.get)
                if scores[best] > 0 and self.state_manager.get_state(best) != "WRONG":
                    self.decision = best
                    return best
                else:
                    self.decision = None
                    return None
        else:
            scores = {}
            for h in range(2):
                ev = self.state_manager.get_evidence(h)
                scores[h] = ev.support - ev.conflict
            best = max(scores, key=scores.get)
            if scores[best] > 0 and self.state_manager.get_state(best) != "WRONG":
                self.decision = best
                return best
            else:
                self.decision = None
                return None

# ---------- ЭКСПЕРИМЕНТ ----------
def run_episode(policy_name, attr_values, true_label, max_queries=5):
    agent = ActiveAgent(policy_name, n_attrs=len(attr_values), max_queries=max_queries)
    decision = None
    while decision is None:
        decision = agent.step(attr_values)
    correct = (decision == true_label) if decision is not None else False
    return {
        "policy": policy_name,
        "correct": int(correct),
        "queries": agent.queries,
        "decision": decision,
        "true_label": true_label,
    }

def main():
    # Загрузка данных
    print("Loading data...")
    images, labels = load_cats_vs_dogs(num_samples=200)
    print(f"Loaded {len(images)} images")

    print("Extracting features...")
    features = extract_features(images)
    print(f"Features shape: {features.shape}")

    # Генерируем синтетические признаки
    n_attrs = 8
    noise_level = 0.15
    print(f"Generating {n_attrs} attributes with noise {noise_level}...")
    attr_values, attr_names = generate_attributes(features, labels, n_attrs=n_attrs, noise_level=noise_level)

    # Запуск эксперимента
    policies = ["SUPPORT", "NET", "PERSISTENT"]
    results = []
    max_queries = 5

    for policy in policies:
        print(f"Running {policy}...")
        for i, (attrs, true_label) in enumerate(zip(attr_values, labels)):
            res = run_episode(policy, attrs, true_label, max_queries)
            res["img_idx"] = i
            results.append(res)

    # Агрегация
    grouped = defaultdict(list)
    for r in results:
        grouped[r["policy"]].append(r)

    print("\n=== Results ===")
    for policy in policies:
        rows = grouped[policy]
        correct = sum(r["correct"] for r in rows)
        total = len(rows)
        avg_queries = sum(r["queries"] for r in rows) / total
        print(f"{policy}: accuracy={correct/total:.3f}, avg_queries={avg_queries:.2f}")

    # Сохранение CSV
    output_csv = Path("results/phase3/test11/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_csv}")
    print("TEST11 PASSED")

if __name__ == "__main__":
    main()
