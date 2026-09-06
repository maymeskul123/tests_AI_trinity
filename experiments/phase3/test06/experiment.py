from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional
import multiprocessing as mp
from functools import partial
import numpy as np

# Попытка импорта matplotlib для графиков (опционально)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots.")

# ============================================================
# TEST06 — СТАТИСТИЧЕСКИЙ БЕНЧМАРК ПОЛИТИК (С MULTIPROCESSING)
# ============================================================

# ---------- ОПРЕДЕЛЕНИЯ ПОЛИТИК ----------
@dataclass(frozen=True)
class Evidence:
    support: int
    conflict: int

    @property
    def total(self) -> int:
        return self.support + self.conflict

    @property
    def net(self) -> int:
        return self.support - self.conflict

    @property
    def consistency(self) -> float:
        if self.total == 0:
            return 0.0
        return self.support / self.total

def unique_best(values: dict[int, float]) -> int | None:
    best = max(values.values())
    winners = [cand for cand, val in values.items() if val == best]
    return winners[0] if len(winners) == 1 else None

def persistent_policy(evidence: dict[int, Evidence]) -> int | None:
    eligible = [c for c, v in evidence.items() if v.support > 0 and v.conflict == 0]
    return eligible[0] if len(eligible) == 1 else None

def net_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {c: v.net for c, v in evidence.items()}
    best = unique_best(scores)
    if best is None or scores[best] <= 0:
        return None
    return best

def consistency_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {c: v.consistency for c, v in evidence.items() if v.total > 0}
    if not scores:
        return None
    return unique_best(scores)

def support_policy(evidence: dict[int, Evidence]) -> int | None:
    scores = {c: v.support for c, v in evidence.items()}
    return unique_best(scores)

POLICIES = {
    "PERSISTENT": persistent_policy,
    "NET": net_policy,
    "CONSISTENCY": consistency_policy,
    "SUPPORT": support_policy,
}

# ---------- ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЦЕНАРИЕВ ----------
@dataclass
class ScenarioConfig:
    n_candidates: int
    n_obs: int
    p_true_support: float
    p_true_conflict: float
    p_false_support: float
    p_false_conflict: float

def generate_scenario(config: ScenarioConfig, seed: int) -> tuple[int, dict[int, Evidence]]:
    rng = np.random.default_rng(seed)
    true_candidate = rng.integers(0, config.n_candidates)
    evidence = {}
    for c in range(config.n_candidates):
        if c == true_candidate:
            p_s = config.p_true_support
            p_f = config.p_true_conflict
        else:
            p_s = config.p_false_support
            p_f = config.p_false_conflict
        support = rng.binomial(config.n_obs, p_s)
        remaining = config.n_obs - support
        # Если p_s близко к 1, конфликт практически невозможен, но формула безопасности
        if p_s < 1.0:
            conflict = rng.binomial(remaining, p_f / (1 - p_s + 1e-12))
        else:
            conflict = 0
        conflict = min(conflict, remaining)
        evidence[c] = Evidence(int(support), int(conflict))
    return true_candidate, evidence

# ---------- ЗАПУСК ОДНОГО СЦЕНАРИЯ (для multiprocessing) ----------
def run_scenario_packed(args: tuple) -> dict:
    scenario_id, config, seed, policy_name = args
    true_cand, evidence = generate_scenario(config, seed)
    policy = POLICIES[policy_name]
    decision = policy(evidence)
    correct = (decision == true_cand) if decision is not None else False
    decision_made = decision is not None
    return {
        "scenario_id": scenario_id,
        "policy": policy_name,
        "true_candidate": true_cand,
        "decision": str(decision) if decision is not None else "ABSTAIN",
        "correct": int(correct),
        "decision_made": int(decision_made),
        "n_candidates": config.n_candidates,
        "n_obs": config.n_obs,
        "p_true_support": config.p_true_support,
        "p_true_conflict": config.p_true_conflict,
        "p_false_support": config.p_false_support,
        "p_false_conflict": config.p_false_conflict,
    }

# ---------- АГРЕГАЦИЯ ----------
def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    decisions = sum(r["decision_made"] for r in results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / decisions if decisions > 0 else 0.0
    coverage = decisions / total
    overall = correct / total
    return {
        "total": total,
        "decisions": decisions,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "overall": overall,
    }

# ---------- ОСНОВНАЯ ФУНКЦИЯ ----------
def main():
    # ---------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ----------
    configs = []
    for n_cand in [3, 5]:
        for n_obs in [10, 30]:
            for p_true_s in [0.4, 0.7]:
                for p_false_s in [0.1, 0.3]:
                    p_true_c = 0.1 if p_true_s > 0.5 else 0.3
                    p_false_c = 0.05
                    configs.append(ScenarioConfig(
                        n_candidates=n_cand,
                        n_obs=n_obs,
                        p_true_support=p_true_s,
                        p_true_conflict=p_true_c,
                        p_false_support=p_false_s,
                        p_false_conflict=p_false_c,
                    ))
    N_SCENARIOS_PER_CONFIG = 1000
    SEED_OFFSET = 42

    # Подготовка задач
    tasks = []
    scenario_id = 0
    for cfg_idx, cfg in enumerate(configs):
        for seed in range(N_SCENARIOS_PER_CONFIG):
            for policy_name in POLICIES:
                tasks.append((
                    scenario_id,
                    cfg,
                    SEED_OFFSET + seed + cfg_idx * 10000 + hash(policy_name) % 1000,
                    policy_name
                ))
                scenario_id += 1

    print(f"Total tasks: {len(tasks)}")

    # Многопроцессорный пул
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(run_scenario_packed, tasks, chunksize=100)):
            results.append(res)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(tasks)} scenarios")

    # Агрегация по политикам
    print("\n=== Overall by Policy ===")
    policy_agg = {}
    for policy_name in POLICIES:
        rows = [r for r in results if r["policy"] == policy_name]
        policy_agg[policy_name] = aggregate_results(rows)
        agg = policy_agg[policy_name]
        print(f"{policy_name:12s} decisions={agg['decisions']:5d}/{agg['total']:5d} "
              f"correct={agg['correct']:5d} acc={agg['accuracy']:.3f} cov={agg['coverage']:.3f} overall={agg['overall']:.3f}")

    # По группам параметров
    print("\n=== By n_candidates and p_true_support ===")
    groups = defaultdict(list)
    for r in results:
        key = (r["n_candidates"], round(r["p_true_support"], 2))
        groups[key].append(r)
    for (n_cand, p_s), rows in sorted(groups.items()):
        print(f"n={n_cand}, p_true_s={p_s:.2f}:")
        for policy_name in POLICIES:
            sub = [r for r in rows if r["policy"] == policy_name]
            agg = aggregate_results(sub)
            print(f"  {policy_name:12s} acc={agg['accuracy']:.3f} cov={agg['coverage']:.3f} overall={agg['overall']:.3f}")

    # Сохранение CSV
    output_csv = Path("results/phase3/test06/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id", "policy", "correct", "decision_made",
        "true_candidate", "decision",
        "n_candidates", "n_obs",
        "p_true_support", "p_true_conflict",
        "p_false_support", "p_false_conflict",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nResults written to {output_csv}")

    # Визуализация (если есть matplotlib)
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axs = axes.flatten()

        # 1. Accuracy by policy
        ax = axs[0]
        data = [[r["correct"] for r in results if r["policy"] == p] for p in POLICIES]
        ax.boxplot(data, labels=POLICIES.keys())
        ax.set_title("Accuracy by Policy")
        ax.set_ylabel("Accuracy")

        # 2. Coverage by policy
        ax = axs[1]
        data = [[r["decision_made"] for r in results if r["policy"] == p] for p in POLICIES]
        ax.boxplot(data, labels=POLICIES.keys())
        ax.set_title("Decision Coverage by Policy")
        ax.set_ylabel("Decision made")

        # 3. Overall by n_candidates
        ax = axs[2]
        for policy_name in POLICIES:
            x_vals = []
            y_vals = []
            for n_cand in sorted(set(r["n_candidates"] for r in results)):
                sub = [r for r in results if r["policy"] == policy_name and r["n_candidates"] == n_cand]
                agg = aggregate_results(sub)
                x_vals.append(n_cand)
                y_vals.append(agg["overall"])
            ax.plot(x_vals, y_vals, marker='o', label=policy_name)
        ax.set_title("Overall by n_candidates")
        ax.set_xlabel("n_candidates")
        ax.set_ylabel("Overall")
        ax.legend()

        # 4. Accuracy vs p_true_support
        ax = axs[3]
        for policy_name in POLICIES:
            x_vals = []
            y_vals = []
            for p_s in sorted(set(round(r["p_true_support"], 2) for r in results)):
                sub = [r for r in results if r["policy"] == policy_name and round(r["p_true_support"], 2) == p_s]
                agg = aggregate_results(sub)
                x_vals.append(p_s)
                y_vals.append(agg["accuracy"])
            ax.plot(x_vals, y_vals, marker='o', label=policy_name)
        ax.set_title("Accuracy vs p_true_support")
        ax.set_xlabel("p_true_support")
        ax.set_ylabel("Accuracy")
        ax.legend()

        plt.tight_layout()
        plot_path = "results/phase3/test06/summary_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")

    print("\n===== FINAL SUMMARY =====")
    for policy, agg in policy_agg.items():
        print(f"{policy:12s} : accuracy={agg['accuracy']:.3f}, coverage={agg['coverage']:.3f}, overall={agg['overall']:.3f}")

    print(f"Total results: {len(results)}")
    print("TEST06 PASSED")

if __name__ == "__main__":
    main()
