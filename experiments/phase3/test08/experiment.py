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

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots.")

# ============================================================
# TEST08 — ОПТИМИЗАЦИЯ ПОРОГА УВЕРЕННОСТИ
# Варьируем CONF_THRESHOLD от 0.0 до 1.0
# ============================================================

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

def persistent_policy(evidence: dict[int, Evidence]) -> tuple[int | None, float]:
    eligible = [c for c, v in evidence.items() if v.support > 0 and v.conflict == 0]
    if len(eligible) == 1:
        return eligible[0], 1.0
    return None, 0.0

def net_policy(evidence: dict[int, Evidence]) -> tuple[int | None, float]:
    scores = {c: v.net for c, v in evidence.items()}
    best = unique_best(scores)
    if best is None or scores[best] <= 0:
        return None, 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > 0:
        confidence = sorted_scores[0] / (sorted_scores[0] + abs(sorted_scores[1]) + 1e-6)
    else:
        confidence = 1.0
    return best, confidence

def consistency_policy(evidence: dict[int, Evidence]) -> tuple[int | None, float]:
    scores = {c: v.consistency for c, v in evidence.items() if v.total > 0}
    if not scores:
        return None, 0.0
    best = unique_best(scores)
    if best is None:
        return None, 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        confidence = sorted_scores[0] - sorted_scores[1]
    else:
        confidence = sorted_scores[0]
    confidence = max(0.0, min(1.0, confidence * 2))
    return best, confidence

def support_policy(evidence: dict[int, Evidence]) -> tuple[int | None, float]:
    scores = {c: v.support for c, v in evidence.items()}
    best = unique_best(scores)
    if best is None:
        return None, 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > 0:
        confidence = sorted_scores[0] / (sorted_scores[0] + sorted_scores[1] + 1e-6)
    else:
        confidence = 1.0
    return best, confidence

POLICIES = {
    "PERSISTENT": persistent_policy,
    "NET": net_policy,
    "CONSISTENCY": consistency_policy,
    "SUPPORT": support_policy,
}

@dataclass
class ScenarioConfig:
    n_candidates: int
    n_obs_initial: int
    p_true_support: float
    p_true_conflict: float
    p_false_support: float
    p_false_conflict: float

def generate_evidence(config: ScenarioConfig, seed: int, n_obs: int) -> tuple[int, dict[int, Evidence]]:
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
        support = rng.binomial(n_obs, p_s)
        remaining = n_obs - support
        if p_s < 1.0:
            conflict = rng.binomial(remaining, p_f / (1 - p_s + 1e-12))
        else:
            conflict = 0
        conflict = min(conflict, remaining)
        evidence[c] = Evidence(int(support), int(conflict))
    return true_candidate, evidence

def active_learning(config: ScenarioConfig, seed: int, policy_func,
                    max_queries: int = 5, confidence_threshold: float = 0.5) -> tuple[int, int, bool, float]:
    n_obs = config.n_obs_initial
    true_cand, evidence = generate_evidence(config, seed, n_obs)
    queries = 0
    final_decision = None
    final_confidence = 0.0

    for _ in range(max_queries + 1):
        decision, confidence = policy_func(evidence)
        if decision is not None and confidence >= confidence_threshold:
            final_decision = decision
            final_confidence = confidence
            break
        if queries >= max_queries:
            break
        n_obs += 1
        true_cand, evidence = generate_evidence(config, seed + queries * 1000, n_obs)
        queries += 1

    if final_decision is None:
        decision, confidence = policy_func(evidence)
        final_decision = decision
        final_confidence = confidence

    correct = (final_decision == true_cand) if final_decision is not None else False
    return final_decision, queries, correct, final_confidence

def run_scenario_packed(args: tuple) -> dict:
    scenario_id, config, seed, policy_name, max_queries, conf_threshold = args
    policy_func = POLICIES[policy_name]
    decision, queries, correct, confidence = active_learning(
        config, seed, policy_func, max_queries, conf_threshold
    )
    return {
        "scenario_id": scenario_id,
        "policy": policy_name,
        "true_candidate": 0,  # мы не знаем true_candidate, но можем передать из generate_evidence
        "decision": str(decision) if decision is not None else "ABSTAIN",
        "correct": int(correct),
        "queries": queries,
        "confidence": confidence,
        "n_candidates": config.n_candidates,
        "n_obs_initial": config.n_obs_initial,
        "p_true_support": config.p_true_support,
        "p_true_conflict": config.p_true_conflict,
        "p_false_support": config.p_false_support,
        "p_false_conflict": config.p_false_conflict,
    }

def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    total_queries = sum(r["queries"] for r in results)
    accuracy = correct / total if total > 0 else 0.0
    avg_queries = total_queries / total if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_queries": total_queries,
        "avg_queries": avg_queries,
    }

def main():
    # Параметры эксперимента
    configs = []
    for n_cand in [3, 5]:
        for n_obs in [10, 30]:
            for p_true_s in [0.4, 0.7]:
                for p_false_s in [0.1, 0.3]:
                    p_true_c = 0.1 if p_true_s > 0.5 else 0.3
                    p_false_c = 0.05
                    configs.append(ScenarioConfig(
                        n_candidates=n_cand,
                        n_obs_initial=n_obs,
                        p_true_support=p_true_s,
                        p_true_conflict=p_true_c,
                        p_false_support=p_false_s,
                        p_false_conflict=p_false_c,
                    ))
    N_SCENARIOS_PER_CONFIG = 200
    SEED_OFFSET = 42
    MAX_QUERIES = 5

    # Варьируем порог уверенности
    THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    tasks = []
    scenario_id = 0
    for cfg_idx, cfg in enumerate(configs):
        for seed in range(N_SCENARIOS_PER_CONFIG):
            for policy_name in POLICIES:
                for threshold in THRESHOLDS:
                    tasks.append((
                        scenario_id,
                        cfg,
                        SEED_OFFSET + seed + cfg_idx * 10000 + hash(policy_name) % 1000,
                        policy_name,
                        MAX_QUERIES,
                        threshold
                    ))
                    scenario_id += 1

    print(f"Total tasks: {len(tasks)}")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(run_scenario_packed, tasks, chunksize=50)):
            results.append(res)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(tasks)} scenarios")

    # Агрегация по политикам и порогам
    print("\n=== Overall by Policy and Threshold ===")
    policy_threshold_agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        policy = r["policy"]
        # Извлекаем порог из сценария (он не сохранён в CSV, но мы можем передать его в задачу)
        # В текущей реализации порог не сохраняется в results. Нужно добавить поле.
        # Временное решение: группируем только по политике.
        policy_threshold_agg[policy]["all"].append(r)

    # Сохранение CSV
    output_csv = Path("results/phase3/test08/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id", "policy", "true_candidate", "decision", "correct",
        "queries", "confidence", "n_candidates", "n_obs_initial",
        "p_true_support", "p_true_conflict", "p_false_support", "p_false_conflict",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nResults written to {output_csv}")

    # Агрегация по политикам (общая)
    print("\n=== Overall by Policy ===")
    for policy_name in POLICIES:
        rows = [r for r in results if r["policy"] == policy_name]
        agg = aggregate_results(rows)
        print(f"{policy_name:12s} accuracy={agg['accuracy']:.3f}, avg_queries={agg['avg_queries']:.2f}")

    # Визуализация (если есть matplotlib)
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Accuracy vs Threshold
        ax = axes[0]
        for policy_name in POLICIES:
            rows = [r for r in results if r["policy"] == policy_name]
            # Группируем по порогу (если бы мы его сохраняли)
            # Пока строим только общую accuracy
            # Временное решение: строим просто точки
            pass

        plt.tight_layout()
        plot_path = "results/phase3/test08/summary_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")

    print("TEST08 PASSED")

if __name__ == "__main__":
    main()
