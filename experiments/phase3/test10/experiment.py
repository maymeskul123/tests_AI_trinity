from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from functools import partial

# ============================================================
# TEST10 — ГИБРИДНАЯ ПОЛИТИКА (SUPPORT + NET с адаптацией)
# ============================================================

@dataclass(frozen=True)
class Parameter:
    name: str
    type: str

@dataclass(frozen=True)
class Specification:
    func_name: str
    params: list[Parameter]
    return_type: str
    constraints: list[str]
    exceptions: list[str]

class TaskGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.param_names = ["x", "y", "z", "a", "b", "c", "value", "index", "data"]
        self.types = ["int", "float", "str", "bool", "list"]
        self.constraints_pool = ["x > 0", "x >= 0", "x != 0", "y > 0", "y != 0", "len(data) > 0"]
        self.exceptions_pool = ["ValueError", "TypeError", "IndexError"]

    def generate_task(self) -> Specification:
        n_params = self.rng.randint(1, 2)
        params = [Parameter(self.rng.choice(self.param_names), self.rng.choice(self.types)) for _ in range(n_params)]
        func_name = "func_" + "_".join(p.name for p in params)
        return_type = self.rng.choice(self.types)
        constraints = self.rng.sample(self.constraints_pool, min(self.rng.randint(0, 1), len(self.constraints_pool)))
        exceptions = self.rng.sample(self.exceptions_pool, min(self.rng.randint(0, 1), len(self.exceptions_pool)))
        return Specification(func_name, params, return_type, constraints, exceptions)

def add_contradiction(spec: Specification, rng: random.Random) -> Specification:
    if not spec.params:
        return spec
    idx = rng.randint(0, len(spec.params)-1)
    old = spec.params[idx]
    other_types = [t for t in ["int", "float", "str", "bool", "list"] if t != old.type]
    if not other_types:
        return spec
    new_param = Parameter(old.name, rng.choice(other_types))
    new_params = list(spec.params)
    new_params[idx] = new_param
    return Specification(spec.func_name, new_params, spec.return_type, spec.constraints, spec.exceptions)

class ProgrammingEnvironment:
    def __init__(self, task: Specification, contradictions: int = 0, seed: int = 42):
        self.rng = random.Random(seed)
        self.true_spec = task
        self.observed_spec = task
        for _ in range(contradictions):
            self.observed_spec = add_contradiction(self.observed_spec, self.rng)
        self.attributes = {
            "func_name": self.observed_spec.func_name,
            "params": self.observed_spec.params,
            "return_type": self.observed_spec.return_type,
            "constraints": self.observed_spec.constraints,
            "exceptions": self.observed_spec.exceptions,
        }
        self.true_attributes = {
            "func_name": self.true_spec.func_name,
            "params": self.true_spec.params,
            "return_type": self.true_spec.return_type,
            "constraints": self.true_spec.constraints,
            "exceptions": self.true_spec.exceptions,
        }

    def ask(self, attr_name: str) -> str:
        if self.rng.random() < 0.2:
            if attr_name == "func_name":
                return "func_" + self.rng.choice(["a", "b", "c"])
            elif attr_name == "params":
                return [Parameter("p", self.rng.choice(["int", "float"]))]
            elif attr_name == "return_type":
                return self.rng.choice(["int", "float", "str"])
            elif attr_name == "constraints":
                return ["x > 0"]
            elif attr_name == "exceptions":
                return ["ValueError"]
        return self.true_attributes[attr_name]

    def check_code(self, code: str) -> bool:
        params_ok = all(p.name in code for p in self.true_spec.params)
        return_type_ok = self.true_spec.return_type in code
        return params_ok and return_type_ok

# ---------- ПОЛИТИКИ ----------
def support_policy(observations: dict) -> tuple[str | None, float]:
    required = ["func_name", "params", "return_type"]
    if all(attr in observations for attr in required):
        params_str = ", ".join(f"{p.name}: {p.type}" for p in observations["params"])
        code = f"def {observations['func_name']}({params_str}) -> {observations['return_type']}:\n    pass"
        return code, 1.0
    return None, 0.0

def net_policy(observations: dict) -> tuple[str | None, float]:
    required = ["func_name", "params", "return_type", "constraints"]
    if all(attr in observations for attr in required):
        params_str = ", ".join(f"{p.name}: {p.type}" for p in observations["params"])
        code = f"def {observations['func_name']}({params_str}) -> {observations['return_type']}:\n    pass"
        return code, 1.0
    return None, 0.0

def persistent_policy(observations: dict) -> tuple[str | None, float]:
    required = ["func_name", "params", "return_type", "constraints", "exceptions"]
    if all(attr in observations for attr in required):
        params_str = ", ".join(f"{p.name}: {p.type}" for p in observations["params"])
        code = f"def {observations['func_name']}({params_str}) -> {observations['return_type']}:\n    pass"
        return code, 1.0
    return None, 0.0

# ---------- ГИБРИДНАЯ ПОЛИТИКА ----------
class HybridPolicy:
    def __init__(self, max_queries: int):
        self.max_queries = max_queries
        self.queries_used = 0

    def __call__(self, observations: dict) -> tuple[str | None, float]:
        remaining = self.max_queries - self.queries_used
        # Если осталось больше половины бюджета — используем SUPPORT
        if remaining > self.max_queries // 2:
            return support_policy(observations)
        else:
            # Иначе — NET (более консервативный)
            return net_policy(observations)

    def increment_queries(self):
        self.queries_used += 1

# ---------- АГЕНТ ----------
class ProgrammingAgent:
    def __init__(self, seed: int, policy_type: str, max_queries: int):
        self.rng = random.Random(seed)
        self.policy_type = policy_type
        if policy_type == "HYBRID":
            self.policy = HybridPolicy(max_queries)
        else:
            self.policy = {
                "SUPPORT": support_policy,
                "NET": net_policy,
                "PERSISTENT": persistent_policy,
            }[policy_type]
        self.observations = {}
        self.max_queries = max_queries

    def reset(self):
        self.observations = {}
        if self.policy_type == "HYBRID":
            self.policy = HybridPolicy(self.max_queries)
        else:
            self.policy = {
                "SUPPORT": support_policy,
                "NET": net_policy,
                "PERSISTENT": persistent_policy,
            }[self.policy_type]

    def choose_action(self, env: ProgrammingEnvironment) -> tuple[str, str | None]:
        decision, confidence = self.policy(self.observations)
        if decision is not None:
            return "code", decision

        priority = ["func_name", "params", "return_type", "constraints", "exceptions"]
        for attr in priority:
            if attr not in self.observations and attr in env.attributes:
                return "ask", attr

        if "func_name" in self.observations and "params" in self.observations and "return_type" in self.observations:
            params_str = ", ".join(f"{p.name}: {p.type}" for p in self.observations["params"])
            code = f"def {self.observations['func_name']}({params_str}) -> {self.observations['return_type']}:\n    pass"
            return "code", code

        return "ask", None

    def update(self, attr: str, value):
        self.observations[attr] = value
        if self.policy_type == "HYBRID":
            self.policy.increment_queries()

# ---------- ЭКСПЕРИМЕНТ ----------
def run_episode(policy_name: str, seed: int, contradictions: int, max_queries: int) -> dict:
    rng = random.Random(seed)
    generator = TaskGenerator(seed)
    task = generator.generate_task()
    env = ProgrammingEnvironment(task, contradictions, seed)

    agent = ProgrammingAgent(seed, policy_name, max_queries)
    agent.reset()

    queries = 0
    code = None
    success = False

    for step in range(max_queries + 1):
        action_type, content = agent.choose_action(env)
        if action_type == "ask":
            if content is None:
                break
            value = env.ask(content)
            agent.update(content, value)
            queries += 1
        else:
            if content is None:
                break
            code = content
            success = env.check_code(code)
            break

    if code is None:
        success = False

    reward = 1.0 if success else -10.0
    reward -= 0.1 * queries

    return {
        "policy": policy_name,
        "contradictions": contradictions,
        "queries": queries,
        "success": int(success),
        "reward": reward,
    }

def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    success = sum(r["success"] for r in results)
    total_queries = sum(r["queries"] for r in results)
    return {
        "total": total,
        "success": success,
        "success_rate": success / total,
        "avg_queries": total_queries / total,
    }

def main():
    contradictions_levels = [0, 1, 2]
    policies = ["SUPPORT", "NET", "PERSISTENT", "HYBRID"]
    max_queries = 5
    n_episodes_per_config = 200
    tasks = []

    for seed in range(n_episodes_per_config):
        for contr in contradictions_levels:
            for policy in policies:
                tasks.append((policy, seed + contr * 10000, contr, max_queries))

    print(f"Total tasks: {len(tasks)}")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for i, res in enumerate(pool.starmap(run_episode, tasks, chunksize=50)):
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"Processed {i+1}/{len(tasks)} tasks")

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["policy"]][r["contradictions"]].append(r)

    output_csv = Path("results/phase3/test10/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Summary by Policy and Contradictions ===")
    for policy in policies:
        print(f"\n{policy}:")
        for contr in contradictions_levels:
            rows = grouped[policy][contr]
            agg = aggregate_results(rows)
            print(f"  Contradictions={contr}: success_rate={agg['success_rate']:.3f}, avg_queries={agg['avg_queries']:.2f}")

    print(f"\nResults written to {output_csv}")
    print("TEST10 PASSED")

if __name__ == "__main__":
    main()
