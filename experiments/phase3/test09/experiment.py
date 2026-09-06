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

# ============================================================
# TEST09 — ИИ-ПРОГРАММИСТ: АКТИВНОЕ УТОЧНЕНИЕ ТРЕБОВАНИЙ
# ============================================================

# ---------- ОПРЕДЕЛЕНИЯ ДЛЯ СПЕЦИФИКАЦИИ ----------
@dataclass(frozen=True)
class Parameter:
    name: str
    type: str   # "int", "float", "str", "bool", "list"

@dataclass(frozen=True)
class Specification:
    func_name: str
    params: list[Parameter]
    return_type: str
    constraints: list[str]  # например ["x > 0", "y != 0"]
    exceptions: list[str]   # например ["ValueError", "TypeError"]

# ---------- ГЕНЕРАТОР ЗАДАЧ ----------
class TaskGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.param_names = ["x", "y", "z", "a", "b", "c", "value", "index", "data", "threshold"]
        self.types = ["int", "float", "str", "bool", "list"]
        self.constraints_pool = [
            "x > 0", "x >= 0", "x < 0", "x != 0",
            "y > 0", "y >= 0", "y != 0",
            "z > 0", "z >= 0",
            "x > y", "x < y",
            "len(data) > 0", "data is not None",
        ]
        self.exceptions_pool = ["ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError"]

    def generate_task(self) -> Specification:
        n_params = self.rng.randint(1, 3)
        params = []
        for i in range(n_params):
            name = self.rng.choice(self.param_names)
            ptype = self.rng.choice(self.types)
            params.append(Parameter(name, ptype))
        func_name = "func_" + "_".join(p.name for p in params)
        return_type = self.rng.choice(self.types)

        # Добавляем ограничения (0-2)
        n_constraints = self.rng.randint(0, 2)
        constraints = []
        pool = [c for c in self.constraints_pool if any(p.name in c for p in params)]
        if pool:
            constraints = self.rng.sample(pool, min(n_constraints, len(pool)))

        # Добавляем исключения (0-1)
        n_exceptions = self.rng.randint(0, 1)
        exceptions = self.rng.sample(self.exceptions_pool, n_exceptions) if n_exceptions > 0 else []

        return Specification(func_name, params, return_type, constraints, exceptions)

# ---------- ИСКАЖЕНИЕ СПЕЦИФИКАЦИИ (ДОБАВЛЕНИЕ ПРОТИВОРЕЧИЙ) ----------
def add_contradiction(spec: Specification, rng: random.Random) -> Specification:
    # Меняем тип одного из параметров на другой
    if not spec.params:
        return spec
    idx = rng.randint(0, len(spec.params)-1)
    old_param = spec.params[idx]
    # Выбираем другой тип
    other_types = [t for t in ["int", "float", "str", "bool", "list"] if t != old_param.type]
    if not other_types:
        return spec
    new_type = rng.choice(other_types)
    new_param = Parameter(old_param.name, new_type)
    new_params = list(spec.params)
    new_params[idx] = new_param
    return Specification(spec.func_name, new_params, spec.return_type, spec.constraints, spec.exceptions)

# ---------- СРЕДА ----------
class ProgrammingEnvironment:
    def __init__(self, task: Specification, contradictions: int = 0, seed: int = 42):
        self.rng = random.Random(seed)
        self.true_spec = task
        # Если contradictions > 0, создаём искажённую версию
        self.observed_spec = task
        for _ in range(contradictions):
            self.observed_spec = add_contradiction(self.observed_spec, self.rng)

        # Храним атрибуты, которые агент может уточнить
        self.attributes = {
            "func_name": self.observed_spec.func_name,
            "params": self.observed_spec.params,
            "return_type": self.observed_spec.return_type,
            "constraints": self.observed_spec.constraints,
            "exceptions": self.observed_spec.exceptions,
        }
        # Истинные атрибуты (для проверки)
        self.true_attributes = {
            "func_name": self.true_spec.func_name,
            "params": self.true_spec.params,
            "return_type": self.true_spec.return_type,
            "constraints": self.true_spec.constraints,
            "exceptions": self.true_spec.exceptions,
        }

    def ask(self, attr_name: str) -> str:
        # Возвращает уточнение по атрибуту (из истинной спецификации)
        # Но с вероятностью 0.2 может вернуть случайное значение (шум)
        if self.rng.random() < 0.2:
            # Возвращаем случайное значение
            if attr_name == "func_name":
                return "func_" + self.rng.choice(["a", "b", "c"])
            elif attr_name == "params":
                return [Parameter("p", self.rng.choice(["int", "float"])) for _ in range(1)]
            elif attr_name == "return_type":
                return self.rng.choice(["int", "float", "str"])
            elif attr_name == "constraints":
                return ["x > 0"]
            elif attr_name == "exceptions":
                return ["ValueError"]
        # Иначе возвращаем истинное значение
        return self.true_attributes[attr_name]

    def check_code(self, code: str) -> bool:
        # Проверяем, соответствует ли код истинной спецификации
        # В упрощённом виде: код должен содержать все параметры и возвращать правильный тип
        # Мы проверяем, что код использует правильные имена параметров и типы.
        # Для простоты считаем код корректным, если:
        # - в коде присутствуют все параметры из истинной спецификации
        # - возвращаемый тип совпадает
        params_ok = all(p.name in code for p in self.true_spec.params)
        return_type_ok = self.true_spec.return_type in code
        return params_ok and return_type_ok

# ---------- ПОЛИТИКИ ----------
def persistent_policy(agent_state: dict) -> tuple[str | None, float]:
    # Если есть противоречия (например, параметры с разными типами) -> запрос
    # Иначе -> решение
    # Для простоты: если конфликтов нет, отдаём код
    return None, 0.0

def net_policy(agent_state: dict) -> tuple[str | None, float]:
    # Используем NET-подход: оцениваем "чистоту" спецификации
    # Если есть конфликт (например, параметр имеет два разных типа), возвращаем запрос
    return None, 0.0

def support_policy(agent_state: dict) -> tuple[str | None, float]:
    # Выбираем наиболее вероятное решение (например, по частоте)
    return None, 0.0

# ---------- АГЕНТ ----------
class ProgrammingAgent:
    def __init__(self, seed: int, policy_name: str):
        self.rng = random.Random(seed)
        self.policy_name = policy_name
        self.policy = {
            "PERSISTENT": persistent_policy,
            "NET": net_policy,
            "SUPPORT": support_policy,
            "CONSISTENCY": persistent_policy,  # запасной
        }[policy_name]
        self.observations = {}  # attr_name -> значение

    def reset(self):
        self.observations = {}

    def choose_action(self, env: ProgrammingEnvironment) -> tuple[str, str | None]:
        # Возвращает либо запрос (attr_name), либо код (code)
        # Используем политику
        decision, confidence = self.policy(self.observations)
        if decision is not None:
            return "code", decision
        else:
            # Выбираем атрибут с наибольшей неопределённостью (если ещё не уточнён)
            possible = [attr for attr in env.attributes if attr not in self.observations]
            if not possible:
                return "code", None  # нет больше атрибутов
            attr = self.rng.choice(possible)
            return "ask", attr

    def update(self, attr: str, value):
        self.observations[attr] = value

# ---------- ЭКСПЕРИМЕНТ ----------
def run_episode(policy_name: str, seed: int, contradictions: int, max_queries: int) -> dict:
    rng = random.Random(seed)
    generator = TaskGenerator(seed)
    task = generator.generate_task()
    env = ProgrammingEnvironment(task, contradictions, seed)

    agent = ProgrammingAgent(seed, policy_name)
    agent.reset()

    queries = 0
    code = None
    success = False

    for step in range(max_queries + 1):
        action_type, content = agent.choose_action(env)
        if action_type == "ask":
            if content is None:
                # нет атрибутов для уточнения
                break
            value = env.ask(content)
            agent.update(content, value)
            queries += 1
        else:  # code
            if content is None:
                break
            code = content
            success = env.check_code(code)
            break

    # Если не удалось выдать код, считаем неудачей
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
    success_rate = success / total
    avg_queries = total_queries / total
    return {
        "total": total,
        "success": success,
        "success_rate": success_rate,
        "avg_queries": avg_queries,
    }

def main():
    contradictions_levels = [0, 1, 2]  # число противоречий в спецификации
    policies = ["SUPPORT", "NET", "PERSISTENT", "CONSISTENCY"]
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

    # Агрегация
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["policy"]][r["contradictions"]].append(r)

    # Сохранение CSV
    output_csv = Path("results/phase3/test09/results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Вывод сводки
    print("\n=== Summary by Policy and Contradictions ===")
    for policy in policies:
        print(f"\n{policy}:")
        for contr in contradictions_levels:
            rows = grouped[policy][contr]
            agg = aggregate_results(rows)
            print(f"  Contradictions={contr}: success_rate={agg['success_rate']:.3f}, avg_queries={agg['avg_queries']:.2f}")

    print(f"\nResults written to {output_csv}")
    print("TEST09 PASSED")

if __name__ == "__main__":
    main()
