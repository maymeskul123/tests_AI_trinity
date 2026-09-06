import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agents import TernaryAgent
from core.constants import UNKNOWN, WRONG, RIGHT


def main():
    print("=" * 60)
    print("TEST 1 — BASIC TERNARY LEARNING")
    print("=" * 60)

    agent = TernaryAgent(seed=42)

    x = 5
    candidates = tuple(range(10))

    print("\nInitial state:")

    for candidate in candidates:
        print(
            f"x={x}, "
            f"candidate={candidate}, "
            f"state={agent.state(x, candidate)}"
        )

    # Агент получает обратную связь:
    # 3 — неправильный вариант
    # 7 — правильный вариант

    agent.learn(x, 3, WRONG)
    agent.learn(x, 7, RIGHT)

    print("\nAfter learning:")

    for candidate in candidates:
        print(
            f"x={x}, "
            f"candidate={candidate}, "
            f"state={agent.state(x, candidate)}"
        )

    prediction = agent.predict(x, candidates)

    print("\nPrediction:")
    print(f"x={x} -> prediction={prediction}")

    assert agent.state(x, 3) == WRONG
    assert agent.state(x, 7) == RIGHT
    assert agent.state(x, 0) == UNKNOWN
    assert prediction == 7

    print("\nPASS")


if __name__ == "__main__":
    main()
