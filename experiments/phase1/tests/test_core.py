from core.agents import BinaryAgent, TernaryAgent
from core.constants import UNKNOWN, WRONG, RIGHT
from core.environment import HiddenRuleEnvironment


def rule(x):
    return (3 * x + 1) % 10


def test_environment_correct_answer():
    env = HiddenRuleEnvironment(rule)

    task = env.make_task(3)

    assert task.answer == 0
    assert env.evaluate(task, 0) == RIGHT
    assert env.evaluate(task, 1) == WRONG


def test_ternary_unknown_state():
    agent = TernaryAgent(seed=1)

    assert agent.state(5, 3) == UNKNOWN


def test_ternary_positive_feedback():
    agent = TernaryAgent(seed=1)

    agent.learn(5, 3, RIGHT)

    assert agent.state(5, 3) == RIGHT


def test_ternary_negative_feedback():
    agent = TernaryAgent(seed=1)

    agent.learn(5, 3, WRONG)

    assert agent.state(5, 3) == WRONG


def test_ternary_known_solution_preferred():
    agent = TernaryAgent(seed=1)

    agent.learn(5, 7, RIGHT)

    prediction = agent.predict(
        5,
        tuple(range(10)),
    )

    assert prediction == 7


def test_binary_learning():
    agent = BinaryAgent(seed=1)

    agent.learn(5, 7, RIGHT)

    assert agent.predict(
        5,
        tuple(range(10)),
    ) == 7


def test_agents_start_empty():
    binary = BinaryAgent(seed=1)
    ternary = TernaryAgent(seed=1)

    assert len(binary.memory) == 0
    assert len(ternary.memory) == 0


def test_learning_changes_prediction():
    agent = TernaryAgent(seed=1)

    agent.learn(5, 7, RIGHT)

    assert agent.predict(
        5,
        tuple(range(10)),
    ) == 7


def test_agent_does_not_know_hidden_rule():
    agent = TernaryAgent(seed=1)

    assert not hasattr(agent, "rule")
    assert not hasattr(agent, "environment")


def test_no_assumed_generalization():
    env = HiddenRuleEnvironment(rule)

    task = env.make_task(9)

    assert task.answer == 8