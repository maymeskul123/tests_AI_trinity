# Experimental Results: Binary, Ternary, and Quaternary Decision Agents

## Overview

This project investigates whether increasing the number of explicit logical states available to an agent can provide practical advantages in decision-making under uncertainty.

The experiments compare three agent architectures:

* **BINARY** — a two-state decision model
* **TERNARY** — a three-state decision model
* **QUATERNARY** — a four-state decision model

The central hypothesis is that a system with more explicit logical states may handle uncertainty, contradictions, incomplete information, and ambiguity more effectively than a conventional binary decision model.

Rather than treating multi-valued logic as a purely theoretical concept, this project evaluates its practical effect through controlled simulations.

---

# 1. Research Question

The main research question is:

> Can an agent with ternary or quaternary logical states make better decisions than a binary agent when operating under uncertainty, conflicting information, and limited information budgets?

The experiments focus on several important factors:

* unknown information;
* conflicting information;
* limited reasoning/query budgets;
* information acquisition;
* false confidence;
* abstention from making a decision;
* overall reward.

---

# 2. Agent Types

## 2.1 Binary Agent

The binary agent operates with two primary logical outcomes:

* TRUE
* FALSE

Conceptually:

```text
TRUE / FALSE
```

A binary system must eventually reduce information to one of two alternatives.

This architecture is simple and efficient but has a fundamental limitation: uncertainty and contradiction must either be ignored, implicitly represented, or forced into one of the two available states.

---

## 2.2 Ternary Agent

The ternary agent introduces an additional logical state:

* TRUE
* FALSE
* UNKNOWN

Conceptually:

```text
TRUE / FALSE / UNKNOWN
```

The `UNKNOWN` state allows the agent to explicitly represent insufficient information.

Instead of guessing, the agent can indicate that the available evidence is not sufficient to support a confident decision.

This is particularly useful in environments with:

* incomplete information;
* noisy observations;
* limited reasoning budgets;
* missing data.

---

## 2.3 Quaternary Agent

The quaternary agent uses four explicit logical states:

* TRUE
* FALSE
* UNKNOWN
* CONFLICT

Conceptually:

```text
TRUE
FALSE
UNKNOWN
CONFLICT
```

The additional `CONFLICT` state distinguishes between two fundamentally different situations:

### Unknown

There is not enough information to determine the answer.

```text
Evidence → insufficient
Result   → UNKNOWN
```

### Conflict

There is information, but the information supports incompatible conclusions.

```text
Evidence A → TRUE
Evidence B → FALSE

Result → CONFLICT
```

This distinction is one of the main theoretical advantages investigated in this project.

A binary system may be forced to collapse both situations into a simple YES/NO decision.

A ternary system can represent uncertainty but may still struggle to distinguish:

```text
"No information"

from

"Contradictory information"
```

The quaternary system explicitly represents both.

---

# 3. Experimental Environment

The agents were tested under controlled conditions with varying levels of:

* unknown information;
* conflicting information;
* reasoning/query budgets.

The experimental parameters included:

```text
unknown_rate
conflict_rate
budget
agent
```

The following agent types were compared:

```text
BINARY
TERNARY
QUATERNARY
```

The main budget values tested were:

```text
1
2
3
4
5
6
8
10
12
```

Unknown and conflict rates were varied across multiple scenarios.

---

# 4. Result Columns

Each experiment produced the following metrics:

| Metric             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `unknown_rate`     | Probability of encountering unknown or missing information   |
| `conflict_rate`    | Probability of encountering conflicting information          |
| `budget`           | Number of information queries or reasoning steps available   |
| `agent`            | Agent architecture being tested                              |
| `correct`          | Fraction of correct decisions                                |
| `queries`          | Average number of queries used                               |
| `information`      | Amount of information obtained or accumulated                |
| `false_confidence` | Incorrect decisions made with unjustified confidence         |
| `abstain`          | Fraction of cases where the agent refused to make a decision |
| `reward`           | Overall reward according to the experiment's reward function |

---

# 5. Understanding the Metrics

## 5.1 Correct

`correct` represents the proportion of decisions that were correct.

Example:

```text
correct = 0.98
```

means that approximately 98% of decisions were correct.

This is one of the most important performance metrics.

However, accuracy alone is not sufficient.

An agent could achieve high accuracy by refusing to answer difficult questions.

Therefore, the results must also be evaluated together with:

* abstention;
* false confidence;
* reward.

---

## 5.2 Queries

`queries` represents the average number of information requests or reasoning operations used by the agent.

A higher query count generally means:

```text
More information
+
More computational cost
```

The budget limits how much information the agent can acquire.

The experiment therefore investigates whether different logical architectures use information more efficiently.

---

## 5.3 Information

`information` represents the amount of useful information accumulated by the agent.

Higher values generally indicate that the agent successfully extracted more information from the available environment.

However, more information is not automatically better.

Conflicting information can increase the amount of information while simultaneously making the final decision more difficult.

Therefore:

```text
Information quantity ≠ decision quality
```

The logical structure of the agent determines how effectively the information can be interpreted.

---

## 5.4 False Confidence

`false_confidence` measures situations where the agent makes an incorrect decision while appearing confident.

This is one of the most important safety metrics.

A system that says:

```text
"I don't know"
```

can be safer than a system that confidently produces an incorrect answer.

High false confidence is particularly dangerous in applications involving:

* autonomous systems;
* medical reasoning;
* financial decisions;
* scientific analysis;
* AI assistants;
* automated decision-making.

The goal is therefore not only to maximize correctness but also to minimize:

```text
Wrong + Confident
```

decisions.

---

## 5.5 Abstain

`abstain` represents the proportion of cases where the agent decides not to provide a definitive answer.

Example:

```text
abstain = 1.0
```

means that the agent abstained in 100% of cases.

At first glance, abstention may appear to be negative.

However, abstention can be rational.

For example:

```text
Insufficient information
→ abstain
```

may be better than:

```text
Insufficient information
→ random confident answer
```

Therefore, abstention must be evaluated together with correctness and reward.

---

## 5.6 Reward

`reward` represents the overall performance according to the experiment's reward function.

The reward function combines different aspects of agent behavior, such as:

* correct decisions;
* incorrect decisions;
* abstention;
* information cost;
* query cost.

In the current experiments, rewards are generally negative because the cost of acquiring information and making decisions exceeds the reward obtained from successful outcomes.

This does not necessarily mean that the agents are malfunctioning.

Instead, it indicates that the reward function strongly penalizes:

```text
queries
+
information cost
+
incorrect decisions
```

The most important comparison is therefore not the absolute reward value alone but:

```text
Relative performance between agents
```

under the same experimental conditions.

---

# 6. Experiment 1: No Unknown Information and No Conflict

Parameters:

```text
unknown_rate = 0.0
conflict_rate = 0.0
```

In this environment, all information is consistent and available.

The results show that:

```text
BINARY ≈ TERNARY ≈ QUATERNARY
```

All three architectures behave almost identically.

For example:

| Budget | Correct | Queries |
| -----: | ------: | ------: |
|      1 |     0.0 |       1 |
|      2 |     0.0 |       2 |
|      3 |     0.0 |       3 |
|      6 |     0.0 |       6 |
|     12 |     0.0 |      12 |

The key observation is that when the environment contains neither uncertainty nor contradiction, additional logical states provide little or no advantage.

### Conclusion

Multi-valued logic is not automatically superior.

Its advantage appears only when the environment contains situations that cannot be efficiently represented by simple binary states.

---

# 7. Experiment 2: Unknown Information

Parameters included scenarios such as:

```text
unknown_rate = 0.2
unknown_rate = 0.4
```

with different reasoning budgets.

As the amount of unknown information increased, the information available to the agents decreased.

For example, with:

```text
unknown_rate = 0.4
```

the information metric dropped dramatically compared with environments where:

```text
unknown_rate = 0.0
```

This demonstrates an important principle:

```text
Less reliable information
→ lower effective information gain
```

However, the logical architectures still behaved similarly in many scenarios.

### Conclusion

Simply adding an `UNKNOWN` state does not automatically create a large advantage.

The architecture must actually use the additional state as part of the decision process.

---

# 8. Experiment 3: Conflicting Information

The most interesting behavior appeared when conflicting information was introduced.

Parameters included:

```text
conflict_rate = 0.1
conflict_rate = 0.2
```

When conflict was present, the architectures began to diverge.

For example, under:

```text
unknown_rate = 0.4
conflict_rate = 0.1
```

small but measurable differences appeared between the agents.

At higher budgets:

```text
Budget = 8
Budget = 10
Budget = 12
```

the ternary and binary agents began to show:

* non-zero incorrect decisions;
* non-zero false confidence;
* increased abstention behavior.

Meanwhile, the quaternary agent frequently maintained:

```text
correct = 0.0
false_confidence = 0.0
abstain = 1.0
```

This means the current quaternary implementation was extremely conservative.

Instead of making an incorrect decision under contradiction, it preferred to abstain.

---

# 9. Important Observation: The Quaternary Agent Is Conservative

One of the strongest observations from the current experiments is that the quaternary agent behaves differently from the other architectures.

When uncertainty or contradiction becomes significant, the quaternary agent frequently chooses:

```text
UNKNOWN
```

or:

```text
CONFLICT
```

instead of forcing a binary answer.

This produces:

```text
false_confidence ≈ 0
```

in many scenarios.

This is potentially valuable.

However, it also produces:

```text
abstain = 1.0
```

in many experimental configurations.

Therefore, the current quaternary implementation demonstrates:

### Strong uncertainty detection

but

### Weak decision conversion

The agent can recognize that the available information is problematic, but the current policy does not yet provide an effective mechanism for resolving uncertainty or contradiction.

---

# 10. Binary and Ternary Results

In the current experiments, binary and ternary agents often produced very similar results.

For example:

```text
BINARY ≈ TERNARY
```

across many low-conflict scenarios.

This suggests that simply introducing an `UNKNOWN` state is not enough.

The ternary architecture requires a policy that actively exploits the additional state.

For example:

```text
UNKNOWN
→ gather additional information
→ evaluate confidence
→ answer or abstain
```

Without such a strategy, the ternary agent may behave almost like a binary system with an additional label.

---

# 11. High Conflict Scenario

The strongest differences appeared under more difficult conditions such as:

```text
unknown_rate = 0.4
conflict_rate = 0.2
```

At higher budgets, binary and ternary agents began to produce measurable levels of:

```text
incorrect decisions
+
false confidence
```

Examples include:

```text
Budget = 8
BINARY false_confidence ≈ 0.0242
TERNARY false_confidence ≈ 0.0266
QUATERNARY false_confidence = 0.0
```

At:

```text
Budget = 12
```

the binary and ternary agents showed even larger false-confidence values.

Meanwhile, the quaternary agent continued to avoid false confidence.

This suggests a possible advantage of explicit conflict representation.

---

# 12. Main Experimental Findings

The experiments currently support several important conclusions.

---

## Finding 1: Additional Logical States Are Not Automatically Better

When the environment contains:

```text
No uncertainty
+
No contradiction
```

all architectures perform similarly.

Therefore:

```text
Binary logic is sufficient
```

for simple, fully consistent environments.

---

## Finding 2: Uncertainty Requires More Than an Additional State

Adding:

```text
UNKNOWN
```

does not automatically improve decision quality.

The agent must have a strategy for using that state.

For example:

```text
UNKNOWN
→ request more information
```

or:

```text
UNKNOWN
→ reduce confidence
```

or:

```text
UNKNOWN
→ abstain
```

The architecture and policy must work together.

---

## Finding 3: Conflict Is Different From Missing Information

This is the most important conceptual distinction in the experiments.

Consider:

### Case A: Unknown

```text
No sufficient evidence exists.
```

### Case B: Conflict

```text
Evidence exists, but supports incompatible conclusions.
```

These situations are fundamentally different.

A quaternary architecture can represent both explicitly:

```text
UNKNOWN
CONFLICT
```

This potentially provides richer reasoning capabilities.

---

## Finding 4: Quaternary Logic Can Reduce False Confidence

The quaternary agent frequently demonstrated:

```text
false_confidence = 0
```

even in difficult scenarios.

This suggests that explicit conflict detection may prevent the system from making unjustified confident decisions.

This is potentially important for reliable AI systems.

---

## Finding 5: The Current Quaternary Agent Abstains Too Often

The main weakness of the current implementation is excessive abstention.

In many scenarios:

```text
abstain = 1.0
```

This means the agent successfully avoids incorrect confident decisions but also fails to convert information into useful decisions.

Therefore, the next stage should focus on:

```text
Conflict detection
+
Conflict resolution
```

rather than conflict detection alone.

---

# 13. Current Architecture Limitation

The current experiments primarily test:

```text
Logical state representation
```

but not yet:

```text
Advanced multi-valued reasoning strategies
```

The agents currently differ mainly in their ability to represent states.

The next generation of experiments should allow the agents to behave differently after detecting:

```text
UNKNOWN
```

or:

```text
CONFLICT
```

For example:

### Binary

```text
TRUE / FALSE
→ choose
```

### Ternary

```text
TRUE / FALSE / UNKNOWN

UNKNOWN
→ gather more information
→ reevaluate
```

### Quaternary

```text
TRUE / FALSE / UNKNOWN / CONFLICT

UNKNOWN
→ gather information

CONFLICT
→ identify conflicting sources
→ compare evidence
→ resolve contradiction
→ estimate confidence
→ answer or abstain
```

This would test the real potential of multi-valued reasoning.

---

# 14. Proposed Next Experiments

The next stage should introduce active reasoning policies.

---

## Experiment A: Active Information Gathering

When the agent encounters:

```text
UNKNOWN
```

it should automatically request additional information.

Example:

```text
UNKNOWN
↓
Query additional source
↓
New evidence
↓
Reevaluate state
```

This would test whether multi-valued agents use information more efficiently.

---

## Experiment B: Conflict Resolution

When the agent encounters:

```text
CONFLICT
```

it should attempt to resolve the contradiction.

Possible strategies:

```text
1. Source reliability weighting
2. Majority evidence
3. Bayesian evidence update
4. Confidence weighting
5. Temporal consistency
6. Causal consistency
7. Additional information requests
```

---

## Experiment C: Confidence Calibration

The agents should maintain an explicit confidence value:

```text
0.0 → no confidence
1.0 → complete confidence
```

The goal is not simply:

```text
Maximum confidence
```

but:

```text
Correctly calibrated confidence
```

Ideally:

```text
High confidence → usually correct
Low confidence  → uncertainty
```

---

## Experiment D: Conflict-Aware Query Strategy

A quaternary agent could allocate its budget dynamically.

Example:

```text
No conflict
→ stop early

UNKNOWN
→ query more

CONFLICT
→ investigate conflicting evidence
```

This could potentially produce a significant efficiency advantage.

---

# 15. Hypothesis for Future Research

The current results suggest the following hypothesis:

> The primary advantage of multi-valued logic does not come from having more states by itself. The advantage comes from allowing an agent to use different reasoning strategies depending on whether information is true, false, unknown, or contradictory.

This distinction is critical.

The architecture:

```text
TRUE / FALSE / UNKNOWN / CONFLICT
```

creates a richer representation of epistemic state.

But the real advantage appears only when the agent can act differently depending on that state.

---

# 16. General Conclusion

The current experiments provide preliminary evidence that multi-valued logical architectures may offer advantages in uncertain and contradictory environments.

The main findings are:

```text
✓ Binary logic works well in simple environments.

✓ Ternary logic can explicitly represent missing information.

✓ Quaternary logic can distinguish missing information from contradictory information.

✓ Explicit conflict representation can reduce false confidence.

✓ The current quaternary implementation is highly conservative.

✓ Additional reasoning policies are required to convert logical-state advantages into higher decision quality.
```

The most important conclusion is:

> More logical states alone do not automatically improve intelligence. The advantage emerges when the additional states allow different information-gathering, reasoning, confidence, and conflict-resolution strategies.

---

# 17. Future Direction

The next stage of the project should move from:

```text
State Representation
```

to:

```text
State-Aware Decision Policies
```

The proposed architecture is:

```text
                ┌───────────────┐
                │   Evidence    │
                └───────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Logical Analyzer │
              └────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
      TRUE          UNKNOWN        CONFLICT
        │              │              │
        │              ▼              ▼
        │       Gather More      Resolve Conflict
        │       Information      Compare Sources
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Decision Policy  │
              └────────┬─────────┘
                       │
                       ▼
             Answer / Abstain / Query
```

The ultimate goal is to determine whether:

```text
Multi-valued representation
+
State-aware reasoning
+
Adaptive information gathering
```

can outperform conventional binary decision systems in complex uncertain environments.

---

# Status

**Current stage:**

```text
✓ Basic simulation completed
✓ Binary agent tested
✓ Ternary agent tested
✓ Quaternary agent tested
✓ Unknown information tested
✓ Conflicting information tested
✓ Multiple budgets tested

Next:

→ Active information gathering
→ Conflict resolution
→ Confidence calibration
→ Adaptive query policies
→ More realistic environments
```

## Project Direction

The project is evolving from a simple comparison of logical state spaces into a broader investigation of:

> **How the structure of logical representation influences information gathering, uncertainty handling, conflict resolution, confidence calibration, and intelligent decision-making.**
