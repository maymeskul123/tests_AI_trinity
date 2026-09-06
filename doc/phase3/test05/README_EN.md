# TEST05 — Decision Policy Comparison

## Goal

TEST05 evaluates not evidence accumulation itself, but the next layer:

> how different decision policies interpret the same accumulated evidence.

All four policies receive exactly the same evidence.

Candidate `0` is the ground-truth candidate in every scenario.

The benchmark contains **20 scenarios × 4 policies = 80 result rows**.

## Policies

### PERSISTENT

Selects a candidate only when:

- at least one supporting observation exists;
- there is no conflicting observation;
- the candidate is unique.

This is a strict "clean evidence only" policy.

### NET

Uses:

`NET = support - conflict`

The policy selects the unique candidate with the highest positive NET score.

It abstains on ties or when the maximum score is not positive.

### CONSISTENCY

Uses:

`CONSISTENCY = support / (support + conflict)`

The unique candidate with the highest consistency is selected.

This emphasizes relative evidence quality rather than absolute evidence volume.

### SUPPORT

Uses only the absolute amount of supporting evidence:

`SUPPORT = support`

The unique candidate with the highest support count is selected.

It abstains on ties.

## Metrics

For every policy:

- `DECISIONS` — number of scenarios where a decision was made;
- `CORRECT` — number of correct decisions;
- `ACCURACY` — correct decisions divided by decisions made;
- `COVERAGE` — fraction of scenarios where a decision was made;
- `OVERALL` — correct decisions divided by all scenarios.

Conflict scenarios are also analyzed separately.

## Methodological interpretation

TEST05 does not establish a universally optimal decision policy.

Each policy answers a different question:

- **PERSISTENT:** "Is there a candidate supported only by clean evidence?"
- **NET:** "Which candidate has the largest support advantage over conflict?"
- **CONSISTENCY:** "Which candidate has the most internally consistent evidence?"
- **SUPPORT:** "Which candidate has the largest amount of positive evidence?"

Therefore TEST05 measures the trade-off between:

**decisiveness ↔ conflict sensitivity ↔ evidence quantity ↔ abstention.**

## Relation to previous tests

TEST01 validated the basic evidence/state semantics.

TEST02 evaluated accumulation under controlled noise.

TEST03 isolated the semantics of persistent conflict versus evidence dominance.

TEST04 showed that adding a fourth state does not by itself solve the decision problem.

TEST05 separates:

**knowledge representation**

from

**decision policy**.

This is an important Phase 3 result.

## Limitations

TEST05 is a controlled synthetic benchmark.

It does not prove that one policy is universally superior.

The next step should be a statistically generated benchmark with controlled:

- probability of correct evidence;
- probability of false evidence;
- conflict rate;
- observation count;
- query budget;
- ambiguity between candidates.

This would allow policies to be compared across distributions of many randomly generated tasks rather than only hand-designed scenarios.
