# Phase 3 — Test 01: Core Consistency

## Purpose

This test validates the fundamental evidence-based state model used by
Phase 3.

The experiment does not measure accuracy, query efficiency, or benchmark
performance.

It verifies that Binary, Ternary, Quaternary, and Dynamic agents receive
the same underlying evidence and interpret it according to their logical
state model.

## Evidence model

For every `(question, candidate)` pair we maintain:

- `support`
- `conflict`

Agents never modify or erase previous evidence.

## Expected semantics

### Binary

Two effective states:

- RIGHT
- WRONG

Contradictory evidence resolves to WRONG when support and conflict are equal.

### Ternary

Three states:

- WRONG
- UNKNOWN
- RIGHT

Equal support and conflict produce UNKNOWN.

### Quaternary

Four states:

- WRONG
- UNKNOWN
- RIGHT
- CONFLICT

Any simultaneous support and conflict produces CONFLICT.

### Dynamic

TEST01 deliberately uses the same state semantics as Quaternary.

Adaptive decision policies will be tested later.

## Recovery

The sequence:

`SUPPORT -> CONFLICT -> SUPPORT`

must preserve the evidence history.

Therefore:

- Binary ends in RIGHT because support is greater than conflict.
- Ternary ends in RIGHT.
- Quaternary remains CONFLICT because both types of evidence exist.

This distinction is intentional and will be investigated in later tests.
