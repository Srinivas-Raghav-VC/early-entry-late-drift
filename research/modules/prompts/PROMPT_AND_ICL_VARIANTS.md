# Prompt Templates and ICL Variants (Phase 0A)

This file defines the experiment factors used to stress-test the transliteration rescue/degradation problem statement.

## Prompt templates

### 1) `canonical`
`Transliterate the following word from English to {script}.`

Purpose:
- Baseline matching the original study framing.

Potential confound isolated:
- none (reference template).

### 2) `output_only`
Adds explicit output constraint and section headers (`Examples:`, `Input:`).

Purpose:
- test whether stronger answer-format instruction changes rescue/degradation behavior.

Potential confound isolated:
- output-format compliance vs transliteration competence.

### 3) `task_tagged`
Uses `Task:` and `Few-shot pairs:` framing.

Purpose:
- test sensitivity to instruction phrasing and task-tag style.

Potential confound isolated:
- prompt framing dependence.

---

## ICL variants

### 1) `helpful`
Correct source-target pairs.

Purpose:
- expected rescue baseline.

### 2) `random`
Random valid pairs sampled from the same `icl_bank`.

Purpose:
- tests whether *any* examples help, vs semantically aligned exemplars.

### 3) `shuffled_targets`
Keeps source tokens but permutes targets.

Purpose:
- isolates mapping correctness from format/token-count effects.

### 4) `corrupted_targets`
Akshara-level corrupted targets (same script, damaged sequence).

Purpose:
- tests robustness under noisy but script-valid exemplars.

---

## Why this matrix matters

A single prompt and a single ICL construction is easy to challenge in review.
This matrix helps show the problem statement is robust to:
- instruction wording,
- output-format constraints,
- exemplar quality,
- exemplar mapping integrity.
