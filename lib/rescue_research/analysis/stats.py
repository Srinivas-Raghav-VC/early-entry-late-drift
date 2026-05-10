from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted = [1.0] * m
    running_max = 0.0
    for rank, (_, p) in enumerate(indexed):
        val = min(1.0, (m - rank) * float(p))
        running_max = max(running_max, val)
        adjusted_sorted[rank] = running_max
    adjusted = [1.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        adjusted[orig_idx] = adjusted_sorted[rank]
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    adjusted_desc = [1.0] * m
    running_min = 1.0
    for rank_desc, (_, p) in enumerate(indexed):
        i = m - rank_desc
        val = min(1.0, float(p) * m / i)
        running_min = min(running_min, val)
        adjusted_desc[rank_desc] = running_min
    adjusted = [1.0] * m
    for rank_desc, (orig_idx, _) in enumerate(indexed):
        adjusted[orig_idx] = adjusted_desc[rank_desc]
    return adjusted


def paired_permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    if len(a) != len(b):
        raise ValueError("Paired permutation requires equal-length samples.")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    obs = abs(sum(diffs) / max(1, len(diffs)))
    if not diffs:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        stat = abs(sum(signed) / len(signed))
        if stat >= obs:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    vals = [float(v) for v in values]
    for _ in range(n_bootstrap):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * (n_bootstrap - 1))
    hi_idx = int((1.0 - alpha / 2.0) * (n_bootstrap - 1))
    return float(means[lo_idx]), float(means[hi_idx])


def paired_standardized_effect(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    mean_d = sum(diffs) / len(diffs)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    sd_d = math.sqrt(var_d)
    if sd_d == 0.0:
        return 0.0
    return float(mean_d / sd_d)

