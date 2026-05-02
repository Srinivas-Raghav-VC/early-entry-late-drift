from __future__ import annotations

import random
from typing import Any

from research.modules.data.row_schema import get_target_text, with_target_text
from research.modules.eval.metrics import segment_aksharas

ICL_VARIANTS = [
    "helpful",
    "random",
    "shuffled_targets",
    "corrupted_targets",
]


def _require_variant(name: str) -> str:
    key = str(name).strip()
    if key not in ICL_VARIANTS:
        raise ValueError(f"Unknown ICL variant {name!r}; expected one of {ICL_VARIANTS}")
    return key


def _permute_targets(
    examples: list[dict[str, str]],
    rng: random.Random,
    *,
    pool_targets: list[str] | None = None,
) -> list[str]:
    n = len(examples)
    if n <= 1:
        if pool_targets:
            return [pool_targets[0]]
        return [get_target_text(examples[0])] if examples else []

    targets = [get_target_text(ex) for ex in examples]
    for _ in range(8):
        shuffled = list(targets)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(targets, shuffled)):
            return shuffled

    if pool_targets:
        sampled = list(pool_targets)
        rng.shuffle(sampled)
        return sampled[:n]
    return targets[1:] + targets[:1]


def _corrupt_target(target: str, rng: random.Random) -> str:
    chunks = segment_aksharas(target)
    if len(chunks) <= 1:
        return target
    shuffled = list(chunks)
    rng.shuffle(shuffled)
    if "".join(shuffled) == target:
        shuffled = shuffled[1:] + shuffled[:1]
    return "".join(shuffled)


def materialize_icl_variant(
    *,
    variant: str,
    n: int,
    helpful_examples: list[dict[str, str]],
    candidate_pool: list[dict[str, str]],
    rng_seed: int,
) -> list[dict[str, str]]:
    key = _require_variant(variant)
    n = int(n)
    if n <= 0:
        return []

    rng = random.Random(int(rng_seed))
    helpful = [dict(ex) for ex in helpful_examples[:n]]

    if key == "helpful":
        return helpful

    if key == "random":
        if len(candidate_pool) < n:
            raise ValueError(f"candidate_pool too small for random ICL (need {n}, got {len(candidate_pool)})")
        sampled = rng.sample(candidate_pool, k=n)
        return [dict(ex) for ex in sampled]

    if key == "shuffled_targets":
        if not helpful:
            return []
        if len(candidate_pool) >= n:
            alt_pool = [get_target_text(r) for r in rng.sample(candidate_pool, k=n)]
        else:
            alt_pool = [get_target_text(r) for r in candidate_pool]
        permuted_targets = _permute_targets(helpful, rng, pool_targets=alt_pool)
        out: list[dict[str, str]] = []
        for ex, tgt in zip(helpful, permuted_targets):
            out.append(with_target_text(ex, str(tgt), keep_legacy_alias=True))
        return out

    if key == "corrupted_targets":
        out: list[dict[str, str]] = []
        for ex in helpful:
            out.append(
                with_target_text(
                    ex,
                    _corrupt_target(get_target_text(ex), rng),
                    keep_legacy_alias=True,
                )
            )
        return out

    raise AssertionError(f"Unhandled ICL variant {key}")


def parse_variant_csv(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_require_variant(part))
    # preserve order, dedupe
    deduped: list[str] = []
    for name in out:
        if name not in deduped:
            deduped.append(name)
    return deduped
