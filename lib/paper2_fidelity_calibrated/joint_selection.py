from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple


def _finite_score(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return float("nan")
    return score if math.isfinite(score) else float("nan")


def layer_best_scores(score_grid: Mapping[int, Mapping[int, float]]) -> dict[int, float]:
    """Return the best finite selection score observed for each layer."""
    best_by_layer: dict[int, float] = {}
    for layer, topk_scores in score_grid.items():
        finite_scores = [
            _finite_score(score)
            for score in (topk_scores or {}).values()
        ]
        finite_scores = [score for score in finite_scores if math.isfinite(score)]
        if finite_scores:
            best_by_layer[int(layer)] = max(finite_scores)
    return best_by_layer


def select_best_joint_config(
    score_grid: Mapping[int, Mapping[int, float]],
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Pick the best `(layer, top-k)` jointly from a nested score grid.

    Ties are broken deterministically by preferring the smaller `(layer, top-k)`
    tuple. This keeps selection stable across repeated runs.
    """
    best_key: Optional[Tuple[int, int]] = None
    best_score = float("nan")

    for layer, topk_scores in score_grid.items():
        layer_key = int(layer)
        for topk, raw_score in (topk_scores or {}).items():
            topk_key = int(topk)
            score = _finite_score(raw_score)
            if not math.isfinite(score):
                continue
            candidate_key = (layer_key, topk_key)
            if (
                best_key is None
                or score > best_score
                or (score == best_score and candidate_key < best_key)
            ):
                best_key = candidate_key
                best_score = score

    if best_key is None:
        return None, None, float("nan")
    return best_key[0], best_key[1], best_score
