from __future__ import annotations

from difflib import SequenceMatcher


def exact_match(pred: str, gold: str) -> float:
    return float(str(pred).strip() == str(gold).strip())


def normalized_similarity(pred: str, gold: str) -> float:
    return float(SequenceMatcher(a=str(pred), b=str(gold)).ratio())


def cer(pred: str, gold: str) -> float:
    """
    Character error rate = Levenshtein distance / max(1, len(gold)).
    """
    a = str(pred)
    b = str(gold)
    if not b:
        return 0.0 if not a else 1.0

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return float(dp[-1] / max(1, len(b)))

