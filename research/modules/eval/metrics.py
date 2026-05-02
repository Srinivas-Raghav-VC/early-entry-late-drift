from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence

INDIC_VIRAMAS = {
    "\u094d",  # Devanagari
    "\u09cd",  # Bengali
    "\u0bcd",  # Tamil
    "\u0c4d",  # Telugu
    "\u0ccd",  # Kannada
    "\u0d4d",  # Malayalam
}
JOINERS = {"\u200c", "\u200d"}

SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "Devanagari": [(0x0900, 0x097F)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
}

REFUSAL_MARKERS = {
    "i can't",
    "i cannot",
    "cannot transliterate",
    "sorry",
    "as an ai",
    "unable",
}


def normalize_text(text: str | None) -> str:
    return unicodedata.normalize("NFC", str(text or "")).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return int(dp[-1])


def segment_aksharas(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    out: list[str] = []
    buf = ""
    carry = False
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
                carry = False
            continue

        if not buf:
            buf = ch
            carry = ch in INDIC_VIRAMAS or ch in JOINERS
            continue

        prev = buf[-1]
        if (
            carry
            or unicodedata.category(ch).startswith("M")
            or ch in JOINERS
            or prev in INDIC_VIRAMAS
            or prev in JOINERS
        ):
            buf += ch
        else:
            out.append(buf)
            buf = ch
        carry = ch in INDIC_VIRAMAS or ch in JOINERS

    if buf:
        out.append(buf)
    return out


def akshara_cer(pred: str, gold: str) -> float:
    pred_aks = segment_aksharas(pred)
    gold_aks = segment_aksharas(gold)
    if not gold_aks:
        return 0.0 if not pred_aks else 1.0
    return float(_levenshtein(pred_aks, gold_aks) / max(1, len(gold_aks)))


def normalized_edit_distance(pred: str, gold: str) -> float:
    a = list(normalize_text(pred))
    b = list(normalize_text(gold))
    if not b:
        return 0.0 if not a else 1.0
    return float(_levenshtein(a, b) / max(1, len(b)))


def classify_char_script(ch: str) -> str:
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return "Other"


def script_chars(text: str) -> list[str]:
    chars: list[str] = []
    for ch in normalize_text(text):
        if ch.isspace() or unicodedata.category(ch).startswith(("P", "N", "S")):
            continue
        chars.append(ch)
    return chars


def infer_primary_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in script_chars(text):
        script = classify_char_script(ch)
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "Unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def script_ratio(text: str, target_script: str) -> float:
    chars = script_chars(text)
    if not chars:
        return 0.0
    hits = sum(1 for ch in chars if classify_char_script(ch) == target_script)
    return float(hits / len(chars))


def script_valid(text: str, target_script: str, *, min_ratio: float = 1.0) -> float:
    ratio = script_ratio(text, target_script)
    return float(ratio >= min_ratio and ratio > 0.0)


def empty_or_refusal(text: str) -> float:
    norm = normalize_text(text).lower()
    if not norm:
        return 1.0
    if any(marker in norm for marker in REFUSAL_MARKERS):
        return 1.0
    return 0.0


def standalone_answer(text: str) -> float:
    norm = normalize_text(text)
    if not norm:
        return 0.0
    if "\n" in norm:
        return 0.0
    if len(norm.split()) > 2:
        return 0.0
    return 1.0


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    summary = {
        "exact_match_rate": mean(r["exact_match"] for r in rows),
        "akshara_CER_mean": mean(r["akshara_CER"] for r in rows),
        "script_validity_rate": mean(r["script_valid"] for r in rows),
        "empty_or_refusal_rate": mean(r["empty_or_refusal"] for r in rows),
    }
    if rows and "standalone_answer" in rows[0]:
        summary["standalone_answer_rate"] = mean(
            float(r.get("standalone_answer", 0.0)) for r in rows
        )
    if rows and "hit_max_new_tokens" in rows[0]:
        summary["hit_max_new_tokens_rate"] = mean(
            float(r.get("hit_max_new_tokens", 0.0)) for r in rows
        )
    if rows and "raw_strict_word_only" in rows[0]:
        summary["raw_strict_word_only_rate"] = mean(
            float(r.get("raw_strict_word_only", 0.0)) for r in rows
        )
    if rows and "has_leading_text" in rows[0]:
        summary["leading_text_rate"] = mean(
            float(r.get("has_leading_text", 0.0)) for r in rows
        )
    if rows and "has_trailing_text" in rows[0]:
        summary["trailing_text_rate"] = mean(
            float(r.get("has_trailing_text", 0.0)) for r in rows
        )
    return summary
