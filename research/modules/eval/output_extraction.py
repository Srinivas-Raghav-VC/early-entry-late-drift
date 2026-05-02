from __future__ import annotations

import re
from typing import Any

from .metrics import normalize_text, script_ratio

_PREFIXES = (
    "Output:",
    "Answer:",
    "Translation:",
    "Transliteration:",
    "Result:",
)

_TOKEN_SPLIT_RE = re.compile(r"[\s,;:.()\[\]{}<>|/\\]+")
_WRAPPER_STRIP_CHARS = " \t\"'`“”‘’()[]{}<>"


def _strip_wrapping_noise(text: str) -> str:
    return normalize_text(text).strip(_WRAPPER_STRIP_CHARS)


def _normalize_line(line: str) -> str:
    out = _strip_wrapping_noise(line)
    if not out:
        return ""

    for prefix in _PREFIXES:
        if out.lower().startswith(prefix.lower()):
            out = out[len(prefix) :].strip()

    if "->" in out:
        out = out.split("->")[-1].strip()

    return normalize_text(out)


def analyze_generation_text(raw_text: str, extracted_text: str) -> dict[str, Any]:
    clean = normalize_text(raw_text)
    extracted = normalize_text(extracted_text)
    if not clean:
        return {
            "raw_normalized": "",
            "strict_word_only": False,
            "has_leading_text": False,
            "has_trailing_text": False,
            "line_count": 0,
        }

    strict_word_only = bool(extracted) and _strip_wrapping_noise(clean) == extracted
    leading = False
    trailing = False
    if extracted:
        idx = clean.find(extracted)
        if idx >= 0:
            leading = bool(clean[:idx].strip())
            trailing = bool(clean[idx + len(extracted) :].strip())
        else:
            leading = bool(clean.strip())

    return {
        "raw_normalized": clean,
        "strict_word_only": strict_word_only,
        "has_leading_text": leading,
        "has_trailing_text": trailing,
        "line_count": len([line for line in clean.splitlines() if normalize_text(line)]),
    }


def extract_transliteration_candidate(
    raw_text: str,
    *,
    script_name: str,
    min_script_ratio: float = 0.80,
) -> str:
    """Extract a strict single-word transliteration candidate.

    Returns empty string when no script-valid candidate is found.
    """
    clean = normalize_text(raw_text)
    if not clean:
        return ""

    candidate_rows: list[str] = []
    for line in clean.splitlines():
        normalized = _normalize_line(line)
        if not normalized:
            continue
        candidate_rows.append(normalized)

    if not candidate_rows:
        return ""

    token_candidates: list[str] = []
    for row in candidate_rows:
        token_candidates.append(row)
        for token in _TOKEN_SPLIT_RE.split(row):
            t = _strip_wrapping_noise(token)
            if t:
                token_candidates.append(t)

    # Prefer script-valid, longer candidates.
    scored: list[tuple[float, int, str]] = []
    for cand in token_candidates:
        ratio = float(script_ratio(cand, script_name))
        scored.append((ratio, len(cand.replace(" ", "")), cand))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_ratio, _, best = scored[0]

    if best_ratio < float(min_script_ratio):
        return ""

    # Ensure single word output.
    if " " in best:
        pieces = [p for p in _TOKEN_SPLIT_RE.split(best) if p]
        if not pieces:
            return ""
        pieces_scored = [
            (float(script_ratio(p, script_name)), len(p), p)
            for p in pieces
        ]
        pieces_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = pieces_scored[0][2]
        if pieces_scored[0][0] < float(min_script_ratio):
            return ""

    return normalize_text(best)


def resolve_generation_stop_ids(tokenizer: Any) -> int | list[int]:
    """Use EOS + newline (if a single token) as generation stop criteria."""
    stop_ids: list[int] = []

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.append(int(eos_id))

    try:
        newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    except Exception:
        newline_ids = []

    if isinstance(newline_ids, list) and len(newline_ids) == 1:
        stop_ids.append(int(newline_ids[0]))

    uniq = sorted(set(stop_ids))
    if not uniq:
        return 1
    if len(uniq) == 1:
        return uniq[0]
    return uniq


def resolve_pad_token_id(tokenizer: Any, *, fallback_stop_ids: int | list[int]) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)

    if isinstance(fallback_stop_ids, list):
        return int(fallback_stop_ids[0])
    return int(fallback_stop_ids)
