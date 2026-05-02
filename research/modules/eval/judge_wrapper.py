from __future__ import annotations

import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .metrics import (
    empty_or_refusal,
    exact_match,
    infer_primary_script,
    normalize_text,
    script_ratio,
)

VALID_LABELS = {
    "exact",
    "acceptable_variant",
    "script_correct_but_wrong",
    "invalid_or_non_answer",
}

LABEL_ALIASES = {
    "exact": "exact",
    "variant": "acceptable_variant",
    "acceptable_variant": "acceptable_variant",
    "wrong": "script_correct_but_wrong",
    "script_correct_but_wrong": "script_correct_but_wrong",
    "invalid": "invalid_or_non_answer",
    "invalid_or_non_answer": "invalid_or_non_answer",
}

JUDGE_PROMPT = """You are a strict transliteration judge.

Language: {language}
Source: {source}
Reference transliteration: {reference}
Model output: {output}

Choose exactly one label:
- exact
- acceptable_variant
- script_correct_but_wrong
- invalid_or_non_answer

Rules:
- exact: output exactly matches reference after NFC normalization.
- acceptable_variant: valid transliteration variant of same word/pronunciation.
- script_correct_but_wrong: output is in the target script but wrong transliteration.
- invalid_or_non_answer: empty, refusal, wrong script, source-copy, or explanatory sentence.

Return JSON only:
{{"label": "...", "reason": "..."}}
"""


def resolve_google_api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for candidate in (Path("api.txt"), Path("research/api.txt")):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"GOOGLE_API_KEY", "GEMINI_API_KEY"} and value.strip():
                return value.strip().strip('"').strip("'")
    return ""


def deterministic_label(source: str, reference: str, output: str) -> dict[str, Any] | None:
    src = normalize_text(source)
    ref = normalize_text(reference)
    pred = normalize_text(output)

    if exact_match(pred, ref) >= 1.0:
        return {
            "label": "exact",
            "acceptable": True,
            "reason": "Deterministic: exact match.",
            "decision_source": "guardrail",
        }

    if empty_or_refusal(pred) >= 1.0:
        return {
            "label": "invalid_or_non_answer",
            "acceptable": False,
            "reason": "Deterministic: empty/refusal.",
            "decision_source": "guardrail",
        }

    target_script = infer_primary_script(ref)
    source_script = infer_primary_script(src)
    pred_script = infer_primary_script(pred)

    if pred and src and pred.lower() == src.lower() and target_script not in {"Unknown", source_script}:
        return {
            "label": "invalid_or_non_answer",
            "acceptable": False,
            "reason": "Deterministic: source-copy output.",
            "decision_source": "guardrail",
        }

    if target_script not in {"Unknown", "Latin"} and script_ratio(pred, target_script) < 0.5:
        return {
            "label": "invalid_or_non_answer",
            "acceptable": False,
            "reason": "Deterministic: wrong script.",
            "decision_source": "guardrail",
        }

    if len(pred.split()) > 2:
        return {
            "label": "invalid_or_non_answer",
            "acceptable": False,
            "reason": "Deterministic: non-standalone answer.",
            "decision_source": "guardrail",
        }

    # Ambiguous middle -> LLM judge.
    return None


def _canonical_model_name(model: str) -> str:
    model = (model or "gemini-2.5-flash").strip()
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _parse_judge_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Unparseable judge response: {text[:180]}")
        obj = json.loads(text[start : end + 1])

    raw_label = normalize_text(str(obj.get("label", ""))).lower()
    label = LABEL_ALIASES.get(raw_label, raw_label)
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid judge label {raw_label!r}")
    reason = normalize_text(str(obj.get("reason", ""))) or "No reason provided."
    return {
        "label": label,
        "acceptable": label in {"exact", "acceptable_variant"},
        "reason": reason,
        "decision_source": "gemini",
        "raw_response": text,
    }


def _gemini_generate(prompt: str, api_key: str, model: str) -> str:
    model_name = _canonical_model_name(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 160,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "label": {
                        "type": "STRING",
                        "enum": [
                            "exact",
                            "acceptable_variant",
                            "script_correct_but_wrong",
                            "invalid_or_non_answer",
                        ],
                    },
                    "reason": {"type": "STRING"},
                },
                "required": ["label", "reason"],
            },
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read())
    return str(payload["candidates"][0]["content"]["parts"][0]["text"])


def judge_transliteration(
    *,
    source: str,
    reference: str,
    output: str,
    language: str,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    guardrail = deterministic_label(source, reference, output)
    if guardrail is not None:
        return guardrail

    resolved_key = resolve_google_api_key(api_key)
    if not resolved_key:
        return {
            "label": "script_correct_but_wrong",
            "acceptable": False,
            "reason": "Judge skipped: no Google API key available.",
            "decision_source": "skipped",
        }

    prompt = JUDGE_PROMPT.format(
        language=language,
        source=normalize_text(source) or "(missing)",
        reference=normalize_text(reference) or "(missing)",
        output=normalize_text(output) or "(empty)",
    )

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _gemini_generate(prompt, resolved_key, model)
            return _parse_judge_response(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:180]}"
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                break
            time.sleep(2**attempt)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)

    return {
        "label": "script_correct_but_wrong",
        "acceptable": False,
        "reason": f"Judge failed: {last_error}",
        "decision_source": "error",
    }


def run_judge_sanity_packet(
    cases: list[dict[str, str]],
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        verdict = judge_transliteration(
            source=case["source"],
            reference=case["reference"],
            output=case["output"],
            language=case.get("language", "unknown"),
            model=model,
            api_key=api_key,
        )
        results.append(
            {
                "case_id": case["case_id"],
                "expected_label": case["expected_label"],
                "predicted_label": verdict["label"],
                "decision_source": verdict["decision_source"],
                "agree": int(verdict["label"] == case["expected_label"]),
                "reason": verdict.get("reason", ""),
            }
        )

    total = len(results)
    agreement = sum(r["agree"] for r in results)
    source_counts: dict[str, int] = {}
    for row in results:
        key = str(row["decision_source"])
        source_counts[key] = source_counts.get(key, 0) + 1

    return {
        "n_cases": total,
        "label_accuracy": float(agreement / total) if total else 0.0,
        "decision_source_counts": source_counts,
        "rows": results,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)
    os.replace(temp_path, path)
