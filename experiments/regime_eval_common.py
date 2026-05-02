#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from core import apply_chat_template
from paper2_fidelity_calibrated.eval_utils import (
    akshara_cer,
    exact_match,
    first_entry_correct,
    normalize_text,
    script_compliance,
)
from research.modules.eval.output_extraction import (
    analyze_generation_text,
    extract_transliteration_candidate,
    resolve_generation_stop_ids,
    resolve_pad_token_id,
)

SCRIPT_NAMES = {
    "DEVANAGARI": "devanagari",
    "TELUGU": "telugu",
    "BENGALI": "bengali",
    "TAMIL": "tamil",
    "LATIN": "latin",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def script_bucket(text: str) -> str:
    stripped = str(text or "").strip()
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return "none"
    counts: Dict[str, int] = {}
    for ch in letters:
        name = unicodedata.name(ch, "")
        matched = "other"
        for key, label in SCRIPT_NAMES.items():
            if key in name:
                matched = label
                break
        counts[matched] = counts.get(matched, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def choose_bank_competitor(meta: Mapping[str, Any], gold: str) -> Dict[str, Any]:
    gold_norm = normalize_text(str(gold))
    ordered = list(meta.get("helpful_similarity_desc") or [])
    for row in ordered:
        target = normalize_text(str(row.get("target", "")))
        if target and target != gold_norm:
            return {
                "target": target,
                "source": str(row.get("source", "")),
                "position": int(row.get("position", -1)),
                "similarity": float(row.get("similarity", float("nan"))),
            }
    if not ordered:
        raise ValueError("No helpful_similarity_desc metadata found.")
    row = ordered[0]
    return {
        "target": normalize_text(str(row.get("target", ""))),
        "source": str(row.get("source", "")),
        "position": int(row.get("position", -1)),
        "similarity": float(row.get("similarity", float("nan"))),
    }


def generate_raw_text(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_ids: int | list[int],
    pad_id: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    sample_seed: Optional[int] = None,
    hooks: Optional[Sequence[Any]] = None,
) -> str:
    active = list(hooks or [])
    try:
        if sample_seed is not None:
            torch.manual_seed(int(sample_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(sample_seed))
        attention_mask = torch.ones_like(input_ids)
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            use_cache=False,
            eos_token_id=stop_ids,
            pad_token_id=int(pad_id),
        )
        if bool(do_sample):
            generate_kwargs.update(
                temperature=float(max(1e-5, temperature)),
                top_p=float(top_p),
            )
        with torch.inference_mode():
            out = model.generate(**generate_kwargs)
        new_tokens = out[0, int(input_ids.shape[1]) :]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def extract_script_word(raw_text: str, *, script_name: str, min_script_ratio: float = 0.80) -> str:
    return extract_transliteration_candidate(
        raw_text,
        script_name=str(script_name),
        min_script_ratio=float(min_script_ratio),
    )


def extract_english_word(raw_text: str) -> str:
    match = _WORD_RE.search(normalize_text(raw_text))
    return match.group(0).lower() if match else ""


def first_step_stats(*, model: Any, input_ids: torch.Tensor, target_id: int, hooks: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, int(input_ids.shape[1] - 1), :].float()
        probs = torch.softmax(logits, dim=-1)
        top1_id = int(torch.argmax(logits).item())
        competitor_logits = logits.clone()
        competitor_logits[int(target_id)] = -float("inf")
        competitor_id = int(torch.argmax(competitor_logits).item())
        return {
            "target_id": int(target_id),
            "target_prob": float(probs[int(target_id)].item()),
            "target_logit": float(logits[int(target_id)].item()),
            "top1_id": int(top1_id),
            "top1_prob": float(probs[int(top1_id)].item()),
            "top1_is_target": bool(top1_id == int(target_id)),
            "competitor_id": int(competitor_id),
            "competitor_logit": float(logits[int(competitor_id)].item()),
            "target_minus_competitor_logit": float(logits[int(target_id)].item() - logits[int(competitor_id)].item()),
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def max_similarity_to_bank(prediction: str, bank_targets: Sequence[str], *, exclude: str = "") -> float:
    pred = normalize_text(prediction)
    if not pred:
        return float("nan")
    vals = [
        float(SequenceMatcher(a=pred, b=normalize_text(t)).ratio())
        for t in bank_targets
        if normalize_text(t) and normalize_text(t) != normalize_text(exclude)
    ]
    return max(vals) if vals else float("nan")


def transliteration_metrics(
    *,
    raw_text: str,
    gold_text: str,
    output_script_name: str,
    bank_targets: Sequence[str],
    nearest_bank_target: str,
    fuzzy_bank_threshold: float,
) -> Dict[str, Any]:
    extracted = extract_script_word(raw_text, script_name=str(output_script_name), min_script_ratio=0.80)
    audit = analyze_generation_text(raw_text, extracted)
    pred = normalize_text(extracted)
    gold = normalize_text(gold_text)
    bank_targets_norm = [normalize_text(t) for t in bank_targets]
    nearest = normalize_text(nearest_bank_target)
    max_bank_similarity = max_similarity_to_bank(pred, bank_targets_norm, exclude=gold)
    return {
        "raw_text": normalize_text(raw_text),
        "prediction": pred,
        "strict_word_only": bool(audit.get("strict_word_only", False)),
        "has_leading_text": bool(audit.get("has_leading_text", False)),
        "has_trailing_text": bool(audit.get("has_trailing_text", False)),
        "exact_match": float(exact_match(pred, gold)),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, str(output_script_name))),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
        "exact_bank_copy": float(pred in {t for t in bank_targets_norm if t and t != gold}),
        "exact_nearest_bank_copy": float(pred == nearest and pred != gold),
        "fuzzy_bank_copy": float(
            np.isfinite(float(max_bank_similarity)) and float(max_bank_similarity) >= float(fuzzy_bank_threshold) and pred != gold
        ),
        "max_bank_similarity": float(max_bank_similarity),
    }


def default_stop_and_pad(tokenizer: Any) -> Dict[str, Any]:
    stop_ids = resolve_generation_stop_ids(tokenizer)
    pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=stop_ids)
    return {"stop_ids": stop_ids, "pad_id": pad_id}


def build_prompt_input_ids(*, tokenizer: Any, prompt_text: str, device: str) -> torch.Tensor:
    rendered = apply_chat_template(tokenizer, prompt_text)
    return tokenizer(rendered, return_tensors="pt").to(device).input_ids
