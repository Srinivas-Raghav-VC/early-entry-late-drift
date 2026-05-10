from __future__ import annotations

import unicodedata
from typing import Any, Dict, Sequence

import numpy as np
import torch

from core import _classify_script, _teacher_forced_metrics_from_input_ids, apply_chat_template


INDIC_VIRAMAS = {
    "\u094d",  # Devanagari
    "\u09cd",  # Bengali
    "\u0bcd",  # Tamil
    "\u0c4d",  # Telugu
    "\u0ccd",  # Kannada
    "\u0d4d",  # Malayalam
}
JOINERS = {"\u200c", "\u200d"}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "").strip())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def build_bare_zs_prompt(query_token: str) -> str:
    return f"{str(query_token).strip()} ->"


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


def akshara_cer(pred: str, gold: str) -> float:
    pred_aks = segment_aksharas(pred)
    gold_aks = segment_aksharas(gold)
    if not gold_aks:
        return 0.0 if not pred_aks else 1.0
    return float(_levenshtein(pred_aks, gold_aks) / max(1, len(gold_aks)))


def first_entry_correct(pred: str, gold: str) -> float:
    pred_aks = segment_aksharas(pred)
    gold_aks = segment_aksharas(gold)
    if not pred_aks or not gold_aks:
        return 0.0
    return float(pred_aks[0] == gold_aks[0])


def continuation_akshara_cer(pred: str, gold: str) -> float:
    pred_aks = segment_aksharas(pred)
    gold_aks = segment_aksharas(gold)
    if not pred_aks or not gold_aks or pred_aks[0] != gold_aks[0]:
        return float("nan")
    pred_tail = pred_aks[1:]
    gold_tail = gold_aks[1:]
    if not gold_tail:
        return 0.0 if not pred_tail else 1.0
    return float(_levenshtein(pred_tail, gold_tail) / max(1, len(gold_tail)))


def script_compliance(pred: str, target_script: str) -> float:
    pred_aks = segment_aksharas(pred)
    if not pred_aks:
        return 0.0
    want = str(target_script or "").strip() or "Unknown"
    hits = 0
    total = 0
    for ak in pred_aks:
        script = _classify_script(ak)
        if script == "Unknown":
            continue
        total += 1
        if script == want:
            hits += 1
    if total <= 0:
        return 0.0
    return float(hits / total)


def _decode_single_token_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    return text.split()[0]


def greedy_decode_single_token(
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 16,
) -> str:
    rendered = apply_chat_template(tokenizer, prompt_text)
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=int(pad_id),
        )
    new_tokens = out[0, input_ids.shape[1] :]
    return _decode_single_token_text(tokenizer.decode(new_tokens, skip_special_tokens=True))


def teacher_forced_prompt_metrics(
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    target_text: str,
    device: str,
) -> Dict[str, float]:
    rendered = apply_chat_template(tokenizer, prompt_text)
    input_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
    target_ids = tokenizer.encode(str(target_text), add_special_tokens=False)
    target_id = int(target_ids[0]) if target_ids else -1
    return _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=input_ids,
        target_ids=target_ids,
        target_id=target_id,
        device=device,
        competitor_id=-1,
    )


def evaluate_prompt_condition(
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    target_text: str,
    target_script: str,
    device: str,
    max_new_tokens: int = 16,
) -> Dict[str, float | str]:
    tf = teacher_forced_prompt_metrics(
        model,
        tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        device=device,
    )
    pred = greedy_decode_single_token(
        model,
        tokenizer,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    gold = normalize_text(target_text)
    cont = continuation_akshara_cer(pred, gold)
    return {
        "prediction": pred,
        "exact_match": float(exact_match(pred, gold)),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, target_script)),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
        "continuation_akshara_cer": float(cont),
        "joint_logprob": float(tf.get("joint_logprob", float("nan"))),
        "target_pos1_nll": float(tf.get("target_pos1_nll", float("nan"))),
        "target_pos2_nll": float(tf.get("target_pos2_nll", float("nan"))),
        "target_pos3_nll": float(tf.get("target_pos3_nll", float("nan"))),
        "first_prob": float(tf.get("first_prob", float("nan"))),
        "first_logit": float(tf.get("first_logit", float("nan"))),
    }


def mean_metric(rows: Sequence[Dict[str, float | str]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))
