#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import apply_chat_template, load_model, set_all_seeds  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import normalize_text  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402

CONDITIONS = ["zs", "icl_helpful", "icl_corrupt"]
SCRIPT_NAMES = {
    "DEVANAGARI": "devanagari",
    "TELUGU": "telugu",
    "BENGALI": "bengali",
    "TAMIL": "tamil",
    "LATIN": "latin",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_tasks(raw: str) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for part in [x.strip() for x in str(raw or "").split(",") if x.strip()]:
        bits = [x.strip() for x in part.split(":") if x.strip()]
        if len(bits) != 3:
            raise ValueError(f"Expected model:pair:n_icl task, got {part!r}")
        out.append((str(bits[0]), str(bits[1]), int(bits[2])))
    return out


def _script_bucket(text: str) -> str:
    stripped = str(text or "").strip()
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return "none"
    seen: Counter[str] = Counter()
    for ch in letters:
        name = unicodedata.name(ch, "")
        matched = None
        for key, label in SCRIPT_NAMES.items():
            if key in name:
                matched = label
                break
        seen[matched or "other"] += 1
    return seen.most_common(1)[0][0]


def _token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return normalize_text(str(text).replace("\n", " ").strip())


def _first_step_distribution(*, model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
    return outputs.logits[0, int(input_ids.shape[1] - 1), :].float()


def _prompt_token_count(tokenizer: Any, prompt_text: str) -> int:
    rendered = apply_chat_template(tokenizer, prompt_text)
    return int(tokenizer(rendered, return_tensors="pt")["input_ids"].shape[1])


def _audit_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    device: str,
) -> Dict[str, Any]:
    rendered = apply_chat_template(tokenizer, prompt_text)
    input_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
    logits = _first_step_distribution(model=model, input_ids=input_ids)
    probs = torch.softmax(logits, dim=-1)
    target_ids = tokenizer.encode(str(target_text), add_special_tokens=False)
    if not target_ids:
        return {
            "target_id": -1,
            "target_token_text": "",
            "target_prob": float("nan"),
            "target_logit": float("nan"),
            "top1_id": int(torch.argmax(logits).item()),
            "top1_token_text": _token_text(tokenizer, int(torch.argmax(logits).item())),
            "top1_script": _script_bucket(_token_text(tokenizer, int(torch.argmax(logits).item()))),
            "top1_prob": float(torch.max(probs).item()),
            "top1_is_target": False,
            "competitor_id": int(torch.argmax(logits).item()),
            "competitor_token_text": _token_text(tokenizer, int(torch.argmax(logits).item())),
            "competitor_script": _script_bucket(_token_text(tokenizer, int(torch.argmax(logits).item()))),
            "target_minus_competitor_logit": float("nan"),
            "prompt_tokens": int(input_ids.shape[1]),
        }
    target_id = int(target_ids[0])
    top1_id = int(torch.argmax(logits).item())
    top1_text = _token_text(tokenizer, top1_id)
    logits_masked = logits.clone()
    logits_masked[target_id] = -float("inf")
    competitor_id = int(torch.argmax(logits_masked).item())
    competitor_text = _token_text(tokenizer, competitor_id)
    return {
        "target_id": int(target_id),
        "target_token_text": _token_text(tokenizer, target_id),
        "target_prob": float(probs[target_id].item()),
        "target_logit": float(logits[target_id].item()),
        "top1_id": int(top1_id),
        "top1_token_text": top1_text,
        "top1_script": _script_bucket(top1_text),
        "top1_prob": float(probs[top1_id].item()),
        "top1_is_target": bool(top1_id == target_id),
        "competitor_id": int(competitor_id),
        "competitor_token_text": competitor_text,
        "competitor_script": _script_bucket(competitor_text),
        "competitor_logit": float(logits[competitor_id].item()),
        "target_minus_competitor_logit": float(logits[target_id].item() - logits[competitor_id].item()),
        "prompt_tokens": int(input_ids.shape[1]),
    }


def _aggregate(rows: List[Dict[str, Any]], *, key_prefix: str) -> Dict[str, Any]:
    top1_script_counts = Counter(str(row.get(f"{key_prefix}_top1_script", "none")) for row in rows)
    competitor_script_counts = Counter(str(row.get(f"{key_prefix}_competitor_script", "none")) for row in rows)
    top1_failed = [row for row in rows if not bool(row.get(f"{key_prefix}_top1_is_target", False))]
    top1_failed_texts = Counter(str(row.get(f"{key_prefix}_top1_token_text", "")) for row in top1_failed)
    return {
        "n_items": int(len(rows)),
        "mean_prompt_tokens": float(np.nanmean([float(row.get(f"{key_prefix}_prompt_tokens", float("nan"))) for row in rows])),
        "mean_target_prob": float(np.nanmean([float(row.get(f"{key_prefix}_target_prob", float("nan"))) for row in rows])),
        "mean_top1_prob": float(np.nanmean([float(row.get(f"{key_prefix}_top1_prob", float("nan"))) for row in rows])),
        "top1_target_rate": float(np.nanmean([1.0 if bool(row.get(f"{key_prefix}_top1_is_target", False)) else 0.0 for row in rows])),
        "mean_target_minus_competitor_logit": float(np.nanmean([float(row.get(f"{key_prefix}_target_minus_competitor_logit", float("nan"))) for row in rows])),
        "top1_script_counts": dict(top1_script_counts),
        "top1_script_rates": {k: float(v) / float(len(rows)) for k, v in top1_script_counts.items()} if rows else {},
        "competitor_script_counts": dict(competitor_script_counts),
        "competitor_script_rates": {k: float(v) / float(len(rows)) for k, v in competitor_script_counts.items()} if rows else {},
        "failed_top1_examples": [
            {
                "word_ood": str(row.get("word_ood", "")),
                "gold": str(row.get("word_hindi", "")),
                "target_token_text": str(row.get(f"{key_prefix}_target_token_text", "")),
                "top1_token_text": str(row.get(f"{key_prefix}_top1_token_text", "")),
                "top1_script": str(row.get(f"{key_prefix}_top1_script", "")),
                "competitor_token_text": str(row.get(f"{key_prefix}_competitor_token_text", "")),
                "competitor_script": str(row.get(f"{key_prefix}_competitor_script", "")),
                "target_prob": float(row.get(f"{key_prefix}_target_prob", float("nan"))),
                "target_minus_competitor_logit": float(row.get(f"{key_prefix}_target_minus_competitor_logit", float("nan"))),
            }
            for row in top1_failed[:8]
        ],
        "most_common_failed_top1_tokens": [
            {"token_text": tok, "count": int(cnt)}
            for tok, cnt in top1_failed_texts.most_common(12)
        ],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit first-token competition on selected transliteration cells.")
    ap.add_argument("--tasks", type=str, default="1b:aksharantar_hin_latin:64,1b:aksharantar_tel_latin:64,4b:aksharantar_tel_latin:64")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/first_token_competition_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_task_summaries: List[Dict[str, Any]] = []
    model_cache: Dict[str, Tuple[Any, Any, str]] = {}

    for model_key, pair, n_icl in _parse_tasks(str(args.tasks)):
        log(f"Running first-token competition audit: model={model_key} pair={pair} n_icl={n_icl}")
        bundle = load_pair_split(
            pair,
            seed=int(args.seed),
            n_icl=int(n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        if model_key not in model_cache:
            model, tokenizer = load_model(model_key, device=str(args.device))
            device = str(next(model.parameters()).device)
            model_cache[model_key] = (model, tokenizer, device)
        model, tokenizer, device = model_cache[model_key]

        eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
        item_rows: List[Dict[str, Any]] = []
        for item_idx, word in enumerate(eval_rows, start=1):
            if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
                log(f"[{item_idx}/{len(eval_rows)}] {model_key} {pair} {n_icl} :: {word['ood']} -> {word['hindi']}")
            prompts, _meta = _condition_prompts(
                tokenizer=tokenizer,
                query=str(word["ood"]),
                icl_examples=bundle["icl_examples"],
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                seed=int(args.seed),
            )
            base_row = {
                "model": str(model_key),
                "pair": str(pair),
                "n_icl": int(n_icl),
                "seed": int(args.seed),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
            }
            for condition in CONDITIONS:
                audit = _audit_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=str(prompts[condition]),
                    target_text=str(word["hindi"]),
                    device=device,
                )
                for key, value in audit.items():
                    base_row[f"{condition}_{key}"] = value
            item_rows.append(base_row)

        summary_by_condition = {
            condition: _aggregate(item_rows, key_prefix=condition)
            for condition in CONDITIONS
        }
        payload = {
            "experiment": "first_token_competition_audit",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": str(model_key),
            "pair": str(pair),
            "n_icl": int(n_icl),
            "seed": int(args.seed),
            "max_items": int(args.max_items),
            "summary_by_condition": summary_by_condition,
            "item_rows": item_rows,
        }
        task_dir = out_root / model_key / pair / f"nicl{n_icl}"
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / "first_token_competition_audit.json", payload)
        all_task_summaries.append(
            {
                "model": str(model_key),
                "pair": str(pair),
                "n_icl": int(n_icl),
                "path": str(task_dir / "first_token_competition_audit.json"),
                "summary_by_condition": summary_by_condition,
            }
        )

    _write_json(out_root / "summary.json", {"tasks": all_task_summaries})
    log(f"Saved: {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
