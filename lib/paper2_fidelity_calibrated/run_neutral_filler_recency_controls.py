#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    _teacher_forced_metrics_from_input_ids,
    apply_chat_template,
    build_corrupted_icl_prompt,
    build_null_icl_prompt,
    build_random_icl_prompt,
    build_task_prompt,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    continuation_akshara_cer,
    first_entry_correct,
    normalize_text,
    script_compliance,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402


CONDITIONS = [
    "zs",
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_reversed",
    "icl_corrupt",
    "icl_random_indic",
    "icl_null_filler",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Neutral-filler and recency-balanced behavioral controls for CFOM rescue.")
    ap.add_argument("--model", type=str, default="4b", choices=["1b", "4b"])
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(a=str(a), b=str(b)).ratio())


def _sort_helpful_examples(query: str, icl_examples: List[Dict[str, str]], *, descending: bool) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    scored = []
    for idx, ex in enumerate(icl_examples):
        s = _similarity(query, str(ex.get("ood", "")))
        scored.append({"original_index": int(idx), "similarity": float(s), "example": dict(ex)})
    scored_sorted = sorted(scored, key=lambda row: (-row["similarity"], row["original_index"]))
    if not descending:
        scored_sorted = list(reversed(scored_sorted))
    ordered_examples = [dict(row["example"]) for row in scored_sorted]
    metadata = [
        {
            "position": int(pos),
            "original_index": int(row["original_index"]),
            "similarity": float(row["similarity"]),
            "source": str(row["example"].get("ood", "")),
            "target": str(row["example"].get("hindi", "")),
        }
        for pos, row in enumerate(scored_sorted)
    ]
    return ordered_examples, metadata


def _generate_text(*, model: Any, tokenizer: Any, input_ids: torch.Tensor, max_new_tokens: int) -> str:
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=int(pad_id),
        )
    new_tokens = out[0, input_ids.shape[1] :]
    return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def _generation_metrics(gold_text: str, pred_text: str, target_script: str) -> Dict[str, float]:
    gold = normalize_text(gold_text)
    pred = normalize_text(pred_text)
    cont = continuation_akshara_cer(pred, gold)
    return {
        "exact_match": float(pred == gold),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, target_script)),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
        "continuation_fidelity": float(cont) if np.isfinite(cont) else float("nan"),
    }


def _condition_prompts(
    *,
    tokenizer: Any,
    query: str,
    icl_examples: List[Dict[str, str]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    helpful_raw = build_task_prompt(
        query,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant="canonical",
    )
    zs_raw = build_task_prompt(
        query,
        None,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant="canonical",
    )
    helpful_rendered = apply_chat_template(tokenizer, helpful_raw)
    zs_rendered = apply_chat_template(tokenizer, zs_raw)
    helpful_budget = int(
        max(
            32,
            int(tokenizer(helpful_rendered, return_tensors="pt")["input_ids"].shape[1])
            - int(tokenizer(zs_rendered, return_tensors="pt")["input_ids"].shape[1]),
        )
    )

    helpful_desc, helpful_desc_meta = _sort_helpful_examples(query, icl_examples, descending=True)
    helpful_asc, helpful_asc_meta = _sort_helpful_examples(query, icl_examples, descending=False)
    helpful_reversed = list(reversed(list(icl_examples)))

    prompts = {
        "zs": zs_raw,
        "icl_helpful": helpful_raw,
        "icl_helpful_similarity_desc": build_task_prompt(
            query,
            helpful_desc,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        ),
        "icl_helpful_similarity_asc": build_task_prompt(
            query,
            helpful_asc,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        ),
        "icl_helpful_reversed": build_task_prompt(
            query,
            helpful_reversed,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        ),
        "icl_corrupt": build_corrupted_icl_prompt(
            query,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            seed=int(seed),
        ),
        "icl_random_indic": build_random_icl_prompt(
            query,
            len(icl_examples),
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            use_indic_control=True,
            length_reference_examples=icl_examples,
            seed=int(seed),
            forbidden_src_texts=[str(query)] + [str(ex.get("ood", "")) for ex in icl_examples],
            forbidden_tgt_texts=[str(ex.get("hindi", "")) for ex in icl_examples],
        ),
        "icl_null_filler": build_null_icl_prompt(
            query,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            seed=int(seed),
            target_token_budget=int(helpful_budget),
        ),
    }
    metadata = {
        "helpful_similarity_desc": helpful_desc_meta,
        "helpful_similarity_asc": helpful_asc_meta,
        "helpful_reversed_original_indices": [
            int(len(icl_examples) - 1 - i) for i in range(len(icl_examples))
        ],
        "null_filler_target_token_budget": int(helpful_budget),
    }
    return prompts, metadata


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    pair_bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    out_root = (
        Path(str(args.out)).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "paper2_fidelity_calibrated" / "results" / "neutral_filler_recency_controls" / str(args.pair) / str(args.model)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    eval_rows = list(pair_bundle["eval_rows"][: max(1, int(args.max_items))])
    if not eval_rows:
        raise RuntimeError("No evaluation rows available for neutral-filler / recency controls")

    item_rows: List[Dict[str, Any]] = []
    ordering_metadata_by_item: List[Dict[str, Any]] = []
    log(
        f"Running neutral-filler / recency controls: pair={args.pair} model={args.model} items={len(eval_rows)} conditions={CONDITIONS}"
    )

    for item_idx, word in enumerate(eval_rows, start=1):
        query = str(word["ood"])
        target_text = str(word["hindi"])
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = int(target_ids[0])
        prompts, prompt_meta = _condition_prompts(
            tokenizer=tokenizer,
            query=query,
            icl_examples=pair_bundle["icl_examples"],
            input_script_name=pair_bundle["input_script_name"],
            source_language=pair_bundle["source_language"],
            output_script_name=pair_bundle["output_script_name"],
            seed=int(args.seed) + int(item_idx),
        )
        ordering_metadata_by_item.append(
            {
                "item_index": int(item_idx - 1),
                "word_ood": query,
                **prompt_meta,
            }
        )
        for condition in CONDITIONS:
            raw_prompt = str(prompts[condition])
            rendered_prompt = apply_chat_template(tokenizer, raw_prompt)
            input_ids = tokenizer(rendered_prompt, return_tensors="pt").to(device).input_ids
            tf = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=input_ids,
                target_ids=list(target_ids),
                target_id=int(target_id),
                device=device,
                competitor_id=-1,
            )
            pred = _generate_text(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
            )
            gen = _generation_metrics(target_text, pred, pair_bundle["output_script_name"])
            item_rows.append(
                {
                    "pair": str(args.pair),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "item_index": int(item_idx - 1),
                    "word_ood": query,
                    "word_hindi": target_text,
                    "condition": str(condition),
                    "prompt_tokens": int(input_ids.shape[1]),
                    "first_prob": float(tf.get("first_prob", float("nan"))),
                    "first_logit": float(tf.get("first_logit", float("nan"))),
                    "joint_logprob": float(tf.get("joint_logprob", float("nan"))),
                    "target_pos1_nll": float(tf.get("target_pos1_nll", float("nan"))),
                    "target_rank": float(tf.get("target_rank", float("nan"))),
                    "prediction": str(pred),
                    **gen,
                }
            )
        if item_idx == 1 or item_idx % 10 == 0:
            log(f"Processed {item_idx}/{len(eval_rows)} items")

    summary_by_condition: Dict[str, Dict[str, float]] = {}
    for condition in CONDITIONS:
        rows_c = [row for row in item_rows if str(row.get("condition")) == condition]
        if not rows_c:
            continue
        def _m(key: str) -> float:
            vals = [float(row.get(key, float("nan"))) for row in rows_c]
            finite = [v for v in vals if np.isfinite(v)]
            if not finite:
                return float("nan")
            return float(np.mean(finite))
        summary_by_condition[str(condition)] = {
            "n_items": float(len(rows_c)),
            "mean_prompt_tokens": _m("prompt_tokens"),
            "mean_first_prob": _m("first_prob"),
            "mean_first_logit": _m("first_logit"),
            "mean_joint_logprob": _m("joint_logprob"),
            "mean_target_pos1_nll": _m("target_pos1_nll"),
            "mean_target_rank": _m("target_rank"),
            "mean_exact_match": _m("exact_match"),
            "mean_akshara_cer": _m("akshara_cer"),
            "mean_script_compliance": _m("script_compliance"),
            "mean_first_entry_correct": _m("first_entry_correct"),
            "mean_continuation_fidelity": _m("continuation_fidelity"),
        }

    def _cond_mean(cond: str, key: str) -> float:
        return float((summary_by_condition.get(cond) or {}).get(key, float("nan")))

    derived = {
        "helpful_minus_zs_first_prob": float(_cond_mean("icl_helpful", "mean_first_prob") - _cond_mean("zs", "mean_first_prob")),
        "helpful_minus_corrupt_first_prob": float(_cond_mean("icl_helpful", "mean_first_prob") - _cond_mean("icl_corrupt", "mean_first_prob")),
        "helpful_minus_random_first_prob": float(_cond_mean("icl_helpful", "mean_first_prob") - _cond_mean("icl_random_indic", "mean_first_prob")),
        "helpful_minus_null_first_prob": float(_cond_mean("icl_helpful", "mean_first_prob") - _cond_mean("icl_null_filler", "mean_first_prob")),
        "similarity_desc_minus_asc_first_prob": float(_cond_mean("icl_helpful_similarity_desc", "mean_first_prob") - _cond_mean("icl_helpful_similarity_asc", "mean_first_prob")),
        "original_minus_reversed_first_prob": float(_cond_mean("icl_helpful", "mean_first_prob") - _cond_mean("icl_helpful_reversed", "mean_first_prob")),
        "similarity_desc_minus_asc_exact_match": float(_cond_mean("icl_helpful_similarity_desc", "mean_exact_match") - _cond_mean("icl_helpful_similarity_asc", "mean_exact_match")),
        "original_minus_reversed_exact_match": float(_cond_mean("icl_helpful", "mean_exact_match") - _cond_mean("icl_helpful_reversed", "mean_exact_match")),
    }

    payload = {
        "experiment": "neutral_filler_recency_controls",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pair": str(args.pair),
        "model": str(args.model),
        "seed": int(args.seed),
        "conditions": list(CONDITIONS),
        "prompt_ordering_metadata_by_item": ordering_metadata_by_item,
        "summary_by_condition": summary_by_condition,
        "derived": derived,
        "item_rows": item_rows,
    }
    _write_json(out_root / "neutral_filler_recency_controls.json", payload)
    log(f"Saved: {out_root / 'neutral_filler_recency_controls.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
