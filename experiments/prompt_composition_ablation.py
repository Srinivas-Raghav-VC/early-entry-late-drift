#!/usr/bin/env python3
"""Reviewer-targeted prompt-composition ablation for transliteration ICL."""
from __future__ import annotations

import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import build_corrupted_icl_prompt, build_task_prompt, load_model, set_all_seeds  # noqa: E402
from experiments.regime_eval_common import (  # noqa: E402
    build_prompt_input_ids,
    choose_bank_competitor,
    default_stop_and_pad,
    first_step_stats,
    generate_raw_text,
    json_safe,
    transliteration_metrics,
    write_json,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402

CONDITIONS = [
    "zs",
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_reversed",
    "icl_helpful_drop_nearest_replace_far",
    "icl_helpful_drop_top2_replace_far",
    "icl_corrupt",
]


def _parse_tasks(raw: str) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for part in [x.strip() for x in str(raw or "").split(",") if x.strip()]:
        bits = [x.strip() for x in part.split(":") if x.strip()]
        if len(bits) != 3:
            raise ValueError(f"Expected model:pair:n_icl task, got {part!r}")
        out.append((str(bits[0]), str(bits[1]), int(bits[2])))
    if not out:
        raise ValueError("No tasks provided.")
    return out


def _similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(a=str(a), b=str(b)).ratio())


def _sort_helpful_examples(
    query: str,
    icl_examples: Sequence[Dict[str, Any]],
    *,
    descending: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored = []
    for idx, ex in enumerate(icl_examples):
        s = _similarity(query, str(ex.get("ood", "")))
        scored.append({
            "original_index": int(idx),
            "similarity": float(s),
            "example": dict(ex),
        })
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


def _select_reserve_replacements(
    query: str,
    icl_examples: Sequence[Dict[str, Any]],
    reserve_rows: Sequence[Dict[str, Any]],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    existing_sources = {str(ex.get("ood", "")) for ex in icl_examples}
    existing_targets = {str(ex.get("hindi", "")) for ex in icl_examples}
    scored = []
    for row in reserve_rows:
        source = str(row.get("ood", ""))
        target = str(row.get("hindi", ""))
        if not source or source == str(query):
            continue
        if source in existing_sources or target in existing_targets:
            continue
        scored.append((float(_similarity(query, source)), dict(row)))
    scored.sort(key=lambda x: (x[0], x[1].get("ood", "")))
    return [dict(row) for _score, row in scored[: max(0, int(k))]]


def _drop_and_replace_examples(
    query: str,
    icl_examples: Sequence[Dict[str, Any]],
    reserve_rows: Sequence[Dict[str, Any]],
    *,
    drop_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    helpful = [dict(ex) for ex in icl_examples]
    scored = []
    for idx, ex in enumerate(helpful):
        scored.append((float(_similarity(query, str(ex.get("ood", "")))), int(idx), dict(ex)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    dropped = scored[: max(0, int(drop_k))]
    replacements = _select_reserve_replacements(query, helpful, reserve_rows, k=len(dropped))
    if len(replacements) < len(dropped):
        raise RuntimeError(
            f"Not enough reserve replacements for query={query!r}: need {len(dropped)}, got {len(replacements)}"
        )

    replaced_rows = [dict(ex) for ex in helpful]
    replacement_meta = []
    for (sim, original_index, removed_ex), replacement in zip(dropped, replacements):
        replaced_rows[int(original_index)] = dict(replacement)
        replacement_meta.append(
            {
                "slot_index": int(original_index),
                "removed_source": str(removed_ex.get("ood", "")),
                "removed_target": str(removed_ex.get("hindi", "")),
                "removed_similarity": float(sim),
                "replacement_source": str(replacement.get("ood", "")),
                "replacement_target": str(replacement.get("hindi", "")),
                "replacement_similarity": float(_similarity(query, str(replacement.get("ood", "")))),
            }
        )
    meta = {
        "drop_k": int(drop_k),
        "removed_targets": [str(row["removed_target"]) for row in replacement_meta],
        "replacement_rows": replacement_meta,
    }
    return replaced_rows, meta


def _condition_prompts(
    *,
    query: str,
    icl_examples: Sequence[Dict[str, Any]],
    reserve_rows: Sequence[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, List[str]], Dict[str, List[str]]]:
    helpful_desc, helpful_desc_meta = _sort_helpful_examples(query, icl_examples, descending=True)
    helpful_asc, helpful_asc_meta = _sort_helpful_examples(query, icl_examples, descending=False)
    helpful_reversed = list(reversed(list(icl_examples)))
    helpful_drop1, helpful_drop1_meta = _drop_and_replace_examples(query, icl_examples, reserve_rows, drop_k=1)
    helpful_drop2, helpful_drop2_meta = _drop_and_replace_examples(query, icl_examples, reserve_rows, drop_k=2)

    condition_examples = {
        "icl_helpful": [dict(ex) for ex in icl_examples],
        "icl_helpful_similarity_desc": helpful_desc,
        "icl_helpful_similarity_asc": helpful_asc,
        "icl_helpful_reversed": [dict(ex) for ex in helpful_reversed],
        "icl_helpful_drop_nearest_replace_far": helpful_drop1,
        "icl_helpful_drop_top2_replace_far": helpful_drop2,
    }
    prompts = {
        "zs": build_task_prompt(
            query,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        ),
        "icl_corrupt": build_corrupted_icl_prompt(
            query,
            list(icl_examples),
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            seed=int(seed),
        ),
    }
    for condition, examples in condition_examples.items():
        prompts[condition] = build_task_prompt(
            query,
            examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        )

    metadata = {
        "helpful_similarity_desc": helpful_desc_meta,
        "helpful_similarity_asc": helpful_asc_meta,
        "helpful_reversed_original_indices": [int(len(icl_examples) - 1 - i) for i in range(len(icl_examples))],
        "helpful_drop_nearest_replace_far": helpful_drop1_meta,
        "helpful_drop_top2_replace_far": helpful_drop2_meta,
    }
    bank_targets_by_condition = {
        "zs": [],
        "icl_corrupt": [str(ex.get("hindi", "")) for ex in icl_examples],
    }
    removed_targets_by_condition = {
        "zs": [],
        "icl_corrupt": [],
    }
    for condition, examples in condition_examples.items():
        bank_targets_by_condition[condition] = [str(ex.get("hindi", "")) for ex in examples]
        removed_targets_by_condition[condition] = list(metadata.get(condition, {}).get("removed_targets", []))
    return prompts, metadata, bank_targets_by_condition, removed_targets_by_condition


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prompt-composition ablation for transliteration controls.")
    ap.add_argument("--tasks", type=str, default="1b:aksharantar_tel_latin:64,4b:aksharantar_tel_latin:64")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--fuzzy-bank-threshold", type=float, default=0.85)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/prompt_composition_ablation_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _parse_tasks(str(args.tasks))
    set_all_seeds(int(args.seed))
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_cache: Dict[str, Tuple[Any, Any, str, Dict[str, Any]]] = {}
    aggregate: List[Dict[str, Any]] = []

    for task_idx, (model_key, pair, n_icl) in enumerate(tasks):
        if model_key not in model_cache:
            model, tokenizer = load_model(str(model_key), device=str(args.device))
            device = str(next(model.parameters()).device)
            model_cache[model_key] = (model, tokenizer, device, default_stop_and_pad(tokenizer))
        model, tokenizer, device, stop_and_pad = model_cache[model_key]

        bundle = load_pair_split(
            str(pair),
            seed=int(args.seed),
            n_icl=int(n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
        reserve_rows = list(bundle["select_rows"])
        log(f"[prompt-composition] model={model_key} pair={pair} n_icl={n_icl} items={len(eval_rows)}")

        item_rows: List[Dict[str, Any]] = []
        ordering_metadata_by_item: List[Dict[str, Any]] = []
        for item_idx, word in enumerate(eval_rows, start=1):
            if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
                log(f"  [{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
            query = str(word["ood"])
            target_text = str(word["hindi"])
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            prompts, prompt_meta, bank_targets_by_condition, removed_targets_by_condition = _condition_prompts(
                query=query,
                icl_examples=bundle["icl_examples"],
                reserve_rows=reserve_rows,
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                seed=int(args.seed) + int(item_idx),
            )
            ordering_metadata_by_item.append({
                "item_index": int(item_idx - 1),
                "word_ood": query,
                **prompt_meta,
            })
            for condition in CONDITIONS:
                prompt_text = str(prompts[condition])
                input_ids = build_prompt_input_ids(tokenizer=tokenizer, prompt_text=prompt_text, device=device)
                first = first_step_stats(model=model, input_ids=input_ids, target_id=target_id)
                raw_text = generate_raw_text(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    stop_ids=stop_and_pad["stop_ids"],
                    pad_id=stop_and_pad["pad_id"],
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
                competitor = choose_bank_competitor(
                    {"helpful_similarity_desc": prompt_meta.get("helpful_similarity_desc", [])},
                    target_text,
                )
                generation = transliteration_metrics(
                    raw_text=raw_text,
                    gold_text=target_text,
                    output_script_name=bundle["output_script_name"],
                    bank_targets=bank_targets_by_condition.get(condition, []),
                    nearest_bank_target=str(competitor.get("target", "")),
                    fuzzy_bank_threshold=float(args.fuzzy_bank_threshold),
                )
                pred = str(generation.get("prediction", ""))
                removed_targets = [str(x) for x in removed_targets_by_condition.get(condition, []) if str(x)]
                generation["exact_removed_target_copy"] = float(bool(pred) and pred in set(removed_targets) and pred != target_text)
                item_rows.append(
                    {
                        "pair": str(pair),
                        "model": str(model_key),
                        "seed": int(args.seed),
                        "n_icl": int(n_icl),
                        "item_index": int(item_idx - 1),
                        "word_ood": query,
                        "word_hindi": target_text,
                        "condition": str(condition),
                        "prompt_tokens": int(input_ids.shape[1]),
                        "removed_targets": removed_targets,
                        **first,
                        **generation,
                    }
                )

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
                "mean_target_prob": _m("target_prob"),
                "first_token_top1_target_rate": _m("top1_is_target"),
                "mean_target_minus_competitor_logit": _m("target_minus_competitor_logit"),
                "mean_exact_match": _m("exact_match"),
                "mean_akshara_cer": _m("akshara_cer"),
                "mean_script_compliance": _m("script_compliance"),
                "mean_first_entry_correct": _m("first_entry_correct"),
                "exact_bank_copy_rate": _m("exact_bank_copy"),
                "exact_nearest_bank_copy_rate": _m("exact_nearest_bank_copy"),
                "fuzzy_bank_copy_rate": _m("fuzzy_bank_copy"),
                "mean_max_bank_similarity": _m("max_bank_similarity"),
                "exact_removed_target_copy_rate": _m("exact_removed_target_copy"),
            }

        def _cond_mean(cond: str, key: str) -> float:
            return float((summary_by_condition.get(cond) or {}).get(key, float("nan")))

        comparison_rows = []
        comparisons = [
            ("drop_nearest_vs_helpful", "icl_helpful_drop_nearest_replace_far", "icl_helpful"),
            ("drop_top2_vs_helpful", "icl_helpful_drop_top2_replace_far", "icl_helpful"),
            ("desc_vs_asc", "icl_helpful_similarity_desc", "icl_helpful_similarity_asc"),
            ("helpful_vs_reversed", "icl_helpful", "icl_helpful_reversed"),
        ]
        for label, a, b in comparisons:
            comparison_rows.append(
                {
                    "comparison": str(label),
                    "cond_a": str(a),
                    "cond_b": str(b),
                    "delta_exact_match": float(_cond_mean(a, "mean_exact_match") - _cond_mean(b, "mean_exact_match")),
                    "delta_first_token_top1_target_rate": float(_cond_mean(a, "first_token_top1_target_rate") - _cond_mean(b, "first_token_top1_target_rate")),
                    "delta_akshara_cer": float(_cond_mean(a, "mean_akshara_cer") - _cond_mean(b, "mean_akshara_cer")),
                    "delta_exact_nearest_bank_copy_rate": float(_cond_mean(a, "exact_nearest_bank_copy_rate") - _cond_mean(b, "exact_nearest_bank_copy_rate")),
                    "delta_fuzzy_bank_copy_rate": float(_cond_mean(a, "fuzzy_bank_copy_rate") - _cond_mean(b, "fuzzy_bank_copy_rate")),
                    "delta_removed_target_copy_rate": float(_cond_mean(a, "exact_removed_target_copy_rate") - _cond_mean(b, "exact_removed_target_copy_rate")),
                }
            )

        payload = {
            "experiment": "prompt_composition_ablation",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pair": str(pair),
            "model": str(model_key),
            "seed": int(args.seed),
            "n_icl": int(n_icl),
            "conditions": list(CONDITIONS),
            "prompt_ordering_metadata_by_item": ordering_metadata_by_item,
            "summary_by_condition": summary_by_condition,
            "comparison_rows": comparison_rows,
            "item_rows": item_rows,
        }
        out_dir = out_root / str(model_key) / str(pair) / f"seed{int(args.seed)}" / f"nicl{int(n_icl)}"
        write_json(out_dir / "prompt_composition_ablation.json", payload)
        aggregate.append(
            {
                "model": str(model_key),
                "pair": str(pair),
                "n_icl": int(n_icl),
                "summary_by_condition": summary_by_condition,
                "comparison_rows": comparison_rows,
            }
        )

    print(json.dumps(json_safe(aggregate), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
