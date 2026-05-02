#!/usr/bin/env python3
"""Prompt-format sensitivity check for the core transliteration regime panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import build_task_prompt, load_model, set_all_seeds  # noqa: E402
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
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402

PROMPT_FORMAT_VARIANTS = ("canonical", "compact", "tagged")
CONDITIONS = ["zs", "icl_helpful", "icl_corrupt"]


def _parse_csv(text: str) -> List[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _parse_int_csv(text: str) -> List[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    return values


def _deranged_examples(
    *,
    icl_examples: Sequence[Dict[str, Any]],
    seed: int,
    input_script_name: str,
    output_script_name: str,
) -> List[Dict[str, Any]]:
    examples = [dict(ex) for ex in icl_examples]
    if len(examples) <= 1:
        return examples

    outputs = [
        str(ex.get("hindi", ex.get("source", ex.get("output", ""))) or "")
        for ex in examples
    ]
    msg = f"prompt_sensitivity_derange::{seed}::{len(outputs)}::{input_script_name}::{output_script_name}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    order = list(range(len(outputs)))
    for _ in range(100):
        perm = rng.permutation(len(outputs)).tolist()
        if all(int(perm[i]) != i for i in order):
            for ex, idx in zip(examples, perm):
                ex["hindi"] = outputs[int(idx)]
            return examples

    rotated = outputs[1:] + outputs[:1]
    for ex, wrong in zip(examples, rotated):
        ex["hindi"] = wrong
    return examples


def _variant_prompts(
    *,
    query: str,
    icl_examples: Sequence[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    prompt_variant: str,
    seed: int,
) -> Dict[str, str]:
    helpful = build_task_prompt(
        query,
        list(icl_examples),
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    zs = build_task_prompt(
        query,
        None,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    corrupt = build_task_prompt(
        query,
        _deranged_examples(
            icl_examples=list(icl_examples),
            seed=int(seed),
            input_script_name=input_script_name,
            output_script_name=output_script_name,
        ),
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    return {
        "zs": zs,
        "icl_helpful": helpful,
        "icl_corrupt": corrupt,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prompt-format sensitivity check for transliteration regimes.")
    ap.add_argument("--models", type=str, default="1b,4b")
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--n-icls", type=str, default="8,64")
    ap.add_argument("--prompt-variants", type=str, default="canonical,compact,tagged")
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
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/prompt_sensitivity_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    models = _parse_csv(args.models)
    pairs = _parse_csv(args.pairs)
    n_icls = _parse_int_csv(args.n_icls)
    prompt_variants = [variant for variant in _parse_csv(args.prompt_variants) if variant in PROMPT_FORMAT_VARIANTS]
    if not prompt_variants:
        raise ValueError(f"No valid prompt variants requested. Valid variants: {PROMPT_FORMAT_VARIANTS}")

    set_all_seeds(int(args.seed))
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    aggregate_summary: Dict[str, Any] = {}
    model_cache: Dict[str, Any] = {}

    for model_key in models:
        if model_key not in model_cache:
            model, tokenizer = load_model(str(model_key), device=str(args.device))
            stop_and_pad = default_stop_and_pad(tokenizer)
            model_cache[model_key] = (model, tokenizer, str(next(model.parameters()).device), stop_and_pad)
        model, tokenizer, device, stop_and_pad = model_cache[model_key]

        for pair in pairs:
            for n_icl in n_icls:
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
                bank_targets = [str(ex.get("hindi", "")) for ex in bundle["icl_examples"]]

                log(f"[prompt-sensitivity] model={model_key} pair={pair} n_icl={n_icl} items={len(eval_rows)} variants={prompt_variants}")

                item_rows: List[Dict[str, Any]] = []
                for item_idx, word in enumerate(eval_rows, start=1):
                    if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
                        log(f"  [{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")

                    _, prompt_meta = _condition_prompts(
                        tokenizer=tokenizer,
                        query=str(word["ood"]),
                        icl_examples=bundle["icl_examples"],
                        input_script_name=bundle["input_script_name"],
                        source_language=bundle["source_language"],
                        output_script_name=bundle["output_script_name"],
                        seed=int(args.seed),
                    )
                    competitor = choose_bank_competitor(prompt_meta, str(word["hindi"]))
                    target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
                    if not target_ids:
                        continue
                    target_id = int(target_ids[0])

                    row: Dict[str, Any] = {
                        "item_index": int(item_idx - 1),
                        "word_ood": str(word["ood"]),
                        "word_target": str(word["hindi"]),
                        "nearest_bank_target": str(competitor["target"]),
                        "nearest_bank_rank": int(competitor["position"] + 1),
                        "nearest_bank_similarity": float(competitor["similarity"]),
                        "variants": {},
                    }
                    for prompt_variant in prompt_variants:
                        prompts = _variant_prompts(
                            query=str(word["ood"]),
                            icl_examples=bundle["icl_examples"],
                            input_script_name=bundle["input_script_name"],
                            source_language=bundle["source_language"],
                            output_script_name=bundle["output_script_name"],
                            prompt_variant=str(prompt_variant),
                            seed=int(args.seed),
                        )
                        row["variants"][prompt_variant] = {}
                        for condition in CONDITIONS:
                            input_ids = build_prompt_input_ids(
                                tokenizer=tokenizer,
                                prompt_text=str(prompts[condition]),
                                device=device,
                            )
                            first = first_step_stats(
                                model=model,
                                input_ids=input_ids,
                                target_id=int(target_id),
                            )
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
                            generation = transliteration_metrics(
                                raw_text=raw_text,
                                gold_text=str(word["hindi"]),
                                output_script_name=bundle["output_script_name"],
                                bank_targets=bank_targets,
                                nearest_bank_target=str(competitor["target"]),
                                fuzzy_bank_threshold=float(args.fuzzy_bank_threshold),
                            )
                            row["variants"][prompt_variant][condition] = {
                                "first_step": first,
                                "generation": generation,
                            }
                    item_rows.append(row)

                summary_by_variant: Dict[str, Dict[str, Any]] = {}
                for prompt_variant in prompt_variants:
                    summary_by_variant[prompt_variant] = {}
                    for condition in CONDITIONS:
                        rows = [item["variants"][prompt_variant][condition] for item in item_rows]
                        summary_by_variant[prompt_variant][condition] = {
                            "n_items": int(len(rows)),
                            "mean_target_prob": float(np.nanmean([float(r["first_step"].get("target_prob", float("nan"))) for r in rows])),
                            "first_token_top1_target_rate": float(np.nanmean([1.0 if bool(r["first_step"].get("top1_is_target", False)) else 0.0 for r in rows])),
                            "mean_target_minus_competitor_logit": float(
                                np.nanmean([float(r["first_step"].get("target_minus_competitor_logit", float("nan"))) for r in rows])
                            ),
                            "mean_exact_match": float(np.nanmean([float(r["generation"].get("exact_match", 0.0)) for r in rows])),
                            "mean_akshara_cer": float(np.nanmean([float(r["generation"].get("akshara_cer", float("nan"))) for r in rows])),
                            "mean_script_compliance": float(np.nanmean([float(r["generation"].get("script_compliance", float("nan"))) for r in rows])),
                            "mean_first_entry_correct": float(np.nanmean([float(r["generation"].get("first_entry_correct", float("nan"))) for r in rows])),
                            "exact_bank_copy_rate": float(np.nanmean([float(r["generation"].get("exact_bank_copy", 0.0)) for r in rows])),
                            "exact_nearest_bank_copy_rate": float(
                                np.nanmean([float(r["generation"].get("exact_nearest_bank_copy", 0.0)) for r in rows])
                            ),
                            "fuzzy_bank_copy_rate": float(np.nanmean([float(r["generation"].get("fuzzy_bank_copy", 0.0)) for r in rows])),
                            "mean_max_bank_similarity": float(
                                np.nanmean([float(r["generation"].get("max_bank_similarity", float("nan"))) for r in rows])
                            ),
                        }

                comparison_rows: List[Dict[str, Any]] = []
                canonical_summary = summary_by_variant.get("canonical", {})
                for prompt_variant in prompt_variants:
                    if prompt_variant == "canonical":
                        continue
                    for condition in CONDITIONS:
                        base = canonical_summary[condition]
                        cur = summary_by_variant[prompt_variant][condition]
                        comparison_rows.append(
                            {
                                "prompt_variant": str(prompt_variant),
                                "condition": str(condition),
                                "delta_mean_exact_match": float(cur["mean_exact_match"] - base["mean_exact_match"]),
                                "delta_mean_akshara_cer": float(cur["mean_akshara_cer"] - base["mean_akshara_cer"]),
                                "delta_first_token_top1_target_rate": float(
                                    cur["first_token_top1_target_rate"] - base["first_token_top1_target_rate"]
                                ),
                                "delta_mean_target_minus_competitor_logit": float(
                                    cur["mean_target_minus_competitor_logit"] - base["mean_target_minus_competitor_logit"]
                                ),
                                "delta_exact_bank_copy_rate": float(cur["exact_bank_copy_rate"] - base["exact_bank_copy_rate"]),
                            }
                        )

                out_dir = (
                    out_root
                    / str(model_key)
                    / str(pair)
                    / f"seed{int(args.seed)}"
                    / f"nicl{int(n_icl)}"
                )
                payload = {
                    "experiment": "prompt_sensitivity_check",
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": str(model_key),
                    "pair": str(pair),
                    "seed": int(args.seed),
                    "n_icl": int(n_icl),
                    "prompt_variants": list(prompt_variants),
                    "conditions": CONDITIONS,
                    "summary_by_variant": summary_by_variant,
                    "comparison_rows": comparison_rows,
                    "item_rows": item_rows,
                }
                write_json(out_dir / "prompt_sensitivity_check.json", payload)
                aggregate_summary[f"{model_key}:{pair}:{n_icl}"] = summary_by_variant

    print(json.dumps(json_safe(aggregate_summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
