#!/usr/bin/env python3
"""Cross-model behavioral check: run the core transliteration behavioral panel
on a non-Gemma model to test whether the regime map generalizes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import apply_chat_template, set_all_seeds  # noqa: E402
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
from paper2_fidelity_calibrated.eval_utils import normalize_text  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402

CONDITIONS = ["zs", "icl_helpful", "icl_corrupt"]


def _load_external_model(hf_id: str, device: str):
    """Load any HF causal LM without going through our Gemma-specific config."""
    log(f"Loading external model: {hf_id}")
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    log(f"Loaded {hf_id} | tokenizer={type(tokenizer).__name__}")
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cross-model behavioral check for transliteration regimes.")
    ap.add_argument("--hf-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model-label", type=str, default="qwen2.5-3b")
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--fuzzy-bank-threshold", type=float, default=0.85)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/cross_model_behavioral_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    pairs = [p.strip() for p in str(args.pairs).split(",") if p.strip()]
    out_root = (REPO_ROOT / str(args.out_root)).resolve()

    model, tokenizer = _load_external_model(str(args.hf_id), str(args.device))
    device = str(next(model.parameters()).device)
    stop_and_pad = default_stop_and_pad(tokenizer)

    all_summaries: Dict[str, Any] = {}

    for pair in pairs:
        log(f"=== {args.model_label} x {pair} ===")
        bundle = load_pair_split(
            str(pair),
            seed=int(args.seed),
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
        bank_targets = [str(ex.get("hindi", "")) for ex in bundle["icl_examples"]]

        item_rows: List[Dict[str, Any]] = []
        for item_idx, word in enumerate(eval_rows, start=1):
            if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
                log(f"  [{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
            prompts, meta = _condition_prompts(
                tokenizer=tokenizer,
                query=str(word["ood"]),
                icl_examples=bundle["icl_examples"],
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                seed=int(args.seed),
            )
            competitor = choose_bank_competitor(meta, str(word["hindi"]))
            target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            row: Dict[str, Any] = {
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_target": str(word["hindi"]),
                "nearest_bank_target": str(competitor["target"]),
                "conditions": {},
            }
            for condition in CONDITIONS:
                input_ids = build_prompt_input_ids(tokenizer=tokenizer, prompt_text=str(prompts[condition]), device=device)
                first = first_step_stats(model=model, input_ids=input_ids, target_id=int(target_id))
                raw_text = generate_raw_text(
                    model=model, tokenizer=tokenizer, input_ids=input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    stop_ids=stop_and_pad["stop_ids"], pad_id=stop_and_pad["pad_id"],
                    do_sample=False, temperature=0.0, top_p=1.0,
                )
                generation = transliteration_metrics(
                    raw_text=raw_text, gold_text=str(word["hindi"]),
                    output_script_name=bundle["output_script_name"],
                    bank_targets=bank_targets, nearest_bank_target=str(competitor["target"]),
                    fuzzy_bank_threshold=float(args.fuzzy_bank_threshold),
                )
                row["conditions"][condition] = {"first_step": first, "generation": generation}
            item_rows.append(row)

        cond_summary: Dict[str, Any] = {}
        for condition in CONDITIONS:
            rows = [r["conditions"][condition] for r in item_rows]
            cond_summary[condition] = {
                "n_items": len(rows),
                "mean_target_prob": float(np.nanmean([float(r["first_step"].get("target_prob", float("nan"))) for r in rows])),
                "first_token_top1_target_rate": float(np.nanmean([1.0 if r["first_step"].get("top1_is_target") else 0.0 for r in rows])),
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
        all_summaries[pair] = cond_summary

        out_dir = out_root / str(args.model_label) / str(pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
        write_json(out_dir / "cross_model_behavioral.json", {
            "experiment": "cross_model_behavioral_check",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_label": str(args.model_label), "hf_id": str(args.hf_id),
            "pair": str(pair), "seed": int(args.seed), "n_icl": int(args.n_icl),
            "conditions": CONDITIONS, "summary": cond_summary, "item_rows": item_rows,
        })
        log(f"Saved {pair}")

    print(json.dumps(json_safe(all_summaries), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
