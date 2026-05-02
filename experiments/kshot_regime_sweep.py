#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import apply_chat_template, load_model, set_all_seeds  # noqa: E402
from experiments.regime_eval_common import (  # noqa: E402
    build_prompt_input_ids,
    choose_bank_competitor,
    default_stop_and_pad,
    first_step_stats,
    generate_raw_text,
    transliteration_metrics,
    write_json,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402

CONDITIONS = ["zs", "icl_helpful", "icl_corrupt"]


def _parse_int_csv(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No integer values provided.")
    out: List[int] = []
    for v in vals:
        if v not in out:
            out.append(int(v))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fixed-split k-shot regime sweep for multilingual transliteration.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=str, default="8,16,32,64,128")
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--fuzzy-bank-threshold", type=float, default=0.85)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/kshot_regime_sweep_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ks = sorted(_parse_int_csv(args.ks))
    max_k = int(max(ks))
    set_all_seeds(int(args.seed))
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(max_k),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    fixed_icl_bank = list(bundle["icl_examples"][: int(max_k)])
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    stop_and_pad = default_stop_and_pad(tokenizer)

    item_rows: List[Dict[str, Any]] = []
    summary_by_k: Dict[str, Dict[str, Any]] = {}

    for k in ks:
        icl_examples = list(fixed_icl_bank[: int(k)])
        k_key = f"k{int(k)}"
        log(f"[kshot] model={args.model} pair={args.pair} seed={args.seed} k={k} items={len(eval_rows)}")
        k_rows: List[Dict[str, Any]] = []
        for item_idx, word in enumerate(eval_rows, start=1):
            if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
                log(f"  [k={k}] {item_idx}/{len(eval_rows)} :: {word['ood']} -> {word['hindi']}")
            prompts, meta = _condition_prompts(
                tokenizer=tokenizer,
                query=str(word["ood"]),
                icl_examples=icl_examples,
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                seed=int(args.seed),
            )
            competitor = choose_bank_competitor(meta, str(word["hindi"])) if icl_examples else {
                "target": "",
                "source": "",
                "position": -1,
                "similarity": float("nan"),
            }
            target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            bank_targets = [str(ex.get("hindi", "")) for ex in icl_examples]
            row: Dict[str, Any] = {
                "k": int(k),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_target": str(word["hindi"]),
                "nearest_bank_target": str(competitor["target"]),
                "nearest_bank_rank": int(competitor["position"] + 1) if int(competitor["position"]) >= 0 else None,
                "nearest_bank_similarity": float(competitor["similarity"]),
                "conditions": {},
            }
            for condition in CONDITIONS:
                input_ids = build_prompt_input_ids(tokenizer=tokenizer, prompt_text=str(prompts[condition]), device=device)
                first = first_step_stats(model=model, input_ids=input_ids, target_id=int(target_id))
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
                row["conditions"][condition] = {
                    "first_step": first,
                    "generation": generation,
                }
            item_rows.append(row)
            k_rows.append(row)

        cond_summary: Dict[str, Any] = {}
        for condition in CONDITIONS:
            rows = [row["conditions"][condition] for row in k_rows]
            cond_summary[condition] = {
                "n_items": int(len(rows)),
                "mean_target_prob": float(np.nanmean([float(r["first_step"].get("target_prob", float("nan"))) for r in rows])),
                "first_token_top1_target_rate": float(np.nanmean([1.0 if bool(r["first_step"].get("top1_is_target", False)) else 0.0 for r in rows])),
                "mean_target_minus_competitor_logit": float(np.nanmean([float(r["first_step"].get("target_minus_competitor_logit", float("nan"))) for r in rows])),
                "mean_exact_match": float(np.nanmean([float(r["generation"].get("exact_match", 0.0)) for r in rows])),
                "mean_akshara_cer": float(np.nanmean([float(r["generation"].get("akshara_cer", float("nan"))) for r in rows])),
                "mean_script_compliance": float(np.nanmean([float(r["generation"].get("script_compliance", float("nan"))) for r in rows])),
                "mean_first_entry_correct": float(np.nanmean([float(r["generation"].get("first_entry_correct", float("nan"))) for r in rows])),
                "exact_bank_copy_rate": float(np.nanmean([float(r["generation"].get("exact_bank_copy", 0.0)) for r in rows])),
                "exact_nearest_bank_copy_rate": float(np.nanmean([float(r["generation"].get("exact_nearest_bank_copy", 0.0)) for r in rows])),
                "fuzzy_bank_copy_rate": float(np.nanmean([float(r["generation"].get("fuzzy_bank_copy", 0.0)) for r in rows])),
                "mean_max_bank_similarity": float(np.nanmean([float(r["generation"].get("max_bank_similarity", float("nan"))) for r in rows])),
            }
        summary_by_k[k_key] = cond_summary

    payload = {
        "experiment": "kshot_regime_sweep",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "ks": ks,
        "fixed_bank_size": int(len(fixed_icl_bank)),
        "fixed_eval_size": int(len(eval_rows)),
        "conditions": CONDITIONS,
        "fuzzy_bank_threshold": float(args.fuzzy_bank_threshold),
        "summary_by_k": summary_by_k,
        "item_rows": item_rows,
    }

    out_dir = out_root / str(args.model) / str(args.pair) / f"seed{int(args.seed)}"
    out_path = out_dir / "kshot_regime_sweep.json"
    write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print(json.dumps(summary_by_k, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
