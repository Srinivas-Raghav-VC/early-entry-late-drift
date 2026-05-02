#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
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


def _safe_mean(values: List[float]) -> float:
    xs = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def _parse_float_csv(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No float values provided.")
    out: List[float] = []
    for v in vals:
        if v not in out:
            out.append(float(v))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Temperature robustness sweep for Telugu transliteration regimes.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=57)
    ap.add_argument("--temperatures", type=str, default="0.0,0.2,0.7")
    ap.add_argument("--samples-per-temp", type=int, default=3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--fuzzy-bank-threshold", type=float, default=0.85)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_temperature_sweep_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    temps = _parse_float_csv(args.temperatures)
    set_all_seeds(int(args.seed))
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    stop_and_pad = default_stop_and_pad(tokenizer)

    item_rows: List[Dict[str, Any]] = []
    summary_by_temp: Dict[str, Dict[str, Any]] = {}

    for item_idx, word in enumerate(eval_rows, start=1):
        if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 10 == 0:
            log(f"[temp] {args.model} {args.pair} {item_idx}/{len(eval_rows)} :: {word['ood']} -> {word['hindi']}")
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
        bank_targets = [str(ex.get("hindi", "")) for ex in bundle["icl_examples"]]
        row: Dict[str, Any] = {
            "item_index": int(item_idx - 1),
            "word_ood": str(word["ood"]),
            "word_target": str(word["hindi"]),
            "nearest_bank_target": str(competitor["target"]),
            "nearest_bank_rank": int(competitor["position"] + 1),
            "nearest_bank_similarity": float(competitor["similarity"]),
            "temperatures": {},
        }
        for temp_idx, temp in enumerate(temps):
            temp_key = f"temp_{temp:.1f}" if float(temp).is_integer() else f"temp_{temp}"
            row["temperatures"][temp_key] = {}
            for condition in CONDITIONS:
                input_ids = build_prompt_input_ids(tokenizer=tokenizer, prompt_text=str(prompts[condition]), device=device)
                first = first_step_stats(model=model, input_ids=input_ids, target_id=int(target_id))
                n_samples = 1 if float(temp) == 0.0 else max(1, int(args.samples_per_temp))
                sample_rows: List[Dict[str, Any]] = []
                for sample_idx in range(n_samples):
                    sample_seed = int(args.seed) + int(temp_idx * 1000) + int(sample_idx * 17)
                    raw_text = generate_raw_text(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        max_new_tokens=int(args.max_new_tokens),
                        stop_ids=stop_and_pad["stop_ids"],
                        pad_id=stop_and_pad["pad_id"],
                        do_sample=bool(float(temp) > 0.0),
                        temperature=float(temp),
                        top_p=float(args.top_p),
                        sample_seed=int(sample_seed),
                    )
                    generation = transliteration_metrics(
                        raw_text=raw_text,
                        gold_text=str(word["hindi"]),
                        output_script_name=bundle["output_script_name"],
                        bank_targets=bank_targets,
                        nearest_bank_target=str(competitor["target"]),
                        fuzzy_bank_threshold=float(args.fuzzy_bank_threshold),
                    )
                    sample_rows.append(
                        {
                            "sample_index": int(sample_idx),
                            "sample_seed": int(sample_seed),
                            "generation": generation,
                        }
                    )
                row["temperatures"][temp_key][condition] = {
                    "first_step": first,
                    "samples": sample_rows,
                }
        item_rows.append(row)

    for temp in temps:
        temp_key = f"temp_{temp:.1f}" if float(temp).is_integer() else f"temp_{temp}"
        summary_by_temp[temp_key] = {}
        for condition in CONDITIONS:
            samples = [sample for row in item_rows for sample in row["temperatures"][temp_key][condition]["samples"]]
            first_rows = [row["temperatures"][temp_key][condition]["first_step"] for row in item_rows]
            summary_by_temp[temp_key][condition] = {
                "n_items": int(len(item_rows)),
                "n_samples": int(len(samples)),
                "mean_target_prob": _safe_mean([float(r.get("target_prob", float("nan"))) for r in first_rows]),
                "first_token_top1_target_rate": _safe_mean([1.0 if bool(r.get("top1_is_target", False)) else 0.0 for r in first_rows]),
                "mean_target_minus_competitor_logit": _safe_mean([float(r.get("target_minus_competitor_logit", float("nan"))) for r in first_rows]),
                "mean_exact_match": _safe_mean([float(s["generation"].get("exact_match", 0.0)) for s in samples]),
                "mean_akshara_cer": _safe_mean([float(s["generation"].get("akshara_cer", float("nan"))) for s in samples]),
                "mean_script_compliance": _safe_mean([float(s["generation"].get("script_compliance", float("nan"))) for s in samples]),
                "mean_first_entry_correct": _safe_mean([float(s["generation"].get("first_entry_correct", float("nan"))) for s in samples]),
                "exact_bank_copy_rate": _safe_mean([float(s["generation"].get("exact_bank_copy", 0.0)) for s in samples]),
                "exact_nearest_bank_copy_rate": _safe_mean([float(s["generation"].get("exact_nearest_bank_copy", 0.0)) for s in samples]),
                "fuzzy_bank_copy_rate": _safe_mean([float(s["generation"].get("fuzzy_bank_copy", 0.0)) for s in samples]),
                "mean_max_bank_similarity": _safe_mean([float(s["generation"].get("max_bank_similarity", float("nan"))) for s in samples]),
            }

    payload = {
        "experiment": "telugu_temperature_sweep",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "conditions": CONDITIONS,
        "temperatures": temps,
        "samples_per_temp": int(args.samples_per_temp),
        "top_p": float(args.top_p),
        "fuzzy_bank_threshold": float(args.fuzzy_bank_threshold),
        "summary_by_temp": summary_by_temp,
        "item_rows": item_rows,
    }

    out_dir = out_root / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_path = out_dir / "telugu_temperature_sweep.json"
    write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print(json.dumps(summary_by_temp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
