#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import (  # noqa: E402
    _teacher_forced_metrics_from_input_ids,
    apply_chat_template,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.eval_utils import normalize_text  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402

CONDITIONS = [
    "zs",
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_reversed",
    "icl_corrupt",
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
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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


def _choose_bank_competitor(meta: Mapping[str, Any], gold: str) -> Dict[str, Any]:
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


def _first_logprob(metrics: Mapping[str, Any]) -> float:
    prob = float(metrics.get("first_prob", float("nan")))
    if not np.isfinite(prob) or prob <= 0.0:
        return float("nan")
    return float(math.log(prob))


def _seq_metrics(
    *,
    model: Any,
    input_ids: torch.Tensor,
    device: str,
    gold_text: str,
    competitor_text: str,
    tokenizer: Any,
) -> Dict[str, Any]:
    gold_ids = tokenizer.encode(str(gold_text), add_special_tokens=False)
    competitor_ids = tokenizer.encode(str(competitor_text), add_special_tokens=False)
    if not gold_ids or not competitor_ids:
        raise ValueError("Gold or competitor tokenization failed.")
    gold_first = int(gold_ids[0])
    comp_first = int(competitor_ids[0])
    gold_metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=input_ids,
        target_ids=[int(x) for x in gold_ids],
        target_id=gold_first,
        competitor_id=comp_first,
        device=device,
    )
    comp_metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=input_ids,
        target_ids=[int(x) for x in competitor_ids],
        target_id=comp_first,
        competitor_id=gold_first,
        device=device,
    )
    gold_first_lp = _first_logprob(gold_metrics)
    comp_first_lp = _first_logprob(comp_metrics)
    seq_gap = float(gold_metrics["joint_logprob"] - comp_metrics["joint_logprob"])
    first_gap = float(gold_first_lp - comp_first_lp) if np.isfinite(gold_first_lp) and np.isfinite(comp_first_lp) else float("nan")
    gold_cont = float(gold_metrics["joint_logprob"] - gold_first_lp) if np.isfinite(gold_first_lp) else float("nan")
    comp_cont = float(comp_metrics["joint_logprob"] - comp_first_lp) if np.isfinite(comp_first_lp) else float("nan")
    cont_gap = float(gold_cont - comp_cont) if np.isfinite(gold_cont) and np.isfinite(comp_cont) else float("nan")
    gold_cont_len = max(0, int(len(gold_ids) - 1))
    comp_cont_len = max(0, int(len(competitor_ids) - 1))
    gold_cont_avg = float(gold_cont / gold_cont_len) if gold_cont_len > 0 and np.isfinite(gold_cont) else float("nan")
    comp_cont_avg = float(comp_cont / comp_cont_len) if comp_cont_len > 0 and np.isfinite(comp_cont) else float("nan")
    cont_gap_avg = float(gold_cont_avg - comp_cont_avg) if np.isfinite(gold_cont_avg) and np.isfinite(comp_cont_avg) else float("nan")
    return {
        "gold": {
            "text": normalize_text(str(gold_text)),
            "n_tokens": int(len(gold_ids)),
            **gold_metrics,
            "first_logprob": gold_first_lp,
            "continuation_logprob": gold_cont,
            "continuation_avg_logprob": gold_cont_avg,
        },
        "competitor": {
            "text": normalize_text(str(competitor_text)),
            "n_tokens": int(len(competitor_ids)),
            **comp_metrics,
            "first_logprob": comp_first_lp,
            "continuation_logprob": comp_cont,
            "continuation_avg_logprob": comp_cont_avg,
        },
        "gaps": {
            "sequence_logprob_gap": seq_gap,
            "first_token_logprob_gap": first_gap,
            "continuation_logprob_gap": cont_gap,
            "continuation_avg_logprob_gap": cont_gap_avg,
        },
        "same_first_token": bool(gold_first == comp_first),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Teacher-forced gold-vs-bank continuation competition audit for Telugu transliteration.")
    ap.add_argument("--model", type=str, default="4b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_continuation_competition_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
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
    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)

    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
    item_rows: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 5 == 0:
            log(f"[{item_idx}/{len(eval_rows)}] {args.model} {args.pair} n{args.n_icl} :: {word['ood']} -> {word['hindi']}")
        prompts, meta = _condition_prompts(
            tokenizer=tokenizer,
            query=str(word["ood"]),
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            seed=int(args.seed),
        )
        competitor = _choose_bank_competitor(meta, str(word["hindi"]))
        row = {
            "item_index": int(item_idx - 1),
            "word_ood": str(word["ood"]),
            "word_telugu": normalize_text(str(word["hindi"])),
            "nearest_bank_target": str(competitor["target"]),
            "nearest_bank_source": str(competitor["source"]),
            "nearest_bank_rank": int(competitor["position"] + 1),
            "nearest_bank_similarity": float(competitor["similarity"]),
            "conditions": {},
        }
        for condition in CONDITIONS:
            rendered = apply_chat_template(tokenizer, str(prompts[condition]))
            input_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            generated = _generate_text(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
            )
            seq = _seq_metrics(
                model=model,
                input_ids=input_ids,
                device=device,
                gold_text=str(word["hindi"]),
                competitor_text=str(competitor["target"]),
                tokenizer=tokenizer,
            )
            row["conditions"][condition] = {
                "generated_prediction": generated,
                "prediction_equals_gold": bool(normalize_text(generated) == normalize_text(str(word["hindi"]))),
                "prediction_equals_nearest_bank": bool(normalize_text(generated) == normalize_text(str(competitor["target"]))),
                **seq,
            }
        item_rows.append(row)

    summary_by_condition: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [item["conditions"][condition] for item in item_rows]
        summary_by_condition[condition] = {
            "n_items": int(len(rows)),
            "mean_sequence_logprob_gap": float(np.nanmean([float(r["gaps"]["sequence_logprob_gap"]) for r in rows])),
            "mean_first_token_logprob_gap": float(np.nanmean([float(r["gaps"]["first_token_logprob_gap"]) for r in rows])),
            "mean_continuation_logprob_gap": float(np.nanmean([float(r["gaps"]["continuation_logprob_gap"]) for r in rows])),
            "mean_continuation_avg_logprob_gap": float(np.nanmean([float(r["gaps"]["continuation_avg_logprob_gap"]) for r in rows])),
            "gold_generation_rate": float(np.mean([1.0 if bool(r["prediction_equals_gold"]) else 0.0 for r in rows])),
            "nearest_bank_generation_rate": float(np.mean([1.0 if bool(r["prediction_equals_nearest_bank"]) else 0.0 for r in rows])),
            "same_first_token_rate": float(np.mean([1.0 if bool(r["same_first_token"]) else 0.0 for r in rows])),
        }

    helpful_ranked = sorted(
        [
            {
                "word_ood": row["word_ood"],
                "gold": row["word_telugu"],
                "nearest_bank_target": row["nearest_bank_target"],
                "nearest_bank_rank": row["nearest_bank_rank"],
                "nearest_bank_similarity": row["nearest_bank_similarity"],
                "generated_prediction": row["conditions"]["icl_helpful"]["generated_prediction"],
                "prediction_equals_gold": row["conditions"]["icl_helpful"]["prediction_equals_gold"],
                "prediction_equals_nearest_bank": row["conditions"]["icl_helpful"]["prediction_equals_nearest_bank"],
                "sequence_logprob_gap": row["conditions"]["icl_helpful"]["gaps"]["sequence_logprob_gap"],
                "first_token_logprob_gap": row["conditions"]["icl_helpful"]["gaps"]["first_token_logprob_gap"],
                "continuation_logprob_gap": row["conditions"]["icl_helpful"]["gaps"]["continuation_logprob_gap"],
                "continuation_avg_logprob_gap": row["conditions"]["icl_helpful"]["gaps"]["continuation_avg_logprob_gap"],
            }
            for row in item_rows
        ],
        key=lambda r: float(r["continuation_logprob_gap"]),
    )

    payload = {
        "experiment": "telugu_continuation_competition_audit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "conditions": CONDITIONS,
        "competitor_definition": "nearest helpful-bank target by query similarity_desc, excluding gold if possible",
        "summary_by_condition": summary_by_condition,
        "worst_helpful_continuation_examples": helpful_ranked[:8],
        "best_helpful_continuation_examples": list(reversed(helpful_ranked[-8:])),
        "item_rows": item_rows,
    }

    out_dir = out_root / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_path = out_dir / "telugu_continuation_competition_audit.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}")
    print("\n=== Summary by condition ===")
    for condition in CONDITIONS:
        row = summary_by_condition[condition]
        print(
            f"  {condition:<28s} seq_gap={row['mean_sequence_logprob_gap']:+.3f} "
            f"first_gap={row['mean_first_token_logprob_gap']:+.3f} cont_gap={row['mean_continuation_logprob_gap']:+.3f} "
            f"gold_gen={row['gold_generation_rate']:.3f} bank_gen={row['nearest_bank_generation_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
