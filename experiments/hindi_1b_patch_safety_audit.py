#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import apply_chat_template, load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_practical_patch_eval import (  # noqa: E402
    _mean_delta_vector,
    _parse_alpha_grid,
    _parse_channels,
    _selection_alpha_sweep,
)
from experiments.hindi_1b_practical_patch_eval import _register_channel_add_hook as _register_channel_add_hook  # noqa: E402
from experiments.hindi_1b_practical_patch_eval import _register_channel_zero_hook as _register_channel_zero_hook  # noqa: E402
from experiments.hindi_1b_practical_patch_eval import _json_safe as _json_safe  # noqa: E402
from experiments.hindi_1b_mlp_channel_subset_panel import _build_latin_mask  # noqa: E402
from experiments.regime_eval_common import (  # noqa: E402
    build_prompt_input_ids,
    extract_english_word,
    extract_script_word,
    first_step_stats,
    generate_raw_text,
    script_bucket,
    write_json,
)
from experiments.safety_prompt_sets import SAFETY_DOMAINS  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import normalize_text  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402
from research.modules.eval.output_extraction import resolve_generation_stop_ids, resolve_pad_token_id  # noqa: E402


INTERVENTIONS = (
    "baseline_no_patch",
    "chosen_mean_shift",
    "chosen_sign_flip",
    "zero_both_channels",
)


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _normalize_answer(raw_text: str, *, domain: str) -> str:
    if str(domain) == "english":
        return extract_english_word(raw_text)
    if str(domain) == "hindi":
        return normalize_text(extract_script_word(raw_text, script_name="Devanagari", min_script_ratio=0.80))
    raise ValueError(f"Unknown domain: {domain}")


def _is_script_valid(prediction: str, *, domain: str) -> bool:
    if str(domain) == "english":
        return bool(prediction) and script_bucket(prediction) == "latin"
    if str(domain) == "hindi":
        return bool(prediction) and script_bucket(prediction) == "devanagari"
    return False


def _best_teacher_forced_metrics(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    answers: Sequence[str],
    hooks: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        from core import _teacher_forced_metrics_from_input_ids  # local import to keep module load simple

        candidates: List[Dict[str, Any]] = []
        for answer in answers:
            token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
            if not token_ids:
                continue
            metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=input_ids,
                target_ids=[int(x) for x in token_ids],
                target_id=int(token_ids[0]),
                device=str(input_ids.device),
                competitor_id=-1,
            )
            candidates.append(
                {
                    "answer": normalize_text(str(answer)),
                    "token_ids": [int(x) for x in token_ids],
                    **metrics,
                }
            )
        if not candidates:
            return {
                "best_answer": "",
                "best_joint_logprob": float("nan"),
                "best_first_prob": float("nan"),
                "best_target_pos1_nll": float("nan"),
                "all_candidates": [],
            }
        best = max(candidates, key=lambda row: float(row.get("joint_logprob", float("-inf"))))
        return {
            "best_answer": str(best["answer"]),
            "best_joint_logprob": float(best.get("joint_logprob", float("nan"))),
            "best_first_prob": float(best.get("first_prob", float("nan"))),
            "best_target_pos1_nll": float(best.get("target_pos1_nll", float("nan"))),
            "all_candidates": candidates,
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _intervention_hook(
    *,
    model: Any,
    intervention: str,
    layer: int,
    patch_position: int,
    channels: Sequence[int],
    chosen_delta: Sequence[float],
):
    if str(intervention) == "baseline_no_patch":
        return None
    if str(intervention) == "chosen_mean_shift":
        return _register_channel_add_hook(
            model,
            layer=int(layer),
            patch_position=int(patch_position),
            channel_indices=[int(c) for c in channels],
            delta_values=[float(x) for x in chosen_delta],
        )
    if str(intervention) == "chosen_sign_flip":
        return _register_channel_add_hook(
            model,
            layer=int(layer),
            patch_position=int(patch_position),
            channel_indices=[int(c) for c in channels],
            delta_values=[-float(x) for x in chosen_delta],
        )
    if str(intervention) == "zero_both_channels":
        return _register_channel_zero_hook(
            model,
            layer=int(layer),
            patch_position=int(patch_position),
            channel_indices=[int(c) for c in channels],
        )
    raise ValueError(f"Unknown intervention: {intervention}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cross-task safety audit for the fixed Hindi 1B practical patch.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--select-max-items", type=int, default=100)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--channels", type=str, default="5486,2299")
    ap.add_argument("--alpha-grid", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/hindi_patch_safety_audit_v1/1b/seed42")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    channels = _parse_channels(args.channels)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

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
    select_rows = list(bundle["select_rows"][: max(1, int(args.select_max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    stop_ids = resolve_generation_stop_ids(tokenizer)
    pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=stop_ids)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)

    chosen_mean = _mean_delta_vector(
        model=model,
        rows=select_rows,
        tokenizer=tokenizer,
        icl_examples=bundle["icl_examples"],
        input_script_name=bundle["input_script_name"],
        source_language=bundle["source_language"],
        output_script_name=bundle["output_script_name"],
        device=device,
        layer=int(args.layer),
        channels=channels,
    )
    selection_rows = _selection_alpha_sweep(
        model=model,
        tokenizer=tokenizer,
        rows=select_rows,
        icl_examples=bundle["icl_examples"],
        input_script_name=bundle["input_script_name"],
        source_language=bundle["source_language"],
        output_script_name=bundle["output_script_name"],
        device=device,
        layer=int(args.layer),
        channels=channels,
        mean_delta=chosen_mean["mean_delta"],
        alpha_grid=alpha_grid,
        latin_mask=latin_mask,
    )
    best_row = max(selection_rows, key=lambda row: (float(row["delta_mean_target_minus_latin_logit"]), float(row["delta_mean_target_prob"])))
    chosen_alpha = float(best_row["alpha"])
    chosen_delta = [chosen_alpha * float(v) for v in chosen_mean["mean_delta"]]

    item_rows: List[Dict[str, Any]] = []
    for domain, items in SAFETY_DOMAINS.items():
        for item_idx, item in enumerate(items, start=1):
            input_ids = build_prompt_input_ids(tokenizer=tokenizer, prompt_text=str(item["prompt"]), device=device)
            patch_position = int(input_ids.shape[1] - 1)
            answer_ids = [
                [int(x) for x in tokenizer.encode(str(ans), add_special_tokens=False)]
                for ans in item["answers"]
                if tokenizer.encode(str(ans), add_special_tokens=False)
            ]
            if not answer_ids:
                continue
            primary_target_id = int(answer_ids[0][0])
            row: Dict[str, Any] = {
                "domain": str(domain),
                "item_id": str(item["id"]),
                "prompt": str(item["prompt"]),
                "answers": [normalize_text(str(a)) for a in item["answers"]],
                "prompt_tokens": int(input_ids.shape[1]),
                "conditions": {},
            }
            for intervention in INTERVENTIONS:
                hook = _intervention_hook(
                    model=model,
                    intervention=str(intervention),
                    layer=int(args.layer),
                    patch_position=int(patch_position),
                    channels=channels,
                    chosen_delta=chosen_delta,
                )
                hooks = [hook] if hook is not None else []
                tf = _best_teacher_forced_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    answers=item["answers"],
                    hooks=hooks,
                )
                hook = _intervention_hook(
                    model=model,
                    intervention=str(intervention),
                    layer=int(args.layer),
                    patch_position=int(patch_position),
                    channels=channels,
                    chosen_delta=chosen_delta,
                )
                hooks = [hook] if hook is not None else []
                first = first_step_stats(
                    model=model,
                    input_ids=input_ids,
                    target_id=int(primary_target_id),
                    hooks=hooks,
                )
                hook = _intervention_hook(
                    model=model,
                    intervention=str(intervention),
                    layer=int(args.layer),
                    patch_position=int(patch_position),
                    channels=channels,
                    chosen_delta=chosen_delta,
                )
                hooks = [hook] if hook is not None else []
                raw_text = generate_raw_text(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    stop_ids=stop_ids,
                    pad_id=pad_id,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    hooks=hooks,
                )
                prediction = _normalize_answer(raw_text, domain=str(domain))
                row["conditions"][str(intervention)] = {
                    "teacher_forced": tf,
                    "first_step": first,
                    "raw_text": normalize_text(raw_text),
                    "prediction": prediction,
                    "exact_match": float(prediction in {normalize_text(str(a).lower() if domain == 'english' else str(a)) for a in item["answers"]}),
                    "script_valid": float(_is_script_valid(prediction, domain=str(domain))),
                }
            item_rows.append(row)
            if item_idx == 1 or item_idx == len(items) or item_idx % 8 == 0:
                print(f"[safety] {domain} {item_idx}/{len(items)} :: {item['id']}", flush=True)

    summaries: Dict[str, Any] = {}
    for domain in list(SAFETY_DOMAINS.keys()) + ["all"]:
        domain_rows = [row for row in item_rows if domain == "all" or str(row["domain"]) == domain]
        bucket: Dict[str, Any] = {}
        for intervention in INTERVENTIONS:
            rows = [row["conditions"][intervention] for row in domain_rows]
            bucket[intervention] = {
                "n_items": int(len(rows)),
                "mean_best_joint_logprob": float(np.nanmean([float(r["teacher_forced"].get("best_joint_logprob", float("nan"))) for r in rows])),
                "mean_best_first_prob": float(np.nanmean([float(r["teacher_forced"].get("best_first_prob", float("nan"))) for r in rows])),
                "mean_target_pos1_nll": float(np.nanmean([float(r["teacher_forced"].get("best_target_pos1_nll", float("nan"))) for r in rows])),
                "first_token_top1_target_rate": float(np.nanmean([1.0 if bool(r["first_step"].get("top1_is_target", False)) else 0.0 for r in rows])),
                "mean_target_minus_competitor_logit": float(np.nanmean([float(r["first_step"].get("target_minus_competitor_logit", float("nan"))) for r in rows])),
                "exact_match_rate": float(np.nanmean([float(r.get("exact_match", 0.0)) for r in rows])),
                "script_valid_rate": float(np.nanmean([float(r.get("script_valid", 0.0)) for r in rows])),
            }
        base = bucket["baseline_no_patch"]
        for intervention in INTERVENTIONS:
            if intervention == "baseline_no_patch":
                continue
            bucket[f"delta_vs_baseline::{intervention}"] = {
                "mean_best_joint_logprob": float(bucket[intervention]["mean_best_joint_logprob"] - base["mean_best_joint_logprob"]),
                "mean_best_first_prob": float(bucket[intervention]["mean_best_first_prob"] - base["mean_best_first_prob"]),
                "first_token_top1_target_rate": float(bucket[intervention]["first_token_top1_target_rate"] - base["first_token_top1_target_rate"]),
                "exact_match_rate": float(bucket[intervention]["exact_match_rate"] - base["exact_match_rate"]),
                "script_valid_rate": float(bucket[intervention]["script_valid_rate"] - base["script_valid_rate"]),
            }
        summaries[domain] = bucket

    payload = {
        "experiment": "hindi_1b_patch_safety_audit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "seed": int(args.seed),
        "patch": {
            "layer": int(args.layer),
            "channels": [int(c) for c in channels],
            "selection_pair": str(args.pair),
            "selection_items": int(len(select_rows)),
            "chosen_alpha": float(chosen_alpha),
            "chosen_delta": [float(x) for x in chosen_delta],
            "selection_rows": selection_rows,
        },
        "domains": list(SAFETY_DOMAINS.keys()),
        "summaries": summaries,
        "item_rows": item_rows,
    }

    out_root = (PROJECT_ROOT / str(args.out_root)).resolve()
    out_path = out_root / "hindi_1b_patch_safety_audit.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print(json.dumps(_json_safe({"summaries": summaries, "patch": payload["patch"]}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
