#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_practical_patch_eval import (  # noqa: E402
    _build_latin_mask,
    _evaluate_generation,
    _first_step_stats,
    _json_safe,
    _mean_delta_vector,
    _parse_alpha_grid,
    _parse_channels,
    _prepare_condition_inputs,
    _register_channel_add_hook,
    _register_channel_zero_hook,
    _selection_alpha_sweep,
    _summarize_intervention,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402
from research.modules.eval.output_extraction import (  # noqa: E402
    resolve_generation_stop_ids,
    resolve_pad_token_id,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Held-out Hindi 1B intervention eval for single-channel ablations and calibrated mean-shift patching.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--select-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--channels", type=str, default="5486,2299")
    ap.add_argument("--alpha-grid", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    channels = _parse_channels(args.channels)
    if len(channels) != 2:
        raise ValueError(f"Expected exactly 2 channels for this evaluation, got {channels}")
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    set_all_seeds(int(args.seed))

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
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)
    stop_ids = resolve_generation_stop_ids(tokenizer)
    pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=stop_ids)

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

    interventions = [
        {
            "name": f"zero_channel_{int(channels[0])}",
            "kind": "zero",
            "channel_indices": [int(channels[0])],
        },
        {
            "name": f"zero_channel_{int(channels[1])}",
            "kind": "zero",
            "channel_indices": [int(channels[1])],
        },
        {
            "name": "zero_both_channels",
            "kind": "zero",
            "channel_indices": [int(c) for c in channels],
        },
        {
            "name": "calibrated_mean_shift",
            "kind": "add",
            "channel_indices": [int(c) for c in channels],
            "delta_values": [float(x) for x in chosen_delta],
        },
        {
            "name": "calibrated_sign_flip",
            "kind": "add",
            "channel_indices": [int(c) for c in channels],
            "delta_values": [-float(x) for x in chosen_delta],
        },
    ]

    print(
        f"Running Hindi intervention eval: model={args.model} pair={args.pair} select_items={len(select_rows)} eval_items={len(eval_rows)} channels={channels} alpha={chosen_alpha}",
        flush=True,
    )

    item_rows: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_rows, start=1):
        if idx == 1 or idx == len(eval_rows) or idx % 5 == 0:
            print(f"[eval {idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)

        target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
        if not target_ids:
            continue
        target_id = int(target_ids[0])
        cond_inputs = _prepare_condition_inputs(
            tokenizer=tokenizer,
            word=word,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
        )
        helpful = cond_inputs["icl_helpful"]

        base_first = _first_step_stats(
            model=model,
            input_ids=helpful["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
        )
        base_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=helpful["input_ids"],
            target_text=str(word["hindi"]),
            output_script_name=bundle["output_script_name"],
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
        )

        item_payload: Dict[str, Any] = {
            "item_index": int(idx - 1),
            "word_ood": str(word["ood"]),
            "word_hindi": str(word["hindi"]),
            "baseline_no_patch": {
                "first_step": base_first,
                "generation": base_gen,
            },
        }

        for spec in interventions:
            if spec["kind"] == "zero":
                hook = _register_channel_zero_hook(
                    model,
                    layer=int(args.layer),
                    patch_position=int(helpful["last_position"]),
                    channel_indices=spec["channel_indices"],
                )
            elif spec["kind"] == "add":
                hook = _register_channel_add_hook(
                    model,
                    layer=int(args.layer),
                    patch_position=int(helpful["last_position"]),
                    channel_indices=spec["channel_indices"],
                    delta_values=spec["delta_values"],
                )
            else:
                raise ValueError(f"Unknown intervention kind: {spec['kind']}")
            patched_first = _first_step_stats(
                model=model,
                input_ids=helpful["input_ids"],
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
                hooks=[hook],
            )

            if spec["kind"] == "zero":
                hook = _register_channel_zero_hook(
                    model,
                    layer=int(args.layer),
                    patch_position=int(helpful["last_position"]),
                    channel_indices=spec["channel_indices"],
                )
            else:
                hook = _register_channel_add_hook(
                    model,
                    layer=int(args.layer),
                    patch_position=int(helpful["last_position"]),
                    channel_indices=spec["channel_indices"],
                    delta_values=spec["delta_values"],
                )
            patched_gen = _evaluate_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=helpful["input_ids"],
                target_text=str(word["hindi"]),
                output_script_name=bundle["output_script_name"],
                max_new_tokens=int(args.max_new_tokens),
                stop_ids=stop_ids,
                pad_id=pad_id,
                hooks=[hook],
            )
            item_payload[str(spec["name"])] = {
                "first_step": patched_first,
                "generation": patched_gen,
            }

        item_rows.append(item_payload)

    summary_rows = [
        _summarize_intervention(name="baseline_no_patch", items=item_rows, key_prefix="baseline_no_patch", seed=int(args.seed)),
    ]
    for i, spec in enumerate(interventions):
        summary_rows.append(
            _summarize_intervention(
                name=str(spec["name"]),
                items=item_rows,
                key_prefix=str(spec["name"]),
                seed=int(args.seed) + 100 * (i + 1),
            )
        )

    summary_map = {row["intervention"]: row for row in summary_rows}
    comparison_rows = [
        {
            "comparison": "calibrated_mean_shift_minus_baseline",
            "delta_exact_match": summary_map["calibrated_mean_shift"]["delta_exact_match"],
            "delta_cer_improvement": summary_map["calibrated_mean_shift"]["delta_cer_improvement"],
            "delta_first_entry_correct": summary_map["calibrated_mean_shift"]["delta_first_entry_correct"],
            "delta_first_token_gap_latin": summary_map["calibrated_mean_shift"]["delta_first_token_gap_latin"],
        },
        {
            "comparison": f"zero_channel_{int(channels[0])}_minus_baseline",
            "delta_exact_match": summary_map[f"zero_channel_{int(channels[0])}"]["delta_exact_match"],
            "delta_cer_improvement": summary_map[f"zero_channel_{int(channels[0])}"]["delta_cer_improvement"],
            "delta_first_entry_correct": summary_map[f"zero_channel_{int(channels[0])}"]["delta_first_entry_correct"],
            "delta_first_token_gap_latin": summary_map[f"zero_channel_{int(channels[0])}"]["delta_first_token_gap_latin"],
        },
        {
            "comparison": f"zero_channel_{int(channels[1])}_minus_baseline",
            "delta_exact_match": summary_map[f"zero_channel_{int(channels[1])}"]["delta_exact_match"],
            "delta_cer_improvement": summary_map[f"zero_channel_{int(channels[1])}"]["delta_cer_improvement"],
            "delta_first_entry_correct": summary_map[f"zero_channel_{int(channels[1])}"]["delta_first_entry_correct"],
            "delta_first_token_gap_latin": summary_map[f"zero_channel_{int(channels[1])}"]["delta_first_token_gap_latin"],
        },
        {
            "comparison": "zero_both_channels_minus_baseline",
            "delta_exact_match": summary_map["zero_both_channels"]["delta_exact_match"],
            "delta_cer_improvement": summary_map["zero_both_channels"]["delta_cer_improvement"],
            "delta_first_entry_correct": summary_map["zero_both_channels"]["delta_first_entry_correct"],
            "delta_first_token_gap_latin": summary_map["zero_both_channels"]["delta_first_token_gap_latin"],
        },
        {
            "comparison": "calibrated_mean_shift_minus_zero_both_channels_exact",
            "value": float(summary_map["calibrated_mean_shift"]["exact_match"]["mean"] - summary_map["zero_both_channels"]["exact_match"]["mean"]),
        },
        {
            "comparison": "calibrated_mean_shift_minus_sign_flip_exact",
            "value": float(summary_map["calibrated_mean_shift"]["exact_match"]["mean"] - summary_map["calibrated_sign_flip"]["exact_match"]["mean"]),
        },
    ]

    payload = {
        "experiment": "hindi_1b_intervention_eval",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "select_max_items": int(args.select_max_items),
        "max_items": int(args.max_items),
        "layer": int(args.layer),
        "channels": [int(c) for c in channels],
        "alpha_grid": [float(a) for a in alpha_grid],
        "selected_alpha": chosen_alpha,
        "selected_mean_delta": chosen_mean,
        "selection_rows": selection_rows,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out_root).resolve()
        if str(args.out_root).strip()
        else PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_intervention_eval_v1" / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "hindi_1b_intervention_eval.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("=== Eval summary ===", flush=True)
    for row in summary_rows:
        print(
            f"  {row['intervention']:<24} EM={row['exact_match']['mean']:.3f} CER={row['akshara_cer']['mean']:.3f} "
            f"dEM={row['delta_exact_match']['mean']:.3f} dCERimp={row['delta_cer_improvement']['mean']:.3f} "
            f"dGap={row['delta_first_token_gap_latin']['mean']:.3f} harm={row['harm_rate_on_base_successes']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
