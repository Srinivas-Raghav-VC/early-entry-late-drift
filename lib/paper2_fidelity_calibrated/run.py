#!/usr/bin/env python3
"""
Paper 2 runner: fidelity-calibrated feature interventions.

Core idea:
  1) Use a disjoint 3-way split per seed: ICL / selection / evaluation.
  2) Measure transcoder fidelity at the intervention target (MLP output) for
     each layer (and variant).
  3) Restrict layer selection to the most faithful layers (configurable).
  4) Select (layer, topk, variant) on the selection split.
  5) Evaluate once on held-out evaluation split.

This produces a single JSON artifact and a per-word CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is importable when running from this subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_config
from core import (
    apply_chat_template,
    build_corrupted_icl_prompt,
    build_task_prompt,
    compute_statistics,
    get_layer_device,
    get_model_layers,
    load_model,
    load_transcoder,
    run_patching_experiment,
    save_json,
    split_data_three_way,
)
from paper2_fidelity_calibrated.joint_selection import (
    layer_best_scores,
    select_best_joint_config,
)
from paper2_fidelity_calibrated.protocol_utils import (
    local_stability_window,
    prompt_fingerprint,
    prompt_template_fingerprint,
    runtime_identity,
)
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata, load_pair_records_bundle


RANK_METRICS = {
    "mean_pe",
    "pe_minus_random",
    "pe_minus_corrupt",
    "pe_minus_shuffle",
    "pe_minus_gauss",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_int_list(raw: str) -> List[int]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _get_model_layers(model):
    return get_model_layers(model)


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if not (np.isfinite(na) and np.isfinite(nb)) or na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / max(1e-12, na * nb))


def _rel_l2(err: torch.Tensor, ref: torch.Tensor) -> float:
    err = err.detach().float().reshape(-1)
    ref = ref.detach().float().reshape(-1)
    denom = float(torch.norm(ref).item())
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(torch.norm(err).item() / max(1e-12, denom))


def _capture_mlp_in_out_last_token(model, tokenizer, *, prompt_text: str, layer: int, device: str):
    from core import apply_chat_template  # local import avoids circular

    text = apply_chat_template(tokenizer, prompt_text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    layers = _get_model_layers(model)

    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs_tuple, output):
        x = inputs_tuple[0]
        y = output[0] if isinstance(output, tuple) else output
        captured["x"] = x.detach()
        captured["y"] = y.detach()

    handle = layers[int(layer)].mlp.register_forward_hook(hook)
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    handle.remove()
    if "x" not in captured or "y" not in captured:
        raise RuntimeError("Failed to capture MLP (input, output).")
    return captured["x"][0, -1, :], captured["y"][0, -1, :]


def _fidelity_curve(
    model,
    tokenizer,
    *,
    model_key: str,
    scope_repo: str,
    variant: str,
    layers: List[int],
    selection_words: List[Dict[str, str]],
    icl_examples: List[Dict[str, str]],
    prompt_meta: Dict[str, str],
    device: str,
    n_fidelity_samples: int,
    seed: int,
) -> Dict[int, Dict[str, float]]:
    """
    Compute mean fidelity metrics per layer for a given variant.

    Returns: {layer: {"mean_cos_y": ..., "mean_relerr_y": ...}}
    """
    from core import build_task_prompt

    rng = np.random.default_rng(int(seed))
    words = selection_words
    if int(n_fidelity_samples) > 0 and len(words) > int(n_fidelity_samples):
        idx = rng.choice(len(words), size=int(n_fidelity_samples), replace=False).tolist()
        words = [words[i] for i in idx]

    source_language, input_script_name, output_script_name = _prompt_naming(prompt_meta)

    out: Dict[int, Dict[str, float]] = {}
    for layer in layers:
        layer_dev = get_layer_device(model, int(layer))
        tc = load_transcoder(
            model,
            scope_repo,
            int(layer),
            layer_dev,
            variant=str(variant),
        )

        cos_y: List[float] = []
        rel_y: List[float] = []
        for w in words:
            prompt = build_task_prompt(
                w["ood"],
                icl_examples=icl_examples,
                input_script_name=input_script_name,
                source_language=source_language,
                output_script_name=output_script_name,
            )
            x, y = _capture_mlp_in_out_last_token(
                model,
                tokenizer,
                prompt_text=prompt,
                layer=int(layer),
                device=str(device),
            )
            feats = tc.encode(x.unsqueeze(0)).squeeze(0)
            recon = tc.decode(feats.unsqueeze(0)).squeeze(0)
            cos_y.append(_safe_cos(recon, y))
            rel_y.append(_rel_l2(recon - y, y))

        out[int(layer)] = {
            "mean_cos_y": float(np.nanmean(cos_y)),
            "mean_relerr_y": float(np.nanmean(rel_y)),
            "n": int(len(words)),
        }
    return out


def _choose_layers_by_fidelity(
    fidelity_by_layer: Dict[int, Dict[str, float]],
    *,
    mode: str,
    keep_top_frac: float,
    keep_top_n: int,
    min_cos_y: float,
) -> List[int]:
    items = []
    for layer, m in fidelity_by_layer.items():
        items.append((int(layer), float(m.get("mean_cos_y", float("nan")))))
    items = [(l, c) for (l, c) in items if np.isfinite(c)]
    if not items:
        return []

    mode = str(mode or "top_frac").strip().lower()
    if mode == "threshold":
        kept = [l for (l, c) in items if c >= float(min_cos_y)]
        return sorted(set(kept))

    items.sort(key=lambda lc: lc[1], reverse=True)
    if mode == "top_n":
        n = max(1, int(keep_top_n))
        return sorted({l for (l, _) in items[:n]})

    # Default: top_frac
    frac = float(keep_top_frac)
    if not np.isfinite(frac) or frac <= 0:
        frac = 0.2
    n = max(1, int(math.ceil(frac * len(items))))
    return sorted({l for (l, _) in items[:n]})


def _selection_score(stats: Dict[str, Any], rank_metric: str) -> float:
    rm = str(rank_metric or "pe_minus_corrupt").strip()
    if rm == "mean_pe":
        return float(stats.get("mean_pe", float("nan")))
    if rm == "pe_minus_random":
        rc = stats.get("random_control", {}) if isinstance(stats, dict) else {}
        return float(rc.get("mean_pe_minus_random", float("nan")))
    if rm == "pe_minus_corrupt":
        tc = stats.get("task_matched_control", {}) if isinstance(stats, dict) else {}
        return float(tc.get("mean_pe_minus_corrupt", float("nan")))
    if rm == "pe_minus_shuffle":
        sc = stats.get("shuffle_control", {}) if isinstance(stats, dict) else {}
        return float(sc.get("mean_pe_minus_shuffle", float("nan")))
    if rm == "pe_minus_gauss":
        gc = stats.get("gauss_control", {}) if isinstance(stats, dict) else {}
        return float(gc.get("mean_pe_minus_gauss", float("nan")))
    raise ValueError(f"Unknown rank metric: {rank_metric!r}")


def _prompt_naming(prompt_meta: Dict[str, str]) -> Tuple[str, str, str]:
    source_language = str(prompt_meta.get("source_language", "")).strip() or "Hindi"
    input_script_name = str(prompt_meta.get("source_script", "")).strip() or "Latin"
    output_script_name = str(prompt_meta.get("target_script", "")).strip() or "Devanagari"
    return source_language, input_script_name, output_script_name


def _load_words(
    pair_id: str,
    *,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    bundle = load_pair_records_bundle(pair_id, include_builtin=not bool(external_only))
    total = int(len(bundle.rows))
    source_names = [s.name for s in bundle.sources]
    external_sources = [n for n in source_names if n and n != "config_multiscript"]

    if bool(require_external_sources) and not external_sources:
        raise RuntimeError(
            f"Pair {pair_id!r} has no external sources (only builtin). "
            "Provide external data under data/transliteration/ or disable --require-external-sources."
        )
    if int(min_pool_size) > 0 and total < int(min_pool_size):
        raise RuntimeError(f"Pair {pair_id!r} pool too small: total={total} < {int(min_pool_size)}")

    words = [{"english": r["english"], "hindi": r["target"], "ood": r["source"]} for r in bundle.rows]
    meta = {
        "pair_id": pair_id,
        "total_rows": total,
        "sources": [asdict(s) for s in bundle.sources],
        "source_counts": dict(bundle.source_counts),
    }
    return words, meta


def _make_protocol_prompt_packet(
    tokenizer,
    *,
    sample_word: Dict[str, str],
    icl_examples: List[Dict[str, str]],
    prompt_meta: Dict[str, str],
    prompt_variant: str,
) -> Dict[str, Any]:
    source_language, input_script_name, output_script_name = _prompt_naming(prompt_meta)
    query_token = str(sample_word["ood"])

    explicit_prompt = build_task_prompt(
        query_token,
        None,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    icl_prompt = build_task_prompt(
        query_token,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    corrupt_prompt = build_corrupted_icl_prompt(
        query_token,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        seed=0,
    )

    return {
        "prompt_variant": str(prompt_variant),
        "prompt_template": prompt_template_fingerprint(tokenizer),
        "explicit_zs": prompt_fingerprint(
            raw_prompt=explicit_prompt,
            rendered_prompt=apply_chat_template(tokenizer, explicit_prompt),
        ),
        "icl": prompt_fingerprint(
            raw_prompt=icl_prompt,
            rendered_prompt=apply_chat_template(tokenizer, icl_prompt),
        ),
        "corrupt_icl": prompt_fingerprint(
            raw_prompt=corrupt_prompt,
            rendered_prompt=apply_chat_template(tokenizer, corrupt_prompt),
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper 2: fidelity-calibrated interventions.")
    ap.add_argument("--model", type=str, default="4b", choices=["270m", "1b", "4b", "12b"])
    ap.add_argument("--pair", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--variants", type=str, default="skipless_or_non_affine")
    ap.add_argument("--layers", type=str, default="", help="Optional comma-separated layer indices.")
    ap.add_argument("--layer-step", type=int, default=1, help="If --layers empty, test every Nth layer.")

    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)

    ap.add_argument("--topk-options", type=str, default="4,8,16,32")
    ap.add_argument("--rank-metric", type=str, default="pe_minus_corrupt", choices=sorted(RANK_METRICS))
    ap.add_argument("--patch-style", type=str, default="sparse", choices=["sparse", "substitute"])
    ap.add_argument("--feature-selection", type=str, default="topk_abs_delta")
    ap.add_argument("--selector-reference", type=str, default="corrupt_icl", choices=["zs", "corrupt_icl"])
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--require-query-span-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--norm-matching", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--n-fidelity-samples", type=int, default=64)
    ap.add_argument("--fidelity-mode", type=str, default="top_frac", choices=["top_frac", "top_n", "threshold"])
    ap.add_argument("--fidelity-keep-top-frac", type=float, default=0.2)
    ap.add_argument("--fidelity-keep-top-n", type=int, default=5)
    ap.add_argument("--fidelity-min-cos-y", type=float, default=0.0)

    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=0)

    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--eval-generation", action="store_true", help="(reserved) included for parity with Paper 1.")
    args = ap.parse_args()

    pair_id = str(args.pair).strip()
    model_key = str(args.model).strip()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    variants = _parse_str_list(args.variants)
    topk_options = [int(x) for x in _parse_int_list(args.topk_options)]
    if not topk_options:
        raise ValueError("Empty --topk-options.")

    words, provenance = _load_words(
        pair_id,
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    prompt_meta = get_pair_prompt_metadata(pair_id)
    source_language, input_script_name, output_script_name = _prompt_naming(prompt_meta)

    cfg = get_model_config(model_key)
    model, tokenizer = load_model(model_key, device=str(args.device))

    layers = _get_model_layers(model)
    n_layers = int(len(layers))
    chosen_layers = _parse_int_list(args.layers)
    if not chosen_layers:
        step = max(1, int(args.layer_step))
        chosen_layers = list(range(0, n_layers, step))
    chosen_layers = [l for l in chosen_layers if 0 <= int(l) < n_layers]
    if not chosen_layers:
        raise ValueError("No valid layers selected.")

    out_root = Path(__file__).resolve().parent
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).resolve() if str(args.out).strip() else (
        results_dir / f"{pair_id}/{model_key}/paper2_fidelity_calibrated_{model_key}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")

    payload: Dict[str, Any] = {
        "paper": "paper2_fidelity_calibrated",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_key": model_key,
        "pair": pair_id,
        "pair_meta": dict(prompt_meta),
        "provenance": provenance,
        "runtime_identity": runtime_identity(
            model_key=model_key,
            hf_id=cfg.hf_id,
            tokenizer=tokenizer,
            model=model,
        ),
        "protocol": {
            "claim_level": "intervention_only",
            "proxy_metrics_not_sufficient": True,
            "prompt_template_frozen_after_stage0": True,
            "local_stability_reporting_only": True,
            "final_paper_outcomes": [
                "Positive mechanism-validity paper",
                "Only affine survives",
                "Only CLT survives",
                "Skeptical negative result",
            ],
        },
        "config": {
            "seeds": seeds,
            "device": str(args.device),
            "variants": variants,
            "layers": chosen_layers,
            "n_icl": int(args.n_icl),
            "n_select": int(args.n_select),
            "n_eval": int(args.n_eval),
            "topk_options": topk_options,
            "rank_metric": str(args.rank_metric),
            "patch_style": str(args.patch_style),
            "feature_selection": str(args.feature_selection),
            "selector_reference": str(args.selector_reference),
            "prompt_variant": str(args.prompt_variant),
            "require_query_span_match": bool(args.require_query_span_match),
            "norm_matching": bool(args.norm_matching),
            "n_fidelity_samples": int(args.n_fidelity_samples),
            "fidelity_mode": str(args.fidelity_mode),
            "fidelity_keep_top_frac": float(args.fidelity_keep_top_frac),
            "fidelity_keep_top_n": int(args.fidelity_keep_top_n),
            "fidelity_min_cos_y": float(args.fidelity_min_cos_y),
            "external_only": bool(args.external_only),
            "require_external_sources": bool(args.require_external_sources),
            "min_pool_size": int(args.min_pool_size),
        },
        "seeds": {},
        "aggregate": {},
    }

    # Per-word CSV records for eval split only (one row per test word per seed).
    csv_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        log(f"Seed {seed}: splitting data")
        icl, sel, ev = split_data_three_way(
            words=words,
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            seed=int(seed),
        )

        seed_info: Dict[str, Any] = {
            "split_sizes": {"icl": len(icl), "select": len(sel), "eval": len(ev)},
            "variants": {},
        }
        sample_word = sel[0] if sel else (ev[0] if ev else (icl[0] if icl else None))
        if sample_word is not None:
            seed_info["prompt_packet"] = _make_protocol_prompt_packet(
                tokenizer,
                sample_word=sample_word,
                icl_examples=icl,
                prompt_meta=prompt_meta,
                prompt_variant=str(args.prompt_variant),
            )

        best_global = {
            "score": float("-inf"),
            "variant": "",
            "layer": None,
            "topk": None,
            "selection_stats": None,
            "eval_stats": None,
        }

        for variant in variants:
            log(f"Seed {seed}: fidelity curve for variant={variant}")
            fidelity = _fidelity_curve(
                model,
                tokenizer,
                model_key=model_key,
                scope_repo=cfg.scope_repo,
                variant=str(variant),
                layers=chosen_layers,
                selection_words=sel,
                icl_examples=icl,
                prompt_meta=prompt_meta,
                device=str(args.device),
                n_fidelity_samples=int(args.n_fidelity_samples),
                seed=int(seed),
            )
            candidate_layers = _choose_layers_by_fidelity(
                fidelity,
                mode=str(args.fidelity_mode),
                keep_top_frac=float(args.fidelity_keep_top_frac),
                keep_top_n=int(args.fidelity_keep_top_n),
                min_cos_y=float(args.fidelity_min_cos_y),
            )
            if not candidate_layers:
                # Conference-friendly: record and skip variant rather than silently widening.
                seed_info["variants"][variant] = {
                    "fidelity": fidelity,
                    "candidate_layers": [],
                    "status": "NO_FIDELITY_LAYERS",
                }
                log(f"Seed {seed}: variant={variant} has no candidate layers after fidelity gate; skipping.")
                continue

            # Jointly select (layer, top-k) on the held-out selection split.
            score_grid: Dict[int, Dict[int, float]] = {}
            stats_grid: Dict[int, Dict[int, Any]] = {}
            for layer in candidate_layers:
                layer_dev = get_layer_device(model, int(layer))
                tc = load_transcoder(
                    model,
                    cfg.scope_repo,
                    int(layer),
                    layer_dev,
                    variant=str(variant),
                )
                score_grid[int(layer)] = {}
                stats_grid[int(layer)] = {}
                for topk in topk_options:
                    results = [
                        run_patching_experiment(
                            model,
                            tokenizer,
                            tc,
                            int(layer),
                            w,
                            icl_examples=icl,
                            topk=int(topk),
                            device=str(args.device),
                            seed=int(seed),
                            input_script_name=input_script_name,
                            source_language=source_language,
                            output_script_name=output_script_name,
                            patch_style=str(args.patch_style),
                            feature_selection=str(args.feature_selection),
                            selector_reference_mode=str(args.selector_reference),
                            require_query_span_match=bool(args.require_query_span_match),
                            use_norm_matching=bool(args.norm_matching),
                            prompt_variant=str(args.prompt_variant),
                        )
                        for w in sel
                    ]
                    stats = compute_statistics(results)
                    score = _selection_score(stats, str(args.rank_metric))
                    score_grid[int(layer)][int(topk)] = float(score)
                    stats_grid[int(layer)][int(topk)] = stats
                    log(f"Seed {seed}: {variant} layer {layer} topk={topk} score={score:+.4f}")

            best_layer, best_topk, best_score = select_best_joint_config(score_grid)
            if best_layer is None or best_topk is None:
                seed_info["variants"][variant] = {
                    "fidelity": fidelity,
                    "candidate_layers": candidate_layers,
                    "selection_grid": {
                        str(layer): {str(topk): score for topk, score in topk_scores.items()}
                        for layer, topk_scores in score_grid.items()
                    },
                    "status": "NO_VALID_SELECTION_SCORE",
                }
                log(
                    f"Seed {seed}: variant={variant} has no finite joint selection score; skipping.",
                )
                continue

            # Held-out evaluation at (variant, layer, topk).
            tc = load_transcoder(
                model,
                cfg.scope_repo,
                int(best_layer),
                get_layer_device(model, int(best_layer)),
                variant=str(variant),
            )
            eval_results = [
                run_patching_experiment(
                    model,
                    tokenizer,
                    tc,
                    int(best_layer),
                    w,
                    icl_examples=icl,
                    topk=int(best_topk),
                    device=str(args.device),
                    seed=int(seed),
                    input_script_name=input_script_name,
                    source_language=source_language,
                    output_script_name=output_script_name,
                    patch_style=str(args.patch_style),
                    feature_selection=str(args.feature_selection),
                    selector_reference_mode=str(args.selector_reference),
                    require_query_span_match=bool(args.require_query_span_match),
                    use_norm_matching=bool(args.norm_matching),
                    prompt_variant=str(args.prompt_variant),
                )
                for w in ev
            ]
            eval_stats = compute_statistics(eval_results)
            sel_score = float(best_score)

            seed_info["variants"][variant] = {
                "status": "OK",
                "fidelity": fidelity,
                "candidate_layers": candidate_layers,
                "selection_grid": {
                    str(layer): {str(topk): score for topk, score in topk_scores.items()}
                    for layer, topk_scores in score_grid.items()
                },
                "layer_scores": layer_best_scores(score_grid),
                "best_layer": int(best_layer),
                "topk_scores": score_grid[int(best_layer)],
                "best_topk": int(best_topk),
                "local_stability_window": local_stability_window(
                    layer=int(best_layer),
                    topk=int(best_topk),
                    valid_layers=candidate_layers,
                    topk_ladder=topk_options,
                ),
                "selection_stats_at_best": stats_grid[int(best_layer)][int(best_topk)],
                "eval_stats_at_best": eval_stats,
            }

            if np.isfinite(sel_score) and sel_score > float(best_global["score"]):
                best_global = {
                    "score": float(sel_score),
                    "variant": str(variant),
                    "layer": int(best_layer),
                    "topk": int(best_topk),
                    "selection_stats": stats_grid[int(best_layer)][int(best_topk)],
                    "eval_stats": eval_stats,
                }

            # Append eval word-level rows for CSV
            for r in eval_results:
                row = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                row.update(
                    {
                        "seed": int(seed),
                        "variant": str(variant),
                        "layer": int(best_layer),
                        "topk": int(best_topk),
                        "split": "eval",
                    }
                )
                csv_rows.append(row)

        seed_info["best"] = best_global
        payload["seeds"][str(seed)] = seed_info

    # Aggregate: mean best score across seeds (selection metric) and mean eval PE at best.
    best_eval_pes: List[float] = []
    best_sel_scores: List[float] = []
    for seed in seeds:
        b = payload["seeds"][str(seed)].get("best", {})
        best_sel_scores.append(float(b.get("score", float("nan"))))
        ev = b.get("eval_stats") or {}
        best_eval_pes.append(float(ev.get("mean_pe", float("nan"))))

    payload["aggregate"] = {
        "best_selection_score_mean": float(np.nanmean(best_sel_scores)),
        "best_selection_score_std": float(np.nanstd(best_sel_scores)),
        "best_eval_mean_pe_mean": float(np.nanmean(best_eval_pes)),
        "best_eval_mean_pe_std": float(np.nanstd(best_eval_pes)),
    }

    save_json(str(out_path), payload)
    log(f"Saved JSON: {out_path}")

    # Write CSV
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            # Use union of keys for stable export.
            keys = sorted({k for row in csv_rows for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        log(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
