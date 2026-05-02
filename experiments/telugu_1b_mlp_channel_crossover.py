#!/usr/bin/env python3
from __future__ import annotations

"""Bounded Telugu compact-channel crossover test.

This is the natural counterpart to the Hindi fixed MLP-channel edit. It asks a
narrow question: if we use the same Gemma-1B MLP channel basis at the Telugu
continuation divergence state, can a small, selection-derived static channel
mean-shift rescue held-out continuation generation?

The script deliberately keeps selection and evaluation splits separate. It may
return a positive, negative, or mixed result; all three are useful for bounding
the paper's Hindi/Telugu intervention asymmetry.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_mlp_channel_subset_panel import (  # noqa: E402
    _extract_mlp_channel_vector,
    _get_mlp,
)
from experiments.hindi_1b_practical_patch_eval import (  # noqa: E402
    _bootstrap_mean_ci,
    _json_safe,
    _register_channel_add_hook,
)
from experiments.telugu_continuation_practical_patch_eval import (  # noqa: E402
    _build_item_setup,
    _divergence_step_stats,
    _evaluate_generation,
    _summarize_intervention,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from research.modules.eval.output_extraction import (  # noqa: E402
    resolve_generation_stop_ids,
    resolve_pad_token_id,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_csv_int(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw or "").split(",") if x.strip()]


def _parse_csv_float(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError("Expected at least one float value")
    return out


def _deterministic_random_channels(*, channel_size: int, k: int, forbidden: Sequence[int], seed: int, label: str) -> List[int]:
    blocked = set(int(x) for x in forbidden)
    pool = np.asarray([i for i in range(int(channel_size)) if i not in blocked], dtype=np.int64)
    if int(k) > int(pool.size):
        raise ValueError(f"Cannot draw {k} random channels from pool of {pool.size}")
    seed32 = int.from_bytes(hashlib.sha256(f"{seed}::{label}::{channel_size}::{k}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed32)
    return [int(x) for x in sorted(rng.choice(pool, size=int(k), replace=False).tolist())]


def _mean_channel_delta(
    *,
    model: Any,
    tokenizer: Any,
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    recipient: str,
    donor: str,
    layer: int,
) -> Dict[str, Any]:
    deltas: List[np.ndarray] = []
    recipient_vecs: List[np.ndarray] = []
    donor_vecs: List[np.ndarray] = []
    skipped = 0
    for word in rows:
        setup = _build_item_setup(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            word=word,
            seed=int(seed),
            recipient=str(recipient),
            donor=str(donor),
        )
        if setup is None:
            skipped += 1
            continue
        recipient_vec = _extract_mlp_channel_vector(
            model,
            setup["full_ids_by_condition"][recipient],
            int(layer),
            int(setup["patch_pos_by_condition"][recipient]),
        ).detach().float().cpu().numpy().astype(np.float64)
        donor_vec = _extract_mlp_channel_vector(
            model,
            setup["full_ids_by_condition"][donor],
            int(layer),
            int(setup["patch_pos_by_condition"][donor]),
        ).detach().float().cpu().numpy().astype(np.float64)
        recipient_vecs.append(recipient_vec)
        donor_vecs.append(donor_vec)
        deltas.append(donor_vec - recipient_vec)
    if not deltas:
        raise RuntimeError("No usable shared-prefix rows for channel-delta estimation")
    delta_arr = np.stack(deltas, axis=0)
    recipient_arr = np.stack(recipient_vecs, axis=0)
    donor_arr = np.stack(donor_vecs, axis=0)
    mean_delta = delta_arr.mean(axis=0)
    return {
        "n_items": int(delta_arr.shape[0]),
        "n_skipped": int(skipped),
        "channel_size": int(delta_arr.shape[1]),
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta_arr))),
        "mean_delta": [float(x) for x in mean_delta.tolist()],
        "mean_recipient": [float(x) for x in recipient_arr.mean(axis=0).tolist()],
        "mean_donor": [float(x) for x in donor_arr.mean(axis=0).tolist()],
    }


def _selection_sweep(
    *,
    model: Any,
    tokenizer: Any,
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    recipient: str,
    donor: str,
    layer: int,
    mean_delta: Sequence[float],
    k_grid: Sequence[int],
    alpha_grid: Sequence[float],
    top_channels: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    delta_vec = torch.tensor([float(x) for x in mean_delta], dtype=torch.float32)
    for k in k_grid:
        channels = [int(x) for x in top_channels[: int(k)]]
        if not channels:
            continue
        for alpha in alpha_grid:
            delta_values = [float(alpha) * float(delta_vec[int(c)].item()) for c in channels]
            gains_gap: List[float] = []
            gains_gold: List[float] = []
            gains_comp: List[float] = []
            gold_top1: List[float] = []
            comp_top1: List[float] = []
            used = 0
            skipped = 0
            for word in rows:
                setup = _build_item_setup(
                    model=model,
                    tokenizer=tokenizer,
                    bundle=bundle,
                    word=word,
                    seed=int(seed),
                    recipient=str(recipient),
                    donor=str(donor),
                )
                if setup is None:
                    skipped += 1
                    continue
                recipient_ids = setup["full_ids_by_condition"][recipient]
                patch_pos = int(setup["patch_pos_by_condition"][recipient])
                base = setup["base_by_condition"][recipient]
                hook = _register_channel_add_hook(
                    model,
                    layer=int(layer),
                    patch_position=int(patch_pos),
                    channel_indices=channels,
                    delta_values=delta_values,
                )
                patched = _divergence_step_stats(
                    model=model,
                    input_ids=recipient_ids,
                    tokenizer=tokenizer,
                    gold_next_id=int(setup["gold_next_id"]),
                    competitor_next_id=int(setup["competitor_next_id"]),
                    hooks=[hook],
                )
                gains_gap.append(float(patched["gold_minus_competitor_logit"] - base["gold_minus_competitor_logit"]))
                gains_gold.append(float(patched["gold_next_prob"] - base["gold_next_prob"]))
                gains_comp.append(float(patched["competitor_next_prob"] - base["competitor_next_prob"]))
                gold_top1.append(float(patched["top1_is_gold"]))
                comp_top1.append(float(patched["top1_is_competitor"]))
                used += 1
            out.append(
                {
                    "selector_kind": "top_abs_mean_delta",
                    "k": int(k),
                    "alpha": float(alpha),
                    "channels": channels,
                    "n_items": int(used),
                    "n_skipped": int(skipped),
                    "delta_mean_gap": float(np.mean(gains_gap)) if gains_gap else float("nan"),
                    "delta_mean_gold_next_prob": float(np.mean(gains_gold)) if gains_gold else float("nan"),
                    "delta_mean_competitor_next_prob": float(np.mean(gains_comp)) if gains_comp else float("nan"),
                    "patched_gold_top1_rate": float(np.mean(gold_top1)) if gold_top1 else float("nan"),
                    "patched_competitor_top1_rate": float(np.mean(comp_top1)) if comp_top1 else float("nan"),
                }
            )
    return out


def _summarize_random_controls(*, item_rows: Sequence[Mapping[str, Any]], random_names: Sequence[str], seed: int) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for idx, name in enumerate(random_names):
        summaries.append(_summarize_intervention(name=name, items=item_rows, key_prefix=name, seed=int(seed) + 1000 + idx * 23))
    return {
        "intervention": "random_channel_shift_avg",
        "n_random": int(len(summaries)),
        "full_exact_match_mean": float(np.mean([s["full_exact_match"]["mean"] for s in summaries])) if summaries else float("nan"),
        "full_akshara_cer_mean": float(np.mean([s["full_akshara_cer"]["mean"] for s in summaries])) if summaries else float("nan"),
        "continuation_exact_match_mean": float(np.mean([s["continuation_exact_match"]["mean"] for s in summaries])) if summaries else float("nan"),
        "continuation_akshara_cer_mean": float(np.mean([s["continuation_akshara_cer"]["mean"] for s in summaries])) if summaries else float("nan"),
        "bank_copy_rate_mean": float(np.mean([s["bank_copy_rate"]["mean"] for s in summaries])) if summaries else float("nan"),
        "delta_continuation_cer_improvement_mean": float(np.mean([s["delta_continuation_cer_improvement"]["mean"] for s in summaries])) if summaries else float("nan"),
        "delta_first_divergence_gap_mean": float(np.mean([s["delta_first_divergence_gap"]["mean"] for s in summaries])) if summaries else float("nan"),
        "per_random_summary_rows": summaries,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Telugu compact MLP-channel crossover at the continuation divergence state.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--select-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--recipient", type=str, default="icl_helpful")
    ap.add_argument("--donor", type=str, default="zs")
    ap.add_argument("--layer", type=int, default=25, help="0-based transformer block index; 25 matches the post-L25 residual boundary used elsewhere.")
    ap.add_argument("--offband-layer", type=int, default=10)
    ap.add_argument("--k-grid", type=str, default="2,4,8,16,32,64,128")
    ap.add_argument("--alpha-grid", type=str, default="0.25,0.5,1.0,1.5,2.0")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    k_grid = _parse_csv_int(args.k_grid)
    alpha_grid = _parse_csv_float(args.alpha_grid)

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
    stop_ids = resolve_generation_stop_ids(tokenizer)
    pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=stop_ids)
    channel_size = int(_get_mlp(model, int(args.layer)).down_proj.in_features)

    log(
        f"Running Telugu MLP-channel crossover: model={args.model} pair={args.pair} select={len(select_rows)} eval={len(eval_rows)} "
        f"recipient={args.recipient} donor={args.donor} layer={args.layer} k_grid={k_grid} alpha_grid={alpha_grid}"
    )

    channel_delta = _mean_channel_delta(
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        rows=select_rows,
        seed=int(args.seed),
        recipient=str(args.recipient),
        donor=str(args.donor),
        layer=int(args.layer),
    )
    mean_delta = np.asarray(channel_delta["mean_delta"], dtype=np.float64)
    top_channels = [int(x) for x in np.argsort(np.abs(mean_delta))[::-1].tolist()]

    selection_rows = _selection_sweep(
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        rows=select_rows,
        seed=int(args.seed),
        recipient=str(args.recipient),
        donor=str(args.donor),
        layer=int(args.layer),
        mean_delta=channel_delta["mean_delta"],
        k_grid=k_grid,
        alpha_grid=alpha_grid,
        top_channels=top_channels,
    )
    usable_selection = [r for r in selection_rows if int(r.get("n_items", 0)) > 0 and np.isfinite(float(r.get("delta_mean_gap", float("nan"))))]
    if not usable_selection:
        raise RuntimeError("No usable selection rows")
    best = max(usable_selection, key=lambda r: (float(r["delta_mean_gap"]), float(r["delta_mean_gold_next_prob"])))
    chosen_channels = [int(x) for x in best["channels"]]
    chosen_alpha = float(best["alpha"])
    chosen_delta_values = [float(chosen_alpha) * float(mean_delta[int(c)]) for c in chosen_channels]
    random_specs = [
        {
            "name": f"random_channel_shift_{i+1}",
            "channels": _deterministic_random_channels(
                channel_size=channel_size,
                k=len(chosen_channels),
                forbidden=chosen_channels,
                seed=int(args.seed) + i * 997,
                label=f"telugu_crossover::{i}",
            ),
            "delta_values": chosen_delta_values,
        }
        for i in range(int(args.n_random))
    ]

    item_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_rows, start=1):
        if idx == 1 or idx == len(eval_rows) or idx % 10 == 0:
            log(f"[eval {idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}")
        setup = _build_item_setup(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            word=word,
            seed=int(args.seed),
            recipient=str(args.recipient),
            donor=str(args.donor),
        )
        if setup is None:
            skipped_rows.append({"item_index": int(idx - 1), "word_ood": str(word.get("ood", "")), "reason": "no_usable_shared_prefix_divergence"})
            continue
        recipient_ids = setup["full_ids_by_condition"][str(args.recipient)]
        patch_pos = int(setup["patch_pos_by_condition"][str(args.recipient)])
        base_first = setup["base_by_condition"][str(args.recipient)]
        base_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=recipient_ids,
            output_script_name=bundle["output_script_name"],
            shared_prefix_text=str(setup["shared_prefix_text"]),
            gold_text=str(setup["gold_text"]),
            gold_suffix_text=str(setup["gold_suffix_text"]),
            competitor_text=str(setup["competitor_text"]),
            competitor_suffix_text=str(setup["competitor_suffix_text"]),
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
        )
        item: Dict[str, Any] = {
            "item_index": int(idx - 1),
            "word_ood": str(setup["word_ood"]),
            "word_target": str(setup["gold_text"]),
            "nearest_bank_target": str(setup["competitor_text"]),
            "nearest_bank_similarity": float(setup["nearest_bank_similarity"]),
            "shared_prefix_len_tokens": int(setup["shared_prefix_len_tokens"]),
            "shared_prefix_text": str(setup["shared_prefix_text"]),
            "baseline_no_patch": {"first_step": base_first, "generation": base_gen},
        }

        intervention_specs: List[Dict[str, Any]] = [
            {"name": "chosen_channel_shift", "layer": int(args.layer), "channels": chosen_channels, "delta_values": chosen_delta_values},
            {"name": "chosen_sign_flip", "layer": int(args.layer), "channels": chosen_channels, "delta_values": [-float(x) for x in chosen_delta_values]},
            {"name": "offband_channel_shift", "layer": int(args.offband_layer), "channels": chosen_channels, "delta_values": chosen_delta_values},
        ] + random_specs

        for spec in intervention_specs:
            hook = _register_channel_add_hook(
                model,
                layer=int(spec.get("layer", args.layer)),
                patch_position=patch_pos,
                channel_indices=spec["channels"],
                delta_values=spec["delta_values"],
            )
            patched_first = _divergence_step_stats(
                model=model,
                input_ids=recipient_ids,
                tokenizer=tokenizer,
                gold_next_id=int(setup["gold_next_id"]),
                competitor_next_id=int(setup["competitor_next_id"]),
                hooks=[hook],
            )
            hook = _register_channel_add_hook(
                model,
                layer=int(spec.get("layer", args.layer)),
                patch_position=patch_pos,
                channel_indices=spec["channels"],
                delta_values=spec["delta_values"],
            )
            patched_gen = _evaluate_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=recipient_ids,
                output_script_name=bundle["output_script_name"],
                shared_prefix_text=str(setup["shared_prefix_text"]),
                gold_text=str(setup["gold_text"]),
                gold_suffix_text=str(setup["gold_suffix_text"]),
                competitor_text=str(setup["competitor_text"]),
                competitor_suffix_text=str(setup["competitor_suffix_text"]),
                max_new_tokens=int(args.max_new_tokens),
                stop_ids=stop_ids,
                pad_id=pad_id,
                hooks=[hook],
            )
            item[str(spec["name"])] = {
                "layer": int(spec.get("layer", args.layer)),
                "channels": [int(x) for x in spec["channels"]],
                "first_step": patched_first,
                "generation": patched_gen,
            }
        item_rows.append(item)

    summary_rows = [
        _summarize_intervention(name="baseline_no_patch", items=item_rows, key_prefix="baseline_no_patch", seed=int(args.seed)),
        _summarize_intervention(name="chosen_channel_shift", items=item_rows, key_prefix="chosen_channel_shift", seed=int(args.seed) + 100),
        _summarize_intervention(name="chosen_sign_flip", items=item_rows, key_prefix="chosen_sign_flip", seed=int(args.seed) + 200),
        _summarize_intervention(name="offband_channel_shift", items=item_rows, key_prefix="offband_channel_shift", seed=int(args.seed) + 300),
    ]
    random_aggregate = _summarize_random_controls(
        item_rows=item_rows,
        random_names=[str(spec["name"]) for spec in random_specs],
        seed=int(args.seed),
    )
    summary_map = {row["intervention"]: row for row in summary_rows}
    chosen = summary_map["chosen_channel_shift"]
    sign = summary_map["chosen_sign_flip"]
    offband = summary_map["offband_channel_shift"]
    comparison_rows = [
        {
            "comparison": "chosen_minus_baseline",
            "delta_full_exact_match": chosen["delta_full_exact_match"],
            "delta_full_cer_improvement": chosen["delta_full_cer_improvement"],
            "delta_continuation_exact_match": chosen["delta_continuation_exact_match"],
            "delta_continuation_cer_improvement": chosen["delta_continuation_cer_improvement"],
            "delta_bank_copy_rate": chosen["delta_bank_copy_rate"],
            "delta_first_divergence_gap": chosen["delta_first_divergence_gap"],
        },
        {"comparison": "chosen_minus_signflip_cont_cer", "value": float(sign["continuation_akshara_cer"]["mean"] - chosen["continuation_akshara_cer"]["mean"])},
        {"comparison": "chosen_minus_offband_cont_cer", "value": float(offband["continuation_akshara_cer"]["mean"] - chosen["continuation_akshara_cer"]["mean"])},
        {"comparison": "chosen_minus_random_avg_cont_cer", "value": float(random_aggregate["continuation_akshara_cer_mean"] - chosen["continuation_akshara_cer"]["mean"])},
    ]

    payload = {
        "experiment": "telugu_1b_mlp_channel_crossover",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "select_max_items": int(args.select_max_items),
        "max_items": int(args.max_items),
        "recipient": str(args.recipient),
        "donor": str(args.donor),
        "layer": int(args.layer),
        "offband_layer": int(args.offband_layer),
        "k_grid": [int(x) for x in k_grid],
        "alpha_grid": [float(x) for x in alpha_grid],
        "selected": {
            "selector_kind": "top_abs_mean_delta_then_select_alpha_and_k_on_gap",
            "k": int(best["k"]),
            "alpha": float(chosen_alpha),
            "channels": chosen_channels,
            "selection_row": best,
        },
        "selected_mean_delta": {
            "channels": chosen_channels,
            "mean_delta": [float(mean_delta[int(c)]) for c in chosen_channels],
            "chosen_delta_values": chosen_delta_values,
            "full_channel_delta_summary": {k: v for k, v in channel_delta.items() if k != "mean_delta" and k != "mean_recipient" and k != "mean_donor"},
        },
        "selection_rows": selection_rows,
        "summary_rows": summary_rows,
        "random_aggregate": random_aggregate,
        "comparison_rows": comparison_rows,
        "item_rows": item_rows,
        "skipped_rows": skipped_rows,
        "oracle": {
            "description": "Crossover test for whether a compact Hindi-style MLP channel mean-shift can rescue Telugu continuation at the shared-prefix divergence state.",
            "success_criterion": "A small selected channel subset improves held-out continuation CER/exactness and beats sign-flip, offband, and random-channel controls.",
            "failure_criterion": "If the selected channel shift is weak, comparable to controls, or only changes the first divergence logit without generation improvement, the compact-channel crossover does not rescue Telugu under this protocol.",
        },
    }

    out_root = (
        Path(args.out_root).resolve()
        if str(args.out_root).strip()
        else REPO_ROOT / "research" / "results" / "autoresearch" / "telugu_mlp_channel_crossover_v1" / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "telugu_1b_mlp_channel_crossover.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("=== Eval summary ===", flush=True)
    for row in summary_rows:
        print(
            f"  {row['intervention']:<24} fullEM={row['full_exact_match']['mean']:.3f} fullCER={row['full_akshara_cer']['mean']:.3f} "
            f"contEM={row['continuation_exact_match']['mean']:.3f} contCER={row['continuation_akshara_cer']['mean']:.3f} "
            f"bank={row['bank_copy_rate']['mean']:.3f} dGap={row['delta_first_divergence_gap']['mean']:.3f}",
            flush=True,
        )
    print(
        f"  {'random_channel_shift_avg':<24} contCER={random_aggregate['continuation_akshara_cer_mean']:.3f} "
        f"dContCERimp={random_aggregate['delta_continuation_cer_improvement_mean']:.3f} dGap={random_aggregate['delta_first_divergence_gap_mean']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
