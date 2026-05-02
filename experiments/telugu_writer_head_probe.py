#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import _extract_layer_output_at_position_from_input_ids, get_model_layers, load_model, register_attention_head_ablation_hook, set_all_seeds  # noqa: E402
from experiments.regime_eval_common import choose_bank_competitor, write_json  # noqa: E402
from experiments.telugu_continuation_practical_patch_eval import _build_item_setup, _divergence_step_stats  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.phase23_common import get_attn_module, infer_num_heads  # noqa: E402


def _parse_layers(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("No layers provided.")
    out: List[int] = []
    for v in vals:
        if v not in out:
            out.append(int(v))
    return out


def _default_probe_config(model_key: str) -> Dict[str, Any]:
    if str(model_key) == "1b":
        return {"recipient_layer": 26, "candidate_layers": [18, 19, 20, 21, 22, 23, 24, 25, 26], "regime_sign": +1}
    if str(model_key) == "4b":
        return {"recipient_layer": 34, "candidate_layers": [30, 31, 32, 33, 34], "regime_sign": -1}
    raise ValueError(f"No default writer-probe config for model={model_key}")


def _extract_layer_output_with_hooks(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layer_index: int,
    position: int,
    hooks: Sequence[Any],
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}
    layer = get_model_layers(model)[int(layer_index)]

    def capture_hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["layer_out"] = y.detach()

    handle = layer.register_forward_hook(capture_hook)
    active = [handle] + list(hooks or [])
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in reversed(active):
            try:
                h.remove()
            except Exception:
                pass
    if "layer_out" not in captured:
        raise RuntimeError("Failed to capture recipient layer output under hooks")
    pos = int(max(0, min(int(position), int(captured["layer_out"].shape[1]) - 1)))
    return captured["layer_out"][0, pos, :]


def _mean(values: Sequence[float]) -> float:
    arr = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(arr)) if arr else float("nan")


def _group_rows_by_head(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    return grouped


def _top_heads_by_score(summary_rows: Sequence[Mapping[str, Any]], top_k: int) -> List[Tuple[int, int]]:
    ordered = sorted(summary_rows, key=lambda row: float(row["writer_score"]), reverse=True)
    return [(int(row["layer"]), int(row["head"])) for row in ordered[: max(1, int(top_k))]]


def _random_matched_group(
    *,
    selected: Sequence[Tuple[int, int]],
    candidate_layers: Sequence[int],
    num_heads: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(int(seed))
    selected_set = {(int(layer), int(head)) for layer, head in selected}
    out: List[Tuple[int, int]] = []
    by_layer: Dict[int, int] = defaultdict(int)
    for layer, _head in selected:
        by_layer[int(layer)] += 1
    for layer in candidate_layers:
        need = int(by_layer.get(int(layer), 0))
        if need <= 0:
            continue
        pool = [(int(layer), int(h)) for h in range(int(num_heads)) if (int(layer), int(h)) not in selected_set]
        rng.shuffle(pool)
        out.extend(pool[:need])
    return out


def _group_probe(
    *,
    model: Any,
    cached_items: Sequence[Mapping[str, Any]],
    heads: Sequence[Tuple[int, int]],
    recipient_layer: int,
) -> Dict[str, Any]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    item_rows: List[Dict[str, Any]] = []
    for item in cached_items:
        handles = [
            register_attention_head_ablation_hook(
                model,
                int(layer) - 1,
                list(hs),
                ablate_position=int(item["patch_position"]),
            )
            for layer, hs in sorted(grouped.items())
        ]
        ablated_stats = _divergence_step_stats(
            model=model,
            input_ids=item["input_ids"],
            tokenizer=item["tokenizer"],
            gold_next_id=int(item["gold_next_id"]),
            competitor_next_id=int(item["competitor_next_id"]),
            hooks=handles,
        )
        handles = [
            register_attention_head_ablation_hook(
                model,
                int(layer) - 1,
                list(hs),
                ablate_position=int(item["patch_position"]),
            )
            for layer, hs in sorted(grouped.items())
        ]
        ablated_recipient = _extract_layer_output_with_hooks(
            model=model,
            input_ids=item["input_ids"],
            layer_index=int(recipient_layer) - 1,
            position=int(item["patch_position"]),
            hooks=handles,
        )
        base_recipient = item["base_recipient_vec"]
        delta_gap = float(ablated_stats["gold_minus_competitor_logit"] - item["base_gap"])
        recipient_shift_l2 = float(torch.linalg.vector_norm((ablated_recipient - base_recipient).float()).item())
        item_rows.append(
            {
                "word_ood": str(item["word_ood"]),
                "word_target": str(item["word_target"]),
                "delta_gap": delta_gap,
                "delta_gold_top1": float(float(ablated_stats["top1_is_gold"]) - float(item["base_top1_gold"])),
                "delta_competitor_top1": float(float(ablated_stats["top1_is_competitor"]) - float(item["base_top1_competitor"])),
                "recipient_shift_l2": recipient_shift_l2,
            }
        )
    return {
        "heads": [{"layer": int(layer), "head": int(head)} for layer, head in heads],
        "n_items": int(len(item_rows)),
        "mean_delta_gap": _mean([row["delta_gap"] for row in item_rows]),
        "mean_delta_gold_top1": _mean([row["delta_gold_top1"] for row in item_rows]),
        "mean_delta_competitor_top1": _mean([row["delta_competitor_top1"] for row in item_rows]),
        "mean_recipient_shift_l2": _mean([row["recipient_shift_l2"] for row in item_rows]),
        "item_rows": item_rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upstream writer-head probe for Telugu late recipient bottlenecks.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=40)
    ap.add_argument("--recipient-layer", type=int, default=-1)
    ap.add_argument("--candidate-layers", type=str, default="")
    ap.add_argument("--top-group-size", type=int, default=3)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_writer_head_probe_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    defaults = _default_probe_config(str(args.model))
    recipient_layer = int(args.recipient_layer) if int(args.recipient_layer) >= 0 else int(defaults["recipient_layer"])
    candidate_layers = _parse_layers(args.candidate_layers) if str(args.candidate_layers).strip() else list(defaults["candidate_layers"])
    regime_sign = int(defaults["regime_sign"])

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

    num_heads = infer_num_heads(model, get_attn_module(model, int(candidate_layers[0]) - 1))
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    cached_items: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        setup = _build_item_setup(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            word=word,
            seed=int(args.seed),
            recipient="icl_helpful",
            donor="zs",
        )
        if setup is None:
            continue
        input_ids = setup["full_ids_by_condition"]["icl_helpful"]
        patch_position = int(setup["patch_pos_by_condition"]["icl_helpful"])
        base_stats = _divergence_step_stats(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            gold_next_id=int(setup["gold_next_id"]),
            competitor_next_id=int(setup["competitor_next_id"]),
        )
        base_recipient_vec = _extract_layer_output_at_position_from_input_ids(
            model,
            input_ids,
            int(recipient_layer) - 1,
            int(patch_position),
        )
        cached_items.append(
            {
                "item_index": int(item_idx - 1),
                "word_ood": str(setup["word_ood"]),
                "word_target": str(setup["gold_text"]),
                "competitor_text": str(setup["competitor_text"]),
                "input_ids": input_ids,
                "patch_position": int(patch_position),
                "gold_next_id": int(setup["gold_next_id"]),
                "competitor_next_id": int(setup["competitor_next_id"]),
                "base_gap": float(base_stats["gold_minus_competitor_logit"]),
                "base_top1_gold": float(base_stats["top1_is_gold"]),
                "base_top1_competitor": float(base_stats["top1_is_competitor"]),
                "base_recipient_vec": base_recipient_vec.detach().clone(),
                "tokenizer": tokenizer,
            }
        )
    log(f"[writer-probe] usable_items={len(cached_items)} candidate_layers={candidate_layers} recipient=L{recipient_layer}")

    item_rows: List[Dict[str, Any]] = []
    for layer in candidate_layers:
        for head in range(int(num_heads)):
            per_item: List[Dict[str, Any]] = []
            for item in cached_items:
                hook = register_attention_head_ablation_hook(
                    model,
                    int(layer) - 1,
                    [int(head)],
                    ablate_position=int(item["patch_position"]),
                )
                ablated_stats = _divergence_step_stats(
                    model=model,
                    input_ids=item["input_ids"],
                    tokenizer=tokenizer,
                    gold_next_id=int(item["gold_next_id"]),
                    competitor_next_id=int(item["competitor_next_id"]),
                    hooks=[hook],
                )
                hook = register_attention_head_ablation_hook(
                    model,
                    int(layer) - 1,
                    [int(head)],
                    ablate_position=int(item["patch_position"]),
                )
                ablated_recipient = _extract_layer_output_with_hooks(
                    model=model,
                    input_ids=item["input_ids"],
                    layer_index=int(recipient_layer) - 1,
                    position=int(item["patch_position"]),
                    hooks=[hook],
                )
                delta_gap = float(ablated_stats["gold_minus_competitor_logit"] - item["base_gap"])
                delta_gold_top1 = float(float(ablated_stats["top1_is_gold"]) - float(item["base_top1_gold"]))
                delta_competitor_top1 = float(float(ablated_stats["top1_is_competitor"]) - float(item["base_top1_competitor"]))
                recipient_shift_l2 = float(torch.linalg.vector_norm((ablated_recipient - item["base_recipient_vec"]).float()).item())
                per_item.append(
                    {
                        "item_index": int(item["item_index"]),
                        "word_ood": str(item["word_ood"]),
                        "word_target": str(item["word_target"]),
                        "delta_gap": delta_gap,
                        "delta_gold_top1": delta_gold_top1,
                        "delta_competitor_top1": delta_competitor_top1,
                        "recipient_shift_l2": recipient_shift_l2,
                    }
                )
            mean_delta_gap = _mean([row["delta_gap"] for row in per_item])
            summary_row = {
                "layer": int(layer),
                "head": int(head),
                "n_items": int(len(per_item)),
                "mean_delta_gap": mean_delta_gap,
                "mean_delta_gold_top1": _mean([row["delta_gold_top1"] for row in per_item]),
                "mean_delta_competitor_top1": _mean([row["delta_competitor_top1"] for row in per_item]),
                "mean_recipient_shift_l2": _mean([row["recipient_shift_l2"] for row in per_item]),
                "writer_score": float(regime_sign * mean_delta_gap),
                "item_rows": per_item,
            }
            item_rows.append(summary_row)
            if head == 0:
                log(f"[writer-probe] scanned layer={layer} / heads=0..{int(num_heads)-1}")

    ranked_rows = sorted(item_rows, key=lambda row: float(row["writer_score"]), reverse=True)
    top_group = _top_heads_by_score(ranked_rows, int(args.top_group_size))
    random_group = _random_matched_group(
        selected=top_group,
        candidate_layers=candidate_layers,
        num_heads=int(num_heads),
        seed=int(args.seed) + 999,
    )
    top_group_probe = _group_probe(
        model=model,
        cached_items=cached_items,
        heads=top_group,
        recipient_layer=int(recipient_layer),
    )
    random_group_probe = _group_probe(
        model=model,
        cached_items=cached_items,
        heads=random_group,
        recipient_layer=int(recipient_layer),
    )

    payload = {
        "experiment": "telugu_writer_head_probe",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "recipient_layer": int(recipient_layer),
        "candidate_layers": [int(x) for x in candidate_layers],
        "num_heads": int(num_heads),
        "regime_sign": int(regime_sign),
        "usable_items": int(len(cached_items)),
        "ranked_heads": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "writer_score": float(row["writer_score"]),
                "mean_delta_gap": float(row["mean_delta_gap"]),
                "mean_delta_gold_top1": float(row["mean_delta_gold_top1"]),
                "mean_delta_competitor_top1": float(row["mean_delta_competitor_top1"]),
                "mean_recipient_shift_l2": float(row["mean_recipient_shift_l2"]),
            }
            for row in ranked_rows
        ],
        "top_group_probe": top_group_probe,
        "random_group_probe": random_group_probe,
        "per_head_rows": item_rows,
    }

    out_dir = out_root / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_path = out_dir / "telugu_writer_head_probe.json"
    write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print(json.dumps({"top_heads": payload["ranked_heads"][:10], "top_group_probe": top_group_probe, "random_group_probe": random_group_probe}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
