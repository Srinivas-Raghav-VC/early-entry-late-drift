#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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

from core import (  # noqa: E402
    _extract_attention_output_at_position_from_input_ids,
    _extract_layer_output_at_position_from_input_ids,
    _extract_mlp_io_at_position_from_input_ids,
    apply_chat_template,
    get_model_layers,
    load_model,
    register_attention_output_replace_hook,
    register_dense_mlp_output_patch_hook,
    register_layer_output_replace_hook,
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
COMPONENTS = ["layer_output", "attention_output", "mlp_output", "final_state"]


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


def _parse_csv_str(raw: str, *, allowed: Optional[Sequence[str]] = None) -> List[str]:
    vals = [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]
    if allowed is not None:
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise ValueError(f"Unexpected values: {bad}; allowed={list(allowed)}")
    return vals


def _parse_csv_int(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw or "").split(",") if x.strip()]


def _parse_patch_pairs(raw: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for part in [x.strip() for x in str(raw or "").split(",") if x.strip()]:
        if "<-" not in part:
            raise ValueError(f"Bad patch pair '{part}'. Expected recipient<-donor.")
        recipient, donor = [x.strip() for x in part.split("<-", 1)]
        if recipient not in CONDITIONS or donor not in CONDITIONS:
            raise ValueError(f"Bad patch pair '{part}'. Conditions must be in {CONDITIONS}.")
        pairs.append((recipient, donor))
    if not pairs:
        raise ValueError("No patch pairs parsed.")
    return pairs


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


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


def _token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).replace("\n", " ").strip()


def _get_final_norm(model: Any) -> torch.nn.Module:
    chains = [
        ("model", "norm"),
        ("model", "language_model", "norm"),
        ("language_model", "norm"),
        ("language_model", "model", "norm"),
        ("model", "model", "norm"),
        ("language_model", "model", "model", "norm"),
        ("model", "language_model", "model", "norm"),
    ]
    for chain in chains:
        cur = model
        ok = True
        for name in chain:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    raise AttributeError(f"Could not locate final norm for model type {type(model).__name__}")


def _get_transformer_layers(model: Any):
    return get_model_layers(model)


def _resolve_layer_specs(model: Any, requested_layer_labels: Sequence[int]) -> Tuple[List[Dict[str, int]], int]:
    layers_mod = _get_transformer_layers(model)
    n_layers = int(len(layers_mod))
    specs: List[Dict[str, int]] = []
    for layer_label in requested_layer_labels:
        label = int(layer_label)
        layer_index = label - 1
        if layer_index < 0 or layer_index >= n_layers:
            raise ValueError(
                f"Requested layer label {label} is out of range for a model with {n_layers} layers. "
                f"This script expects 1-based human-facing layer labels."
            )
        specs.append({"label": label, "index": layer_index})
    return specs, n_layers


def _register_final_state_replace_hook(
    model: Any,
    patch_vector: torch.Tensor,
    *,
    patch_position: int,
):
    norm_module = _get_final_norm(model)
    patch_vector = patch_vector.detach()

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output
        seq_len = int(y.shape[1])
        if int(patch_position) >= seq_len:
            return output
        pv = patch_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if pv.shape[1] != y.shape[2]:
            return output
        y_new = y.clone()
        y_new[:, int(patch_position), :] = pv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return norm_module.register_forward_hook(hook)


def _extract_final_state_at_position_from_input_ids(
    model: Any,
    input_ids: torch.Tensor,
    position: int,
) -> torch.Tensor:
    norm_module = _get_final_norm(model)
    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["final_state"] = y.detach()

    handle = norm_module.register_forward_hook(hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()
    if "final_state" not in captured:
        raise RuntimeError("Failed to capture final norm output sequence.")
    seq_len = int(captured["final_state"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["final_state"][0, pos, :]


def _extract_component_vector(
    *,
    model: Any,
    component: str,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    if component == "layer_output":
        return _extract_layer_output_at_position_from_input_ids(model, input_ids, int(layer), int(position))
    if component == "attention_output":
        return _extract_attention_output_at_position_from_input_ids(model, input_ids, int(layer), int(position))
    if component == "mlp_output":
        _mlp_in, mlp_out = _extract_mlp_io_at_position_from_input_ids(model, input_ids, int(layer), int(position))
        return mlp_out
    if component == "final_state":
        return _extract_final_state_at_position_from_input_ids(model, input_ids, int(position))
    raise ValueError(f"Unknown component: {component}")


def _build_patch_hook(
    *,
    model: Any,
    component: str,
    layer: int,
    patch_vector: torch.Tensor,
    patch_position: int,
):
    if component == "layer_output":
        return register_layer_output_replace_hook(model, int(layer), patch_vector, patch_position=int(patch_position))
    if component == "attention_output":
        return register_attention_output_replace_hook(model, int(layer), patch_vector, patch_position=int(patch_position))
    if component == "mlp_output":
        return register_dense_mlp_output_patch_hook(model, int(layer), patch_vector, patch_position=int(patch_position))
    if component == "final_state":
        return _register_final_state_replace_hook(model, patch_vector, patch_position=int(patch_position))
    raise ValueError(f"Unknown component: {component}")


def _divergence_step_stats(
    *,
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    gold_next_id: int,
    competitor_next_id: int,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, int(input_ids.shape[1] - 1), :].float()
        probs = torch.softmax(logits, dim=-1)
        top1_id = int(torch.argmax(logits).item())
        return {
            "gold_next_id": int(gold_next_id),
            "gold_next_token_text": _token_text(tokenizer, int(gold_next_id)),
            "gold_next_prob": float(probs[int(gold_next_id)].item()),
            "gold_next_logit": float(logits[int(gold_next_id)].item()),
            "competitor_next_id": int(competitor_next_id),
            "competitor_next_token_text": _token_text(tokenizer, int(competitor_next_id)),
            "competitor_next_prob": float(probs[int(competitor_next_id)].item()),
            "competitor_next_logit": float(logits[int(competitor_next_id)].item()),
            "gold_minus_competitor_logit": float(logits[int(gold_next_id)].item() - logits[int(competitor_next_id)].item()),
            "top1_id": int(top1_id),
            "top1_token_text": _token_text(tokenizer, top1_id),
            "top1_is_gold": bool(top1_id == int(gold_next_id)),
            "top1_is_competitor": bool(top1_id == int(competitor_next_id)),
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Telugu continuation component patch panel at the divergence token.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--components", type=str, default="layer_output,attention_output,mlp_output,final_state")
    ap.add_argument("--patch-pairs", type=str, default="")
    ap.add_argument("--min-shared-prefix", type=int, default=1)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_continuation_component_panel_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    default_layers = [18, 19, 20, 21, 22, 23, 24, 25, 26] if str(args.model) == "1b" else [30, 31, 32, 33, 34]
    default_patch_pairs = "icl_helpful<-zs,zs<-icl_helpful,icl_helpful<-icl_corrupt" if str(args.model) == "1b" else "icl_corrupt<-icl_helpful,icl_helpful<-icl_corrupt"

    layer_labels = _parse_csv_int(str(args.layers)) or list(default_layers)
    components = _parse_csv_str(str(args.components), allowed=COMPONENTS) or list(COMPONENTS)
    patch_pairs = _parse_patch_pairs(str(args.patch_pairs) or default_patch_pairs)

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
    layer_specs, n_model_layers = _resolve_layer_specs(model, layer_labels)

    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
    out_root = (REPO_ROOT / str(args.out_root)).resolve() / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running Telugu continuation component panel: model={args.model} pair={args.pair} items={len(eval_rows)} layer_labels={[spec['label'] for spec in layer_specs]} components={components} patch_pairs={patch_pairs}",
        flush=True,
    )

    item_rows: List[Dict[str, Any]] = []
    skipped_items: List[Dict[str, Any]] = []

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
        gold_text = normalize_text(str(word["hindi"]))
        competitor_text = normalize_text(str(competitor["target"]))
        gold_ids = [int(x) for x in tokenizer.encode(gold_text, add_special_tokens=False)]
        competitor_ids = [int(x) for x in tokenizer.encode(competitor_text, add_special_tokens=False)]

        if not gold_ids or not competitor_ids:
            skipped_items.append({
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "gold": gold_text,
                "nearest_bank_target": competitor_text,
                "reason": "empty_tokenization",
            })
            continue

        prefix_len = int(_common_prefix_len(gold_ids, competitor_ids))
        if prefix_len < int(args.min_shared_prefix):
            skipped_items.append({
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "gold": gold_text,
                "nearest_bank_target": competitor_text,
                "shared_prefix_len": prefix_len,
                "reason": "shared_prefix_too_short",
            })
            continue
        if prefix_len >= min(len(gold_ids), len(competitor_ids)):
            skipped_items.append({
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "gold": gold_text,
                "nearest_bank_target": competitor_text,
                "shared_prefix_len": prefix_len,
                "reason": "no_divergence_after_shared_prefix",
            })
            continue

        shared_prefix_ids = gold_ids[:prefix_len]
        gold_next_id = int(gold_ids[prefix_len])
        competitor_next_id = int(competitor_ids[prefix_len])

        full_ids_by_condition: Dict[str, torch.Tensor] = {}
        base_by_condition: Dict[str, Dict[str, Any]] = {}
        donor_vectors: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {c: {} for c in CONDITIONS}
        patch_pos_by_condition: Dict[str, int] = {}

        for condition in CONDITIONS:
            rendered = apply_chat_template(tokenizer, str(prompts[condition]))
            prompt_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            if shared_prefix_ids:
                shared_prefix_tensor = torch.tensor([shared_prefix_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
                full_ids = torch.cat([prompt_ids, shared_prefix_tensor], dim=1)
            else:
                full_ids = prompt_ids
            full_ids_by_condition[condition] = full_ids
            patch_pos_by_condition[condition] = int(full_ids.shape[1] - 1)
            base_by_condition[condition] = _divergence_step_stats(
                model=model,
                input_ids=full_ids,
                tokenizer=tokenizer,
                gold_next_id=int(gold_next_id),
                competitor_next_id=int(competitor_next_id),
            )

        for donor_condition in CONDITIONS:
            donor_vectors[donor_condition] = {}
            donor_input_ids = full_ids_by_condition[donor_condition]
            donor_pos = int(patch_pos_by_condition[donor_condition])
            for layer_spec in layer_specs:
                layer_label = int(layer_spec["label"])
                layer_index = int(layer_spec["index"])
                donor_vectors[donor_condition][layer_label] = {}
                for component in components:
                    donor_vectors[donor_condition][layer_label][component] = _extract_component_vector(
                        model=model,
                        component=component,
                        input_ids=donor_input_ids,
                        layer=int(layer_index),
                        position=int(donor_pos),
                    )

        interventions: List[Dict[str, Any]] = []
        for recipient_condition, donor_condition in patch_pairs:
            recipient_input_ids = full_ids_by_condition[recipient_condition]
            recipient_pos = int(patch_pos_by_condition[recipient_condition])
            base_stats = base_by_condition[recipient_condition]
            for layer_spec in layer_specs:
                layer_label = int(layer_spec["label"])
                layer_index = int(layer_spec["index"])
                for component in components:
                    patch_vector = donor_vectors[donor_condition][layer_label][component]
                    hook = _build_patch_hook(
                        model=model,
                        component=component,
                        layer=int(layer_index),
                        patch_vector=patch_vector,
                        patch_position=int(recipient_pos),
                    )
                    patched_stats = _divergence_step_stats(
                        model=model,
                        input_ids=recipient_input_ids,
                        tokenizer=tokenizer,
                        gold_next_id=int(gold_next_id),
                        competitor_next_id=int(competitor_next_id),
                        hooks=[hook],
                    )
                    interventions.append({
                        "recipient_condition": str(recipient_condition),
                        "donor_condition": str(donor_condition),
                        "layer": int(layer_label),
                        "layer_index": int(layer_index),
                        "component": str(component),
                        "base": base_stats,
                        "patched": patched_stats,
                        "delta": {
                            "gold_next_prob": float(patched_stats["gold_next_prob"] - base_stats["gold_next_prob"]),
                            "competitor_next_prob": float(patched_stats["competitor_next_prob"] - base_stats["competitor_next_prob"]),
                            "gold_minus_competitor_logit": float(patched_stats["gold_minus_competitor_logit"] - base_stats["gold_minus_competitor_logit"]),
                            "top1_is_gold": float(float(patched_stats["top1_is_gold"]) - float(base_stats["top1_is_gold"])),
                            "top1_is_competitor": float(float(patched_stats["top1_is_competitor"]) - float(base_stats["top1_is_competitor"])),
                        },
                    })

        item_rows.append({
            "item_index": int(item_idx - 1),
            "word_ood": str(word["ood"]),
            "word_target": gold_text,
            "nearest_bank_target": competitor_text,
            "nearest_bank_source": str(competitor["source"]),
            "nearest_bank_rank": int(competitor["position"] + 1),
            "nearest_bank_similarity": float(competitor["similarity"]),
            "shared_prefix_len_tokens": int(prefix_len),
            "shared_prefix_text": tokenizer.decode(shared_prefix_ids).strip(),
            "divergence_token_index": int(prefix_len),
            "gold_next_token_text": _token_text(tokenizer, int(gold_next_id)),
            "competitor_next_token_text": _token_text(tokenizer, int(competitor_next_id)),
            "patch_position_by_condition": patch_pos_by_condition,
            "base_by_condition": base_by_condition,
            "interventions": interventions,
        })

    summary_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in patch_pairs:
        for layer_spec in layer_specs:
            layer_label = int(layer_spec["label"])
            layer_index = int(layer_spec["index"])
            for component in components:
                rows = []
                for item in item_rows:
                    for intr in item["interventions"]:
                        if (
                            intr["recipient_condition"] == recipient_condition
                            and intr["donor_condition"] == donor_condition
                            and int(intr["layer"]) == int(layer_label)
                            and intr["component"] == component
                        ):
                            rows.append((item, intr))
                if not rows:
                    continue
                base_gold_top1 = [1.0 if bool(intr["base"]["top1_is_gold"]) else 0.0 for _item, intr in rows]
                patched_gold_top1 = [1.0 if bool(intr["patched"]["top1_is_gold"]) else 0.0 for _item, intr in rows]
                base_comp_top1 = [1.0 if bool(intr["base"]["top1_is_competitor"]) else 0.0 for _item, intr in rows]
                patched_comp_top1 = [1.0 if bool(intr["patched"]["top1_is_competitor"]) else 0.0 for _item, intr in rows]
                base_comp_rows = [(item, intr) for item, intr in rows if bool(intr["base"]["top1_is_competitor"])]
                base_gold_rows = [(item, intr) for item, intr in rows if bool(intr["base"]["top1_is_gold"])]
                summary_rows.append({
                    "recipient_condition": str(recipient_condition),
                    "donor_condition": str(donor_condition),
                    "layer": int(layer_label),
                    "layer_index": int(layer_index),
                    "component": str(component),
                    "n_items": int(len(rows)),
                    "base_mean_gap": float(np.nanmean([intr["base"]["gold_minus_competitor_logit"] for _item, intr in rows])),
                    "patched_mean_gap": float(np.nanmean([intr["patched"]["gold_minus_competitor_logit"] for _item, intr in rows])),
                    "delta_mean_gap": float(np.nanmean([intr["delta"]["gold_minus_competitor_logit"] for _item, intr in rows])),
                    "base_gold_top1_rate": float(np.mean(base_gold_top1)),
                    "patched_gold_top1_rate": float(np.mean(patched_gold_top1)),
                    "delta_gold_top1_rate": float(np.mean(patched_gold_top1) - np.mean(base_gold_top1)),
                    "base_competitor_top1_rate": float(np.mean(base_comp_top1)),
                    "patched_competitor_top1_rate": float(np.mean(patched_comp_top1)),
                    "delta_competitor_top1_rate": float(np.mean(patched_comp_top1) - np.mean(base_comp_top1)),
                    "rescue_rate_on_base_competitor_top1": float(
                        np.mean([1.0 if bool(intr["patched"]["top1_is_gold"]) else 0.0 for _item, intr in base_comp_rows])
                    ) if base_comp_rows else 0.0,
                    "harm_rate_on_base_gold_top1": float(
                        np.mean([1.0 if not bool(intr["patched"]["top1_is_gold"]) else 0.0 for _item, intr in base_gold_rows])
                    ) if base_gold_rows else 0.0,
                })

    payload = {
        "experiment": "telugu_continuation_component_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "n_model_layers": int(n_model_layers),
        "layers": [int(spec["label"]) for spec in layer_specs],
        "layer_indices": [int(spec["index"]) for spec in layer_specs],
        "components": [str(x) for x in components],
        "patch_pairs": [{"recipient": r, "donor": d} for r, d in patch_pairs],
        "min_shared_prefix": int(args.min_shared_prefix),
        "n_items_used": int(len(item_rows)),
        "n_items_skipped": int(len(skipped_items)),
        "oracle": {
            "description": "At the divergence token between the gold continuation and nearest-bank continuation, which component and layer carry the donor-vs-recipient competition state?",
            "success_criterion": "Find a late layer/component where donor replacement moves the gold-vs-bank gap and top1 competition in the expected direction without relying on a tiny number of pathological items.",
            "failure_criterion": "If no component/layer meaningfully moves the divergence-token gap, the current localization is too diffuse or the donor/recipient contrast is wrong.",
        },
        "summary_rows": summary_rows,
        "item_rows": item_rows,
        "skipped_items": skipped_items,
    }
    out_path = out_root / "telugu_continuation_component_panel.json"
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
