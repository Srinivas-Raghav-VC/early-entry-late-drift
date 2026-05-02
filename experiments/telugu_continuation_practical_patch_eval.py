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
    _extract_layer_output_at_position_from_input_ids,
    apply_chat_template,
    get_model_layers,
    load_model,
    register_layer_output_replace_hook,
    set_all_seeds,
)
from experiments.hindi_1b_practical_patch_eval import _bootstrap_mean_ci, _json_safe  # noqa: E402
from paper2_fidelity_calibrated.eval_utils import akshara_cer, exact_match, normalize_text  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split, log  # noqa: E402
from paper2_fidelity_calibrated.run_neutral_filler_recency_controls import _condition_prompts  # noqa: E402
from research.modules.eval.output_extraction import (  # noqa: E402
    analyze_generation_text,
    extract_transliteration_candidate,
    resolve_generation_stop_ids,
    resolve_pad_token_id,
)

CONDITIONS = [
    "zs",
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_reversed",
    "icl_corrupt",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_site(raw: str) -> Dict[str, Any]:
    parts = [x.strip() for x in str(raw).split(":") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Bad site spec '{raw}'. Expected <layer>:layer_output.")
    layer = int(parts[0])
    component = str(parts[1])
    if component != "layer_output":
        raise ValueError("This practical Telugu patch currently supports layer_output only.")
    return {"layer": layer, "component": component, "name": f"L{layer}_{component}"}


def _token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).replace("\n", " ").strip()


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


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


def _build_layer_output_add_hook(*, model: Any, layer_label: int, delta_vector: torch.Tensor, patch_position: int):
    layers = get_model_layers(model)
    delta_vector = delta_vector.detach()

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output
        seq_len = int(y.shape[1])
        if int(patch_position) >= seq_len:
            return output
        dv = delta_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if dv.shape[1] != y.shape[2]:
            return output
        y_new = y.clone()
        y_new[:, int(patch_position), :] = y_new[:, int(patch_position), :] + dv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[int(layer_label) - 1].register_forward_hook(hook)


def _extract_layer_output(*, model: Any, input_ids: torch.Tensor, layer_label: int, position: int) -> torch.Tensor:
    return _extract_layer_output_at_position_from_input_ids(model, input_ids, int(layer_label) - 1, int(position))


def _fixed_random_delta(*, hidden_size: int, target_norm: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    rand = torch.tensor(rng.standard_normal(size=(int(hidden_size),)), dtype=torch.float32)
    rand = rand / torch.linalg.vector_norm(rand)
    return rand * float(target_norm)


def _generate_continuation(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_ids: int | list[int],
    pad_id: int,
    hooks: Optional[List[Any]] = None,
) -> str:
    active = list(hooks or [])
    try:
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=False,
                eos_token_id=stop_ids,
                pad_token_id=int(pad_id),
            )
        new_tokens = out[0, int(input_ids.shape[1]) :]
        return normalize_text(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _evaluate_generation(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    output_script_name: str,
    shared_prefix_text: str,
    gold_text: str,
    gold_suffix_text: str,
    competitor_text: str,
    competitor_suffix_text: str,
    max_new_tokens: int,
    stop_ids: int | list[int],
    pad_id: int,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    raw_continuation = _generate_continuation(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        pad_id=pad_id,
        hooks=hooks,
    )
    extracted_continuation = extract_transliteration_candidate(
        raw_continuation,
        script_name=str(output_script_name),
        min_script_ratio=0.80,
    )
    audit = analyze_generation_text(raw_continuation, extracted_continuation)
    full_prediction = normalize_text(str(shared_prefix_text) + str(extracted_continuation))
    return {
        "raw_continuation": raw_continuation,
        "generated_continuation": extracted_continuation,
        "full_prediction": full_prediction,
        "full_exact_match": float(exact_match(full_prediction, gold_text)),
        "full_akshara_cer": float(akshara_cer(full_prediction, gold_text)),
        "continuation_exact_match": float(exact_match(extracted_continuation, gold_suffix_text)),
        "continuation_akshara_cer": float(akshara_cer(extracted_continuation, gold_suffix_text)),
        "prediction_equals_bank": float(exact_match(full_prediction, competitor_text)),
        "prediction_equals_bank_suffix": float(exact_match(extracted_continuation, competitor_suffix_text)),
        "raw_strict_word_only": float(int(audit.get("strict_word_only", False))),
        "has_leading_text": float(int(audit.get("has_leading_text", False))),
        "has_trailing_text": float(int(audit.get("has_trailing_text", False))),
    }


def _build_item_setup(
    *,
    model: Any,
    tokenizer: Any,
    bundle: Mapping[str, Any],
    word: Mapping[str, Any],
    seed: int,
    recipient: str,
    donor: str,
) -> Optional[Dict[str, Any]]:
    prompts, meta = _condition_prompts(
        tokenizer=tokenizer,
        query=str(word["ood"]),
        icl_examples=bundle["icl_examples"],
        input_script_name=bundle["input_script_name"],
        source_language=bundle["source_language"],
        output_script_name=bundle["output_script_name"],
        seed=int(seed),
    )
    competitor = _choose_bank_competitor(meta, str(word["hindi"]))
    gold_text = normalize_text(str(word["hindi"]))
    competitor_text = normalize_text(str(competitor["target"]))
    gold_ids = [int(x) for x in tokenizer.encode(gold_text, add_special_tokens=False)]
    competitor_ids = [int(x) for x in tokenizer.encode(competitor_text, add_special_tokens=False)]
    if not gold_ids or not competitor_ids:
        return None

    prefix_len = int(_common_prefix_len(gold_ids, competitor_ids))
    if prefix_len < 1 or prefix_len >= min(len(gold_ids), len(competitor_ids)):
        return None

    shared_prefix_ids = gold_ids[:prefix_len]
    gold_next_id = int(gold_ids[prefix_len])
    competitor_next_id = int(competitor_ids[prefix_len])
    gold_suffix_text = normalize_text(tokenizer.decode(gold_ids[prefix_len:], skip_special_tokens=True).strip())
    competitor_suffix_text = normalize_text(tokenizer.decode(competitor_ids[prefix_len:], skip_special_tokens=True).strip())
    shared_prefix_text = normalize_text(tokenizer.decode(shared_prefix_ids, skip_special_tokens=True).strip())

    full_ids_by_condition: Dict[str, torch.Tensor] = {}
    patch_pos_by_condition: Dict[str, int] = {}
    base_by_condition: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        rendered = apply_chat_template(tokenizer, str(prompts[condition]))
        prompt_ids = tokenizer(rendered, return_tensors="pt").to(next(model.parameters()).device).input_ids
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

    return {
        "word_ood": str(word["ood"]),
        "gold_text": gold_text,
        "competitor_text": competitor_text,
        "nearest_bank_source": str(competitor["source"]),
        "nearest_bank_rank": int(competitor["position"] + 1),
        "nearest_bank_similarity": float(competitor["similarity"]),
        "shared_prefix_len_tokens": int(prefix_len),
        "shared_prefix_text": shared_prefix_text,
        "gold_suffix_text": gold_suffix_text,
        "competitor_suffix_text": competitor_suffix_text,
        "gold_next_id": int(gold_next_id),
        "competitor_next_id": int(competitor_next_id),
        "full_ids_by_condition": full_ids_by_condition,
        "patch_pos_by_condition": patch_pos_by_condition,
        "base_by_condition": base_by_condition,
        "recipient": str(recipient),
        "donor": str(donor),
    }


def _mean_delta_vector(
    *,
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    bundle: Mapping[str, Any],
    seed: int,
    recipient: str,
    donor: str,
    site: Mapping[str, Any],
) -> Dict[str, Any]:
    deltas: List[np.ndarray] = []
    recipient_vecs: List[np.ndarray] = []
    donor_vecs: List[np.ndarray] = []
    used = 0
    for word in rows:
        setup = _build_item_setup(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            word=word,
            seed=int(seed),
            recipient=recipient,
            donor=donor,
        )
        if setup is None:
            continue
        recipient_ids = setup["full_ids_by_condition"][recipient]
        donor_ids = setup["full_ids_by_condition"][donor]
        recipient_pos = int(setup["patch_pos_by_condition"][recipient])
        donor_pos = int(setup["patch_pos_by_condition"][donor])
        recipient_vec = _extract_layer_output(model=model, input_ids=recipient_ids, layer_label=int(site["layer"]), position=int(recipient_pos))
        donor_vec = _extract_layer_output(model=model, input_ids=donor_ids, layer_label=int(site["layer"]), position=int(donor_pos))
        recipient_np = recipient_vec.detach().float().cpu().numpy().astype(np.float64)
        donor_np = donor_vec.detach().float().cpu().numpy().astype(np.float64)
        recipient_vecs.append(recipient_np)
        donor_vecs.append(donor_np)
        deltas.append((donor_np - recipient_np).astype(np.float64))
        used += 1
    if not deltas:
        raise RuntimeError("No usable items for Telugu mean-delta estimation.")
    delta_arr = np.stack(deltas, axis=0)
    recipient_arr = np.stack(recipient_vecs, axis=0)
    donor_arr = np.stack(donor_vecs, axis=0)
    mean_delta = delta_arr.mean(axis=0)
    mean_recipient = recipient_arr.mean(axis=0)
    mean_donor = donor_arr.mean(axis=0)
    return {
        "n_items": int(used),
        "hidden_size": int(delta_arr.shape[1]),
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta_arr))),
        "mean_delta": [float(x) for x in mean_delta.tolist()],
        "mean_recipient": [float(x) for x in mean_recipient.tolist()],
        "mean_donor": [float(x) for x in mean_donor.tolist()],
    }


def _selection_alpha_sweep(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    seed: int,
    recipient: str,
    donor: str,
    site: Mapping[str, Any],
    mean_delta: Sequence[float],
    alpha_grid: Sequence[float],
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    mean_delta_tensor = torch.tensor([float(x) for x in mean_delta], dtype=torch.float32)
    for alpha in alpha_grid:
        delta = mean_delta_tensor * float(alpha)
        gains_gap: List[float] = []
        gains_gold: List[float] = []
        gains_comp: List[float] = []
        patched_gold_top1: List[float] = []
        patched_comp_top1: List[float] = []
        used = 0
        for word in rows:
            setup = _build_item_setup(
                model=model,
                tokenizer=tokenizer,
                bundle=bundle,
                word=word,
                seed=int(seed),
                recipient=recipient,
                donor=donor,
            )
            if setup is None:
                continue
            recipient_ids = setup["full_ids_by_condition"][recipient]
            recipient_pos = int(setup["patch_pos_by_condition"][recipient])
            base = setup["base_by_condition"][recipient]
            hook = _build_layer_output_add_hook(
                model=model,
                layer_label=int(site["layer"]),
                delta_vector=delta,
                patch_position=int(recipient_pos),
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
            patched_gold_top1.append(float(patched["top1_is_gold"]))
            patched_comp_top1.append(float(patched["top1_is_competitor"]))
            used += 1
        out_rows.append(
            {
                "alpha": float(alpha),
                "n_items": int(used),
                "delta_mean_gap": float(np.mean(gains_gap)) if gains_gap else float("nan"),
                "delta_mean_gold_next_prob": float(np.mean(gains_gold)) if gains_gold else float("nan"),
                "delta_mean_competitor_next_prob": float(np.mean(gains_comp)) if gains_comp else float("nan"),
                "patched_gold_top1_rate": float(np.mean(patched_gold_top1)) if patched_gold_top1 else float("nan"),
                "patched_competitor_top1_rate": float(np.mean(patched_comp_top1)) if patched_comp_top1 else float("nan"),
            }
        )
    return out_rows


def _summarize_intervention(
    *,
    name: str,
    items: Sequence[Mapping[str, Any]],
    key_prefix: str,
    base_prefix: str = "baseline_no_patch",
    seed: int,
) -> Dict[str, Any]:
    full_exact = [float(item[key_prefix]["generation"]["full_exact_match"]) for item in items]
    full_cer = [float(item[key_prefix]["generation"]["full_akshara_cer"]) for item in items]
    cont_exact = [float(item[key_prefix]["generation"]["continuation_exact_match"]) for item in items]
    cont_cer = [float(item[key_prefix]["generation"]["continuation_akshara_cer"]) for item in items]
    bank_rate = [float(item[key_prefix]["generation"]["prediction_equals_bank"]) for item in items]
    gap = [float(item[key_prefix]["first_step"]["gold_minus_competitor_logit"]) for item in items]
    gold_top1 = [float(item[key_prefix]["first_step"]["top1_is_gold"]) for item in items]
    comp_top1 = [float(item[key_prefix]["first_step"]["top1_is_competitor"]) for item in items]

    delta_full_exact = [float(item[key_prefix]["generation"]["full_exact_match"] - item[base_prefix]["generation"]["full_exact_match"]) for item in items]
    delta_full_cer_imp = [float(item[base_prefix]["generation"]["full_akshara_cer"] - item[key_prefix]["generation"]["full_akshara_cer"]) for item in items]
    delta_cont_exact = [float(item[key_prefix]["generation"]["continuation_exact_match"] - item[base_prefix]["generation"]["continuation_exact_match"]) for item in items]
    delta_cont_cer_imp = [float(item[base_prefix]["generation"]["continuation_akshara_cer"] - item[key_prefix]["generation"]["continuation_akshara_cer"]) for item in items]
    delta_bank = [float(item[key_prefix]["generation"]["prediction_equals_bank"] - item[base_prefix]["generation"]["prediction_equals_bank"]) for item in items]
    delta_gap = [float(item[key_prefix]["first_step"]["gold_minus_competitor_logit"] - item[base_prefix]["first_step"]["gold_minus_competitor_logit"]) for item in items]
    delta_gold_top1 = [float(item[key_prefix]["first_step"]["top1_is_gold"] - item[base_prefix]["first_step"]["top1_is_gold"]) for item in items]
    delta_comp_top1 = [float(item[key_prefix]["first_step"]["top1_is_competitor"] - item[base_prefix]["first_step"]["top1_is_competitor"]) for item in items]

    base_bank = [item for item in items if float(item[base_prefix]["generation"]["prediction_equals_bank"]) > 0.5]
    base_full_exact = [item for item in items if float(item[base_prefix]["generation"]["full_exact_match"]) > 0.5]
    base_comp = [item for item in items if bool(item[base_prefix]["first_step"]["top1_is_competitor"])]
    base_gold = [item for item in items if bool(item[base_prefix]["first_step"]["top1_is_gold"])]

    return {
        "intervention": str(name),
        "n_items": int(len(items)),
        "full_exact_match": _bootstrap_mean_ci(full_exact, seed=seed + 11),
        "full_akshara_cer": _bootstrap_mean_ci(full_cer, seed=seed + 17),
        "continuation_exact_match": _bootstrap_mean_ci(cont_exact, seed=seed + 23),
        "continuation_akshara_cer": _bootstrap_mean_ci(cont_cer, seed=seed + 29),
        "bank_copy_rate": _bootstrap_mean_ci(bank_rate, seed=seed + 31),
        "first_divergence_gap": _bootstrap_mean_ci(gap, seed=seed + 37),
        "first_divergence_gold_top1_rate": _bootstrap_mean_ci(gold_top1, seed=seed + 41),
        "first_divergence_competitor_top1_rate": _bootstrap_mean_ci(comp_top1, seed=seed + 43),
        "delta_full_exact_match": _bootstrap_mean_ci(delta_full_exact, seed=seed + 47),
        "delta_full_cer_improvement": _bootstrap_mean_ci(delta_full_cer_imp, seed=seed + 53),
        "delta_continuation_exact_match": _bootstrap_mean_ci(delta_cont_exact, seed=seed + 59),
        "delta_continuation_cer_improvement": _bootstrap_mean_ci(delta_cont_cer_imp, seed=seed + 61),
        "delta_bank_copy_rate": _bootstrap_mean_ci(delta_bank, seed=seed + 67),
        "delta_first_divergence_gap": _bootstrap_mean_ci(delta_gap, seed=seed + 71),
        "delta_first_divergence_gold_top1_rate": _bootstrap_mean_ci(delta_gold_top1, seed=seed + 73),
        "delta_first_divergence_competitor_top1_rate": _bootstrap_mean_ci(delta_comp_top1, seed=seed + 79),
        "rescue_rate_on_base_bank_copy": float(np.mean([1.0 - float(item[key_prefix]["generation"]["prediction_equals_bank"]) for item in base_bank])) if base_bank else float("nan"),
        "harm_rate_on_base_full_exact": float(np.mean([1.0 - float(item[key_prefix]["generation"]["full_exact_match"]) for item in base_full_exact])) if base_full_exact else float("nan"),
        "rescue_rate_on_base_competitor_top1": float(np.mean([float(item[key_prefix]["first_step"]["top1_is_gold"]) for item in base_comp])) if base_comp else float("nan"),
        "harm_rate_on_base_gold_top1": float(np.mean([1.0 - float(item[key_prefix]["first_step"]["top1_is_gold"]) for item in base_gold])) if base_gold else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Practical Telugu late-state steering eval from a fixed mean delta at the final continuation bottleneck.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--select-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--recipient", type=str, default="")
    ap.add_argument("--donor", type=str, default="")
    ap.add_argument("--site", type=str, default="")
    ap.add_argument("--offband-site", type=str, default="")
    ap.add_argument("--alpha-grid", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    default_recipient = "icl_helpful" if str(args.model) == "1b" else "icl_corrupt"
    default_donor = "zs" if str(args.model) == "1b" else "icl_helpful"
    default_site = "26:layer_output" if str(args.model) == "1b" else "34:layer_output"
    default_offband = "10:layer_output" if str(args.model) == "1b" else "20:layer_output"

    recipient = str(args.recipient or default_recipient)
    donor = str(args.donor or default_donor)
    if recipient not in CONDITIONS or donor not in CONDITIONS:
        raise ValueError(f"Bad recipient/donor: {recipient}, {donor}")
    site = _parse_site(str(args.site or default_site))
    offband_site = _parse_site(str(args.offband_site or default_offband))
    alpha_grid = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    if not alpha_grid:
        raise ValueError("alpha-grid is empty")

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

    mean_delta = _mean_delta_vector(
        model=model,
        rows=select_rows,
        tokenizer=tokenizer,
        bundle=bundle,
        seed=int(args.seed),
        recipient=recipient,
        donor=donor,
        site=site,
    )
    selection_rows = _selection_alpha_sweep(
        model=model,
        tokenizer=tokenizer,
        rows=select_rows,
        bundle=bundle,
        seed=int(args.seed),
        recipient=recipient,
        donor=donor,
        site=site,
        mean_delta=mean_delta["mean_delta"],
        alpha_grid=alpha_grid,
    )
    best_row = max(selection_rows, key=lambda row: (float(row["delta_mean_gap"]), float(row["delta_mean_gold_next_prob"])))
    selected_alpha = float(best_row["alpha"])
    chosen_delta = torch.tensor([float(x) for x in mean_delta["mean_delta"]], dtype=torch.float32) * float(selected_alpha)
    random_delta = _fixed_random_delta(hidden_size=int(mean_delta["hidden_size"]), target_norm=float(torch.linalg.vector_norm(chosen_delta).item()), seed=int(args.seed) * 8191 + int(site["layer"]))

    print(
        f"Running Telugu practical patch eval: model={args.model} pair={args.pair} recipient={recipient} donor={donor} select_items={len(select_rows)} eval_items={len(eval_rows)} site={site['name']} alpha={selected_alpha}",
        flush=True,
    )

    item_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_rows, start=1):
        if idx == 1 or idx == len(eval_rows) or idx % 5 == 0:
            log(f"[eval {idx}/{len(eval_rows)}] {args.model} {args.pair} :: {word['ood']} -> {word['hindi']}")
        setup = _build_item_setup(
            model=model,
            tokenizer=tokenizer,
            bundle=bundle,
            word=word,
            seed=int(args.seed),
            recipient=recipient,
            donor=donor,
        )
        if setup is None:
            skipped_rows.append({"item_index": int(idx - 1), "word_ood": str(word["ood"]), "reason": "no_usable_shared_prefix_divergence"})
            continue

        recipient_ids = setup["full_ids_by_condition"][recipient]
        recipient_pos = int(setup["patch_pos_by_condition"][recipient])
        base_first = setup["base_by_condition"][recipient]
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
        item_payload: Dict[str, Any] = {
            "item_index": int(idx - 1),
            "word_ood": str(setup["word_ood"]),
            "word_target": str(setup["gold_text"]),
            "nearest_bank_target": str(setup["competitor_text"]),
            "nearest_bank_source": str(setup["nearest_bank_source"]),
            "nearest_bank_rank": int(setup["nearest_bank_rank"]),
            "nearest_bank_similarity": float(setup["nearest_bank_similarity"]),
            "shared_prefix_len_tokens": int(setup["shared_prefix_len_tokens"]),
            "shared_prefix_text": str(setup["shared_prefix_text"]),
            "baseline_no_patch": {"first_step": base_first, "generation": base_gen},
        }

        specs: List[Tuple[str, Dict[str, Any]]] = [
            (
                "chosen_mean_shift",
                {
                    "hook_kind": "add",
                    "hook_site": site,
                    "delta": chosen_delta,
                },
            ),
            (
                "chosen_sign_flip",
                {
                    "hook_kind": "add",
                    "hook_site": site,
                    "delta": -chosen_delta,
                },
            ),
            (
                "offband_mean_shift",
                {
                    "hook_kind": "add",
                    "hook_site": offband_site,
                    "delta": chosen_delta,
                },
            ),
            (
                "site_random_delta",
                {
                    "hook_kind": "add",
                    "hook_site": site,
                    "delta": random_delta,
                },
            ),
            (
                "site_mean_donor_replace",
                {
                    "hook_kind": "replace",
                    "hook_site": site,
                    "patch_vector": torch.tensor(mean_delta["mean_donor"], dtype=torch.float32),
                },
            ),
        ]

        for name, spec in specs:
            if spec["hook_kind"] == "add":
                hook = _build_layer_output_add_hook(
                    model=model,
                    layer_label=int(spec["hook_site"]["layer"]),
                    delta_vector=spec["delta"],
                    patch_position=int(recipient_pos),
                )
            else:
                hook = register_layer_output_replace_hook(
                    model,
                    int(spec["hook_site"]["layer"] - 1),
                    spec["patch_vector"],
                    patch_position=int(recipient_pos),
                )
            patched_first = _divergence_step_stats(
                model=model,
                input_ids=recipient_ids,
                tokenizer=tokenizer,
                gold_next_id=int(setup["gold_next_id"]),
                competitor_next_id=int(setup["competitor_next_id"]),
                hooks=[hook],
            )
            if spec["hook_kind"] == "add":
                hook = _build_layer_output_add_hook(
                    model=model,
                    layer_label=int(spec["hook_site"]["layer"]),
                    delta_vector=spec["delta"],
                    patch_position=int(recipient_pos),
                )
            else:
                hook = register_layer_output_replace_hook(
                    model,
                    int(spec["hook_site"]["layer"] - 1),
                    spec["patch_vector"],
                    patch_position=int(recipient_pos),
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
            item_payload[str(name)] = {
                "hook_site": spec["hook_site"],
                "first_step": patched_first,
                "generation": patched_gen,
            }

        item_rows.append(item_payload)

    summary_rows = [
        _summarize_intervention(name="baseline_no_patch", items=item_rows, key_prefix="baseline_no_patch", seed=int(args.seed)),
        _summarize_intervention(name="chosen_mean_shift", items=item_rows, key_prefix="chosen_mean_shift", seed=int(args.seed) + 100),
        _summarize_intervention(name="chosen_sign_flip", items=item_rows, key_prefix="chosen_sign_flip", seed=int(args.seed) + 200),
        _summarize_intervention(name="offband_mean_shift", items=item_rows, key_prefix="offband_mean_shift", seed=int(args.seed) + 300),
        _summarize_intervention(name="site_random_delta", items=item_rows, key_prefix="site_random_delta", seed=int(args.seed) + 400),
        _summarize_intervention(name="site_mean_donor_replace", items=item_rows, key_prefix="site_mean_donor_replace", seed=int(args.seed) + 500),
    ]

    summary_map = {row["intervention"]: row for row in summary_rows}
    comparison_rows = [
        {
            "comparison": "chosen_minus_baseline",
            "delta_full_exact_match": summary_map["chosen_mean_shift"]["delta_full_exact_match"],
            "delta_full_cer_improvement": summary_map["chosen_mean_shift"]["delta_full_cer_improvement"],
            "delta_continuation_exact_match": summary_map["chosen_mean_shift"]["delta_continuation_exact_match"],
            "delta_continuation_cer_improvement": summary_map["chosen_mean_shift"]["delta_continuation_cer_improvement"],
            "delta_bank_copy_rate": summary_map["chosen_mean_shift"]["delta_bank_copy_rate"],
            "delta_first_divergence_gap": summary_map["chosen_mean_shift"]["delta_first_divergence_gap"],
        },
        {
            "comparison": "chosen_minus_offband_full_exact",
            "value": float(summary_map["chosen_mean_shift"]["full_exact_match"]["mean"] - summary_map["offband_mean_shift"]["full_exact_match"]["mean"]),
        },
        {
            "comparison": "chosen_minus_random_full_exact",
            "value": float(summary_map["chosen_mean_shift"]["full_exact_match"]["mean"] - summary_map["site_random_delta"]["full_exact_match"]["mean"]),
        },
        {
            "comparison": "chosen_minus_signflip_full_exact",
            "value": float(summary_map["chosen_mean_shift"]["full_exact_match"]["mean"] - summary_map["chosen_sign_flip"]["full_exact_match"]["mean"]),
        },
        {
            "comparison": "mean_replace_minus_baseline",
            "delta_full_exact_match": summary_map["site_mean_donor_replace"]["delta_full_exact_match"],
            "delta_full_cer_improvement": summary_map["site_mean_donor_replace"]["delta_full_cer_improvement"],
            "delta_continuation_exact_match": summary_map["site_mean_donor_replace"]["delta_continuation_exact_match"],
            "delta_continuation_cer_improvement": summary_map["site_mean_donor_replace"]["delta_continuation_cer_improvement"],
            "delta_bank_copy_rate": summary_map["site_mean_donor_replace"]["delta_bank_copy_rate"],
            "delta_first_divergence_gap": summary_map["site_mean_donor_replace"]["delta_first_divergence_gap"],
        },
    ]

    payload = {
        "experiment": "telugu_continuation_practical_patch_eval",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "select_max_items": int(args.select_max_items),
        "max_items": int(args.max_items),
        "recipient": str(recipient),
        "donor": str(donor),
        "site": site,
        "offband_site": offband_site,
        "alpha_grid": [float(a) for a in alpha_grid],
        "selected_alpha": float(selected_alpha),
        "selected_mean_delta": mean_delta,
        "selection_rows": selection_rows,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "item_rows": item_rows,
        "skipped_rows": skipped_rows,
        "oracle": {
            "description": "Can a fixed mean delta at the final Telugu continuation bottleneck improve held-out continuation generation without per-example donor states?",
            "success_criterion": "Chosen site mean-shift improves continuation/full metrics and beats sign-flip, off-band, and random-direction controls on held-out items.",
            "failure_criterion": "If chosen mean-shift is comparable to off-band or random controls, or only changes first-step diagnostics without generation improvement, the practical steering claim is weak.",
        },
    }

    out_root = (
        Path(args.out_root).resolve()
        if str(args.out_root).strip()
        else REPO_ROOT / "research" / "results" / "autoresearch" / "telugu_continuation_practical_patch_eval_v1" / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "telugu_continuation_practical_patch_eval.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("=== Eval summary ===", flush=True)
    for row in summary_rows:
        print(
            f"  {row['intervention']:<22} fullEM={row['full_exact_match']['mean']:.3f} fullCER={row['full_akshara_cer']['mean']:.3f} "
            f"contEM={row['continuation_exact_match']['mean']:.3f} contCER={row['continuation_akshara_cer']['mean']:.3f} "
            f"bank={row['bank_copy_rate']['mean']:.3f} dGap={row['delta_first_divergence_gap']['mean']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
