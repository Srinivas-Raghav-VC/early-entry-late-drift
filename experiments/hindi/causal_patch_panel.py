#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRAFT_ROOT = PROJECT_ROOT / "lib"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import (  # noqa: E402
    _extract_attention_output_at_position_from_input_ids,
    _extract_layer_output_at_position_from_input_ids,
    _extract_mlp_io_at_position_from_input_ids,
    _find_last_subsequence,
    apply_chat_template,
    build_corrupted_icl_prompt,
    build_task_prompt,
    load_model,
    register_attention_output_replace_hook,
    register_dense_mlp_output_patch_hook,
    register_layer_output_replace_hook,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402


CONDITIONS = ("zs", "icl_helpful", "icl_corrupt")
COMPONENTS = ("layer_output", "attention_output", "mlp_output", "final_state")
DEFAULT_LAYERS = (20, 23, 24, 25)
DEFAULT_PATCHES = (
    ("icl_helpful", "zs"),
    ("zs", "icl_helpful"),
)
DEFAULT_SAME_LENGTH_PATCHES = (
    ("icl_helpful", "icl_corrupt"),
    ("icl_corrupt", "icl_helpful"),
)


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
    vals = [int(x.strip()) for x in str(raw or "").split(",") if x.strip()]
    return vals


def _parse_patch_pairs(raw: str) -> List[Tuple[str, str]]:
    """Parse recipient:donor patch-pair specs.

    Examples:
        "default" keeps the historical helpful<->zero-shot panel.
        "default,same_length" adds helpful<->corrupt prompt-state controls.
        "icl_helpful:icl_corrupt" runs one explicit same-length direction.
    """

    allowed = set(CONDITIONS)
    specs = [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]
    if not specs:
        specs = ["default"]
    pairs: List[Tuple[str, str]] = []

    def add_pair(recipient: str, donor: str) -> None:
        if recipient not in allowed or donor not in allowed:
            raise ValueError(
                f"Bad patch pair {recipient!r}:{donor!r}; expected conditions in {sorted(allowed)}"
            )
        pair = (recipient, donor)
        if pair not in pairs:
            pairs.append(pair)

    for spec in specs:
        lowered = spec.lower()
        if lowered == "default":
            for recipient, donor in DEFAULT_PATCHES:
                add_pair(recipient, donor)
            continue
        if lowered in {"same_length", "same-length", "helpful_corrupt"}:
            for recipient, donor in DEFAULT_SAME_LENGTH_PATCHES:
                add_pair(recipient, donor)
            continue
        parts = [x.strip() for x in spec.split(":")]
        if len(parts) != 2:
            raise ValueError(f"Bad patch-pair spec {spec!r}; expected recipient:donor, default, or same_length")
        add_pair(parts[0], parts[1])
    return pairs


def _detect_script(text: str) -> str:
    scripts: Counter[str] = Counter()
    for ch in str(text or ""):
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            name = ""
        if "DEVANAGARI" in name:
            scripts["devanagari"] += 1
        elif "TELUGU" in name:
            scripts["telugu"] += 1
        elif "BENGALI" in name or "BANGLA" in name:
            scripts["bengali"] += 1
        elif "TAMIL" in name:
            scripts["tamil"] += 1
        elif "LATIN" in name or ch.isascii():
            scripts["latin"] += 1
        else:
            scripts["other"] += 1
    if not scripts:
        return "unknown"
    return scripts.most_common(1)[0][0]


def _token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).replace("\n", " ").strip()


def _build_latin_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    flags: List[bool] = []
    for token_id in range(int(vocab_size)):
        text = _token_text(tokenizer, token_id)
        flags.append(_detect_script(text) == "latin")
    # Guarantee the mask is non-empty.
    if not any(flags):
        raise RuntimeError("Latin token mask is empty; token classification failed.")
    return torch.tensor(flags, dtype=torch.bool)


def _first_step_stats(
    *,
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_id: int,
    latin_mask: torch.Tensor,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, int(input_ids.shape[1] - 1), :].float()
        probs = torch.softmax(logits, dim=-1)

        top1_id = int(torch.argmax(logits).item())
        top1_text = _token_text(tokenizer, top1_id)

        logits_no_target = logits.clone()
        logits_no_target[int(target_id)] = -float("inf")
        competitor_id = int(torch.argmax(logits_no_target).item())
        competitor_text = _token_text(tokenizer, competitor_id)

        lat_mask = latin_mask.to(device=logits.device)
        logits_latin = logits.clone()
        logits_latin[~lat_mask] = -float("inf")
        latin_id = int(torch.argmax(logits_latin).item())
        latin_text = _token_text(tokenizer, latin_id)

        return {
            "target_id": int(target_id),
            "target_token_text": _token_text(tokenizer, int(target_id)),
            "target_prob": float(probs[int(target_id)].item()),
            "target_logit": float(logits[int(target_id)].item()),
            "top1_id": int(top1_id),
            "top1_token_text": top1_text,
            "top1_script": _detect_script(top1_text),
            "top1_prob": float(probs[top1_id].item()),
            "top1_is_target": bool(top1_id == int(target_id)),
            "competitor_id": int(competitor_id),
            "competitor_token_text": competitor_text,
            "competitor_script": _detect_script(competitor_text),
            "competitor_logit": float(logits[competitor_id].item()),
            "target_minus_competitor_logit": float(logits[int(target_id)].item() - logits[competitor_id].item()),
            "latin_competitor_id": int(latin_id),
            "latin_competitor_token_text": latin_text,
            "latin_competitor_script": _detect_script(latin_text),
            "latin_competitor_logit": float(logits[latin_id].item()),
            "target_minus_latin_logit": float(logits[int(target_id)].item() - logits[latin_id].item()),
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _register_final_state_replace_hook(
    model: Any,
    patch_vector: torch.Tensor,
    *,
    patch_position: int,
):
    norm_module = getattr(getattr(model, "model", None), "norm", None)
    if norm_module is None:
        raise AttributeError("Could not locate model.model.norm for final-state patching.")
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


def _extract_final_state_at_position_from_input_ids(
    model: Any,
    input_ids: torch.Tensor,
    position: int,
) -> torch.Tensor:
    norm_module = getattr(getattr(model, "model", None), "norm", None)
    if norm_module is None:
        raise AttributeError("Could not locate model.model.norm for final-state extraction.")
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hindi 1B causal patch panel around the localized 23-25 band.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--layers", type=str, default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--components", type=str, default=",".join(COMPONENTS))
    ap.add_argument(
        "--patch-pairs",
        type=str,
        default="default",
        help=(
            "Comma-separated recipient:donor patch pairs, or aliases 'default' "
            "(helpful<->zs) and 'same_length' (helpful<->corrupt). "
            "Example: --patch-pairs default,same_length"
        ),
    )
    ap.add_argument(
        "--patch-position-mode",
        type=str,
        default="query_last_subtoken",
        choices=["query_last_subtoken", "last_token"],
        help="Where to extract donor vectors and apply patches. For first-token competition at generation time, last_token is usually the correct locus.",
    )
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    layers = _parse_csv_int(str(args.layers)) or list(DEFAULT_LAYERS)
    components = _parse_csv_str(str(args.components), allowed=COMPONENTS) or list(COMPONENTS)
    patch_pairs = _parse_patch_pairs(str(args.patch_pairs))

    bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=True,
        require_external_sources=True,
        min_pool_size=500,
    )
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)
    layer_types = list(getattr(model.config, "layer_types", []))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "research" / "results" / "hindi_patch_panel_v1" / str(args.model) / str(args.pair) / f"nicl{int(args.n_icl)}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running Hindi causal patch panel: model={args.model} pair={args.pair} items={len(eval_rows)} layers={layers} components={components} patch_pairs={patch_pairs}",
        flush=True,
    )

    item_rows: List[Dict[str, Any]] = []

    for item_idx, word in enumerate(eval_rows, start=1):
        if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 5 == 0:
            print(f"[{item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)

        prompt_by_condition = {
            "zs": build_task_prompt(
                str(word["ood"]),
                None,
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                prompt_variant="canonical",
            ),
            "icl_helpful": build_task_prompt(
                str(word["ood"]),
                bundle["icl_examples"],
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                prompt_variant="canonical",
            ),
            "icl_corrupt": build_corrupted_icl_prompt(
                str(word["ood"]),
                bundle["icl_examples"],
                input_script_name=bundle["input_script_name"],
                source_language=bundle["source_language"],
                output_script_name=bundle["output_script_name"],
                seed=int(args.seed),
            ),
        }

        query_ids = tokenizer.encode(str(word["ood"]), add_special_tokens=False)
        if not query_ids:
            continue
        target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
        if not target_ids:
            continue
        target_id = int(target_ids[0])

        rendered_by_condition: Dict[str, str] = {}
        input_ids_by_condition: Dict[str, torch.Tensor] = {}
        query_pos_by_condition: Dict[str, int] = {}
        last_pos_by_condition: Dict[str, int] = {}
        pos_by_condition: Dict[str, int] = {}
        base_stats_by_condition: Dict[str, Dict[str, Any]] = {}

        for condition in CONDITIONS:
            rendered = apply_chat_template(tokenizer, str(prompt_by_condition[condition]))
            input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
            span = _find_last_subsequence(
                input_ids[0].detach().cpu().tolist(),
                [int(x) for x in query_ids],
            )
            if span is None:
                raise RuntimeError(f"Failed to locate query span for condition={condition} item={word['ood']}")
            rendered_by_condition[condition] = rendered
            input_ids_by_condition[condition] = input_ids
            query_pos_by_condition[condition] = int(span[1] - 1)
            last_pos_by_condition[condition] = int(input_ids.shape[1] - 1)
            pos_by_condition[condition] = (
                int(query_pos_by_condition[condition])
                if str(args.patch_position_mode) == "query_last_subtoken"
                else int(last_pos_by_condition[condition])
            )
            base_stats_by_condition[condition] = _first_step_stats(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
            )

        donor_vectors: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {cond: {} for cond in CONDITIONS}
        for donor_condition in CONDITIONS:
            donor_vectors[donor_condition] = {}
            for layer in layers:
                donor_vectors[donor_condition][int(layer)] = {}
                donor_input_ids = input_ids_by_condition[donor_condition]
                donor_pos = int(pos_by_condition[donor_condition])
                for component in components:
                    donor_vectors[donor_condition][int(layer)][component] = _extract_component_vector(
                        model=model,
                        component=component,
                        input_ids=donor_input_ids,
                        layer=int(layer),
                        position=donor_pos,
                    )

        interventions: List[Dict[str, Any]] = []
        for recipient_condition, donor_condition in patch_pairs:
            recipient_input_ids = input_ids_by_condition[recipient_condition]
            recipient_pos = int(pos_by_condition[recipient_condition])
            base_stats = base_stats_by_condition[recipient_condition]
            for layer in layers:
                for component in components:
                    layer_type = "final_norm" if component == "final_state" else (layer_types[int(layer)] if layer_types and int(layer) < len(layer_types) else "unknown")
                    patch_vector = donor_vectors[donor_condition][int(layer)][component]
                    hook = _build_patch_hook(
                        model=model,
                        component=component,
                        layer=int(layer),
                        patch_vector=patch_vector,
                        patch_position=int(recipient_pos),
                    )
                    patched_stats = _first_step_stats(
                        model=model,
                        input_ids=recipient_input_ids,
                        tokenizer=tokenizer,
                        target_id=target_id,
                        latin_mask=latin_mask,
                        hooks=[hook],
                    )
                    interventions.append(
                        {
                            "recipient_condition": str(recipient_condition),
                            "donor_condition": str(donor_condition),
                            "layer": int(layer),
                            "layer_type": str(layer_type),
                            "component": str(component),
                            "base": base_stats,
                            "patched": patched_stats,
                            "delta": {
                                "target_prob": float(patched_stats["target_prob"] - base_stats["target_prob"]),
                                "target_minus_competitor_logit": float(
                                    patched_stats["target_minus_competitor_logit"] - base_stats["target_minus_competitor_logit"]
                                ),
                                "target_minus_latin_logit": float(
                                    patched_stats["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]
                                ),
                                "top1_is_target": float(float(patched_stats["top1_is_target"]) - float(base_stats["top1_is_target"])),
                            },
                        }
                    )

        item_rows.append(
            {
                "model": str(args.model),
                "pair": str(args.pair),
                "seed": int(args.seed),
                "n_icl": int(args.n_icl),
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "target_id": int(target_id),
                "target_token_text": _token_text(tokenizer, int(target_id)),
                "patch_position_mode": str(args.patch_position_mode),
                "query_position_by_condition": query_pos_by_condition,
                "last_position_by_condition": last_pos_by_condition,
                "patch_position_by_condition": pos_by_condition,
                "base_by_condition": base_stats_by_condition,
                "interventions": interventions,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in patch_pairs:
        for layer in layers:
            for component in components:
                layer_type = "final_norm" if component == "final_state" else (layer_types[int(layer)] if layer_types and int(layer) < len(layer_types) else "unknown")
                rows = []
                for item in item_rows:
                    for intr in item["interventions"]:
                        if (
                            intr["recipient_condition"] == recipient_condition
                            and int(intr["layer"]) == int(layer)
                            and intr["component"] == component
                            and intr["donor_condition"] == donor_condition
                        ):
                            rows.append((item, intr))
                if not rows:
                    continue
                base_success = [1.0 if bool(intr["base"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                patched_success = [1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                failed_base_rows = [(item, intr) for item, intr in rows if not bool(intr["base"]["top1_is_target"])]
                succeeded_base_rows = [(item, intr) for item, intr in rows if bool(intr["base"]["top1_is_target"])]
                patched_script_counts = Counter(str(intr["patched"]["top1_script"]) for _item, intr in rows)
                summary_rows.append(
                    {
                        "recipient_condition": str(recipient_condition),
                        "donor_condition": str(donor_condition),
                        "layer": int(layer),
                        "layer_type": str(layer_type),
                        "component": str(component),
                        "n_items": int(len(rows)),
                        "base_mean_target_prob": float(np.nanmean([intr["base"]["target_prob"] for _item, intr in rows])),
                        "patched_mean_target_prob": float(np.nanmean([intr["patched"]["target_prob"] for _item, intr in rows])),
                        "delta_mean_target_prob": float(np.nanmean([intr["delta"]["target_prob"] for _item, intr in rows])),
                        "base_mean_gap_overall": float(np.nanmean([intr["base"]["target_minus_competitor_logit"] for _item, intr in rows])),
                        "patched_mean_gap_overall": float(np.nanmean([intr["patched"]["target_minus_competitor_logit"] for _item, intr in rows])),
                        "delta_mean_gap_overall": float(np.nanmean([intr["delta"]["target_minus_competitor_logit"] for _item, intr in rows])),
                        "base_mean_gap_latin": float(np.nanmean([intr["base"]["target_minus_latin_logit"] for _item, intr in rows])),
                        "patched_mean_gap_latin": float(np.nanmean([intr["patched"]["target_minus_latin_logit"] for _item, intr in rows])),
                        "delta_mean_gap_latin": float(np.nanmean([intr["delta"]["target_minus_latin_logit"] for _item, intr in rows])),
                        "base_top1_target_rate": float(np.nanmean(base_success)),
                        "patched_top1_target_rate": float(np.nanmean(patched_success)),
                        "delta_top1_target_rate": float(np.nanmean([p - b for p, b in zip(patched_success, base_success)])),
                        "rescue_rate_on_base_failures": float(
                            np.nanmean([1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in failed_base_rows])
                        ) if failed_base_rows else None,
                        "harm_rate_on_base_successes": float(
                            np.nanmean([0.0 if bool(intr["patched"]["top1_is_target"]) else 1.0 for _item, intr in succeeded_base_rows])
                        ) if succeeded_base_rows else None,
                        "patched_top1_script_counts": dict(patched_script_counts),
                    }
                )

    payload = {
        "experiment": "hindi_1b_causal_patch_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "layers": [int(x) for x in layers],
        "components": list(components),
        "patches": [{"recipient_condition": r, "donor_condition": d} for r, d in patch_pairs],
        "layer_types": {str(i): str(t) for i, t in enumerate(layer_types)} if layer_types else {},
        "latin_mask_count": int(latin_mask.sum().item()),
        "patch_position_mode": str(args.patch_position_mode),
        "oracle": {
            "description": "Localized causal replacement around the Hindi 1B band using target-vs-Latin and target-vs-overall competitor margins.",
            "expected_signal": "If the harmful high-shot state is localized at the patched locus, replacing helpful states with zs states should improve the gold-vs-Latin margin and target rate, while replacing zs states with helpful states should harm them.",
            "control_expectation": "Off-band layers should be weaker than the main band; last-token patching is the correct direct test for next-token competition, while query-position patching only tests earlier mediation.",
        },
        "summary_rows": summary_rows,
        "item_rows": item_rows,
    }

    out_path = out_root / "hindi_1b_causal_patch_panel.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)

    print("\n=== Top summary rows by helpful<-zs delta_mean_gap_latin ===", flush=True)
    top_rows = [
        row for row in summary_rows
        if row["recipient_condition"] == "icl_helpful" and row["donor_condition"] == "zs"
    ]
    top_rows = sorted(top_rows, key=lambda row: float(row["delta_mean_gap_latin"]), reverse=True)
    for row in top_rows[:8]:
        print(
            f"  L{row['layer']:02d} {row['component']:>16s} {row['layer_type']:<18s} "
            f"dgapLatin={row['delta_mean_gap_latin']:.3f} dtop1={row['delta_top1_target_rate']:.3f} "
            f"rescue_fail={row['rescue_rate_on_base_failures']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
