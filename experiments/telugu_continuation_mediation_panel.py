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
COMPONENTS = ["layer_output", "attention_output", "mlp_output"]


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


def _parse_site_spec(raw: str) -> Dict[str, Any]:
    parts = [x.strip() for x in str(raw).split(":") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Bad site spec '{raw}'. Expected <layer>:<component>.")
    layer = int(parts[0])
    component = str(parts[1])
    if component not in COMPONENTS:
        raise ValueError(f"Bad component in '{raw}'. Allowed: {COMPONENTS}")
    return {"layer": layer, "component": component, "name": f"L{layer}_{component}"}


def _parse_site_list(raw: str) -> List[Dict[str, Any]]:
    vals = [_parse_site_spec(x.strip()) for x in str(raw or "").split(";") if x.strip()]
    if not vals:
        raise ValueError("No site specs parsed.")
    return vals


def _build_patch_hook(
    *,
    model: Any,
    component: str,
    layer_label: int,
    patch_vector: torch.Tensor,
    patch_position: int,
):
    layer_index = int(layer_label) - 1
    if component == "layer_output":
        return register_layer_output_replace_hook(model, int(layer_index), patch_vector, patch_position=int(patch_position))
    if component == "attention_output":
        return register_attention_output_replace_hook(model, int(layer_index), patch_vector, patch_position=int(patch_position))
    if component == "mlp_output":
        return register_dense_mlp_output_patch_hook(model, int(layer_index), patch_vector, patch_position=int(patch_position))
    raise ValueError(f"Unknown component: {component}")


def _extract_component_vector(
    *,
    model: Any,
    component: str,
    input_ids: torch.Tensor,
    layer_label: int,
    position: int,
) -> torch.Tensor:
    layer_index = int(layer_label) - 1
    if component == "layer_output":
        return _extract_layer_output_at_position_from_input_ids(model, input_ids, int(layer_index), int(position))
    if component == "attention_output":
        return _extract_attention_output_at_position_from_input_ids(model, input_ids, int(layer_index), int(position))
    if component == "mlp_output":
        _mlp_in, mlp_out = _extract_mlp_io_at_position_from_input_ids(model, input_ids, int(layer_index), int(position))
        return mlp_out
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
    ap = argparse.ArgumentParser(description="Telugu continuation mediation panel at the divergence token.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--recipient", type=str, default="")
    ap.add_argument("--donor", type=str, default="")
    ap.add_argument("--writer-sites", type=str, default="")
    ap.add_argument("--mediator-site", type=str, default="")
    ap.add_argument("--offband-site", type=str, default="")
    ap.add_argument("--min-shared-prefix", type=int, default=1)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_continuation_mediation_panel_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    default_recipient = "icl_helpful" if str(args.model) == "1b" else "icl_corrupt"
    default_donor = "zs" if str(args.model) == "1b" else "icl_helpful"
    default_writer_sites = "18:attention_output;18:layer_output" if str(args.model) == "1b" else "30:layer_output;32:layer_output"
    default_mediator_site = "26:layer_output" if str(args.model) == "1b" else "34:layer_output"
    default_offband_site = "10:layer_output" if str(args.model) == "1b" else "20:layer_output"

    recipient = str(args.recipient or default_recipient)
    donor = str(args.donor or default_donor)
    if recipient not in CONDITIONS or donor not in CONDITIONS:
        raise ValueError(f"Bad recipient/donor: {recipient}, {donor}")
    writer_sites = _parse_site_list(str(args.writer_sites or default_writer_sites))
    mediator_site = _parse_site_spec(str(args.mediator_site or default_mediator_site))
    offband_site = _parse_site_spec(str(args.offband_site or default_offband_site))

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
    out_root = (REPO_ROOT / str(args.out_root)).resolve() / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running Telugu continuation mediation panel: model={args.model} pair={args.pair} items={len(eval_rows)} recipient={recipient} donor={donor} writer_sites={writer_sites} mediator={mediator_site} offband={offband_site}",
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
        patch_pos_by_condition: Dict[str, int] = {}
        base_by_condition: Dict[str, Dict[str, Any]] = {}
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

        recipient_ids = full_ids_by_condition[recipient]
        recipient_pos = int(patch_pos_by_condition[recipient])
        donor_ids = full_ids_by_condition[donor]
        donor_pos = int(patch_pos_by_condition[donor])

        # Precompute vectors at all required sites for donor and recipient conditions.
        needed_sites = list(writer_sites) + [mediator_site, offband_site]
        vectors: Dict[str, Dict[str, torch.Tensor]] = {"recipient": {}, "donor": {}}
        for site in needed_sites:
            site_name = str(site["name"])
            vectors["recipient"][site_name] = _extract_component_vector(
                model=model,
                component=str(site["component"]),
                input_ids=recipient_ids,
                layer_label=int(site["layer"]),
                position=int(recipient_pos),
            )
            vectors["donor"][site_name] = _extract_component_vector(
                model=model,
                component=str(site["component"]),
                input_ids=donor_ids,
                layer_label=int(site["layer"]),
                position=int(donor_pos),
            )

        interventions: List[Dict[str, Any]] = []
        for writer_site in writer_sites:
            writer_name = str(writer_site["name"])
            hook_specs = {
                "writer_only": [
                    (writer_site, vectors["donor"][writer_name]),
                ],
                "mediator_only": [
                    (mediator_site, vectors["donor"][str(mediator_site["name"])]),
                ],
                "writer_plus_mediator_recipient_overwrite": [
                    (writer_site, vectors["donor"][writer_name]),
                    (mediator_site, vectors["recipient"][str(mediator_site["name"])]),
                ],
                "writer_plus_offband_recipient_overwrite": [
                    (writer_site, vectors["donor"][writer_name]),
                    (offband_site, vectors["recipient"][str(offband_site["name"])]),
                ],
                "mediator_only_plus_writer_recipient_overwrite": [
                    (mediator_site, vectors["donor"][str(mediator_site["name"])]),
                    (writer_site, vectors["recipient"][writer_name]),
                ],
                "recipient_noop": [
                    (writer_site, vectors["recipient"][writer_name]),
                    (mediator_site, vectors["recipient"][str(mediator_site["name"])]),
                ],
            }
            for intervention_name, spec_list in hook_specs.items():
                hooks = []
                for site, patch_vector in spec_list:
                    hooks.append(
                        _build_patch_hook(
                            model=model,
                            component=str(site["component"]),
                            layer_label=int(site["layer"]),
                            patch_vector=patch_vector,
                            patch_position=int(recipient_pos),
                        )
                    )
                patched_stats = _divergence_step_stats(
                    model=model,
                    input_ids=recipient_ids,
                    tokenizer=tokenizer,
                    gold_next_id=int(gold_next_id),
                    competitor_next_id=int(competitor_next_id),
                    hooks=hooks,
                )
                base_stats = base_by_condition[recipient]
                interventions.append({
                    "writer_site": writer_site,
                    "mediator_site": mediator_site,
                    "offband_site": offband_site,
                    "intervention_name": str(intervention_name),
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
    for writer_site in writer_sites:
        writer_name = str(writer_site["name"])
        for intervention_name in [
            "writer_only",
            "mediator_only",
            "writer_plus_mediator_recipient_overwrite",
            "writer_plus_offband_recipient_overwrite",
            "mediator_only_plus_writer_recipient_overwrite",
            "recipient_noop",
        ]:
            rows = []
            for item in item_rows:
                for intr in item["interventions"]:
                    if intr["writer_site"]["name"] == writer_name and intr["intervention_name"] == intervention_name:
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
                "recipient": str(recipient),
                "donor": str(donor),
                "writer_site": writer_site,
                "mediator_site": mediator_site,
                "offband_site": offband_site,
                "intervention_name": str(intervention_name),
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
        "experiment": "telugu_continuation_mediation_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "recipient": str(recipient),
        "donor": str(donor),
        "writer_sites": writer_sites,
        "mediator_site": mediator_site,
        "offband_site": offband_site,
        "min_shared_prefix": int(args.min_shared_prefix),
        "n_items_used": int(len(item_rows)),
        "n_items_skipped": int(len(skipped_items)),
        "oracle": {
            "description": "Does an earlier Telugu candidate site act through the final late layer-output mediator state?",
            "success_criterion": "Writer-only patch helps; writer+mediator-recipient-overwrite substantially removes that help; writer+offband-recipient-overwrite leaves it largely intact; mediator-only remains strong even when writer is overwritten to recipient.",
            "failure_criterion": "If mediator overwrite does not remove writer rescue or offband overwrite removes it equally, the mediation interpretation is weak.",
        },
        "summary_rows": summary_rows,
        "item_rows": item_rows,
        "skipped_items": skipped_items,
    }
    out_path = out_root / "telugu_continuation_mediation_panel.json"
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
