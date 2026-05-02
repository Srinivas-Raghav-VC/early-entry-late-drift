#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    register_layer_output_replace_hook,
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


def _parse_site(raw: str) -> Dict[str, Any]:
    parts = [x.strip() for x in str(raw).split(":") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Bad site spec '{raw}'. Expected <layer>:layer_output.")
    layer = int(parts[0])
    component = str(parts[1])
    if component != "layer_output":
        raise ValueError("This sufficiency probe currently supports layer_output only.")
    return {"layer": layer, "component": component, "name": f"L{layer}_{component}"}


def _extract_layer_output(*, model: Any, input_ids: torch.Tensor, layer_label: int, position: int) -> torch.Tensor:
    return _extract_layer_output_at_position_from_input_ids(model, input_ids, int(layer_label) - 1, int(position))


def _build_layer_output_hook(*, model: Any, layer_label: int, patch_vector: torch.Tensor, patch_position: int):
    return register_layer_output_replace_hook(model, int(layer_label) - 1, patch_vector, patch_position=int(patch_position))


def _make_random_delta_control(
    *,
    recipient_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    rec = recipient_vec.detach().float().cpu()
    donor = donor_vec.detach().float().cpu()
    delta = donor - rec
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    if delta_norm == 0.0:
        return recipient_vec.detach().clone()
    rng = np.random.default_rng(int(seed))
    rand = torch.tensor(rng.standard_normal(size=tuple(rec.shape)), dtype=rec.dtype)
    rand = rand / torch.linalg.vector_norm(rand)
    random_vec = rec + rand * delta_norm
    return random_vec.to(device=recipient_vec.device, dtype=recipient_vec.dtype)


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
    ap = argparse.ArgumentParser(description="Telugu continuation final-site sufficiency transplant.")
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
    ap.add_argument("--site", type=str, default="")
    ap.add_argument("--offband-site", type=str, default="")
    ap.add_argument("--min-shared-prefix", type=int, default=1)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_continuation_final_site_sufficiency_v1")
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
        f"Running Telugu final-site sufficiency panel: model={args.model} pair={args.pair} items={len(eval_rows)} recipient={recipient} donor={donor} site={site} offband={offband_site}",
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

        recipient_site_vec = _extract_layer_output(model=model, input_ids=recipient_ids, layer_label=int(site["layer"]), position=int(recipient_pos))
        donor_site_vec = _extract_layer_output(model=model, input_ids=donor_ids, layer_label=int(site["layer"]), position=int(donor_pos))
        donor_offband_vec = _extract_layer_output(model=model, input_ids=donor_ids, layer_label=int(offband_site["layer"]), position=int(donor_pos))
        random_site_vec = _make_random_delta_control(
            recipient_vec=recipient_site_vec,
            donor_vec=donor_site_vec,
            seed=int(args.seed) * 1000003 + int(item_idx) * 97 + int(site["layer"]),
        )

        base_stats = base_by_condition[recipient]
        interventions: List[Dict[str, Any]] = []
        specs = {
            "site_donor": (site, donor_site_vec),
            "offband_donor": (offband_site, donor_offband_vec),
            "site_random_delta": (site, random_site_vec),
            "recipient_noop": (site, recipient_site_vec),
        }
        for name, (patch_site, patch_vec) in specs.items():
            hook = _build_layer_output_hook(
                model=model,
                layer_label=int(patch_site["layer"]),
                patch_vector=patch_vec,
                patch_position=int(recipient_pos),
            )
            patched_stats = _divergence_step_stats(
                model=model,
                input_ids=recipient_ids,
                tokenizer=tokenizer,
                gold_next_id=int(gold_next_id),
                competitor_next_id=int(competitor_next_id),
                hooks=[hook],
            )
            interventions.append({
                "intervention_name": str(name),
                "patch_site": patch_site,
                "base": base_stats,
                "patched": patched_stats,
                "delta": {
                    "gold_next_prob": float(patched_stats["gold_next_prob"] - base_stats["gold_next_prob"]),
                    "competitor_next_prob": float(patched_stats["competitor_next_prob"] - base_stats["competitor_next_prob"]),
                    "gold_minus_competitor_logit": float(patched_stats["gold_minus_competitor_logit"] - base_stats["gold_minus_competitor_logit"]),
                    "top1_is_gold": float(float(patched_stats["top1_is_gold"]) - float(base_stats["top1_is_gold"])),
                    "top1_is_competitor": float(float(patched_stats["top1_is_competitor"]) - float(base_stats["top1_is_competitor"])),
                },
                "vector_stats": {
                    "recipient_norm": float(torch.linalg.vector_norm(recipient_site_vec.float()).item()),
                    "donor_norm": float(torch.linalg.vector_norm(donor_site_vec.float()).item()),
                    "delta_norm": float(torch.linalg.vector_norm((donor_site_vec.float() - recipient_site_vec.float())).item()),
                    "random_norm": float(torch.linalg.vector_norm(random_site_vec.float()).item()),
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
    for intervention_name in ["site_donor", "offband_donor", "site_random_delta", "recipient_noop"]:
        rows = []
        for item in item_rows:
            for intr in item["interventions"]:
                if intr["intervention_name"] == intervention_name:
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
            "intervention_name": str(intervention_name),
            "site": site,
            "offband_site": offband_site,
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

    comparisons: Dict[str, Any] = {}
    by_name = {row["intervention_name"]: row for row in summary_rows}
    if all(name in by_name for name in ["site_donor", "offband_donor", "site_random_delta"]):
        comparisons = {
            "site_minus_offband_delta_mean_gap": float(by_name["site_donor"]["delta_mean_gap"] - by_name["offband_donor"]["delta_mean_gap"]),
            "site_minus_random_delta_mean_gap": float(by_name["site_donor"]["delta_mean_gap"] - by_name["site_random_delta"]["delta_mean_gap"]),
            "site_minus_offband_delta_gold_top1_rate": float(by_name["site_donor"]["delta_gold_top1_rate"] - by_name["offband_donor"]["delta_gold_top1_rate"]),
            "site_minus_random_delta_gold_top1_rate": float(by_name["site_donor"]["delta_gold_top1_rate"] - by_name["site_random_delta"]["delta_gold_top1_rate"]),
        }

    payload = {
        "experiment": "telugu_continuation_final_site_sufficiency",
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
        "site": site,
        "offband_site": offband_site,
        "min_shared_prefix": int(args.min_shared_prefix),
        "n_items_used": int(len(item_rows)),
        "n_items_skipped": int(len(skipped_items)),
        "oracle": {
            "description": "Can the final late layer-output site alone transplant meaningful continuation preference from donor to recipient?",
            "success_criterion": "True-site donor patch materially beats off-band and random-direction controls on gold-vs-bank gap and/or gold top1 movement, while recipient no-op is exactly zero.",
            "failure_criterion": "If off-band or random-direction controls are comparable to the true-site donor patch, the current mediator claim is more about generic perturbation than content-carrying sufficiency.",
        },
        "summary_rows": summary_rows,
        "comparisons": comparisons,
        "item_rows": item_rows,
        "skipped_items": skipped_items,
    }
    out_path = out_root / "telugu_continuation_final_site_sufficiency.json"
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
