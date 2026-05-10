#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
    _find_last_subsequence,
    apply_chat_template,
    build_task_prompt,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402

CONDITIONS = ("zs", "icl_helpful")
PATCH_DIRECTIONS = (("icl_helpful", "zs"), ("zs", "icl_helpful"))


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


def _get_mlp(model: Any, layer: int):
    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate model.model.layers")
    return layers[int(layer)].mlp


def _extract_mlp_channel_vector(
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    mlp = _get_mlp(model, int(layer))
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(module, inputs_tuple):
        x = inputs_tuple[0] if inputs_tuple else None
        if torch.is_tensor(x):
            captured["channel_in"] = x.detach()

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()
    if "channel_in" not in captured:
        raise RuntimeError("Failed to capture MLP channel vector at down_proj input.")
    seq_len = int(captured["channel_in"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["channel_in"][0, pos, :]


def _register_partial_mlp_channel_replace_hook(
    model: Any,
    layer: int,
    donor_channel_vector: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    patch_position: int,
):
    mlp = _get_mlp(model, int(layer))
    donor_channel_vector = donor_channel_vector.detach()
    selected_idx = selected_idx.detach().to(dtype=torch.long)

    def pre_hook(module, inputs_tuple):
        if not inputs_tuple:
            return None
        x = inputs_tuple[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None
        seq_len = int(x.shape[1])
        pos = int(max(0, min(int(patch_position), seq_len - 1)))
        idx = selected_idx.to(device=x.device)
        if idx.numel() == 0:
            return None
        idx = idx[(idx >= 0) & (idx < int(x.shape[2]))]
        if idx.numel() == 0:
            return None
        donor = donor_channel_vector.to(device=x.device, dtype=x.dtype).view(-1)
        x_new = x.clone()
        x_new[:, pos, idx] = donor[idx].view(1, -1)
        if len(inputs_tuple) == 1:
            return (x_new,)
        return (x_new,) + tuple(inputs_tuple[1:])

    return mlp.down_proj.register_forward_pre_hook(pre_hook)


def _prepare_condition_inputs(
    *,
    tokenizer: Any,
    word: Mapping[str, Any],
    icl_examples: List[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
    prompt_variant: str = "canonical",
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    query_text = str(word["ood"])
    query_id_candidates = []
    for candidate in (query_text, f" {query_text}", f"\n{query_text}"):
        cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if cand_ids:
            query_id_candidates.append([int(x) for x in cand_ids])
    if not query_id_candidates:
        raise ValueError(f"Failed to tokenize query text: {query_text!r}")
    for condition in CONDITIONS:
        prompt = build_task_prompt(
            query_text,
            None if condition == "zs" else icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=str(prompt_variant),
        )
        rendered = apply_chat_template(tokenizer, prompt)
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
        token_list = input_ids[0].detach().cpu().tolist()
        span = None
        for cand_ids in query_id_candidates:
            span = _find_last_subsequence(token_list, cand_ids)
            if span is not None:
                break
        if span is None:
            # Prompt-format variants can change boundary tokenization enough that the
            # isolated query token sequence is not recoverable exactly. Keep the last
            # prompt position path usable and fall back conservatively for the query site.
            span = (max(0, int(input_ids.shape[1]) - 1), int(input_ids.shape[1]))
        out[condition] = {
            "rendered": rendered,
            "input_ids": input_ids,
            "query_position": int(span[1] - 1),
            "last_position": int(input_ids.shape[1] - 1),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Subset panel for top negative L25 MLP channels in Hindi 1B.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--selector-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--n-top", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

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
    select_rows = list(bundle["select_rows"][: max(1, int(args.selector_max_items))])
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)
    mlp = _get_mlp(model, int(args.layer))
    channel_size = int(getattr(mlp.down_proj, "in_features", 0))

    print(
        f"Running Hindi L{args.layer} top-negative channel subset panel: model={args.model} pair={args.pair} select_items={len(select_rows)} eval_items={len(eval_rows)} n_top={args.n_top}",
        flush=True,
    )

    # Selection split ranking in channel basis.
    abs_delta_sum = torch.zeros(channel_size, dtype=torch.float32)
    signed_delta_sum = torch.zeros(channel_size, dtype=torch.float32)
    n_selector_items = 0
    for idx, word in enumerate(select_rows, start=1):
        if idx == 1 or idx == len(select_rows) or idx % 25 == 0:
            print(f"[selector {idx}/{len(select_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
        cond_inputs = _prepare_condition_inputs(
            tokenizer=tokenizer,
            word=word,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
        )
        zs_vec = _extract_mlp_channel_vector(model, cond_inputs["zs"]["input_ids"], int(args.layer), int(cond_inputs["zs"]["last_position"])).detach().float().cpu()
        helpful_vec = _extract_mlp_channel_vector(model, cond_inputs["icl_helpful"]["input_ids"], int(args.layer), int(cond_inputs["icl_helpful"]["last_position"])).detach().float().cpu()
        delta = zs_vec - helpful_vec
        abs_delta_sum += torch.abs(delta)
        signed_delta_sum += delta
        n_selector_items += 1
    if n_selector_items <= 0:
        raise RuntimeError("No selector items were processed.")
    mean_abs_delta = abs_delta_sum / float(n_selector_items)
    mean_signed_delta = signed_delta_sum / float(n_selector_items)
    neg_rank = torch.nonzero(mean_signed_delta < 0, as_tuple=False).flatten()
    if neg_rank.numel() <= 0:
        raise RuntimeError("No negative-signed channels found on selection split.")
    neg_scores = torch.abs(mean_signed_delta[neg_rank])
    neg_rank = neg_rank[torch.argsort(neg_scores, descending=True)]
    top_neg = [int(x) for x in neg_rank[: int(args.n_top)].tolist()]

    subset_specs: List[Dict[str, Any]] = []
    for k in range(1, int(args.n_top) + 1):
        subset_specs.append({"name": f"neg_top{k}", "channels": top_neg[:k]})
    for channel in top_neg:
        subset_specs.append(
            {
                "name": f"neg_single_{int(channel)}",
                "channels": [int(channel)],
                "single_channel": int(channel),
            }
        )
    if len(top_neg) > 1:
        for a, b in itertools.combinations(top_neg, 2):
            subset_specs.append(
                {
                    "name": f"neg_pair_{int(a)}_{int(b)}",
                    "channels": [int(a), int(b)],
                }
            )
        for idx, channel in enumerate(top_neg):
            subset_specs.append(
                {
                    "name": f"neg_leaveout_{int(channel)}",
                    "channels": [int(c) for j, c in enumerate(top_neg) if j != idx],
                    "left_out": int(channel),
                }
            )

    item_rows: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 5 == 0:
            print(f"[eval {item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
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
        base_by_condition: Dict[str, Dict[str, Any]] = {}
        donor_channels_by_condition: Dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            input_ids = cond_inputs[condition]["input_ids"]
            base_by_condition[condition] = _first_step_stats(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
            )
            donor_channels_by_condition[condition] = _extract_mlp_channel_vector(
                model,
                input_ids,
                int(args.layer),
                int(cond_inputs[condition]["last_position"]),
            ).detach()

        interventions: List[Dict[str, Any]] = []
        for recipient_condition, donor_condition in PATCH_DIRECTIONS:
            recipient_input_ids = cond_inputs[recipient_condition]["input_ids"]
            patch_position = int(cond_inputs[recipient_condition]["last_position"])
            donor_channels = donor_channels_by_condition[donor_condition]
            base_stats = base_by_condition[recipient_condition]
            for spec in subset_specs:
                idx_selected = torch.tensor(spec["channels"], dtype=torch.long)
                hook = _register_partial_mlp_channel_replace_hook(
                    model,
                    int(args.layer),
                    donor_channels,
                    idx_selected,
                    patch_position=patch_position,
                )
                patched = _first_step_stats(
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
                        "subset_name": str(spec["name"]),
                        "channels": [int(x) for x in spec["channels"]],
                        "k": int(len(spec["channels"])),
                        "base": base_stats,
                        "patched": patched,
                        "delta": {
                            "target_prob": float(patched["target_prob"] - base_stats["target_prob"]),
                            "target_minus_competitor_logit": float(patched["target_minus_competitor_logit"] - base_stats["target_minus_competitor_logit"]),
                            "target_minus_latin_logit": float(patched["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]),
                            "top1_is_target": float(float(patched["top1_is_target"]) - float(base_stats["top1_is_target"])),
                        },
                    }
                )
        item_rows.append(
            {
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "target_id": int(target_id),
                "patch_position_mode": "last_token",
                "base_by_condition": base_by_condition,
                "interventions": interventions,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in PATCH_DIRECTIONS:
        for spec in subset_specs:
            rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for item in item_rows:
                for intr in item["interventions"]:
                    if (
                        intr["recipient_condition"] == recipient_condition
                        and intr["donor_condition"] == donor_condition
                        and intr["subset_name"] == spec["name"]
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
                    "subset_name": str(spec["name"]),
                    "channels": [int(x) for x in spec["channels"]],
                    "k": int(len(spec["channels"])),
                    "n_items": int(len(rows)),
                    "delta_mean_target_prob": float(np.nanmean([intr["delta"]["target_prob"] for _item, intr in rows])),
                    "delta_mean_gap_overall": float(np.nanmean([intr["delta"]["target_minus_competitor_logit"] for _item, intr in rows])),
                    "delta_mean_gap_latin": float(np.nanmean([intr["delta"]["target_minus_latin_logit"] for _item, intr in rows])),
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
        "experiment": "hindi_1b_mlp_channel_subset_panel",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "selector_max_items": int(args.selector_max_items),
        "max_items": int(args.max_items),
        "layer": int(args.layer),
        "component": "mlp_channel_basis_subset_panel",
        "patch_position_mode": "last_token",
        "channel_size": int(channel_size),
        "top_negative_channels": [int(x) for x in top_neg],
        "subset_specs": subset_specs,
        "oracle": {
            "description": "Test whether the strong negative-signed L25 MLP channel effect is driven by a single channel or by a small synergistic set.",
            "primary_hypothesis": "If a small subset is truly causal, top negative channels and leave-one-out subsets should show strong structure rather than flat effects.",
        },
        "summary_rows": summary_rows,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "research" / "results" / "hindi_mlp_channel_subset_panel_v1" / str(args.model) / str(args.pair) / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "hindi_1b_mlp_channel_subset_panel.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)

    print("\n=== Helpful<-zs subset summary ===", flush=True)
    rows = [r for r in summary_rows if r['recipient_condition']=='icl_helpful' and r['donor_condition']=='zs']
    for row in sorted(rows, key=lambda r: (int(r['k']), str(r['subset_name']))):
        print(f"  {row['subset_name']:<20s} k={row['k']:>2d} dgapLatin={row['delta_mean_gap_latin']:.3f} dtop1={row['delta_top1_target_rate']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
