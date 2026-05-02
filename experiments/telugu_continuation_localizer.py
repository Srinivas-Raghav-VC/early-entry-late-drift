#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = REPO_ROOT / "Draft_Results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import _get_unembedding_weight, apply_chat_template, load_model, set_all_seeds  # noqa: E402
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


def _parse_conditions(raw: str) -> List[str]:
    if not str(raw or "").strip():
        return list(CONDITIONS)
    wanted = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in wanted if x not in CONDITIONS]
    if bad:
        raise ValueError(f"Unknown conditions: {bad}")
    return wanted


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


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    target_logit = logits[int(token_id)]
    return int(torch.sum(logits > target_logit).item()) + 1


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)]).strip()


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


def _layerwise_divergence_trace(
    *,
    model: Any,
    input_ids: torch.Tensor,
    gold_next_id: int,
    competitor_next_id: int,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    unembed: torch.Tensor,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hidden_states = out.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states")

    rows: List[Dict[str, Any]] = []
    last_idx = len(hidden_states) - 1
    for layer_idx, h in enumerate(hidden_states):
        if layer_idx == last_idx:
            logits = out.logits[0, -1, :].float()
        else:
            vec = h[0, -1, :].detach().to(dtype=unembed.dtype)
            normed = final_norm(vec.unsqueeze(0))
            logits = torch.nn.functional.linear(normed, unembed).float()[0]
        probs = torch.softmax(logits, dim=-1)

        gold_prob = float(probs[int(gold_next_id)].item())
        comp_prob = float(probs[int(competitor_next_id)].item())
        gold_logit = float(logits[int(gold_next_id)].item())
        comp_logit = float(logits[int(competitor_next_id)].item())
        gold_rank = int(_rank_of_token(logits, int(gold_next_id)))
        comp_rank = int(_rank_of_token(logits, int(competitor_next_id)))

        topk = torch.topk(logits, k=min(int(top_k), int(logits.numel())))
        topk_ids = [int(i) for i in topk.indices.tolist()]
        topk_tokens = [_decode_token(tokenizer, i) for i in topk_ids]
        topk_logits = [float(x) for x in topk.values.tolist()]
        topk_probs = [float(probs[i].item()) for i in topk_ids]

        rows.append(
            {
                "layer": int(layer_idx),
                "gold_next_id": int(gold_next_id),
                "gold_next_token": _decode_token(tokenizer, int(gold_next_id)),
                "gold_next_prob": gold_prob,
                "gold_next_logit": gold_logit,
                "gold_next_rank": gold_rank,
                "competitor_next_id": int(competitor_next_id),
                "competitor_next_token": _decode_token(tokenizer, int(competitor_next_id)),
                "competitor_next_prob": comp_prob,
                "competitor_next_logit": comp_logit,
                "competitor_next_rank": comp_rank,
                "gold_minus_competitor_logit": float(gold_logit - comp_logit),
                "gold_minus_competitor_prob": float(gold_prob - comp_prob),
                "top1_is_gold": bool(topk_ids and topk_ids[0] == int(gold_next_id)),
                "top1_is_competitor": bool(topk_ids and topk_ids[0] == int(competitor_next_id)),
                "top_k_tokens": topk_tokens,
                "top_k_logits": topk_logits,
                "top_k_probs": topk_probs,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Layerwise localizer for Telugu continuation competition at the first divergence token.")
    ap.add_argument("--model", type=str, default="4b")
    ap.add_argument("--pair", type=str, default="aksharantar_tel_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    ap.add_argument("--min-shared-prefix", type=int, default=1)
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="research/results/autoresearch/telugu_continuation_localizer_v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))
    condition_names = _parse_conditions(args.conditions)

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
    final_norm = _get_final_norm(model).to(device)
    unembed = _get_unembedding_weight(model)
    if unembed is None:
        raise RuntimeError("Could not locate output embedding / lm_head weight.")
    unembed = unembed.detach().to(device)

    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])
    out_root = (REPO_ROOT / str(args.out_root)).resolve() / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    out_root.mkdir(parents=True, exist_ok=True)

    used_items: List[Dict[str, Any]] = []
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
            skipped_items.append(
                {
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "gold": gold_text,
                    "nearest_bank_target": competitor_text,
                    "reason": "empty_tokenization",
                }
            )
            continue

        prefix_len = int(_common_prefix_len(gold_ids, competitor_ids))
        if prefix_len < int(args.min_shared_prefix):
            skipped_items.append(
                {
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "gold": gold_text,
                    "nearest_bank_target": competitor_text,
                    "shared_prefix_len": prefix_len,
                    "reason": "shared_prefix_too_short",
                }
            )
            continue
        if prefix_len >= min(len(gold_ids), len(competitor_ids)):
            skipped_items.append(
                {
                    "item_index": int(item_idx - 1),
                    "word_ood": str(word["ood"]),
                    "gold": gold_text,
                    "nearest_bank_target": competitor_text,
                    "shared_prefix_len": prefix_len,
                    "reason": "no_divergence_after_shared_prefix",
                }
            )
            continue

        shared_prefix_ids = gold_ids[:prefix_len]
        gold_next_id = int(gold_ids[prefix_len])
        competitor_next_id = int(competitor_ids[prefix_len])

        item_row: Dict[str, Any] = {
            "item_index": int(item_idx - 1),
            "word_ood": str(word["ood"]),
            "word_target": gold_text,
            "nearest_bank_target": competitor_text,
            "nearest_bank_source": str(competitor["source"]),
            "nearest_bank_rank": int(competitor["position"] + 1),
            "nearest_bank_similarity": float(competitor["similarity"]),
            "shared_prefix_len_tokens": int(prefix_len),
            "shared_prefix_text": tokenizer.decode(shared_prefix_ids).strip(),
            "same_first_token": bool(int(gold_ids[0]) == int(competitor_ids[0])),
            "divergence_token_index": int(prefix_len),
            "gold_next_token_id": int(gold_next_id),
            "gold_next_token_text": _decode_token(tokenizer, int(gold_next_id)),
            "competitor_next_token_id": int(competitor_next_id),
            "competitor_next_token_text": _decode_token(tokenizer, int(competitor_next_id)),
            "conditions": {},
        }

        for condition in condition_names:
            rendered = apply_chat_template(tokenizer, str(prompts[condition]))
            prompt_ids = tokenizer(rendered, return_tensors="pt").to(device).input_ids
            if shared_prefix_ids:
                shared_prefix_tensor = torch.tensor([shared_prefix_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
                full_ids = torch.cat([prompt_ids, shared_prefix_tensor], dim=1)
            else:
                full_ids = prompt_ids

            layer_rows = _layerwise_divergence_trace(
                model=model,
                input_ids=full_ids,
                gold_next_id=int(gold_next_id),
                competitor_next_id=int(competitor_next_id),
                tokenizer=tokenizer,
                final_norm=final_norm,
                unembed=unembed,
                top_k=5,
            )
            item_row["conditions"][condition] = {
                "prompt_tokens": int(prompt_ids.shape[1]),
                "full_input_tokens": int(full_ids.shape[1]),
                "layers": layer_rows,
            }

        used_items.append(item_row)

    summary_by_condition: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in condition_names}
    comparison_rows: List[Dict[str, Any]] = []
    if used_items:
        n_layers = len(used_items[0]["conditions"][condition_names[0]]["layers"])
        for condition in condition_names:
            for layer_idx in range(n_layers):
                gold_probs = []
                comp_probs = []
                gold_ranks = []
                comp_ranks = []
                gaps = []
                gold_top1 = 0
                comp_top1 = 0
                for item in used_items:
                    row = item["conditions"][condition]["layers"][layer_idx]
                    gold_probs.append(float(row["gold_next_prob"]))
                    comp_probs.append(float(row["competitor_next_prob"]))
                    gold_ranks.append(float(row["gold_next_rank"]))
                    comp_ranks.append(float(row["competitor_next_rank"]))
                    gaps.append(float(row["gold_minus_competitor_logit"]))
                    gold_top1 += int(bool(row["top1_is_gold"]))
                    comp_top1 += int(bool(row["top1_is_competitor"]))
                n = len(gold_probs)
                summary_by_condition[condition].append(
                    {
                        "condition": condition,
                        "layer": int(layer_idx),
                        "n_items": int(n),
                        "mean_gold_next_prob": float(np.mean(gold_probs)),
                        "mean_competitor_next_prob": float(np.mean(comp_probs)),
                        "mean_gold_minus_competitor_logit": float(np.mean(gaps)),
                        "mean_gold_next_rank": float(np.mean(gold_ranks)),
                        "mean_competitor_next_rank": float(np.mean(comp_ranks)),
                        "gold_top1_rate": float(gold_top1 / max(n, 1)),
                        "competitor_top1_rate": float(comp_top1 / max(n, 1)),
                    }
                )

        if {"zs", "icl_helpful"}.issubset(set(condition_names)):
            for layer_idx in range(n_layers):
                zs = summary_by_condition["zs"][layer_idx]
                helpful = summary_by_condition["icl_helpful"][layer_idx]
                comparison_rows.append(
                    {
                        "comparison": "icl_helpful_minus_zs",
                        "layer": int(layer_idx),
                        "delta_mean_gold_minus_competitor_logit": float(helpful["mean_gold_minus_competitor_logit"] - zs["mean_gold_minus_competitor_logit"]),
                        "delta_gold_top1_rate": float(helpful["gold_top1_rate"] - zs["gold_top1_rate"]),
                        "delta_competitor_top1_rate": float(helpful["competitor_top1_rate"] - zs["competitor_top1_rate"]),
                    }
                )
        if {"icl_helpful", "icl_corrupt"}.issubset(set(condition_names)):
            for layer_idx in range(n_layers):
                helpful = summary_by_condition["icl_helpful"][layer_idx]
                corrupt = summary_by_condition["icl_corrupt"][layer_idx]
                comparison_rows.append(
                    {
                        "comparison": "icl_helpful_minus_icl_corrupt",
                        "layer": int(layer_idx),
                        "delta_mean_gold_minus_competitor_logit": float(helpful["mean_gold_minus_competitor_logit"] - corrupt["mean_gold_minus_competitor_logit"]),
                        "delta_gold_top1_rate": float(helpful["gold_top1_rate"] - corrupt["gold_top1_rate"]),
                        "delta_competitor_top1_rate": float(helpful["competitor_top1_rate"] - corrupt["competitor_top1_rate"]),
                    }
                )

    payload = {
        "experiment": "telugu_continuation_localizer",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "conditions": list(condition_names),
        "min_shared_prefix": int(args.min_shared_prefix),
        "n_used_items": int(len(used_items)),
        "n_skipped_items": int(len(skipped_items)),
        "oracle": {
            "description": "At the first divergence token after the shared gold-vs-nearest-bank prefix, does the model prefer the gold continuation or the bank continuation as the residual stream evolves layer by layer?",
            "success_criterion": "Recover a clean layer band where 4B helpful stays positive on gold-vs-bank competition while 1B helpful turns negative or is sharply weaker.",
            "failure_criterion": "If no layerwise separation emerges or if used items are too few because the competitor does not share a prefix, this localizer is not the right stage-isolation instrument.",
        },
        "summary_by_condition": summary_by_condition,
        "comparison_rows": comparison_rows,
        "used_items": used_items,
        "skipped_items": skipped_items,
    }
    out_path = out_root / "telugu_continuation_localizer.json"
    _write_json(out_path, payload)
    log(f"Saved: {out_path}")

    for condition in condition_names:
        rows = summary_by_condition.get(condition) or []
        if not rows:
            continue
        last_rows = rows[-8:]
        condensed = ", ".join(
            f"L{int(r['layer'])}=gap{r['mean_gold_minus_competitor_logit']:+.3f}/gold@1={r['gold_top1_rate']:.2f}"
            for r in last_rows
        )
        print(f"{condition}: {condensed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
