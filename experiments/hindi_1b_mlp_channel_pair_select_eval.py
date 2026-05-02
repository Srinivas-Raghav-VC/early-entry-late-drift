#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_mlp_channel_subset_panel import (  # noqa: E402
    CONDITIONS,
    PATCH_DIRECTIONS,
    _build_latin_mask,
    _extract_mlp_channel_vector,
    _first_step_stats,
    _prepare_condition_inputs,
    _register_partial_mlp_channel_replace_hook,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402


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


def _deterministic_random_pair(*, size: int, seed: int, label: str) -> torch.Tensor:
    import hashlib

    msg = f"{seed}::{label}::{size}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    perm = rng.permutation(int(size))
    return torch.tensor(perm[:2], dtype=torch.long)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Choose best top-negative channel pair on selection split; evaluate on held-out eval split.")
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
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _mean(xs: List[float]) -> float:
    return float(np.nanmean(xs)) if xs else float("nan")


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
    channel_size = int(getattr(model.model.layers[int(args.layer)].mlp.down_proj, "in_features", 0))

    print(
        f"Running Hindi pair select->eval: model={args.model} pair={args.pair} select_items={len(select_rows)} eval_items={len(eval_rows)} n_top={args.n_top} n_random={args.n_random}",
        flush=True,
    )

    # First rank negative channels on selection split.
    signed_delta_sum = torch.zeros(channel_size, dtype=torch.float32)
    for idx, word in enumerate(select_rows, start=1):
        if idx == 1 or idx == len(select_rows) or idx % 25 == 0:
            print(f"[neg-rank {idx}/{len(select_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
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
        signed_delta_sum += (zs_vec - helpful_vec)
    mean_signed_delta = signed_delta_sum / float(len(select_rows))
    neg_rank = torch.nonzero(mean_signed_delta < 0, as_tuple=False).flatten()
    neg_scores = torch.abs(mean_signed_delta[neg_rank])
    neg_rank = neg_rank[torch.argsort(neg_scores, descending=True)]
    top_neg = [int(x) for x in neg_rank[: int(args.n_top)].tolist()]
    candidate_pairs = [(int(a), int(b)) for a, b in itertools.combinations(top_neg, 2)]

    # Score candidate pairs on selection split using only helpful<-zs delta_mean_gap_latin.
    pair_selection_scores: List[Dict[str, Any]] = []
    for pair_idx, pair in enumerate(candidate_pairs, start=1):
        pair_tensor = torch.tensor(list(pair), dtype=torch.long)
        deltas: List[float] = []
        for word in select_rows:
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
            recipient_input_ids = cond_inputs["icl_helpful"]["input_ids"]
            patch_position = int(cond_inputs["icl_helpful"]["last_position"])
            donor_channels = _extract_mlp_channel_vector(model, cond_inputs["zs"]["input_ids"], int(args.layer), int(cond_inputs["zs"]["last_position"])).detach()
            base_stats = _first_step_stats(
                model=model,
                input_ids=recipient_input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
            )
            hook = _register_partial_mlp_channel_replace_hook(
                model,
                int(args.layer),
                donor_channels,
                pair_tensor,
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
            deltas.append(float(patched["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]))
        pair_selection_scores.append(
            {
                "pair_name": f"neg_pair_{pair[0]}_{pair[1]}",
                "channels": [int(x) for x in pair],
                "selection_mean_delta_gap_latin": _mean(deltas),
            }
        )
    pair_selection_scores = sorted(pair_selection_scores, key=lambda row: float(row["selection_mean_delta_gap_latin"]), reverse=True)
    chosen = pair_selection_scores[0]
    chosen_pair = [int(x) for x in chosen["channels"]]
    chosen_pair_tensor = torch.tensor(chosen_pair, dtype=torch.long)

    # Eval chosen pair + random pairs.
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

            hook = _register_partial_mlp_channel_replace_hook(
                model,
                int(args.layer),
                donor_channels,
                chosen_pair_tensor,
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
                    "subset_kind": "chosen_pair",
                    "subset_name": str(chosen["pair_name"]),
                    "channels": chosen_pair,
                    "random_repeat": None,
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

            for random_repeat in range(int(args.n_random)):
                rand_pair = _deterministic_random_pair(size=channel_size, seed=int(args.seed) + int(random_repeat), label=f"{recipient_condition}<-{donor_condition}::random_pair::{random_repeat}")
                hook = _register_partial_mlp_channel_replace_hook(
                    model,
                    int(args.layer),
                    donor_channels,
                    rand_pair,
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
                        "subset_kind": "random_pair",
                        "subset_name": f"random_pair_{random_repeat}",
                        "channels": [int(x) for x in rand_pair.tolist()],
                        "random_repeat": int(random_repeat),
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
                "base_by_condition": base_by_condition,
                "interventions": interventions,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in PATCH_DIRECTIONS:
        for subset_kind in ("chosen_pair", "random_pair"):
            repeat_ids = [None] if subset_kind == "chosen_pair" else list(range(int(args.n_random)))
            for repeat_id in repeat_ids:
                rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                for item in item_rows:
                    for intr in item["interventions"]:
                        if (
                            intr["recipient_condition"] == recipient_condition
                            and intr["donor_condition"] == donor_condition
                            and intr["subset_kind"] == subset_kind
                            and intr.get("random_repeat") == repeat_id
                        ):
                            rows.append((item, intr))
                if not rows:
                    continue
                base_success = [1.0 if bool(intr["base"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                patched_success = [1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                failed_base_rows = [(item, intr) for item, intr in rows if not bool(intr["base"]["top1_is_target"])]
                succeeded_base_rows = [(item, intr) for item, intr in rows if bool(intr["base"]["top1_is_target"])]
                summary_rows.append(
                    {
                        "recipient_condition": str(recipient_condition),
                        "donor_condition": str(donor_condition),
                        "subset_kind": str(subset_kind),
                        "subset_name": rows[0][1]["subset_name"],
                        "channels": rows[0][1]["channels"],
                        "random_repeat": repeat_id,
                        "delta_mean_gap_latin": _mean([intr["delta"]["target_minus_latin_logit"] for _item, intr in rows]),
                        "delta_top1_target_rate": _mean([float(intr["delta"]["top1_is_target"]) for _item, intr in rows]),
                        "rescue_rate_on_base_failures": _mean([1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in failed_base_rows]) if failed_base_rows else None,
                        "harm_rate_on_base_successes": _mean([0.0 if bool(intr["patched"]["top1_is_target"]) else 1.0 for _item, intr in succeeded_base_rows]) if succeeded_base_rows else None,
                    }
                )

    comparison_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in PATCH_DIRECTIONS:
        chosen_rows = [r for r in summary_rows if r['recipient_condition']==recipient_condition and r['donor_condition']==donor_condition and r['subset_kind']=='chosen_pair']
        random_rows = [r for r in summary_rows if r['recipient_condition']==recipient_condition and r['donor_condition']==donor_condition and r['subset_kind']=='random_pair']
        if not chosen_rows:
            continue
        chosen_row = chosen_rows[0]
        comparison_rows.append(
            {
                'recipient_condition': recipient_condition,
                'donor_condition': donor_condition,
                'chosen_pair_name': chosen_row['subset_name'],
                'chosen_channels': chosen_row['channels'],
                'chosen_delta_mean_gap_latin': chosen_row['delta_mean_gap_latin'],
                'chosen_delta_top1_target_rate': chosen_row['delta_top1_target_rate'],
                'random_mean_delta_gap_latin': _mean([r['delta_mean_gap_latin'] for r in random_rows]),
                'random_std_delta_gap_latin': float(np.nanstd([r['delta_mean_gap_latin'] for r in random_rows])) if random_rows else None,
                'chosen_minus_random_mean_gap_latin': float(chosen_row['delta_mean_gap_latin'] - _mean([r['delta_mean_gap_latin'] for r in random_rows])) if random_rows else None,
            }
        )

    payload = {
        'experiment': 'hindi_1b_mlp_channel_pair_select_eval',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': str(args.model),
        'pair': str(args.pair),
        'seed': int(args.seed),
        'n_icl': int(args.n_icl),
        'n_select': int(args.n_select),
        'n_eval': int(args.n_eval),
        'selector_max_items': int(args.selector_max_items),
        'max_items': int(args.max_items),
        'layer': int(args.layer),
        'component': 'mlp_channel_basis_pair_select_eval',
        'top_negative_channels': top_neg,
        'pair_selection_scores': pair_selection_scores,
        'chosen_pair': chosen,
        'n_random': int(args.n_random),
        'summary_rows': summary_rows,
        'comparison_rows': comparison_rows,
        'item_rows': item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / 'research' / 'results' / 'autoresearch' / 'hindi_mlp_channel_pair_select_eval_v1' / str(args.model) / str(args.pair) / f'nicl{int(args.n_icl)}'
    )
    out_path = out_root / 'hindi_1b_mlp_channel_pair_select_eval.json'
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print('\n=== Chosen pair ===', chosen, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
