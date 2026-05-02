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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_mlp_channel_subset_panel import (  # noqa: E402
    _build_latin_mask,
    _extract_mlp_channel_vector,
    _first_step_stats,
    _get_mlp,
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


def _parse_channels(text: str) -> List[int]:
    out = [int(p.strip()) for p in str(text).split(",") if p.strip()]
    if not out:
        raise ValueError("No channels provided.")
    return out


def _subtype(base_helpful: Mapping[str, Any]) -> str:
    if bool(base_helpful.get("top1_is_target")):
        return "base_success"
    script = str(base_helpful.get("top1_script", "unknown"))
    if script == "latin":
        return "latin_collapse"
    if script == "devanagari":
        return "devanagari_wrong"
    return f"other_{script}"


def _capture_output_margin_grad(
    *,
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
    target_id: int,
    latin_id: int,
) -> torch.Tensor:
    mlp = _get_mlp(model, int(layer))
    captured: Dict[str, torch.Tensor] = {}

    def fwd_hook(module, inputs, output):
        if torch.is_tensor(output):
            output.retain_grad()
            captured["y"] = output

    handle = mlp.down_proj.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, int(input_ids.shape[1] - 1), :].float()
        margin = logits[int(target_id)] - logits[int(latin_id)]
        margin.backward()
        if "y" not in captured or captured["y"].grad is None:
            raise RuntimeError("Failed to capture gradient on down_proj output.")
        grad = captured["y"].grad.detach().clone()[0, int(input_ids.shape[1] - 1), :]
        return grad.float().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit L25 MLP channel-to-readout geometry for Hindi 1B.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--channels", type=str, default="5486,2299,6015,789")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    channels = _parse_channels(args.channels)
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
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)
    mlp = _get_mlp(model, int(args.layer))
    down_weight = mlp.down_proj.weight.detach().float().cpu()  # [resid, channels]

    print(
        f"Running Hindi channel readout geometry audit: model={args.model} pair={args.pair} eval_items={len(eval_rows)} channels={channels}",
        flush=True,
    )

    item_rows: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_rows, start=1):
        if idx == 1 or idx == len(eval_rows) or idx % 5 == 0:
            print(f"[eval {idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
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
        helpful_input_ids = cond_inputs["icl_helpful"]["input_ids"]
        zs_input_ids = cond_inputs["zs"]["input_ids"]
        helpful_pos = int(cond_inputs["icl_helpful"]["last_position"])
        zs_pos = int(cond_inputs["zs"]["last_position"])

        base_helpful = _first_step_stats(
            model=model,
            input_ids=helpful_input_ids,
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
        )
        helpful_vec = _extract_mlp_channel_vector(model, helpful_input_ids, int(args.layer), helpful_pos).detach().float().cpu()
        zs_vec = _extract_mlp_channel_vector(model, zs_input_ids, int(args.layer), zs_pos).detach().float().cpu()
        grad_y = _capture_output_margin_grad(
            model=model,
            input_ids=helpful_input_ids,
            layer=int(args.layer),
            target_id=target_id,
            latin_id=int(base_helpful["latin_competitor_id"]),
        )
        grad_y_norm = float(torch.norm(grad_y).item())

        channel_rows: List[Dict[str, Any]] = []
        for channel in channels:
            column = down_weight[:, int(channel)]
            col_norm = float(torch.norm(column).item())
            dot = float(torch.dot(grad_y, column).item())
            cosine = float(dot / (grad_y_norm * col_norm)) if grad_y_norm > 0 and col_norm > 0 else None
            delta = float(zs_vec[int(channel)].item() - helpful_vec[int(channel)].item())
            predicted = float(dot * delta)
            hook = _register_partial_mlp_channel_replace_hook(
                model,
                int(args.layer),
                zs_vec,
                torch.tensor([int(channel)], dtype=torch.long),
                patch_position=helpful_pos,
            )
            patched = _first_step_stats(
                model=model,
                input_ids=helpful_input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
                hooks=[hook],
            )
            actual = float(patched["target_minus_latin_logit"] - base_helpful["target_minus_latin_logit"])
            channel_rows.append(
                {
                    "channel": int(channel),
                    "down_proj_column_norm": col_norm,
                    "helpful_value": float(helpful_vec[int(channel)].item()),
                    "zs_value": float(zs_vec[int(channel)].item()),
                    "zs_minus_helpful": delta,
                    "readout_grad_dot_column": dot,
                    "readout_grad_column_cosine": cosine,
                    "first_order_predicted_effect": predicted,
                    "actual_singleton_patch_effect": actual,
                    "approx_error": float(actual - predicted),
                }
            )
        item_rows.append(
            {
                "item_index": int(idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "subtype": _subtype(base_helpful),
                "base_helpful": base_helpful,
                "readout_grad_norm": grad_y_norm,
                "channels": channel_rows,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for subtype in sorted({row["subtype"] for row in item_rows} | {"overall"}):
        rows_for_subtype = item_rows if subtype == "overall" else [row for row in item_rows if row["subtype"] == subtype]
        for channel in channels:
            rows = [next(ch for ch in row["channels"] if int(ch["channel"]) == int(channel)) for row in rows_for_subtype]
            predicted = np.array([r["first_order_predicted_effect"] for r in rows], dtype=float)
            actual = np.array([r["actual_singleton_patch_effect"] for r in rows], dtype=float)
            dots = np.array([r["readout_grad_dot_column"] for r in rows], dtype=float)
            cosines = np.array([r["readout_grad_column_cosine"] for r in rows], dtype=float)
            delta = np.array([r["zs_minus_helpful"] for r in rows], dtype=float)
            corr = None
            if len(rows) >= 2 and float(np.std(predicted)) > 0 and float(np.std(actual)) > 0:
                corr = float(np.corrcoef(predicted, actual)[0, 1])
            summary_rows.append(
                {
                    "subtype": subtype,
                    "channel": int(channel),
                    "n_items": int(len(rows)),
                    "mean_down_proj_column_norm": float(np.mean([r["down_proj_column_norm"] for r in rows])),
                    "mean_readout_grad_dot_column": float(np.mean(dots)),
                    "mean_readout_grad_column_cosine": float(np.mean(cosines)),
                    "mean_zs_minus_helpful": float(np.mean(delta)),
                    "mean_first_order_predicted_effect": float(np.mean(predicted)),
                    "mean_actual_singleton_patch_effect": float(np.mean(actual)),
                    "mean_approx_error": float(np.mean(actual - predicted)),
                    "predicted_actual_corr": corr,
                }
            )

    payload = {
        "experiment": "hindi_1b_channel_readout_geometry_audit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "max_items": int(args.max_items),
        "layer": int(args.layer),
        "channels": channels,
        "oracle": {
            "description": "Explain channel ranking by direct geometry from MLP channels through down_proj into the local target-vs-Latin readout direction.",
            "primary_hypothesis": "Causally important channels should have the right-sign readout-gradient projection, while shifted but inert channels should have near-zero projection and wrong-sign channels should have opposite-sign projection.",
        },
        "summary_rows": summary_rows,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_channel_readout_geometry_v1" / str(args.model) / str(args.pair) / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "hindi_1b_channel_readout_geometry_audit.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("\n=== Overall geometry summary ===", flush=True)
    for row in [r for r in summary_rows if r['subtype'] == 'overall']:
        print(
            f"  ch={row['channel']:>4d} dot={row['mean_readout_grad_dot_column']:+.4f} cos={row['mean_readout_grad_column_cosine']:+.4f} delta={row['mean_zs_minus_helpful']:+.3f} pred={row['mean_first_order_predicted_effect']:+.3f} actual={row['mean_actual_singleton_patch_effect']:+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
