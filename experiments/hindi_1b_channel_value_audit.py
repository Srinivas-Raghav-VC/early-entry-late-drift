#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
    _prepare_condition_inputs,
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
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out = [int(p) for p in parts]
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit raw L25 MLP channel values for candidate Hindi channels.")
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

    print(
        f"Running Hindi channel value audit: model={args.model} pair={args.pair} eval_items={len(eval_rows)} channels={channels}",
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
        base_helpful = _first_step_stats(
            model=model,
            input_ids=cond_inputs["icl_helpful"]["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
        )
        base_zs = _first_step_stats(
            model=model,
            input_ids=cond_inputs["zs"]["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
        )
        vectors = {}
        for condition in ("zs", "icl_helpful"):
            vectors[condition] = _extract_mlp_channel_vector(
                model,
                cond_inputs[condition]["input_ids"],
                int(args.layer),
                int(cond_inputs[condition]["last_position"]),
            ).detach().float().cpu()
        channel_rows = []
        for channel in channels:
            z = float(vectors["zs"][int(channel)].item())
            h = float(vectors["icl_helpful"][int(channel)].item())
            channel_rows.append(
                {
                    "channel": int(channel),
                    "zs_value": z,
                    "icl_helpful_value": h,
                    "zs_minus_helpful": float(z - h),
                    "helpful_minus_zs": float(h - z),
                }
            )
        item_rows.append(
            {
                "item_index": int(idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "subtype": _subtype(base_helpful),
                "base_helpful": base_helpful,
                "base_zs": base_zs,
                "channels": channel_rows,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for subtype in sorted({row["subtype"] for row in item_rows}):
        subset = [row for row in item_rows if row["subtype"] == subtype]
        for channel in channels:
            rows = [next(ch for ch in row["channels"] if int(ch["channel"]) == int(channel)) for row in subset]
            summary_rows.append(
                {
                    "subtype": subtype,
                    "channel": int(channel),
                    "n_items": int(len(rows)),
                    "mean_zs_value": float(np.mean([r["zs_value"] for r in rows])),
                    "mean_icl_helpful_value": float(np.mean([r["icl_helpful_value"] for r in rows])),
                    "mean_zs_minus_helpful": float(np.mean([r["zs_minus_helpful"] for r in rows])),
                    "mean_helpful_minus_zs": float(np.mean([r["helpful_minus_zs"] for r in rows])),
                }
            )

    overall_rows: List[Dict[str, Any]] = []
    for channel in channels:
        rows = [next(ch for ch in row["channels"] if int(ch["channel"]) == int(channel)) for row in item_rows]
        overall_rows.append(
            {
                "channel": int(channel),
                "n_items": int(len(rows)),
                "mean_zs_value": float(np.mean([r["zs_value"] for r in rows])),
                "mean_icl_helpful_value": float(np.mean([r["icl_helpful_value"] for r in rows])),
                "mean_zs_minus_helpful": float(np.mean([r["zs_minus_helpful"] for r in rows])),
                "mean_helpful_minus_zs": float(np.mean([r["helpful_minus_zs"] for r in rows])),
            }
        )

    payload = {
        "experiment": "hindi_1b_channel_value_audit",
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
        "overall_rows": overall_rows,
        "summary_rows": summary_rows,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_channel_value_audit_v1" / str(args.model) / str(args.pair) / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "hindi_1b_channel_value_audit.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("\n=== Overall channel means ===", flush=True)
    for row in overall_rows:
        print(
            f"  ch={row['channel']:>4d} zs={row['mean_zs_value']:+.3f} helpful={row['mean_icl_helpful_value']:+.3f} helpful-zs={row['mean_helpful_minus_zs']:+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
