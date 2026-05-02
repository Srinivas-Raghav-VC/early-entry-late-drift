#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONTROL_CONDITIONS = ("icl_corrupt", "icl_random_indic", "icl_null_filler")
DEFAULT_MODELS = ("1b", "4b")
DEFAULT_PAIRS = ("aksharantar_hin_latin", "aksharantar_tel_latin")
DEFAULT_NICLS = (8, 64)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _mean(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_payload_path(results_root: Path, model: str, pair: str, n_icl: int) -> Path:
    base = results_root / model / pair / f"nicl{n_icl}"
    candidates = [
        base / "neutral_filler_recency_controls.json",
        base / "neutral_filler_recency_controls.json" / "neutral_filler_recency_controls.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Backward-compatible fallback for partially downloaded or malformed trees.
    if (base / "neutral_filler_recency_controls.json").is_dir():
        nested = base / "neutral_filler_recency_controls.json" / "neutral_filler_recency_controls.json"
        if nested.exists():
            return nested
    return candidates[0]


def _condition_metrics(summary_by_condition: Dict[str, Any], condition: str) -> Dict[str, float]:
    bucket = dict(summary_by_condition.get(condition) or {})
    return {
        "exact": _safe_float(bucket.get("mean_exact_match")),
        "cer": _safe_float(bucket.get("mean_akshara_cer")),
        "script": _safe_float(bucket.get("mean_script_compliance")),
        "first_prob": _safe_float(bucket.get("mean_first_prob")),
        "first_entry": _safe_float(bucket.get("mean_first_entry_correct")),
    }


def _pairwise_shot_delta(rows: List[Dict[str, Any]], model: str, pair: str, key: str) -> float:
    row8 = next((r for r in rows if r["model"] == model and r["pair"] == pair and int(r["n_icl"]) == 8), None)
    row64 = next((r for r in rows if r["model"] == model and r["pair"] == pair and int(r["n_icl"]) == 64), None)
    if row8 is None or row64 is None:
        return float("nan")
    return _safe_float(row64.get(key)) - _safe_float(row8.get(key))


def _split_tokens(raw: str) -> List[str]:
    if not str(raw).strip():
        return []
    return [tok for tok in str(raw).replace(",", " ").split() if tok]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score Loop 2 helpful-vs-control outputs.")
    ap.add_argument("--results-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--models", type=str, default=" ".join(DEFAULT_MODELS))
    ap.add_argument("--pairs", type=str, default=" ".join(DEFAULT_PAIRS))
    ap.add_argument("--nicls", type=str, default=" ".join(str(x) for x in DEFAULT_NICLS))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    out_path = Path(args.out).resolve()
    models = tuple(_split_tokens(args.models)) or DEFAULT_MODELS
    pairs = tuple(_split_tokens(args.pairs)) or DEFAULT_PAIRS
    nicls = tuple(int(x) for x in _split_tokens(args.nicls)) or DEFAULT_NICLS
    expected = [(model, pair, n_icl) for model in models for pair in pairs for n_icl in nicls]

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for model, pair, n_icl in expected:
        path = _resolve_payload_path(results_root, model, pair, n_icl)
        if not path.exists() or not path.is_file():
            missing.append(str(path))
            continue
        obj = _load_json(path)
        summary_by_condition = dict(obj.get("summary_by_condition") or {})

        helpful = _condition_metrics(summary_by_condition, "icl_helpful")
        zs = _condition_metrics(summary_by_condition, "zs")
        desc = _condition_metrics(summary_by_condition, "icl_helpful_similarity_desc")
        asc = _condition_metrics(summary_by_condition, "icl_helpful_similarity_asc")
        reversed_helpful = _condition_metrics(summary_by_condition, "icl_helpful_reversed")
        controls = {c: _condition_metrics(summary_by_condition, c) for c in CONTROL_CONDITIONS}

        best_control_exact = max((_safe_float(v["exact"]) for v in controls.values()), default=float("nan"))
        best_control_cer = min((_safe_float(v["cer"]) for v in controls.values()), default=float("nan"))
        best_control_script = max((_safe_float(v["script"]) for v in controls.values()), default=float("nan"))
        best_control_first_prob = max((_safe_float(v["first_prob"]) for v in controls.values()), default=float("nan"))

        row = {
            "model": model,
            "pair": pair,
            "n_icl": int(n_icl),
            "path": str(path),
            "helpful_exact": helpful["exact"],
            "helpful_cer": helpful["cer"],
            "helpful_script": helpful["script"],
            "helpful_first_prob": helpful["first_prob"],
            "zs_exact": zs["exact"],
            "zs_cer": zs["cer"],
            "best_control_exact": best_control_exact,
            "best_control_cer": best_control_cer,
            "best_control_script": best_control_script,
            "best_control_first_prob": best_control_first_prob,
            "helpful_minus_zs_exact": helpful["exact"] - zs["exact"],
            "helpful_minus_zs_cer": zs["cer"] - helpful["cer"],
            "helpful_control_exact_margin": helpful["exact"] - best_control_exact,
            "helpful_control_cer_margin": best_control_cer - helpful["cer"],
            "helpful_control_script_margin": helpful["script"] - best_control_script,
            "helpful_control_first_prob_margin": helpful["first_prob"] - best_control_first_prob,
            "desc_minus_asc_exact": desc["exact"] - asc["exact"],
            "desc_minus_asc_first_prob": desc["first_prob"] - asc["first_prob"],
            "helpful_minus_reversed_exact": helpful["exact"] - reversed_helpful["exact"],
            "helpful_minus_reversed_first_prob": helpful["first_prob"] - reversed_helpful["first_prob"],
        }
        rows.append(row)

    helpful_control_exact_margin_mean = _mean([_safe_float(r["helpful_control_exact_margin"]) for r in rows])
    helpful_control_cer_margin_mean = _mean([_safe_float(r["helpful_control_cer_margin"]) for r in rows])
    helpful_minus_zs_exact_mean = _mean([_safe_float(r["helpful_minus_zs_exact"]) for r in rows])
    helpful_minus_zs_cer_mean = _mean([_safe_float(r["helpful_minus_zs_cer"]) for r in rows])
    desc_minus_asc_exact_mean = _mean([_safe_float(r["desc_minus_asc_exact"]) for r in rows])
    helpful_minus_reversed_exact_mean = _mean([_safe_float(r["helpful_minus_reversed_exact"]) for r in rows])

    one_b_highN_helpful_exact_delta_mean = _mean([
        _pairwise_shot_delta(rows, "1b", pair, "helpful_exact")
        for pair in ("aksharantar_hin_latin", "aksharantar_tel_latin")
    ])
    one_b_highN_helpful_cer_regret_mean = _mean([
        _pairwise_shot_delta(rows, "1b", pair, "helpful_cer")
        for pair in ("aksharantar_hin_latin", "aksharantar_tel_latin")
    ])
    four_b_highN_helpful_exact_delta_mean = _mean([
        _pairwise_shot_delta(rows, "4b", pair, "helpful_exact")
        for pair in ("aksharantar_hin_latin", "aksharantar_tel_latin")
    ])
    four_b_highN_helpful_cer_gain_mean = _mean([
        -_pairwise_shot_delta(rows, "4b", pair, "helpful_cer")
        for pair in ("aksharantar_hin_latin", "aksharantar_tel_latin")
    ])

    summary = {
        "helpful_control_exact_margin_mean": helpful_control_exact_margin_mean,
        "helpful_control_cer_margin_mean": helpful_control_cer_margin_mean,
        "helpful_minus_zs_exact_mean": helpful_minus_zs_exact_mean,
        "helpful_minus_zs_cer_mean": helpful_minus_zs_cer_mean,
        "desc_minus_asc_exact_mean": desc_minus_asc_exact_mean,
        "helpful_minus_reversed_exact_mean": helpful_minus_reversed_exact_mean,
        "one_b_highN_helpful_exact_delta_mean": one_b_highN_helpful_exact_delta_mean,
        "one_b_highN_helpful_cer_regret_mean": one_b_highN_helpful_cer_regret_mean,
        "four_b_highN_helpful_exact_delta_mean": four_b_highN_helpful_exact_delta_mean,
        "four_b_highN_helpful_cer_gain_mean": four_b_highN_helpful_cer_gain_mean,
        "positive_helpful_control_tasks": sum(1 for r in rows if _safe_float(r["helpful_control_exact_margin"]) > 0.0),
        "positive_helpful_vs_zs_tasks": sum(1 for r in rows if _safe_float(r["helpful_minus_zs_exact"]) > 0.0),
    }

    payload = {
        "metric_name": "helpful_control_exact_margin_mean",
        "metric_unit": "exact_match_points",
        "direction": "higher",
        "results_root": str(results_root),
        "found_tasks": len(rows),
        "expected_tasks": len(expected),
        "missing": missing,
        "summary": summary,
        "rows": rows,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
