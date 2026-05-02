#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

TASKS = [
    ("270m", "aksharantar_hin_latin"),
    ("270m", "aksharantar_tel_latin"),
    ("1b", "aksharantar_hin_latin"),
    ("1b", "aksharantar_tel_latin"),
    ("4b", "aksharantar_hin_latin"),
    ("4b", "aksharantar_tel_latin"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_summary(results_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model, pair in TASKS:
        path = results_root / "premise_gate" / model / f"{pair}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload = _load_json(path)
        summary = payload.get("summary", {})
        full_set = summary.get("full_set", {})
        gaps = summary.get("gaps", {}).get("full_set", {})
        em_gap = gaps.get("exact_match", {})
        cer_gap = gaps.get("akshara_cer", {})
        row = {
            "model": model,
            "pair": pair,
            "path": str(path),
            "explicit_zs_exact_match": _safe_float(full_set.get("explicit_zs", {}).get("exact_match")),
            "icl64_exact_match": _safe_float(full_set.get("icl64", {}).get("exact_match")),
            "explicit_zs_akshara_cer": _safe_float(full_set.get("explicit_zs", {}).get("akshara_cer")),
            "icl64_akshara_cer": _safe_float(full_set.get("icl64", {}).get("akshara_cer")),
            "premise_gap_exact": _safe_float(em_gap.get("gap_mean")),
            "premise_gap_exact_ci_excludes_zero": bool(em_gap.get("ci_excludes_zero", False)),
            "premise_gap_cer": _safe_float(cer_gap.get("gap_mean")),
            "premise_gap_cer_ci_excludes_zero": bool(cer_gap.get("ci_excludes_zero", False)),
            "runbook_gate_full_set": bool(summary.get("runbook_gate", {}).get("full_set_clearly_above_floor_noise", False)),
        }
        rows.append(row)

    if rows:
        anchor_gap_exact = mean(r["premise_gap_exact"] for r in rows)
        anchor_gap_cer = mean(r["premise_gap_cer"] for r in rows)
        positive_exact_tasks = sum(1 for r in rows if r["premise_gap_exact"] > 0)
        ci_positive_exact_tasks = sum(1 for r in rows if r["premise_gap_exact_ci_excludes_zero"])
        gate_positive_tasks = sum(1 for r in rows if r["runbook_gate_full_set"])
    else:
        anchor_gap_exact = float("nan")
        anchor_gap_cer = float("nan")
        positive_exact_tasks = 0
        ci_positive_exact_tasks = 0
        gate_positive_tasks = 0

    return {
        "metric_name": "premise_gap_exact_mean",
        "metric_unit": "exact_match_points",
        "direction": "higher",
        "results_root": str(results_root),
        "found_tasks": len(rows),
        "expected_tasks": len(TASKS),
        "missing": missing,
        "summary": {
            "premise_gap_exact_mean": anchor_gap_exact,
            "premise_gap_cer_mean": anchor_gap_cer,
            "positive_exact_tasks": positive_exact_tasks,
            "ci_positive_exact_tasks": ci_positive_exact_tasks,
            "runbook_gate_positive_tasks": gate_positive_tasks,
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score cross-scale premise-gate outputs for Loop 1 autoresearch.")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_summary(Path(args.results_root).resolve())
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    if str(args.out).strip():
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob, encoding="utf-8")
    print(blob)
    return 0 if payload["found_tasks"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
