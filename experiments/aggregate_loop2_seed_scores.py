#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: List[float]) -> float:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return statistics.fmean(xs)


def _stdev(vals: List[float]) -> float:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(xs) < 2:
        return 0.0 if xs else float("nan")
    return statistics.pstdev(xs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate Loop 2 score.json files across seeds.")
    ap.add_argument("--score", action="append", required=True, help="Path to a loop2 score.json; repeat for multiple seeds")
    ap.add_argument("--out", required=True, help="Output JSON path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    score_paths = [Path(p).resolve() for p in args.score]
    payloads = [_load(p) for p in score_paths]

    summary_keys = sorted({k for payload in payloads for k in (payload.get("summary") or {}).keys()})
    summary_agg: Dict[str, Dict[str, Any]] = {}
    for key in summary_keys:
        vals = [float((payload.get("summary") or {}).get(key, float("nan"))) for payload in payloads]
        summary_agg[key] = {
            "mean": _mean(vals),
            "stdev": _stdev(vals),
            "min": min([v for v in vals if math.isfinite(v)], default=float("nan")),
            "max": max([v for v in vals if math.isfinite(v)], default=float("nan")),
            "n": sum(1 for v in vals if math.isfinite(v)),
        }

    row_buckets: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for payload in payloads:
        for row in payload.get("rows") or []:
            key = (str(row.get("model")), str(row.get("pair")), int(row.get("n_icl")))
            row_buckets.setdefault(key, []).append(dict(row))

    row_summary: List[Dict[str, Any]] = []
    metric_keys = [
        "helpful_exact",
        "helpful_cer",
        "zs_exact",
        "zs_cer",
        "best_control_exact",
        "best_control_cer",
        "helpful_minus_zs_exact",
        "helpful_minus_zs_cer",
        "helpful_control_exact_margin",
        "helpful_control_cer_margin",
        "helpful_control_first_prob_margin",
        "desc_minus_asc_exact",
        "helpful_minus_reversed_exact",
    ]
    for (model, pair, n_icl), rows in sorted(row_buckets.items()):
        cell = {"model": model, "pair": pair, "n_icl": n_icl, "n_seeds": len(rows)}
        for mk in metric_keys:
            vals = [float(r.get(mk, float("nan"))) for r in rows]
            cell[mk] = {
                "mean": _mean(vals),
                "stdev": _stdev(vals),
                "min": min([v for v in vals if math.isfinite(v)], default=float("nan")),
                "max": max([v for v in vals if math.isfinite(v)], default=float("nan")),
            }
        row_summary.append(cell)

    out = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_paths": [str(p) for p in score_paths],
        "summary_aggregate": summary_agg,
        "row_aggregate": row_summary,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
