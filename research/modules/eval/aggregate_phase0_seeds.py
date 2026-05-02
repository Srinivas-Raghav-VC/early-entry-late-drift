from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate Phase 0A per-seed result packets.")
    ap.add_argument("--input-dir", default="research/results/phase0")
    ap.add_argument("--glob", default="phase0a_packet_results_seed*.json")
    ap.add_argument("--out", default="research/results/phase0/table_phase0_seed_aggregate.csv")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No seed packets found under {input_dir} with pattern {args.glob}")

    behavior_buckets: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}
    judge_buckets: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}

    seeds_seen: list[int] = []
    for fp in files:
        payload = _read_json(fp)
        seed = int(payload.get("run_config", {}).get("split_seed", -1))
        seeds_seen.append(seed)

        for row in payload.get("behavior_summary", []):
            key = (
                str(row.get("language", "")),
                str(row.get("prompt_template", "canonical")),
                str(row.get("icl_variant", "helpful")),
                int(row.get("N", 0)),
            )
            bucket = behavior_buckets.setdefault(
                key,
                {
                    "exact_match_rate": [],
                    "akshara_CER_mean": [],
                    "script_validity_rate": [],
                    "empty_or_refusal_rate": [],
                },
            )
            for metric in list(bucket.keys()):
                bucket[metric].append(float(row.get(metric, 0.0)))

        for row in payload.get("judge_curve", []):
            key = (
                str(row.get("language", "")),
                str(row.get("prompt_template", "canonical")),
                str(row.get("icl_variant", "helpful")),
                int(row.get("N", 0)),
            )
            bucket = judge_buckets.setdefault(
                key,
                {
                    "judge_acceptable_rate": [],
                    "judge_exact_rate": [],
                    "judge_script_wrong_rate": [],
                    "judge_invalid_rate": [],
                },
            )
            for metric in list(bucket.keys()):
                value = row.get(metric, None)
                if value is None:
                    continue
                bucket[metric].append(float(value))

    out_rows: list[dict[str, Any]] = []
    all_keys = sorted(set(behavior_buckets.keys()) | set(judge_buckets.keys()))
    for language, prompt_template, icl_variant, n in all_keys:
        row: dict[str, Any] = {
            "language": language,
            "prompt_template": prompt_template,
            "icl_variant": icl_variant,
            "N": n,
            "n_seed_packets": len(files),
            "seeds_seen": ",".join(str(s) for s in sorted(set(seeds_seen))),
        }

        for metric, vals in behavior_buckets.get((language, prompt_template, icl_variant, n), {}).items():
            m, s = _agg(vals)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s

        for metric, vals in judge_buckets.get((language, prompt_template, icl_variant, n), {}).items():
            m, s = _agg(vals)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s

        out_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, out_rows)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
