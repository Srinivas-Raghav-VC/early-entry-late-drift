#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "research" / "results" / "autoresearch" / "loop2_vm_controls" / "loop2_full" / "raw" / "1b" / "aksharantar_tel_latin" / "nicl64" / "neutral_filler_recency_controls.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "loop2_telugu_retrieval_conditions_2026-03-29.json"
CONDITIONS = [
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_reversed",
    "icl_corrupt",
    "icl_random_indic",
    "icl_null_filler",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value == value and value not in (float("inf"), -float("inf")):
            return value
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _summary_for_condition(payload: Dict[str, Any], condition: str) -> Dict[str, Any]:
    rows = [row for row in payload.get("item_rows", []) if str(row.get("condition")) == condition]
    helpful_rows = [row for row in payload.get("item_rows", []) if str(row.get("condition")) == "icl_helpful"]
    helpful_pred = {str(row.get("word_ood", "")): str(row.get("prediction", "")).strip() for row in helpful_rows}
    meta_by_item = {str(item.get("word_ood", "")): item for item in payload.get("prompt_ordering_metadata_by_item", [])}

    copied: List[Dict[str, Any]] = []
    same_as_helpful = 0
    for row in rows:
        word = str(row.get("word_ood", ""))
        pred = str(row.get("prediction", "")).strip()
        if pred == helpful_pred.get(word, ""):
            same_as_helpful += 1
        meta = meta_by_item.get(word)
        if not meta:
            continue
        desc = list(meta.get("helpful_similarity_desc") or [])
        matches = [d for d in desc if str(d.get("target", "")).strip() == pred]
        if not matches:
            continue
        best = min(matches, key=lambda d: int(d.get("position", 10**9)))
        copied.append(
            {
                "word_ood": word,
                "gold": str(row.get("word_hindi", "")),
                "prediction": pred,
                "rank_by_helpful_similarity_desc": int(best.get("position", 0)) + 1,
                "matched_source": str(best.get("source", "")),
                "matched_similarity": float(best.get("similarity", 0.0)),
                "exact_match": float(row.get("exact_match", 0.0)),
                "first_entry_correct": float(row.get("first_entry_correct", 0.0)),
                "akshara_cer": float(row.get("akshara_cer", 0.0)),
            }
        )

    ranks = [int(x["rank_by_helpful_similarity_desc"]) for x in copied]
    summary_cond = dict(payload.get("summary_by_condition", {}).get(condition, {}))
    return {
        "condition": condition,
        "n_items": int(len(rows)),
        "same_prediction_as_helpful_count": int(same_as_helpful),
        "same_prediction_as_helpful_rate": float(same_as_helpful) / float(len(rows)) if rows else 0.0,
        "copy_rate": float(len(copied)) / float(len(rows)) if rows else 0.0,
        "rank_summary": {
            "mean": float(statistics.mean(ranks)) if ranks else None,
            "median": float(statistics.median(ranks)) if ranks else None,
            "top1_rate_among_copies": float(sum(1 for r in ranks if r == 1)) / float(len(ranks)) if ranks else 0.0,
            "top5_rate_among_copies": float(sum(1 for r in ranks if r <= 5)) / float(len(ranks)) / 1.0 if ranks else 0.0,
            "top10_rate_among_copies": float(sum(1 for r in ranks if r <= 10)) / float(len(ranks)) if ranks else 0.0,
        },
        "behavior_summary": {
            "mean_exact_match": summary_cond.get("mean_exact_match"),
            "mean_akshara_cer": summary_cond.get("mean_akshara_cer"),
            "mean_first_prob": summary_cond.get("mean_first_prob"),
            "mean_first_entry_correct": summary_cond.get("mean_first_entry_correct"),
        },
        "copied_examples": copied[:20],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Telugu retrieval/copy behavior across existing Loop 2 conditions.")
    ap.add_argument("--path", type=str, default=str(DEFAULT_PATH))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    results = [_summary_for_condition(payload, condition) for condition in CONDITIONS]
    out = Path(args.out).resolve()
    _write_json(out, {"source_path": str(Path(args.path).resolve()), "conditions": results})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
