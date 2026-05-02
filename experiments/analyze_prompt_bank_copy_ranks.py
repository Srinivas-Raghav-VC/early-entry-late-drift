#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "research" / "results" / "autoresearch" / "loop2_vm_controls"

DEFAULT_TASKS = [
    (ROOT / "loop2_full" / "raw", "1b", "aksharantar_tel_latin", 64),
    (ROOT / "loop2_full" / "raw", "4b", "aksharantar_tel_latin", 64),
    (ROOT / "loop2_full" / "raw", "1b", "aksharantar_hin_latin", 64),
    (ROOT / "threshold_1b_seed42" / "raw", "1b", "aksharantar_tel_latin", 48),
    (ROOT / "threshold_1b_seed42" / "raw", "1b", "aksharantar_tel_latin", 56),
    (ROOT / "threshold_1b_seed42" / "raw", "1b", "aksharantar_tel_latin", 64),
]

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "loop2_bank_copy_rank_2026-03-29.json"


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


def _payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(root: Path, model: str, pair: str, n_icl: int) -> Path:
    return root / model / pair / f"nicl{n_icl}" / "neutral_filler_recency_controls.json"


def _cell_summary(root: Path, model: str, pair: str, n_icl: int) -> Dict[str, Any]:
    path = _path(root, model, pair, n_icl)
    payload = _payload(path)
    by_item = {str(item.get("word_ood", "")): item for item in payload.get("prompt_ordering_metadata_by_item", [])}
    rows = [row for row in payload.get("item_rows", []) if str(row.get("condition")) == "icl_helpful"]
    copied: List[Dict[str, Any]] = []
    for row in rows:
        pred = str(row.get("prediction", "")).strip()
        meta = by_item.get(str(row.get("word_ood", "")))
        if not meta:
            continue
        desc = list(meta.get("helpful_similarity_desc") or [])
        matches = [d for d in desc if str(d.get("target", "")).strip() == pred]
        if not matches:
            continue
        # If multiple entries share the same target string, keep all candidates but report the best rank.
        best = min(matches, key=lambda d: int(d.get("position", 10**9)))
        copied.append(
            {
                "word_ood": str(row.get("word_ood", "")),
                "gold": str(row.get("word_hindi", "")),
                "prediction": pred,
                "rank_by_source_similarity": int(best.get("position", 0)) + 1,
                "matched_source": str(best.get("source", "")),
                "matched_target": str(best.get("target", "")),
                "matched_similarity": float(best.get("similarity", 0.0)),
                "candidate_match_count": int(len(matches)),
                "exact_match": float(row.get("exact_match", 0.0)),
                "first_entry_correct": float(row.get("first_entry_correct", 0.0)),
                "akshara_cer": float(row.get("akshara_cer", 0.0)),
            }
        )
    ranks = [int(row["rank_by_source_similarity"]) for row in copied]
    n_rows = len(rows)
    return {
        "path": str(path),
        "model": str(model),
        "pair": str(pair),
        "n_icl": int(n_icl),
        "n_items": int(n_rows),
        "copied_bank_targets": int(len(copied)),
        "copy_rate": float(len(copied)) / float(n_rows) if n_rows else 0.0,
        "rank_summary": {
            "mean": float(statistics.mean(ranks)) if ranks else None,
            "median": float(statistics.median(ranks)) if ranks else None,
            "top1_rate_among_copies": float(sum(1 for r in ranks if r == 1)) / float(len(ranks)) if ranks else 0.0,
            "top5_rate_among_copies": float(sum(1 for r in ranks if r <= 5)) / float(len(ranks)) if ranks else 0.0,
            "top10_rate_among_copies": float(sum(1 for r in ranks if r <= 10)) / float(len(ranks)) if ranks else 0.0,
        },
        "copied_examples": copied[:20],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze similarity ranks of copied prompt-bank targets from existing Loop 2 artifacts.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [_cell_summary(*task) for task in DEFAULT_TASKS]
    out = Path(args.out).resolve()
    _write_json(out, {"cells": summaries})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
