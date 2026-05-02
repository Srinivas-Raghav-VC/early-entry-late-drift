#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "").strip())


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(xs) if xs else float("nan")


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_items(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    meta_by_index = {
        int(row["item_index"]): row
        for row in (payload.get("prompt_ordering_metadata_by_item") or [])
        if "item_index" in row
    }
    merged: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("item_rows") or []:
        idx = int(row["item_index"])
        item = merged.setdefault(
            idx,
            {
                "item_index": idx,
                "word_ood": str(row.get("word_ood", "")),
                "word_hindi": str(row.get("word_hindi", "")),
                "conditions": {},
                "prompt_metadata": meta_by_index.get(idx, {}),
            },
        )
        item["conditions"][str(row.get("condition"))] = dict(row)
    return [merged[k] for k in sorted(merged)]


def _bank_match_info(item: Mapping[str, Any], prediction: str) -> Dict[str, Any]:
    pred = normalize_text(prediction)
    if not pred:
        return {"is_bank_copy": False, "rank": None, "source": None, "target": None, "similarity": None}
    desc = (((item.get("prompt_metadata") or {}).get("helpful_similarity_desc")) or [])
    for rank, ex in enumerate(desc, start=1):
        tgt = normalize_text(str(ex.get("target", "")))
        if tgt and pred == tgt:
            return {
                "is_bank_copy": True,
                "rank": int(rank),
                "source": str(ex.get("source", "")),
                "target": str(ex.get("target", "")),
                "similarity": float(ex.get("similarity")) if ex.get("similarity") is not None else None,
            }
    return {"is_bank_copy": False, "rank": None, "source": None, "target": None, "similarity": None}


def _extract_case(item: Mapping[str, Any]) -> Dict[str, Any]:
    conds = item.get("conditions") or {}
    helpful = conds.get("icl_helpful", {})
    zs = conds.get("zs", {})
    corrupt = conds.get("icl_corrupt", {})
    helpful_pred = str(helpful.get("prediction", ""))
    gold = str(item.get("word_hindi", ""))
    source = str(item.get("word_ood", ""))
    bank_info = _bank_match_info(item, helpful_pred)
    return {
        "item_index": int(item.get("item_index", -1)),
        "word_ood": source,
        "gold": gold,
        "helpful_prediction": helpful_pred,
        "zs_prediction": str(zs.get("prediction", "")),
        "corrupt_prediction": str(corrupt.get("prediction", "")),
        "helpful_exact": float(helpful.get("exact_match", float("nan"))),
        "zs_exact": float(zs.get("exact_match", float("nan"))),
        "helpful_cer": float(helpful.get("akshara_cer", float("nan"))),
        "zs_cer": float(zs.get("akshara_cer", float("nan"))),
        "helpful_first_entry_correct": float(helpful.get("first_entry_correct", float("nan"))),
        "zs_first_entry_correct": float(zs.get("first_entry_correct", float("nan"))),
        "helpful_script_compliance": float(helpful.get("script_compliance", float("nan"))),
        "helpful_prompt_tokens": int(helpful.get("prompt_tokens", -1)) if helpful.get("prompt_tokens") is not None else None,
        "helpful_is_source_copy": normalize_text(helpful_pred) == normalize_text(source),
        **bank_info,
    }


def _sample(cases: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return cases[: max(0, int(limit))]


def _format_case(case: Mapping[str, Any]) -> str:
    bits = [
        f"item={case.get('item_index')}",
        f"src={case.get('word_ood')}",
        f"gold={case.get('gold')}",
        f"helpful={case.get('helpful_prediction')}",
        f"zs={case.get('zs_prediction')}",
        f"helpful_exact={case.get('helpful_exact')}",
        f"zs_exact={case.get('zs_exact')}",
        f"helpful_cer={case.get('helpful_cer')}",
    ]
    if case.get("is_bank_copy"):
        bits.append(f"bank_rank={case.get('rank')}")
        bits.append(f"bank_source={case.get('source')}")
        bits.append(f"bank_target={case.get('target')}")
    if case.get("helpful_is_source_copy"):
        bits.append("SOURCE_COPY")
    return " | ".join(str(x) for x in bits)


def build_audit(payload: Mapping[str, Any], *, max_examples: int) -> Dict[str, Any]:
    items = _merge_items(payload)
    cases = [_extract_case(item) for item in items]

    helpful_beats_zs = [
        c for c in cases
        if math.isfinite(c["helpful_exact"]) and math.isfinite(c["zs_exact"]) and c["helpful_exact"] > c["zs_exact"]
    ]
    helpful_loses_to_zs = [
        c for c in cases
        if math.isfinite(c["helpful_exact"]) and math.isfinite(c["zs_exact"]) and c["helpful_exact"] < c["zs_exact"]
    ]
    helpful_bank_copy = [c for c in cases if bool(c.get("is_bank_copy"))]
    helpful_source_copy = [c for c in cases if bool(c.get("helpful_is_source_copy"))]
    helpful_first_correct_exact_wrong = [
        c for c in cases
        if c.get("helpful_first_entry_correct") == 1.0 and c.get("helpful_exact") == 0.0
    ]
    helpful_near_miss = [
        c for c in cases
        if c.get("helpful_first_entry_correct") == 1.0
        and c.get("helpful_exact") == 0.0
        and c.get("helpful_script_compliance") == 1.0
        and not bool(c.get("is_bank_copy"))
    ]

    helpful_bank_copy_sorted = sorted(
        helpful_bank_copy,
        key=lambda c: (999999 if c.get("rank") is None else int(c["rank"]), float(c.get("helpful_cer", 999.0))),
    )
    helpful_loses_to_zs_sorted = sorted(helpful_loses_to_zs, key=lambda c: (float(c.get("helpful_exact", 0.0)) - float(c.get("zs_exact", 0.0)), float(c.get("helpful_cer", 999.0))))
    helpful_beats_zs_sorted = sorted(helpful_beats_zs, key=lambda c: (float(c.get("zs_exact", 0.0)) - float(c.get("helpful_exact", 0.0)), float(c.get("helpful_cer", 999.0))))

    summary = {
        "n_items": int(len(cases)),
        "helpful_exact_rate": _safe_mean([c["helpful_exact"] for c in cases]),
        "zs_exact_rate": _safe_mean([c["zs_exact"] for c in cases]),
        "helpful_cer_mean": _safe_mean([c["helpful_cer"] for c in cases]),
        "zs_cer_mean": _safe_mean([c["zs_cer"] for c in cases]),
        "helpful_first_entry_correct_rate": _safe_mean([c["helpful_first_entry_correct"] for c in cases]),
        "zs_first_entry_correct_rate": _safe_mean([c["zs_first_entry_correct"] for c in cases]),
        "helpful_beats_zs_count": int(len(helpful_beats_zs)),
        "helpful_loses_to_zs_count": int(len(helpful_loses_to_zs)),
        "helpful_bank_copy_count": int(len(helpful_bank_copy)),
        "helpful_source_copy_count": int(len(helpful_source_copy)),
        "helpful_first_correct_exact_wrong_count": int(len(helpful_first_correct_exact_wrong)),
        "helpful_near_miss_count": int(len(helpful_near_miss)),
        "bank_copy_median_rank": statistics.median([int(c["rank"]) for c in helpful_bank_copy if c.get("rank") is not None]) if helpful_bank_copy else None,
    }

    exemplar_sets = {
        "helpful_beats_zs": _sample(helpful_beats_zs_sorted, max_examples),
        "helpful_loses_to_zs": _sample(helpful_loses_to_zs_sorted, max_examples),
        "helpful_bank_copy": _sample(helpful_bank_copy_sorted, max_examples),
        "helpful_source_copy": _sample(helpful_source_copy, max_examples),
        "helpful_first_correct_exact_wrong": _sample(helpful_first_correct_exact_wrong, max_examples),
        "helpful_near_miss": _sample(helpful_near_miss, max_examples),
    }

    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": str(payload.get("experiment", "")),
        "pair": str(payload.get("pair", "")),
        "model": str(payload.get("model", "")),
        "seed": int(payload.get("seed", -1)),
        "summary": summary,
        "exemplar_sets": exemplar_sets,
    }


def render_markdown(audit: Mapping[str, Any], *, source_path: Path) -> str:
    s = audit.get("summary") or {}
    lines: List[str] = []
    lines.append(f"# Manual audit packet: {audit.get('model')} × {audit.get('pair')} × seed {audit.get('seed')}")
    lines.append("")
    lines.append(f"Source artifact: `{source_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in [
        "n_items",
        "helpful_exact_rate",
        "zs_exact_rate",
        "helpful_cer_mean",
        "zs_cer_mean",
        "helpful_first_entry_correct_rate",
        "zs_first_entry_correct_rate",
        "helpful_beats_zs_count",
        "helpful_loses_to_zs_count",
        "helpful_bank_copy_count",
        "helpful_source_copy_count",
        "helpful_first_correct_exact_wrong_count",
        "helpful_near_miss_count",
        "bank_copy_median_rank",
    ]:
        lines.append(f"- **{key}**: {s.get(key)}")
    lines.append("")

    pretty_names = {
        "helpful_beats_zs": "Helpful beats zero-shot",
        "helpful_loses_to_zs": "Helpful loses to zero-shot",
        "helpful_bank_copy": "Helpful prompt-bank copies",
        "helpful_source_copy": "Helpful source copies",
        "helpful_first_correct_exact_wrong": "Helpful first-correct but exact-wrong",
        "helpful_near_miss": "Helpful likely near-misses (not bank copies)",
    }

    for key, title in pretty_names.items():
        lines.append(f"## {title}")
        lines.append("")
        rows = (audit.get("exemplar_sets") or {}).get(key) or []
        if not rows:
            lines.append("_No examples in this slice._")
            lines.append("")
            continue
        for row in rows:
            lines.append(f"- {_format_case(row)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a manual-audit packet from a neutral_filler_recency_controls artifact.")
    ap.add_argument("--input", required=True, help="Path to neutral_filler_recency_controls.json")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--out-md", required=True, help="Output Markdown path")
    ap.add_argument("--max-examples", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    payload = _load_payload(input_path)
    audit = build_audit(payload, max_examples=int(args.max_examples))

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    _write_json(out_json, audit)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(audit, source_path=input_path), encoding="utf-8")
    print(json.dumps(_json_safe(audit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
