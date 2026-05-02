#!/usr/bin/env python3
"""Local CI aggregation for retained behavioral and intervention artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACTS = {
    "hindi_practical_patch_v1": REPO_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "hindi_intervention_v1": REPO_ROOT / "research/results/autoresearch/hindi_intervention_eval_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json",
    "telugu_practical_patch_1b_v1": REPO_ROOT / "research/results/autoresearch/telugu_continuation_practical_patch_eval_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json",
    "telugu_practical_patch_4b_v1": REPO_ROOT / "research/results/autoresearch/telugu_continuation_practical_patch_eval_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json",
}
CONTROL_CONDITIONS = ["icl_corrupt", "icl_random_indic", "icl_null_filler"]
META_ITEM_KEYS = {
    "item_index",
    "word_ood",
    "word_hindi",
    "word_target",
    "nearest_bank_target",
    "nearest_bank_source",
    "nearest_bank_rank",
    "nearest_bank_similarity",
    "shared_prefix_len_tokens",
    "shared_prefix_text",
    "random_controls",
}


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, n_boot: int = 5000) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if arr.size == 1:
        value = float(arr[0])
        return {"mean": value, "ci_low": value, "ci_high": value, "n": 1}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(arr.size),
    }


def _parse_artifact_arg(text: str) -> Tuple[str, Path]:
    if "=" not in str(text):
        raise ValueError(f"Artifact spec must be label=path, got: {text!r}")
    label, raw_path = str(text).split("=", 1)
    return label.strip(), Path(raw_path.strip()).resolve()


def _format_ci(ci: Mapping[str, float], digits: int = 3) -> str:
    if not ci or not math.isfinite(float(ci.get("mean", float("nan")))):
        return "nan"
    return f"{ci['mean']:.{digits}f} [{ci['ci_low']:.{digits}f}, {ci['ci_high']:.{digits}f}]"


def _extract_path_int(path: Path, prefix: str, default: int = 0) -> int:
    for part in path.parts:
        if part.startswith(prefix):
            suffix = part[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return int(default)


def _generation_cer(payload: Mapping[str, Any]) -> float:
    primary = float(payload.get("akshara_cer", float("nan")))
    if math.isfinite(primary):
        return primary
    fallback = float(payload.get("continuation_akshara_cer", float("nan")))
    if math.isfinite(fallback):
        return fallback
    return float("nan")


def _group_control_rows(item_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[int, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in item_rows:
        item_idx = int(row["item_index"])
        grouped[item_idx][str(row["condition"])] = row
    return grouped


def _extract_four_lang_item_metrics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped = _group_control_rows(payload.get("item_rows") or [])
    per_item: Dict[str, List[float]] = defaultdict(list)
    for item_idx in sorted(grouped):
        by_condition = grouped[item_idx]
        helpful = by_condition.get("icl_helpful")
        zs = by_condition.get("zs")
        if helpful is None or zs is None:
            continue
        controls = [by_condition[name] for name in CONTROL_CONDITIONS if name in by_condition]
        if not controls:
            continue
        helpful_exact = float(helpful.get("exact_match", 0.0))
        helpful_cer = _generation_cer(helpful)
        helpful_first = float(helpful.get("first_entry_correct", float("nan")))
        helpful_prob = float(helpful.get("first_prob", float("nan")))
        zs_exact = float(zs.get("exact_match", 0.0))
        zs_cer = _generation_cer(zs)
        control_best_exact = max(float(row.get("exact_match", 0.0)) for row in controls)
        control_best_cer = min(_generation_cer(row) for row in controls)
        per_item["helpful_exact"].append(helpful_exact)
        per_item["helpful_cer"].append(helpful_cer)
        per_item["helpful_first_entry_correct"].append(helpful_first)
        per_item["helpful_first_prob"].append(helpful_prob)
        per_item["helpful_minus_zs_exact"].append(helpful_exact - zs_exact)
        per_item["helpful_minus_zs_cer_improvement"].append(zs_cer - helpful_cer)
        per_item["helpful_control_exact_margin"].append(helpful_exact - control_best_exact)
        per_item["helpful_control_cer_improvement"].append(control_best_cer - helpful_cer)

    seed = int(payload.get("seed", _extract_path_int(path, "seed", 0)))
    cell_summary = {
        "model": str(payload.get("model", "")),
        "pair": str(payload.get("pair", "")),
        "n_icl": int(payload.get("n_icl", _extract_path_int(path, "nicl", 0))),
        "seed": seed,
        "path": str(path),
        "n_items": len(per_item.get("helpful_exact", [])),
        "metrics": {
            name: _bootstrap_mean_ci(values, seed=seed + 17 * (idx + 1))
            for idx, (name, values) in enumerate(sorted(per_item.items()))
        },
        "per_item": {name: [float(v) for v in values] for name, values in per_item.items()},
    }
    return cell_summary


def _scan_four_lang(root: Path) -> Dict[str, Any]:
    cell_summaries = []
    pooled: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob('neutral_filler_recency_controls.json')):
        cell = _extract_four_lang_item_metrics(path)
        cell_summaries.append(cell)
        key = (cell["model"], cell["pair"], int(cell["n_icl"]))
        for metric_name, values in cell["per_item"].items():
            pooled[key][metric_name].extend(values)

    pooled_rows = []
    for idx, key in enumerate(sorted(pooled)):
        model, pair, n_icl = key
        metric_map = pooled[key]
        pooled_rows.append(
            {
                "model": model,
                "pair": pair,
                "n_icl": int(n_icl),
                "n_items": int(len(metric_map.get("helpful_exact", []))),
                "metrics": {
                    metric_name: _bootstrap_mean_ci(values, seed=1000 + 97 * idx + j)
                    for j, (metric_name, values) in enumerate(sorted(metric_map.items()))
                },
            }
        )

    return {
        "root": str(root),
        "n_cells": int(len(cell_summaries)),
        "cell_summaries": cell_summaries,
        "pooled_rows": pooled_rows,
    }


def _intervention_names(item_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if not item_rows:
        return []
    names = []
    for key in item_rows[0].keys():
        if key not in META_ITEM_KEYS:
            value = item_rows[0][key]
            if isinstance(value, Mapping):
                names.append(str(key))
    return names


def _extract_intervention_metrics(label: str, path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item_rows = payload.get("item_rows") or []
    interventions = _intervention_names(item_rows)
    if "baseline_no_patch" not in interventions:
        raise ValueError(f"Missing baseline_no_patch in {path}")

    rows = []
    for idx, name in enumerate(interventions):
        exact = []
        cer = []
        first_entry = []
        first_gap = []
        delta_exact = []
        delta_cer_improvement = []
        delta_first_entry = []
        delta_gap = []
        for item in item_rows:
            current = item[name]
            base = item["baseline_no_patch"]
            cur_gen = current["generation"]
            base_gen = base["generation"]
            cur_first = current["first_step"]
            base_first = base["first_step"]
            exact.append(float(cur_gen.get("exact_match", 0.0)))
            cer.append(_generation_cer(cur_gen))
            first_entry.append(float(cur_gen.get("first_entry_correct", float("nan"))))
            first_gap.append(float(cur_first.get("target_minus_latin_logit", float("nan"))))
            delta_exact.append(float(cur_gen.get("exact_match", 0.0)) - float(base_gen.get("exact_match", 0.0)))
            delta_cer_improvement.append(_generation_cer(base_gen) - _generation_cer(cur_gen))
            delta_first_entry.append(float(cur_gen.get("first_entry_correct", float("nan"))) - float(base_gen.get("first_entry_correct", float("nan"))))
            delta_gap.append(float(cur_first.get("target_minus_latin_logit", float("nan"))) - float(base_first.get("target_minus_latin_logit", float("nan"))))
        rows.append(
            {
                "intervention": str(name),
                "n_items": int(len(exact)),
                "exact_match": _bootstrap_mean_ci(exact, seed=2000 + idx),
                "akshara_cer": _bootstrap_mean_ci(cer, seed=3000 + idx),
                "first_entry_correct": _bootstrap_mean_ci(first_entry, seed=4000 + idx),
                "first_token_gap_latin": _bootstrap_mean_ci(first_gap, seed=5000 + idx),
                "delta_exact_match": _bootstrap_mean_ci(delta_exact, seed=6000 + idx),
                "delta_cer_improvement": _bootstrap_mean_ci(delta_cer_improvement, seed=7000 + idx),
                "delta_first_entry_correct": _bootstrap_mean_ci(delta_first_entry, seed=8000 + idx),
                "delta_first_token_gap_latin": _bootstrap_mean_ci(delta_gap, seed=9000 + idx),
            }
        )

    return {
        "label": str(label),
        "path": str(path),
        "experiment": str(payload.get("experiment", "")),
        "model": str(payload.get("model", "")),
        "pair": str(payload.get("pair", "")),
        "seed": int(payload.get("seed", 0)),
        "n_icl": int(payload.get("n_icl", 0)),
        "rows": rows,
    }


def _markdown_report(four_lang: Mapping[str, Any], interventions: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Review Confidence Interval Analysis")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append("")
    lines.append("## Four-language pooled behavioral cells")
    lines.append("")
    lines.append("| Cell | N | Helpful EM | Helpful-control EM margin | Helpful-control CER improvement | Helpful-ZS EM delta | Helpful-ZS CER improvement |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in four_lang.get("pooled_rows", []):
        metrics = row["metrics"]
        cell = f"{row['model']} {row['pair']} nicl={row['n_icl']}"
        lines.append(
            "| "
            + cell
            + f" | {row['n_items']}"
            + f" | {_format_ci(metrics['helpful_exact'])}"
            + f" | {_format_ci(metrics['helpful_control_exact_margin'])}"
            + f" | {_format_ci(metrics['helpful_control_cer_improvement'])}"
            + f" | {_format_ci(metrics['helpful_minus_zs_exact'])}"
            + f" | {_format_ci(metrics['helpful_minus_zs_cer_improvement'])} |\n"
        )
    lines.append("")
    lines.append("## Intervention artifacts")
    lines.append("")
    for artifact in interventions:
        lines.append(f"### {artifact['label']}")
        lines.append("")
        lines.append(f"Artifact: `{artifact['path']}`")
        lines.append("")
        lines.append("| Intervention | N | EM | dEM vs baseline | CER | CER improvement vs baseline | First-entry delta | Gap delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in artifact.get("rows", []):
            lines.append(
                "| "
                + str(row["intervention"])
                + f" | {row['n_items']}"
                + f" | {_format_ci(row['exact_match'])}"
                + f" | {_format_ci(row['delta_exact_match'])}"
                + f" | {_format_ci(row['akshara_cer'])}"
                + f" | {_format_ci(row['delta_cer_improvement'])}"
                + f" | {_format_ci(row['delta_first_entry_correct'])}"
                + f" | {_format_ci(row['delta_first_token_gap_latin'])} |\n"
            )
        lines.append("")
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate local confidence intervals for retained artifacts.")
    ap.add_argument(
        "--four-lang-root",
        type=str,
        default=str(REPO_ROOT / "research/results/autoresearch/four_lang_thesis_panel"),
    )
    ap.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Extra intervention artifact as label=/abs/or/relative/path.json . Can be repeated.",
    )
    ap.add_argument(
        "--write-prefix",
        type=str,
        default=str(REPO_ROOT / "outputs/review_confidence_intervals_2026-03-31"),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    four_lang_root = Path(args.four_lang_root).resolve()
    artifact_specs = []
    for label, path in DEFAULT_ARTIFACTS.items():
        if path.exists():
            artifact_specs.append((label, path))
    for spec in args.artifact:
        artifact_specs.append(_parse_artifact_arg(spec))

    four_lang = _scan_four_lang(four_lang_root)
    interventions = []
    for label, path in artifact_specs:
        if path.exists():
            interventions.append(_extract_intervention_metrics(label, path))

    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "four_lang": four_lang,
        "interventions": interventions,
    }
    prefix = Path(args.write_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix('.json')
    md_path = prefix.with_suffix('.md')
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    md_path.write_text(_markdown_report(four_lang, interventions), encoding='utf-8')
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "n_four_lang_cells": four_lang.get("n_cells", 0),
        "n_intervention_artifacts": len(interventions),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
