#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "research/results/autoresearch/prompt_composition_ablation_v1"
OUT_JSON = PROJECT_ROOT / "outputs/fuzzy_bank_threshold_sensitivity_2026-04-01.json"
OUT_MD = PROJECT_ROOT / "outputs/fuzzy_bank_threshold_sensitivity_2026-04-01.md"

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
KEEP_CONDITIONS = [
    "icl_helpful",
    "icl_helpful_similarity_desc",
    "icl_helpful_similarity_asc",
    "icl_helpful_drop_nearest_replace_far",
    "icl_helpful_drop_top2_replace_far",
    "icl_corrupt",
]
LABELS = {
    "icl_helpful": "helpful",
    "icl_helpful_similarity_desc": "sim-front",
    "icl_helpful_similarity_asc": "sim-back",
    "icl_helpful_drop_nearest_replace_far": "drop-nearest",
    "icl_helpful_drop_top2_replace_far": "drop-top2",
    "icl_corrupt": "corrupt",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_rates(payload: dict[str, Any]) -> dict[str, Any]:
    item_rows = payload["item_rows"]
    out: dict[str, Any] = {}
    for condition in KEEP_CONDITIONS:
        cond_rows = [row for row in item_rows if row["condition"] == condition]
        vals = []
        for thr in THRESHOLDS:
            positives = 0
            usable = 0
            for row in cond_rows:
                sim = row.get("max_bank_similarity")
                if sim is None:
                    continue
                usable += 1
                if float(row.get("exact_match", 0.0)) < 0.5 and float(sim) >= float(thr):
                    positives += 1
            vals.append(
                {
                    "threshold": float(thr),
                    "n_usable": int(usable),
                    "fuzzy_bank_copy_rate": float(positives / usable) if usable else None,
                }
            )
        out[condition] = {
            "label": LABELS[condition],
            "rates": vals,
            "stored_rate_at_0_85": float(payload["summary_by_condition"][condition]["fuzzy_bank_copy_rate"]),
        }
    return out


def main() -> int:
    payloads = {
        "Gemma 1B Telugu": load_json(RESULT_ROOT / "1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"),
        "Gemma 4B Telugu": load_json(RESULT_ROOT / "4b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"),
    }
    results = {model: recompute_rates(payload) for model, payload in payloads.items()}

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Fuzzy bank-copy threshold sensitivity (2026-04-01)",
        "",
        "## Objective",
        "",
        "Check whether the Telugu prompt-composition conclusions depend strongly on the chosen fuzzy bank-copy threshold (default paper threshold = 0.85).",
        "",
    ]
    for model, model_res in results.items():
        lines.extend([f"## {model}", ""])
        header = "| condition | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |"
        sep = "|---|---:|---:|---:|---:|---:|"
        lines.extend([header, sep])
        for condition in KEEP_CONDITIONS:
            row = model_res[condition]
            vals = [f"{r['fuzzy_bank_copy_rate']:.3f}" for r in row["rates"]]
            lines.append(f"| {row['label']} | " + " | ".join(vals) + " |")
        lines.append("")
        if model == "Gemma 1B Telugu":
            lines.extend(
                [
                    "**Readout.** The threshold sweep preserves the same broad picture as the paper: fuzzy-copy stays high for the main helpful / similarity-heavy 1B conditions, `sim-back` becomes materially lower once the threshold is moderately strict, and dropping nearest examples weakens fuzzy copy without restoring exact success. The qualitative prompt-composition conclusion is therefore not an artifact of choosing 0.85 specifically.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Readout.** All fuzzy-copy rates remain low for the helpful 4B conditions across thresholds, while `corrupt` stays clearly higher. This preserves the paper's 4B contrast: low-copy composition remains stable under threshold variation.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Sources",
            "",
            "- `research/results/autoresearch/prompt_composition_ablation_v1/1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json`",
            "- `research/results/autoresearch/prompt_composition_ablation_v1/4b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
