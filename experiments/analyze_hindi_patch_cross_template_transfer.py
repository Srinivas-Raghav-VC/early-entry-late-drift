#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "within_canonical": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "within_tagged": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_tagged_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "taggedvec_on_canonical": PROJECT_ROOT / "research/results/autoresearch/hindi_patch_cross_template_taggedvec_on_canonical_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "canonicalvec_on_tagged": PROJECT_ROOT / "research/results/autoresearch/hindi_patch_cross_template_canonicalvec_on_tagged_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
}
OUT_JSON = PROJECT_ROOT / "outputs/hindi_patch_cross_template_transfer_2026-04-01.json"
OUT_MD = PROJECT_ROOT / "outputs/hindi_patch_cross_template_transfer_2026-04-01.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {row["intervention"]: row for row in payload["summary_rows"]}
    base = rows["baseline_no_patch"]
    chosen = rows["chosen_mean_shift"]
    return {
        "prompt_variant": payload.get("prompt_variant", ""),
        "external_patch_source": payload.get("external_patch_source", ""),
        "selection_mode": payload.get("selection_mode", ""),
        "selected_alpha": float(payload["selected_alpha"]),
        "delta_exact_match": float(chosen["delta_exact_match"]["mean"]),
        "delta_cer_improvement": float(chosen["delta_cer_improvement"]["mean"]),
        "delta_first_entry_correct": float(chosen["delta_first_entry_correct"]["mean"]),
        "delta_first_token_gap_latin": float(chosen["delta_first_token_gap_latin"]["mean"]),
        "baseline_exact_match": float(base["exact_match"]["mean"]),
        "chosen_exact_match": float(chosen["exact_match"]["mean"]),
        "baseline_akshara_cer": float(base["akshara_cer"]["mean"]),
        "chosen_akshara_cer": float(chosen["akshara_cer"]["mean"]),
    }


def main() -> int:
    summary = {name: extract_summary(load_json(path)) for name, path in PATHS.items()}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Hindi cross-template patch transfer summary (2026-04-01)",
        "",
        "| run | prompt variant | external patch source | selection mode | alpha | ΔEM | ΔCER improvement | Δfirst-entry | Δgap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        src = Path(row["external_patch_source"]).name if row["external_patch_source"] else "self-fit"
        lines.append(
            f"| {name} | {row['prompt_variant']} | {src} | {row['selection_mode']} | {row['selected_alpha']:.2f} | {row['delta_exact_match']:+.3f} | {row['delta_cer_improvement']:+.3f} | {row['delta_first_entry_correct']:+.3f} | {row['delta_first_token_gap_latin']:+.3f} |"
        )
    lines.append("")
    lines.append("Interpretation target: compare whether behavior tracks the prompt template or the imported vector source more strongly.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
