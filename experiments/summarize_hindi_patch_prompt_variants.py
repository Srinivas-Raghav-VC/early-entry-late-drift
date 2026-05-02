#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_PATHS = {
    "canonical": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "compact": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_compact_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "tagged": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_tagged_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
}
OUT_JSON = PROJECT_ROOT / "outputs/hindi_patch_prompt_variant_summary_2026-04-01.json"
OUT_MD = PROJECT_ROOT / "outputs/hindi_patch_prompt_variant_summary_2026-04-01.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    payloads = {name: load_json(path) for name, path in VARIANT_PATHS.items()}
    rows = []
    for variant, payload in payloads.items():
        summary = {row["intervention"]: row for row in payload["summary_rows"]}
        chosen = summary["chosen_mean_shift"]
        sign_flip = summary["chosen_sign_flip"]
        rows.append(
            {
                "variant": variant,
                "selected_alpha": float(payload["selected_alpha"]),
                "delta_exact": float(chosen["delta_exact_match"]["mean"]),
                "delta_cer": float(chosen["delta_cer_improvement"]["mean"]),
                "delta_first": float(chosen["delta_first_entry_correct"]["mean"]),
                "delta_gap": float(chosen["delta_first_token_gap_latin"]["mean"]),
                "sign_flip_delta_cer": float(sign_flip["delta_cer_improvement"]["mean"]),
            }
        )
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Hindi fixed-patch prompt-variant summary (2026-04-01)",
        "",
        "| variant | selected α | mean-shift ΔEM | mean-shift ΔCER | mean-shift Δentry | mean-shift Δgap | sign-flip ΔCER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['selected_alpha']:.2f} | {row['delta_exact']:+.3f} | {row['delta_cer']:+.3f} | {row['delta_first']:+.3f} | {row['delta_gap']:+.3f} | {row['sign_flip_delta_cer']:+.3f} |"
        )
    lines.append("")
    lines.append("The main robustness question is whether the bounded fixed patch stays directionally helpful outside the canonical template. This summary is meant to be merged into the paper once the compact and tagged runs finish.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
