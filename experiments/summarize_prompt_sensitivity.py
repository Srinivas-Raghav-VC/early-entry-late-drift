#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "research/results/autoresearch/prompt_sensitivity_64_v1"
OUT_JSON = PROJECT_ROOT / "outputs/prompt_sensitivity_64_summary_2026-04-01.json"
OUT_MD = PROJECT_ROOT / "outputs/prompt_sensitivity_64_summary_2026-04-01.md"
PAIRS = [("aksharantar_hin_latin", "Hindi"), ("aksharantar_tel_latin", "Telugu")]
MODELS = [("1b", "Gemma 1B"), ("4b", "Gemma 4B")]
VARIANTS = ["canonical", "compact", "tagged"]
CONDITIONS = ["zs", "icl_helpful", "icl_corrupt"]
KEYS = ["mean_exact_match", "mean_akshara_cer", "mean_first_entry_correct", "exact_bank_copy_rate"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    summary: dict[str, Any] = {}
    lines = ["# Prompt-format sensitivity summary (2026-04-01)", ""]
    for model_key, model_label in MODELS:
        lines.append(f"## {model_label}")
        lines.append("")
        for pair_key, pair_label in PAIRS:
            path = ROOT / model_key / pair_key / "seed42" / "nicl64" / "prompt_sensitivity_check.json"
            payload = load_json(path)
            s = payload["summary_by_variant"]
            summary[f"{model_key}:{pair_key}"] = s
            lines.append(f"### {pair_label}")
            lines.append("")
            for condition in CONDITIONS:
                lines.append(f"#### {condition}")
                lines.append("")
                lines.append("| variant | exact | CER | first-entry | bank-copy |")
                lines.append("|---|---:|---:|---:|---:|")
                for variant in VARIANTS:
                    row = s[variant][condition]
                    lines.append(
                        f"| {variant} | {row['mean_exact_match']:.3f} | {row['mean_akshara_cer']:.3f} | {row['mean_first_entry_correct']:.3f} | {row['exact_bank_copy_rate']:.3f} |"
                    )
                lines.append("")
                if condition != "canonical":
                    pass
            helpful = s["icl_helpful"] if False else None
            comp = []
            for variant in ("compact", "tagged"):
                base = s["canonical"]["icl_helpful"]
                cur = s[variant]["icl_helpful"]
                comp.append(
                    {
                        "variant": variant,
                        "delta_exact": cur["mean_exact_match"] - base["mean_exact_match"],
                        "delta_cer": cur["mean_akshara_cer"] - base["mean_akshara_cer"],
                        "delta_first": cur["mean_first_entry_correct"] - base["mean_first_entry_correct"],
                        "delta_bank": cur["exact_bank_copy_rate"] - base["exact_bank_copy_rate"],
                    }
                )
            lines.append("**Helpful-vs-canonical deltas.**")
            lines.append("")
            lines.append("| variant | Δ exact | Δ CER | Δ first-entry | Δ bank-copy |")
            lines.append("|---|---:|---:|---:|---:|")
            for row in comp:
                lines.append(
                    f"| {row['variant']} | {row['delta_exact']:+.3f} | {row['delta_cer']:+.3f} | {row['delta_first']:+.3f} | {row['delta_bank']:+.3f} |"
                )
            lines.append("")
        lines.append("")
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
