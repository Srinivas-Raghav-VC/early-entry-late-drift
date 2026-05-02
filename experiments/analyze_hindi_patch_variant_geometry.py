#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_PATHS = {
    "canonical": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "compact": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_compact_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "tagged": PROJECT_ROOT / "research/results/autoresearch/hindi_practical_patch_eval_tagged_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
}
OUT_JSON = PROJECT_ROOT / "outputs/hindi_patch_variant_geometry_2026-04-01.json"
OUT_MD = PROJECT_ROOT / "outputs/hindi_patch_variant_geometry_2026-04-01.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def main() -> int:
    payloads = {name: load_json(path) for name, path in VARIANT_PATHS.items()}
    vectors = {name: [float(x) for x in payload["selected_mean_delta"]["mean_delta"]] for name, payload in payloads.items()}
    pairwise = {}
    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairwise[f"{a}__{b}"] = {
                "cosine": cosine(vectors[a], vectors[b]),
                "a": vectors[a],
                "b": vectors[b],
            }

    per_variant = {}
    for name, payload in payloads.items():
        rows = {row["intervention"]: row for row in payload["summary_rows"]}
        base = rows["baseline_no_patch"]
        chosen = rows["chosen_mean_shift"]
        stats = {
            "selected_alpha": float(payload["selected_alpha"]),
            "mean_delta": vectors[name],
            "delta_exact_match": float(chosen["delta_exact_match"]["mean"]),
            "delta_cer_improvement": float(chosen["delta_cer_improvement"]["mean"]),
            "delta_first_entry_correct": float(chosen["delta_first_entry_correct"]["mean"]),
            "delta_first_token_gap_latin": float(chosen["delta_first_token_gap_latin"]["mean"]),
            "baseline_first_entry_correct": float(base["first_entry_correct"]["mean"]),
            "chosen_first_entry_correct": float(chosen["first_entry_correct"]["mean"]),
            "baseline_first_token_top1_target_rate": float(base["first_token_top1_target_rate"]["mean"]),
            "chosen_first_token_top1_target_rate": float(chosen["first_token_top1_target_rate"]["mean"]),
            "baseline_first_token_top1_latin_rate": float(base["first_token_top1_latin_rate"]["mean"]),
            "chosen_first_token_top1_latin_rate": float(chosen["first_token_top1_latin_rate"]["mean"]),
        }
        gap_up = 0
        entry_up = 0
        entry_down = 0
        gap_up_and_entry_up = 0
        gap_up_and_entry_down = 0
        strict_word_up = 0
        for item in payload["item_rows"]:
            b = item["baseline_no_patch"]
            c = item["chosen_mean_shift"]
            is_gap_up = float(c["first_step"]["target_minus_latin_logit"]) > float(b["first_step"]["target_minus_latin_logit"])
            is_entry_up = float(c["generation"]["first_entry_correct"]) > float(b["generation"]["first_entry_correct"])
            is_entry_down = float(c["generation"]["first_entry_correct"]) < float(b["generation"]["first_entry_correct"])
            if is_gap_up:
                gap_up += 1
            if is_entry_up:
                entry_up += 1
            if is_entry_down:
                entry_down += 1
            if is_gap_up and is_entry_up:
                gap_up_and_entry_up += 1
            if is_gap_up and is_entry_down:
                gap_up_and_entry_down += 1
            if float(c["generation"]["raw_strict_word_only"]) > float(b["generation"]["raw_strict_word_only"]):
                strict_word_up += 1
        stats["item_level"] = {
            "n_items": len(payload["item_rows"]),
            "gap_up": gap_up,
            "entry_up": entry_up,
            "entry_down": entry_down,
            "gap_up_and_entry_up": gap_up_and_entry_up,
            "gap_up_and_entry_down": gap_up_and_entry_down,
            "strict_word_up": strict_word_up,
        }
        per_variant[name] = stats

    output = {"pairwise": pairwise, "per_variant": per_variant}
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Hindi patch prompt-variant geometry and conversion audit (2026-04-01)",
        "",
        "## Pairwise vector similarity",
        "",
        "| pair | cosine |",
        "|---|---:|",
    ]
    for name, row in pairwise.items():
        lines.append(f"| {name} | {row['cosine']:.6f} |")
    lines.extend(["", "## Per-variant summary", ""])
    for name, row in per_variant.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- selected alpha: {row['selected_alpha']:.2f}\n"
            f"- mean delta: `{row['mean_delta']}`\n"
            f"- ΔEM: {row['delta_exact_match']:+.3f}\n"
            f"- ΔCER improvement: {row['delta_cer_improvement']:+.3f}\n"
            f"- Δfirst-entry: {row['delta_first_entry_correct']:+.3f}\n"
            f"- Δfirst-token target-vs-Latin gap: {row['delta_first_token_gap_latin']:+.3f}"
        )
        item = row["item_level"]
        lines.append("")
        lines.append(
            f"Item-level conversion audit: gap improved on {item['gap_up']}/{item['n_items']} items; "
            f"first-entry improved on {item['entry_up']}/{item['n_items']} and worsened on {item['entry_down']}/{item['n_items']}; "
            f"among items with improved gap, first-entry improved on {item['gap_up_and_entry_up']} and worsened on {item['gap_up_and_entry_down']}."
        )
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
