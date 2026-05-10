#!/usr/bin/env python3
from __future__ import annotations

"""Summarize minimal interpretability evidence for Hindi channels 5486/2299.

This script does not try to name the channels semantically. It extracts the
stronger, defensible statement available from retained artifacts: helpful
high-shot prompts push these coordinates upward relative to zero-shot, and the
local readout geometry makes moving them back toward the zero-shot mean increase
the gold-first-token-vs-Latin margin.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VALUE_PATH = ROOT / "research/results/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json"
GEOM_PATH = ROOT / "research/results/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json"
OUT_DIR = ROOT / "research/submission"
OUT_JSON = OUT_DIR / "hindi_channel_interpretation_summary_2026-04-28.json"
OUT_MD = OUT_DIR / "hindi_channel_interpretation_summary_2026-04-28.md"
CHANNELS = [5486, 2299]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(rows: list[dict[str, Any]], *, channel: int, subtype: str | None = None) -> dict[str, Any]:
    for row in rows:
        if int(row.get("channel", -1)) == int(channel) and (subtype is None or row.get("subtype") == subtype):
            return row
    raise KeyError(f"Missing row channel={channel} subtype={subtype}")


def main() -> int:
    value = _read(VALUE_PATH)
    geom = _read(GEOM_PATH)
    def corr(xs: list[float], ys: list[float]) -> float:
        import math

        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))

    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        overall_value = _find(value["overall_rows"], channel=channel)
        overall_geom = _find(geom["summary_rows"], channel=channel, subtype="overall")
        latin_geom = _find(geom["summary_rows"], channel=channel, subtype="latin_collapse")
        item_pairs = [
            (item, next(ch for ch in item["channels"] if int(ch["channel"]) == int(channel)))
            for item in value["item_rows"]
        ]
        helpful_values = [float(ch["icl_helpful_value"]) for _item, ch in item_pairs]
        helpful_gaps = [float(item["base_helpful"]["target_minus_latin_logit"]) for item, _ch in item_pairs]
        helpful_latin_top1 = [1.0 if str(item["base_helpful"]["top1_script"]) == "latin" else 0.0 for item, _ch in item_pairs]
        subtype_rows: list[dict[str, Any]] = []
        for subtype in sorted({str(item["subtype"]) for item, _ch in item_pairs}):
            subset = [(item, ch) for item, ch in item_pairs if str(item["subtype"]) == subtype]
            subtype_rows.append(
                {
                    "subtype": subtype,
                    "n_items": len(subset),
                    "mean_helpful_value": sum(float(ch["icl_helpful_value"]) for _item, ch in subset) / len(subset),
                    "mean_helpful_minus_zs": sum(float(ch["helpful_minus_zs"]) for _item, ch in subset) / len(subset),
                }
            )
        top_contexts = []
        for item, ch in sorted(item_pairs, key=lambda pair: float(pair[1]["icl_helpful_value"]), reverse=True)[:5]:
            top_contexts.append(
                {
                    "word_ood": str(item["word_ood"]),
                    "word_hindi": str(item["word_hindi"]),
                    "subtype": str(item["subtype"]),
                    "helpful_value": float(ch["icl_helpful_value"]),
                    "helpful_minus_zs": float(ch["helpful_minus_zs"]),
                    "helpful_top1_text": str(item["base_helpful"]["top1_token_text"]),
                    "helpful_top1_script": str(item["base_helpful"]["top1_script"]),
                    "helpful_target_minus_latin": float(item["base_helpful"]["target_minus_latin_logit"]),
                }
            )
        rows.append(
            {
                "channel": channel,
                "n_items": int(overall_value["n_items"]),
                "zs_value": float(overall_value["mean_zs_value"]),
                "helpful_value": float(overall_value["mean_icl_helpful_value"]),
                "zs_minus_helpful": float(overall_value["mean_zs_minus_helpful"]),
                "readout_grad_dot_column": float(overall_geom["mean_readout_grad_dot_column"]),
                "readout_grad_column_cosine": float(overall_geom["mean_readout_grad_column_cosine"]),
                "first_order_predicted_effect": float(overall_geom["mean_first_order_predicted_effect"]),
                "actual_singleton_patch_effect": float(overall_geom["mean_actual_singleton_patch_effect"]),
                "predicted_actual_corr": float(overall_geom["predicted_actual_corr"]),
                "latin_collapse_actual_effect": float(latin_geom["mean_actual_singleton_patch_effect"]),
                "corr_helpful_value_with_helpful_target_minus_latin": corr(helpful_values, helpful_gaps),
                "corr_helpful_value_with_latin_top1": corr(helpful_values, helpful_latin_top1),
                "subtype_rows": subtype_rows,
                "top_helpful_value_contexts": top_contexts,
            }
        )

    payload = {
        "value_artifact": str(VALUE_PATH.relative_to(ROOT)),
        "geometry_artifact": str(GEOM_PATH.relative_to(ROOT)),
        "interpretation_status": "supported but bounded",
        "summary_rows": rows,
        "claim": (
            "Channels 5486 and 2299 are not assigned human-semantic labels. "
            "They are a small high-shot state/readout direction: helpful prompts move the coordinates upward relative to zero-shot, "
            "and because their down-projection columns have negative local dot product with the target-vs-Latin readout gradient, "
            "moving those coordinates back toward zero-shot increases the gold-first-token margin."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Hindi channel interpretation summary (2026-04-28)\n\n")
    lines.append("Status: **supported but bounded**. The evidence below characterizes a local readout direction; it does not name a monosemantic feature or a complete circuit.\n\n")
    lines.append("Artifacts:\n\n")
    lines.append(f"- `{payload['value_artifact']}`\n")
    lines.append(f"- `{payload['geometry_artifact']}`\n\n")
    lines.append("## Main claim\n\n")
    lines.append(payload["claim"] + "\n\n")
    lines.append("## Extracted rows\n\n")
    lines.append("| channel | helpful − zs value | grad·column | predicted Δgap | actual singleton Δgap | corr(pred, actual) | Latin-collapse actual Δgap |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        lines.append(
            f"| {row['channel']} | {row['helpful_value'] - row['zs_value']:+.2f} | "
            f"{row['readout_grad_dot_column']:+.3f} | {row['first_order_predicted_effect']:+.2f} | "
            f"{row['actual_singleton_patch_effect']:+.2f} | {row['predicted_actual_corr']:.2f} | "
            f"{row['latin_collapse_actual_effect']:+.2f} |\n"
        )
    lines.append("\nInterpretation: helpful high-shot prompting increases both coordinates by roughly 8--11 units relative to zero-shot. The local readout gradient is negative along both down-projection columns, so the zero-shot-minus-helpful shift has a positive first-order effect on the target-vs-Latin margin. The first-order estimate closely tracks actual singleton channel replacement, especially for channel 5486.\n\n")
    lines.append("## Item-level characterization\n\n")
    lines.append("This is still **not** a semantic feature label. The item-level audit is useful mainly for ruling in a prompt-state/readout interpretation and ruling out a clean monosemantic story. Channel 2299 is more associated with Latin top-1 pressure than 5486; channel 5486 is broader and appears in both base-success and Latin-collapse contexts.\n\n")
    lines.append("| channel | corr(value, target−Latin gap) | corr(value, Latin top-1) | largest-value contexts |\n")
    lines.append("|---:|---:|---:|---|\n")
    for row in rows:
        contexts = "; ".join(
            f"{ctx['word_ood']}→{ctx['word_hindi']} ({ctx['subtype']}, top1={ctx['helpful_top1_text']})"
            for ctx in row["top_helpful_value_contexts"][:3]
        )
        lines.append(
            f"| {row['channel']} | {row['corr_helpful_value_with_helpful_target_minus_latin']:+.2f} | "
            f"{row['corr_helpful_value_with_latin_top1']:+.2f} | {contexts} |\n"
        )
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
