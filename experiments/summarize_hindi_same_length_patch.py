#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "research/results/autoresearch/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_causal_patch_panel.json"
OUT_JSON = ROOT / "outputs/hindi_same_length_patch_summary_2026-04-28.json"
OUT_MD = ROOT / "outputs/hindi_same_length_patch_summary_2026-04-28.md"

PAIRS = [
    ("icl_helpful", "zs", "helpful ← zero-shot"),
    ("icl_helpful", "icl_corrupt", "helpful ← corrupt"),
    ("icl_corrupt", "icl_helpful", "corrupt ← helpful"),
]


def _top(rows: list[dict[str, Any]], recipient: str, donor: str, n: int = 6) -> list[dict[str, Any]]:
    sub = [r for r in rows if r["recipient_condition"] == recipient and r["donor_condition"] == donor]
    return sorted(sub, key=lambda r: float(r["delta_mean_gap_latin"]), reverse=True)[:n]


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipient_condition": row["recipient_condition"],
        "donor_condition": row["donor_condition"],
        "layer": int(row["layer"]),
        "component": row["component"],
        "layer_type": row["layer_type"],
        "n_items": int(row["n_items"]),
        "delta_mean_gap_latin": float(row["delta_mean_gap_latin"]),
        "delta_mean_target_prob": float(row["delta_mean_target_prob"]),
        "delta_top1_target_rate": float(row["delta_top1_target_rate"]),
        "base_top1_target_rate": float(row["base_top1_target_rate"]),
        "patched_top1_target_rate": float(row["patched_top1_target_rate"]),
    }


def main() -> int:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rows = payload["summary_rows"]
    summary: dict[str, Any] = {
        "artifact": str(ARTIFACT.relative_to(ROOT)),
        "n_items": payload["max_items"],
        "patch_position_mode": payload["patch_position_mode"],
        "patches": payload["patches"],
        "top_by_pair": {},
    }
    for recipient, donor, label in PAIRS:
        summary["top_by_pair"][label] = [_compact(r) for r in _top(rows, recipient, donor)]

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md: list[str] = []
    md.append("# Hindi same-length helpful/corrupt patch summary (2026-04-28)\n")
    md.append(f"Artifact: `{summary['artifact']}`\n")
    md.append(f"Panel: n={summary['n_items']}, patch position = `{summary['patch_position_mode']}`.\n")
    md.append("\n## Main readout\n")
    md.append("The original prompt-final helpful←zero-shot sweep still peaks at `L25 mlp_output` (+3.30 target-vs-Latin logit-gap, +0.20 top-1 target rate). The same-length helpful/corrupt control does **not** reproduce that compact L25 MLP rescue. Instead, corrupt←helpful shows a broader layer-output signal peaking at `L23 layer_output` (+2.37 gap, +0.033 top-1), while helpful←corrupt is small and does not change top-1 target rate.\n")
    md.append("\nInterpretation: the same-length control reduces the prompt-length objection, but it also narrows the claim. The L25 MLP patch is best described as a compact high-shot/prompt-state competition bottleneck, not as a clean semantic helpful-vs-corrupt retrieval site.\n")
    for label, top_rows in summary["top_by_pair"].items():
        md.append(f"\n## {label}\n")
        md.append("| site | Δ target-vs-Latin gap | Δ target prob | Δ top-1 target rate | top-1 base→patched |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for row in top_rows[:6]:
            site = f"L{row['layer']} {row['component']}"
            md.append(
                f"| `{site}` | {row['delta_mean_gap_latin']:+.3f} | {row['delta_mean_target_prob']:+.3f} | "
                f"{row['delta_top1_target_rate']:+.3f} | {row['base_top1_target_rate']:.3f}→{row['patched_top1_target_rate']:.3f} |\n"
            )
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
