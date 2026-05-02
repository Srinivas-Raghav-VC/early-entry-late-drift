#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_ROOT / "outputs/icml_v11_telugu_followups_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/icml_v11_telugu_followups_2026-04-09.md"

SITE_PATHS = {
    "1B": PROJECT_ROOT / "research/results/autoresearch/telugu_continuation_final_site_sufficiency_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json",
    "4B": PROJECT_ROOT / "research/results/autoresearch/telugu_continuation_final_site_sufficiency_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json",
}
LAYER_BAND_4B = PROJECT_ROOT / "research/results/autoresearch/telugu_continuation_layer_band_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_layer_band_panel.json"
WRITER_PATHS = {
    "1B": PROJECT_ROOT / "research/results/autoresearch/telugu_writer_head_probe_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_writer_head_probe.json",
    "4B": PROJECT_ROOT / "research/results/autoresearch/telugu_writer_head_probe_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_writer_head_probe.json",
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    site_rows: dict[str, Any] = {}
    for model, path in SITE_PATHS.items():
        obj = load_json(path)
        by_name = {row["intervention_name"]: row for row in obj["summary_rows"]}
        site_rows[model] = {
            "n_items": int(obj["n_items_used"]),
            "site": str(obj["site"]["name"]),
            "offband_site": str(obj["offband_site"]["name"]),
            "site_delta_gap": float(by_name["site_donor"]["delta_mean_gap"]),
            "offband_delta_gap": float(by_name["offband_donor"]["delta_mean_gap"]),
            "random_delta_gap": float(by_name["site_random_delta"]["delta_mean_gap"]),
            "noop_delta_gap": float(by_name["recipient_noop"]["delta_mean_gap"]),
            "site_delta_gold_top1": float(by_name["site_donor"]["delta_gold_top1_rate"]),
            "offband_delta_gold_top1": float(by_name["offband_donor"]["delta_gold_top1_rate"]),
        }

    layer_band_obj = load_json(LAYER_BAND_4B)
    layer_band_help = [
        row for row in layer_band_obj["summary_rows"]
        if row["recipient_condition"] == "icl_corrupt" and row["donor_condition"] == "icl_helpful"
    ]
    best_4b = max(layer_band_help, key=lambda row: float(row["delta_mean_gap"]))

    writer_rows: dict[str, Any] = {}
    for model, path in WRITER_PATHS.items():
        obj = load_json(path)
        top3 = obj["ranked_heads"][:3]
        writer_rows[model] = {
            "usable_items": int(obj["usable_items"]),
            "recipient_layer": int(obj["recipient_layer"]),
            "top3": [
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "writer_score": float(row["writer_score"]),
                    "mean_delta_gap": float(row["mean_delta_gap"]),
                }
                for row in top3
            ],
            "top_group_heads": obj["top_group_probe"]["heads"],
            "top_group_mean_delta_gap": float(obj["top_group_probe"]["mean_delta_gap"]),
            "random_group_heads": obj["random_group_probe"]["heads"],
            "random_group_mean_delta_gap": float(obj["random_group_probe"]["mean_delta_gap"]),
        }

    payload = {
        "late_site_summary": site_rows,
        "late_site_4b_layer_band_best": {
            "n_items": int(layer_band_obj["n_items_used"]),
            "best_layer_set_name": str(best_4b["layer_set_name"]),
            "best_delta_gap": float(best_4b["delta_mean_gap"]),
            "best_delta_gold_top1": float(best_4b["delta_gold_top1_rate"]),
        },
        "writer_head_summary": writer_rows,
        "source_paths": {
            "site_paths": {k: str(v) for k, v in SITE_PATHS.items()},
            "layer_band_4b": str(LAYER_BAND_4B),
            "writer_paths": {k: str(v) for k, v in WRITER_PATHS.items()},
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ICML v11 Telugu late-site and writer-head follow-ups (2026-04-09)",
        "",
        "## Late residual site summary",
    ]
    for model in ["1B", "4B"]:
        row = payload["late_site_summary"][model]
        lines.append(
            f"- {model}: true site {row['site']} on n={row['n_items']} gives Δgap {row['site_delta_gap']:.3f}; "
            f"off-band {row['offband_site']} gives {row['offband_delta_gap']:.3f}; random-direction control gives {row['random_delta_gap']:.3f}; no-op gives {row['noop_delta_gap']:.3f}."
        )
    lines.append(
        f"- 4B layer-band follow-up: strongest tested late site is {payload['late_site_4b_layer_band_best']['best_layer_set_name']} on n={payload['late_site_4b_layer_band_best']['n_items']} with Δgap {payload['late_site_4b_layer_band_best']['best_delta_gap']:.3f}."
    )
    lines.extend(["", "## Writer-head probe summary"])
    for model in ["1B", "4B"]:
        row = payload["writer_head_summary"][model]
        top_txt = ", ".join(
            f"L{r['layer']}H{r['head']} (score {r['writer_score']:.3f})" for r in row['top3']
        )
        lines.append(
            f"- {model}: usable_items={row['usable_items']}; top heads = {top_txt}; top-group mean Δgap = {row['top_group_mean_delta_gap']:.3f}; matched-random mean Δgap = {row['random_group_mean_delta_gap']:.3f}."
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
