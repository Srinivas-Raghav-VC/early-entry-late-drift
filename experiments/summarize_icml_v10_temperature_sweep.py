#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "research/results/autoresearch/telugu_temperature_sweep_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_temperature_sweep.json"
OUT_JSON = PROJECT_ROOT / "outputs/icml_v10_temperature_sweep_summary_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/icml_v10_temperature_sweep_summary_2026-04-09.md"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    obj = load_json(SRC)
    rows = []
    for key in ["temp_0.0", "temp_0.2", "temp_0.7"]:
        help_row = obj["summary_by_temp"][key]["icl_helpful"]
        rows.append(
            {
                "temperature": float(key.split("_")[1]),
                "n_items": int(help_row["n_items"]),
                "n_samples": int(help_row["n_samples"]),
                "first_akshara_correct": float(help_row["mean_first_entry_correct"]),
                "akshara_cer": float(help_row["mean_akshara_cer"]),
                "exact_bank_copy": float(help_row["exact_bank_copy_rate"]),
                "fuzzy_bank_copy": float(help_row["fuzzy_bank_copy_rate"]),
            }
        )
    payload = {
        "source_path": str(SRC),
        "rows": rows,
        "notes": {
            "panel_type": "separate helpful Telugu 1B follow-up panel",
            "seed": int(obj["seed"]),
            "n_icl": int(obj["n_icl"]),
            "nonzero_temps_use_multiple_samples": True,
            "samples_per_temp": int(obj["samples_per_temp"]),
            "top_p": float(obj["top_p"]),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ICML v10 Telugu temperature-sweep summary (2026-04-09)",
        "",
        f"Source: `{SRC}`",
        "",
        "Separate helpful Telugu 1B follow-up panel. Greedy uses one sample per item at `T=0.0`; nonzero temperatures use three sampled continuations per item.",
        "",
        "| T | n_items | n_samples | E_ak | CER | Exact bank-copy | Fuzzy bank-copy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['temperature']:.1f} | {row['n_items']} | {row['n_samples']} | {row['first_akshara_correct']:.3f} | {row['akshara_cer']:.3f} | {row['exact_bank_copy']:.3f} | {row['fuzzy_bank_copy']:.3f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
