#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_ROOT / "outputs/icml_v8_axes_kshot_summaries_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/icml_v8_axes_kshot_summaries_2026-04-09.md"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_mean(vals: list[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def stage_metrics_from_raw(path: Path) -> Dict[str, Dict[str, float | int | None]]:
    obj = load_json(path)
    meta_by_item = {int(row["item_index"]): row for row in obj["prompt_ordering_metadata_by_item"]}
    out: Dict[str, Dict[str, float | int | None]] = {}
    for condition in ["zs", "icl_helpful"]:
        rows = [row for row in obj["item_rows"] if str(row["condition"]) == condition]
        bank_hits = []
        for row in rows:
            meta = meta_by_item[int(row["item_index"])]
            bank_targets = [str(x.get("target", "")) for x in meta.get("helpful_similarity_desc", [])]
            bank_hits.append(1.0 if str(row.get("prediction", "")) in bank_targets else 0.0)
        out[condition] = {
            "n_items": int(len(rows)),
            "first_akshara_correct": safe_mean([row.get("first_entry_correct") for row in rows]),
            "akshara_cer": safe_mean([row.get("akshara_cer") for row in rows]),
            "continuation_tail_cer": safe_mean([row.get("continuation_fidelity") for row in rows]),
            "exact_bank_copy_rate": safe_mean(bank_hits),
        }
    return out


def main() -> None:
    stage_paths = {
        "Hindi": PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        "Marathi": PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/expansion_core64_seed42/raw/1b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json",
        "Telugu": PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
    }
    stage = {lang: stage_metrics_from_raw(path) for lang, path in stage_paths.items()}

    kshot_hin = load_json(
        PROJECT_ROOT / "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_hin_latin/seed42/kshot_regime_sweep.json"
    )
    kshot_tel = load_json(
        PROJECT_ROOT / "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_tel_latin/seed42/kshot_regime_sweep.json"
    )
    kshot_rows = []
    for k in [8, 16, 32, 64, 128]:
        hin = kshot_hin["summary_by_k"][f"k{k}"]["icl_helpful"]
        tel = kshot_tel["summary_by_k"][f"k{k}"]["icl_helpful"]
        kshot_rows.append(
            {
                "k": int(k),
                "hindi_first_akshara": float(hin["mean_first_entry_correct"]),
                "hindi_cer": float(hin["mean_akshara_cer"]),
                "telugu_first_akshara": float(tel["mean_first_entry_correct"]),
                "telugu_exact_bank_copy": float(tel["exact_bank_copy_rate"]),
                "telugu_cer": float(tel["mean_akshara_cer"]),
            }
        )

    payload = {
        "stage_table_inputs": stage,
        "kshot_sweep_rows": kshot_rows,
        "source_paths": {k: str(v) for k, v in stage_paths.items()},
        "kshot_paths": {
            "Hindi": str(PROJECT_ROOT / "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_hin_latin/seed42/kshot_regime_sweep.json"),
            "Telugu": str(PROJECT_ROOT / "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_tel_latin/seed42/kshot_regime_sweep.json"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ICML v8 stage-axis and k-shot summaries (2026-04-09)",
        "",
        "## Stage-sensitive 1B k=64 table inputs",
    ]
    for lang in ["Hindi", "Marathi", "Telugu"]:
        zs = stage[lang]["zs"]
        help_row = stage[lang]["icl_helpful"]
        lines.append(
            f"- {lang}: E_ak zs->help = {zs['first_akshara_correct']:.3f} -> {help_row['first_akshara_correct']:.3f}; tail CER_help = {help_row['continuation_tail_cer'] if help_row['continuation_tail_cer'] is not None else 'NA'}; bank_help = {help_row['exact_bank_copy_rate']:.3f}; CER_help = {help_row['akshara_cer']:.3f}"
        )
    lines.extend([
        "",
        "## Fixed-split 1B k-shot sweep (helpful condition)",
    ])
    for row in kshot_rows:
        lines.append(
            f"- k={row['k']}: Hindi E_ak={row['hindi_first_akshara']:.3f}, CER={row['hindi_cer']:.3f}; Telugu E_ak={row['telugu_first_akshara']:.3f}, bank={row['telugu_exact_bank_copy']:.3f}, CER={row['telugu_cer']:.3f}"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
