#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_behavior_uncertainty() -> list[dict[str, Any]]:
    four = _load(PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed_aggregate.json")
    rows = {(r["model"], r["pair"]): r for r in four["row_aggregate"] if int(r["n_icl"]) == 64}

    mar_paths = {
        ("1b", "Marathi"): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/expansion_core64_seed42/raw/1b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json",
        ("4b", "Marathi"): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/expansion_core64_seed42/raw/4b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json",
    }

    lang_map = {
        "aksharantar_hin_latin": "Hindi",
        "aksharantar_ben_latin": "Bengali",
        "aksharantar_tam_latin": "Tamil",
        "aksharantar_tel_latin": "Telugu",
    }

    out: list[dict[str, Any]] = []
    for (model, pair), row in rows.items():
        out.append(
            {
                "model": model.upper(),
                "language": lang_map[pair],
                "seed_support": "3 seeds (11,42,101)",
                "exact_gain_mean": float(row["helpful_minus_zs_exact"]["mean"]),
                "exact_gain_min": float(row["helpful_minus_zs_exact"]["min"]),
                "exact_gain_max": float(row["helpful_minus_zs_exact"]["max"]),
                "cer_gain_mean": float(row["helpful_minus_zs_cer"]["mean"]),
                "cer_gain_min": float(row["helpful_minus_zs_cer"]["min"]),
                "cer_gain_max": float(row["helpful_minus_zs_cer"]["max"]),
            }
        )

    for (model, language), path in mar_paths.items():
        obj = _load(path)
        h = obj["summary_by_condition"]["icl_helpful"]
        z = obj["summary_by_condition"]["zs"]
        out.append(
            {
                "model": model.upper(),
                "language": language,
                "seed_support": "seed 42 only",
                "exact_gain_mean": float(h["mean_exact_match"] - z["mean_exact_match"]),
                "exact_gain_min": None,
                "exact_gain_max": None,
                "cer_gain_mean": float(z["mean_akshara_cer"] - h["mean_akshara_cer"]),
                "cer_gain_min": None,
                "cer_gain_max": None,
            }
        )
    order_model = {"1B": 0, "4B": 1}
    order_lang = {"Hindi": 0, "Marathi": 1, "Bengali": 2, "Tamil": 3, "Telugu": 4}
    return sorted(out, key=lambda r: (order_model[r["model"]], order_lang[r["language"]]))


def build_panel_summary() -> list[dict[str, str]]:
    return [
        {
            "panel": "Figure 3 broad regime map",
            "n": "30 items per seed per cell",
            "seeds": "11, 42, 101 for Hindi/Bengali/Tamil/Telugu; Marathi seed 42 only",
            "purpose": "broad behavioral summary",
            "selection_eval": "evaluation",
            "notes": "Helpful vs zero-shot EM gain and CER gain. Appendix uncertainty table reports seed ranges where available.",
        },
        {
            "panel": "Table 1 stage-sensitive diagnostics",
            "n": "30 items per language",
            "seeds": "42",
            "purpose": "axis definition (Hindi/Marathi/Telugu, 1B)",
            "selection_eval": "evaluation",
            "notes": "Matched means the same first 30 held-out evaluation words are scored across conditions within each language.",
        },
        {
            "panel": "Hindi localization sweep",
            "n": "30 items",
            "seeds": "42",
            "purpose": "site localization at last prompt token",
            "selection_eval": "exploratory localization",
            "notes": "Prompt-length-aware alignment uses each condition's own prompt-final token.",
        },
        {
            "panel": "Hindi channel ranking / pair selection",
            "n": "100 items",
            "seeds": "42",
            "purpose": "select harmful L25 channel subset",
            "selection_eval": "selection",
            "notes": "Used for pair ranking; repeat-seed confirmation reported separately.",
        },
        {
            "panel": "Hindi alpha tuning (Figure 5A)",
            "n": "200 items",
            "seeds": "42",
            "purpose": "select deployed patch scale",
            "selection_eval": "selection",
            "notes": "Separate from the 100-item channel-selection split; alpha chosen by mean ΔL with mean p_target as tie-break.",
        },
        {
            "panel": "Hindi held-out patch evaluation",
            "n": "200 items",
            "seeds": "42",
            "purpose": "confirmatory fixed-patch evaluation",
            "selection_eval": "evaluation",
            "notes": "Reports mean-shift, sign-flip, and zero-ablate controls with bootstrap CIs.",
        },
        {
            "panel": "Telugu alpha tuning",
            "n": "100 items",
            "seeds": "42",
            "purpose": "select oracle-positioned static-patch scale",
            "selection_eval": "selection",
            "notes": "Objective is mean first-divergence gold-vs-bank gap ΔL_div.",
        },
        {
            "panel": "Telugu held-out oracle patch",
            "n": "191 usable / 200 starting items",
            "seeds": "42",
            "purpose": "confirmatory static-patch evaluation",
            "selection_eval": "evaluation",
            "notes": "Nine items excluded for having no usable non-empty shared-prefix-then-divergence pattern; usable items all pass the first-akshara boundary check.",
        },
        {
            "panel": "Telugu temperature sweep",
            "n": "57 items",
            "seeds": "42",
            "purpose": "check copy robustness beyond greedy decoding",
            "selection_eval": "follow-up evaluation",
            "notes": "T=0 uses 57 samples; T=0.2/0.7 use 171 samples each (3 per item).",
        },
        {
            "panel": "Telugu late-site follow-ups",
            "n": "57 items",
            "seeds": "42",
            "purpose": "late-site confirmation / controls",
            "selection_eval": "follow-up evaluation",
            "notes": "Includes off-band, random-direction, and no-op comparisons; 4B strongest tested late site is L34.",
        },
        {
            "panel": "Telugu writer-head probe",
            "n": "38 usable of first 40 words",
            "seeds": "42",
            "purpose": "exploratory upstream head screen",
            "selection_eval": "exploratory follow-up",
            "notes": "Hypothesis-generating only; usable subset restricted to items with a well-defined first-divergence comparison.",
        },
    ]


def main() -> int:
    payload = {
        "behavior_uncertainty": build_behavior_uncertainty(),
        "panel_summary": build_panel_summary(),
    }
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "icml_v18_panels_2026-04-10.json"
    md_path = out_dir / "icml_v18_panels_2026-04-10.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# ICML v18 panel and uncertainty summary (2026-04-10)", "", "## Figure 3 uncertainty support", ""]
    for row in payload["behavior_uncertainty"]:
        if row["exact_gain_min"] is None:
            exact = f"{row['exact_gain_mean']:+.3f} (seed 42 only)"
            cer = f"{row['cer_gain_mean']:+.3f} (seed 42 only)"
        else:
            exact = f"{row['exact_gain_mean']:+.3f} [{row['exact_gain_min']:+.3f}, {row['exact_gain_max']:+.3f}]"
            cer = f"{row['cer_gain_mean']:+.3f} [{row['cer_gain_min']:+.3f}, {row['cer_gain_max']:+.3f}]"
        lines.append(f"- {row['model']} {row['language']}: EM gain {exact}; CER gain {cer}; support = {row['seed_support']}")
    lines.extend(["", "## Panel summary", "", "| panel | n | seeds | purpose | status | notes |", "|---|---:|---|---|---|---|"])
    for row in payload["panel_summary"]:
        lines.append(
            f"| {row['panel']} | {row['n']} | {row['seeds']} | {row['purpose']} | {row['selection_eval']} | {row['notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
