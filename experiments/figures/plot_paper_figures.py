#!/usr/bin/env python3
from __future__ import annotations

"""
Generate publication-style paper figures with a more declarative / Altair-like aesthetic.

Recommended invocation:
    uv run --with altair --with pandas --with vl-convert-python \
        python3 experiments/plot_final_paper_figures.py

The script overwrites the canonical paper figure PNGs used by the manuscript and also
emits SVG companions for easier visual inspection.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import altair as alt
except ImportError as exc:  # pragma: no cover - runtime guidance only
    raise SystemExit(
        "Missing Altair stack. Run with: uv run --with altair --with pandas --with vl-convert-python "
        "python3 experiments/plot_final_paper_figures.py"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANG_ORDER = ["Hindi", "Marathi", "Bengali", "Tamil", "Telugu"]
MODEL_ORDER = ["4B", "1B"]
INTERVENTION_ORDER = ["Mean shift", "Sign flip", "Zero ablate"]
LESION_ORDER = ["Zero 5486", "Zero 2299", "Zero both", "Mean shift", "Sign flip"]
TELUGU_ORDER = [
    "Writer only",
    "+ recipient overwrite",
    "+ off-band overwrite",
    "Site donor",
    "Off-band donor",
    "Random direction",
]
CROSS_FAMILY_ORDER = [
    "Gemma 1B",
    "Gemma 4B",
    "Qwen 2.5 1.5B",
    "Qwen 2.5 3B",
    "Llama 3.2 1B",
    "Llama 3.2 3B",
]
CROSS_CELL_ORDER = ["Hin-8", "Hin-64", "Tel-8", "Tel-64"]
PROMPT_VARIANT_ORDER = [
    "helpful",
    "sim-front",
    "sim-back",
    "reversed",
    "drop-nearest",
    "drop-top2",
    "corrupt",
]

COLORS = {
    "ink": "#1C1C1E",
    "muted": "#6E6E73",
    "grid": "#E4E4E9",
    "paper": "#FFFFFF",
    "one_b": "#D55E00",
    "four_b": "#0072B2",
    "good": "#009E73",
    "harm": "#CC79A7",
    "control": "#B2B2BA",
    "light_blue": "#56B4E9",
    "neg": "#B2182B",
    "mid": "#F5F5F7",
    "pos": "#1B9E77",
}

alt.data_transformers.disable_max_rows()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _styled(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return chart.configure(
        background=COLORS["paper"],
        padding={"left": 12, "right": 12, "top": 12, "bottom": 12},
    ).configure_view(
        stroke=None,
    ).configure_axis(
        labelColor=COLORS["ink"],
        titleColor=COLORS["ink"],
        domainColor="#D8D8DE",
        tickColor="#D8D8DE",
        gridColor=COLORS["grid"],
        gridOpacity=0.7,
        labelFont="Inter, Noto Sans, DejaVu Sans, Arial",
        titleFont="Inter, Noto Sans, DejaVu Sans, Arial",
        labelFontSize=13,
        titleFontSize=14,
        titleFontWeight="bold",
    ).configure_title(
        anchor="start",
        color=COLORS["ink"],
        font="Inter, Noto Sans, DejaVu Sans, Arial",
        fontSize=24,
        fontWeight="bold",
        subtitleColor=COLORS["muted"],
        subtitleFont="Inter, Noto Sans, DejaVu Sans, Arial",
        subtitleFontSize=13,
        subtitlePadding=6,
    ).configure_legend(
        labelColor=COLORS["ink"],
        titleColor=COLORS["ink"],
        labelFont="Inter, Noto Sans, DejaVu Sans, Arial",
        titleFont="Inter, Noto Sans, DejaVu Sans, Arial",
        labelFontSize=12,
        titleFontSize=12,
        symbolSize=140,
        orient="top",
        direction="horizontal",
    ).configure_header(
        labelColor=COLORS["ink"],
        titleColor=COLORS["ink"],
        labelFont="Inter, Noto Sans, DejaVu Sans, Arial",
        titleFont="Inter, Noto Sans, DejaVu Sans, Arial",
        labelFontSize=13,
        titleFontSize=14,
        titleFontWeight="bold",
    )


def _save(chart: alt.TopLevelMixin, filename: str) -> None:
    png_path = FIG_DIR / filename
    svg_path = png_path.with_suffix(".svg")
    chart.save(str(png_path), scale_factor=2.4)
    chart.save(str(svg_path))
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def _prepare_behavior_df() -> pd.DataFrame:
    agg = _read_json(PROJECT_ROOT / "research/results/four_lang_thesis_panel/seed_aggregate.json")
    rows = {(r["model"], r["pair"]): r for r in agg["row_aggregate"] if int(r["n_icl"]) == 64}

    mar_1b = _read_json(
        PROJECT_ROOT
        / "research/results/loop2_vm_controls/expansion_core64_seed42/raw/1b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json"
    )
    mar_4b = _read_json(
        PROJECT_ROOT
        / "research/results/loop2_vm_controls/expansion_core64_seed42/raw/4b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json"
    )

    def mar_delta(payload: Dict[str, Any]) -> Dict[str, float]:
        h = payload["summary_by_condition"]["icl_helpful"]
        z = payload["summary_by_condition"]["zs"]
        return {
            "exact_gain": float(h["mean_exact_match"] - z["mean_exact_match"]),
            "cer_improvement": float(z["mean_akshara_cer"] - h["mean_akshara_cer"]),
        }

    mar_rows = {
        ("1b", "aksharantar_mar_latin"): mar_delta(mar_1b),
        ("4b", "aksharantar_mar_latin"): mar_delta(mar_4b),
    }

    lang_pairs = [
        ("aksharantar_hin_latin", "Hindi"),
        ("aksharantar_mar_latin", "Marathi"),
        ("aksharantar_ben_latin", "Bengali"),
        ("aksharantar_tam_latin", "Tamil"),
        ("aksharantar_tel_latin", "Telugu"),
    ]
    records = []
    for pair, label in lang_pairs:
        for model_raw, model_label in [("1b", "1B"), ("4b", "4B")]:
            if pair == "aksharantar_mar_latin":
                exact_gain = mar_rows[(model_raw, pair)]["exact_gain"]
                cer_improvement = mar_rows[(model_raw, pair)]["cer_improvement"]
            else:
                r = rows[(model_raw, pair)]
                exact_gain = float(r["helpful_exact"]["mean"] - r["zs_exact"]["mean"])
                cer_improvement = float(r["zs_cer"]["mean"] - r["helpful_cer"]["mean"])
            records.extend(
                [
                    {
                        "language": label,
                        "model": model_label,
                        "metric": "Exact-match gain",
                        "value": exact_gain,
                    },
                    {
                        "language": label,
                        "model": model_label,
                        "metric": "CER improvement",
                        "value": cer_improvement,
                    },
                ]
            )
    return pd.DataFrame.from_records(records)


def plot_behavioral_regime_summary() -> None:
    df = _prepare_behavior_df()
    pieces = []
    limits = {
        "Exact-match gain": (-0.05, 0.28),
        "CER improvement": (-1.30, 2.30),
    }
    subtitles = {
        "Exact-match gain": "helpful − zero-shot exact match",
        "CER improvement": "zero-shot CER − helpful CER",
    }

    for metric in ["Exact-match gain", "CER improvement"]:
        sub = df[df["metric"] == metric].copy()
        lo, hi = limits[metric]
        base = alt.Chart(sub).encode(
            x=alt.X(
                "language:N",
                sort=LANG_ORDER,
                title=None,
                axis=alt.Axis(labelAngle=0, labelPadding=10, labelFontWeight="bold"),
            ),
            y=alt.Y("model:N", sort=MODEL_ORDER, title=None),
        )
        rect = base.mark_rect(cornerRadius=8, stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(domain=[lo, 0, hi], range=[COLORS["neg"], COLORS["mid"], COLORS["pos"]]),
                legend=None,
            )
        )
        text = base.mark_text(fontSize=14, fontWeight="bold", color=COLORS["ink"]).encode(
            text=alt.Text("value:Q", format="+.2f"),
        )
        pieces.append(
            (rect + text)
            .properties(
                width=340,
                height=110,
                title=alt.TitleParams(metric, subtitle=subtitles[metric], anchor="start"),
            )
        )

    chart = alt.hconcat(*pieces, spacing=28).resolve_scale(color="independent").properties(
        title=alt.TitleParams(
            "Behavioral phase structure at high shot",
            subtitle="Gemma 3 at n_icl = 64. Each cell reports a helpful-versus-zero-shot delta; red is harmful and green is helpful.",
            anchor="start",
        )
    )
    _save(_styled(chart), "fig_behavioral_regime_summary.png")


def _prepare_hindi_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patch_obj = _read_json(
        PROJECT_ROOT
        / "research/results/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
    )
    intervention_obj = _read_json(
        PROJECT_ROOT
        / "research/results/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json"
    )

    selection_rows = []
    for row in patch_obj["selection_rows"]:
        selection_rows.extend(
            [
                {
                    "alpha": float(row["alpha"]),
                    "metric": "Target − Latin gap",
                    "value": float(row["delta_mean_target_minus_latin_logit"]),
                },
                {
                    "alpha": float(row["alpha"]),
                    "metric": "Target probability",
                    "value": float(row["delta_mean_target_prob"]),
                },
            ]
        )
    selection_df = pd.DataFrame(selection_rows)

    summary = {row["intervention"]: row for row in patch_obj["summary_rows"]}
    random_agg = patch_obj["random_aggregate"]
    eval_rows = []
    eval_metrics = [
        ("Delta exact match", "delta_exact_match"),
        ("Delta CER improvement", "delta_cer_improvement"),
        ("Delta first-entry correct", "delta_first_entry_correct"),
    ]
    for raw_name, pretty_name, color_key in [
        ("chosen_mean_shift", "Mean shift", "good"),
        ("chosen_sign_flip", "Sign flip", "harm"),
        ("chosen_zero_ablate", "Zero ablate", "control"),
    ]:
        for metric_name, metric_key in eval_metrics:
            value = float(summary[raw_name][metric_key]["mean"])
            eval_rows.append(
                {
                    "intervention": pretty_name,
                    "metric": metric_name,
                    "value": value,
                    "color": COLORS[color_key],
                }
            )
    eval_df = pd.DataFrame(eval_rows)

    intervention_summary = {row["intervention"]: row for row in intervention_obj["summary_rows"]}
    lesion_rows = []
    for raw_name, pretty_name, color_key in [
        ("zero_channel_5486", "Zero 5486", "control"),
        ("zero_channel_2299", "Zero 2299", "control"),
        ("zero_both_channels", "Zero both", "control"),
        ("calibrated_mean_shift", "Mean shift", "good"),
        ("calibrated_sign_flip", "Sign flip", "harm"),
    ]:
        row = intervention_summary[raw_name]
        exact_match = float(row["delta_exact_match"]["mean"])
        lesion_rows.append(
            {
                "intervention": pretty_name,
                "cer_improvement": float(row["delta_cer_improvement"]["mean"]),
                "exact_match": exact_match,
                "exact_match_label": f"ΔEM {exact_match:+.02f}",
                "color": COLORS[color_key],
            }
        )
    lesion_df = pd.DataFrame(lesion_rows)
    return selection_df, eval_df, lesion_df


def plot_hindi_practical_patch() -> None:
    selection_df, eval_df, lesion_df = _prepare_hindi_data()
    selected_alpha = 2.0

    selection_charts = []
    for metric, color in [("Target − Latin gap", COLORS["four_b"]), ("Target probability", COLORS["good"])]:
        sub = selection_df[selection_df["metric"] == metric]
        line = alt.Chart(sub).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=80),
            strokeWidth=3,
            color=color,
        ).encode(
            x=alt.X("alpha:Q", title="Patch scale α", scale=alt.Scale(domain=[0.2, 2.05])),
            y=alt.Y("value:Q", title=None),
        ).properties(width=260, height=135, title=metric)
        rule = alt.Chart(pd.DataFrame({"alpha": [selected_alpha]})).mark_rule(color=COLORS["one_b"], strokeDash=[8, 5], strokeWidth=2).encode(x="alpha:Q")
        label_y = float(sub["value"].max()) * 0.96
        label = alt.Chart(pd.DataFrame({"alpha": [selected_alpha], "value": [label_y], "label": [f"selected α = {selected_alpha:g}"]})).mark_text(
            dx=10,
            dy=-8,
            align="left",
            color=COLORS["one_b"],
            fontSize=12,
        ).encode(x="alpha:Q", y="value:Q", text="label:N")
        selection_charts.append(line + rule + label)
    selection_panel = alt.vconcat(*selection_charts, spacing=10).properties(
        title=alt.TitleParams(
            "A. Selection split sweep",
            subtitle="Tune the patch on local first-token criteria only, then carry it to held-out generation.",
            anchor="start",
        )
    )

    eval_base = alt.Chart(eval_df).encode(
        y=alt.Y("intervention:N", sort=INTERVENTION_ORDER, title=None),
        x=alt.X("value:Q", title=None),
    )
    eval_rule = eval_base.mark_rule(color=COLORS["grid"], strokeWidth=3).encode(x2=alt.value(0))
    eval_point = eval_base.mark_point(filled=True, size=110).encode(color=alt.Color("color:N", scale=None, legend=None))
    eval_text = eval_base.mark_text(dx=10, align="left", fontSize=11, color=COLORS["ink"]).encode(text=alt.Text("value:Q", format="+.02f"))
    eval_panel = (eval_rule + eval_point + eval_text).properties(width=125, height=82).facet(
        column=alt.Column("metric:N", sort=["Delta exact match", "Delta CER improvement", "Delta first-entry correct"], title=None),
    ).resolve_scale(x="independent").properties(
        title=alt.TitleParams(
            "B. Held-out generation deltas",
            subtitle="Reviewer rerun on 200 held-out items: the chosen patch helps, while sign flip hurts.",
            anchor="start",
        )
    )

    lesion_base = alt.Chart(lesion_df).encode(
        y=alt.Y("intervention:N", sort=LESION_ORDER, title=None),
        x=alt.X("cer_improvement:Q", title="Δ CER improvement vs baseline"),
    )
    lesion_rule = lesion_base.mark_rule(color=COLORS["grid"], strokeWidth=3).encode(x2=alt.value(0))
    lesion_point = lesion_base.mark_point(filled=True, size=140).encode(color=alt.Color("color:N", scale=None, legend=None))
    lesion_value = lesion_base.mark_text(dx=10, align="left", fontSize=11, color=COLORS["ink"]).encode(text=alt.Text("cer_improvement:Q", format="+.02f"))
    lesion_em = lesion_base.mark_text(dx=98, align="left", fontSize=11, color=COLORS["muted"]).encode(text="exact_match_label:N")
    lesion_panel = (lesion_rule + lesion_point + lesion_value + lesion_em).properties(
        width=740,
        height=155,
        title=alt.TitleParams(
            "C. Signed steering is stronger than simple lesioning",
            subtitle="Lesions are weak; the calibrated signed shift is the only substantial positive intervention.",
            anchor="start",
        ),
    )

    chart = alt.vconcat(
        alt.hconcat(selection_panel, eval_panel, spacing=30),
        lesion_panel,
        spacing=18,
    ).properties(
        title=alt.TitleParams(
            "Practical Hindi patch from a bounded channel mechanism",
            subtitle="Held-out intervention view for Gemma 3 1B Hindi using the 200-item reviewer rerun. Positive values indicate improvement over the helpful baseline.",
            anchor="start",
        )
    )
    _save(_styled(chart), "fig_hindi_practical_patch.png")


def _prepare_telugu_df() -> pd.DataFrame:
    med_1b = _read_json(
        PROJECT_ROOT
        / "research/results/telugu_continuation_mediation_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_mediation_panel.json"
    )
    med_4b = _read_json(
        PROJECT_ROOT
        / "research/results/telugu_continuation_mediation_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_mediation_panel.json"
    )
    suf_1b = _read_json(
        PROJECT_ROOT
        / "research/results/telugu_continuation_final_site_sufficiency_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json"
    )
    suf_4b = _read_json(
        PROJECT_ROOT
        / "research/results/telugu_continuation_final_site_sufficiency_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json"
    )

    def pick_row(summary_rows: list[Dict[str, Any]], site_name: str, intervention: str) -> Dict[str, Any]:
        for row in summary_rows:
            site = row.get("writer_site") or row.get("site") or {}
            if str(site.get("name", "")) == str(site_name) and str(row.get("intervention_name", row.get("intervention", ""))) == str(intervention):
                return row
        raise KeyError((site_name, intervention))

    records = []
    specs = [
        (
            "1B · final site L26",
            med_1b,
            suf_1b,
            "L18_layer_output",
            {
                "Writer only": ("writer_only", "four_b"),
                "+ recipient overwrite": ("writer_plus_mediator_recipient_overwrite", "harm"),
                "+ off-band overwrite": ("writer_plus_offband_recipient_overwrite", "good"),
            },
        ),
        (
            "4B · final site L34",
            med_4b,
            suf_4b,
            "L30_layer_output",
            {
                "Writer only": ("writer_only", "four_b"),
                "+ recipient overwrite": ("writer_plus_mediator_recipient_overwrite", "harm"),
                "+ off-band overwrite": ("writer_plus_offband_recipient_overwrite", "good"),
            },
        ),
    ]
    for model_label, med, suf, site_name, mapping in specs:
        for label, (raw_name, color_key) in mapping.items():
            records.append(
                {
                    "model": model_label,
                    "group": "Necessity",
                    "intervention": label,
                    "value": float(pick_row(med["summary_rows"], site_name, raw_name)["delta_mean_gap"]),
                    "color": COLORS[color_key],
                }
            )
        for label, raw_name, color_key in [
            ("Site donor", "site_donor", "four_b"),
            ("Off-band donor", "offband_donor", "light_blue"),
            ("Random direction", "site_random_delta", "harm"),
        ]:
            row = next(r for r in suf["summary_rows"] if r["intervention_name"] == raw_name)
            records.append(
                {
                    "model": model_label,
                    "group": "Sufficiency",
                    "intervention": label,
                    "value": float(row["delta_mean_gap"]),
                    "color": COLORS[color_key],
                }
            )
    return pd.DataFrame.from_records(records)


def plot_telugu_bottleneck_summary() -> None:
    df = _prepare_telugu_df()

    base = alt.Chart(df).encode(
        y=alt.Y("intervention:N", sort=TELUGU_ORDER, title=None),
        x=alt.X("value:Q", title="Δ mean gold-vs-bank gap"),
    )
    bars = base.mark_bar(size=20, cornerRadiusEnd=6).encode(color=alt.Color("color:N", scale=None, legend=None))
    text = base.mark_text(dx=8, align="left", fontSize=11, color=COLORS["ink"]).encode(text=alt.Text("value:Q", format=".1f"))

    chart = alt.layer(bars, text).properties(width=290, height=112).facet(
        row=alt.Row("group:N", sort=["Necessity", "Sufficiency"], title=None, header=alt.Header(labelOrient="left")),
        column=alt.Column("model:N", sort=["1B · final site L26", "4B · final site L34"], title=None),
    ).resolve_scale(y="independent").properties(
        title=alt.TitleParams(
            "Telugu late bottleneck: necessity and sufficiency controls",
            subtitle="Overwrite kills upstream rescue; true-site donor transfer beats off-band and random controls in both models.",
            anchor="start",
        )
    )
    _save(_styled(chart), "fig_telugu_bottleneck_summary.png")


def _read_control_summary(path: Path) -> Dict[str, Any]:
    return _read_json(path)["summary_by_condition"]


def _prepare_cross_family_df() -> pd.DataFrame:
    records = []
    gemma_paths = {
        ("Gemma 1B", "Hindi", 8): PROJECT_ROOT / "research/results/loop2_vm_controls/loop2_full/raw/1b/aksharantar_hin_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Hindi", 64): PROJECT_ROOT / "research/results/four_lang_thesis_panel/seed42/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Telugu", 8): PROJECT_ROOT / "research/results/loop2_vm_controls/loop2_full/raw/1b/aksharantar_tel_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Telugu", 64): PROJECT_ROOT / "research/results/four_lang_thesis_panel/seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Hindi", 8): PROJECT_ROOT / "research/results/loop2_vm_controls/loop2_full/raw/4b/aksharantar_hin_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Hindi", 64): PROJECT_ROOT / "research/results/four_lang_thesis_panel/seed42/raw/4b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Telugu", 8): PROJECT_ROOT / "research/results/loop2_vm_controls/loop2_full/raw/4b/aksharantar_tel_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Telugu", 64): PROJECT_ROOT / "research/results/four_lang_thesis_panel/seed42/raw/4b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
    }
    for (model_name, language, k), path in gemma_paths.items():
        summary = _read_control_summary(path)
        helpful = summary["icl_helpful"]
        corrupt = summary["icl_corrupt"]
        cell = f"{'Hin' if language == 'Hindi' else 'Tel'}-{k}"
        records.extend(
            [
                {"model": model_name, "cell": cell, "metric": "Helpful exact match", "value": float(helpful["mean_exact_match"])},
                {"model": model_name, "cell": cell, "metric": "Helpful first-entry", "value": float(helpful["mean_first_entry_correct"])},
                {"model": model_name, "cell": cell, "metric": "CER gain vs corrupt", "value": float(corrupt["mean_akshara_cer"] - helpful["mean_akshara_cer"])},
            ]
        )

    external_specs = [
        ("Qwen 2.5 1.5B", "qwen2.5-1.5b"),
        ("Qwen 2.5 3B", "qwen2.5-3b"),
        ("Llama 3.2 1B", "llama3.2-1b"),
        ("Llama 3.2 3B", "llama3.2-3b"),
    ]
    pair_map = {"Hindi": "aksharantar_hin_latin", "Telugu": "aksharantar_tel_latin"}
    for model_name, model_dir in external_specs:
        for language, pair in pair_map.items():
            for k in [8, 64]:
                path = PROJECT_ROOT / "research/results/cross_model_behavioral_v1" / model_dir / pair / "seed42" / f"nicl{k}" / "cross_model_behavioral.json"
                payload = _read_json(path)["summary"]
                helpful = payload["icl_helpful"]
                corrupt = payload["icl_corrupt"]
                cell = f"{'Hin' if language == 'Hindi' else 'Tel'}-{k}"
                records.extend(
                    [
                        {"model": model_name, "cell": cell, "metric": "Helpful exact match", "value": float(helpful["mean_exact_match"])},
                        {"model": model_name, "cell": cell, "metric": "Helpful first-entry", "value": float(helpful["mean_first_entry_correct"])},
                        {"model": model_name, "cell": cell, "metric": "CER gain vs corrupt", "value": float(corrupt["mean_akshara_cer"] - helpful["mean_akshara_cer"])},
                    ]
                )
    return pd.DataFrame.from_records(records)


def plot_cross_family_behavior() -> None:
    df = _prepare_cross_family_df()
    limits = {
        "Helpful exact match": (0.0, 0.50),
        "Helpful first-entry": (0.0, 1.0),
        "CER gain vs corrupt": (-0.10, 0.45),
    }
    subtitles = {
        "Helpful exact match": "end-to-end task success under helpful prompting",
        "Helpful first-entry": "first akshara correct under helpful prompting",
        "CER gain vs corrupt": "positive means helpful beats corrupt on CER",
    }
    pieces = []
    for metric in ["Helpful exact match", "Helpful first-entry", "CER gain vs corrupt"]:
        sub = df[df["metric"] == metric].copy()
        lo, hi = limits[metric]
        base = alt.Chart(sub).encode(
            x=alt.X("cell:N", sort=CROSS_CELL_ORDER, title=None, axis=alt.Axis(labelAngle=0, labelPadding=8)),
            y=alt.Y("model:N", sort=CROSS_FAMILY_ORDER, title=None),
        )
        rect = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=1.5).encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(domain=[lo, (lo + hi) / 2.0, hi], range=[COLORS["mid"], COLORS["light_blue"], COLORS["pos"]]),
                legend=None,
            )
        )
        text = base.mark_text(fontSize=11, fontWeight="bold", color=COLORS["ink"]).encode(text=alt.Text("value:Q", format=".2f"))
        pieces.append(
            (rect + text).properties(
                width=230,
                height=180,
                title=alt.TitleParams(metric, subtitle=subtitles[metric], anchor="start"),
            )
        )
    chart = alt.hconcat(*pieces, spacing=18).resolve_scale(color="independent").properties(
        title=alt.TitleParams(
            "Cross-family behavioral follow-up",
            subtitle="Gemma included for reference at seed 42. Above the capability floor, Telugu keeps a larger entry→completion gap than Hindi across families.",
            anchor="start",
        )
    )
    _save(_styled(chart), "fig_cross_family_behavior.png")


def _prepare_prompt_tradeoff_df() -> pd.DataFrame:
    label_map = {
        "icl_helpful": "helpful",
        "icl_helpful_similarity_desc": "sim-front",
        "icl_helpful_similarity_asc": "sim-back",
        "icl_helpful_reversed": "reversed",
        "icl_helpful_drop_nearest_replace_far": "drop-1",
        "icl_helpful_drop_top2_replace_far": "drop-2",
        "icl_corrupt": "corrupt",
    }
    order_map = {
        "icl_helpful": 1,
        "icl_helpful_similarity_desc": 2,
        "icl_helpful_similarity_asc": 3,
        "icl_helpful_reversed": 4,
        "icl_helpful_drop_nearest_replace_far": 5,
        "icl_helpful_drop_top2_replace_far": 6,
        "icl_corrupt": 7,
    }
    label_pos = {
        ("Gemma 1B Telugu", "icl_helpful"): (0.64, 0.43),
        ("Gemma 1B Telugu", "icl_helpful_similarity_desc"): (0.66, 0.475),
        ("Gemma 1B Telugu", "icl_helpful_similarity_asc"): (0.24, 0.155),
        ("Gemma 1B Telugu", "icl_helpful_reversed"): (0.64, 0.345),
        ("Gemma 1B Telugu", "icl_helpful_drop_nearest_replace_far"): (0.71, 0.398),
        ("Gemma 1B Telugu", "icl_helpful_drop_top2_replace_far"): (0.41, 0.318),
        ("Gemma 1B Telugu", "icl_corrupt"): (0.49, 0.392),
        ("Gemma 4B Telugu", "icl_helpful"): (1.102, 0.048),
        ("Gemma 4B Telugu", "icl_helpful_similarity_desc"): (1.102, 0.082),
        ("Gemma 4B Telugu", "icl_helpful_similarity_asc"): (1.102, 0.034),
        ("Gemma 4B Telugu", "icl_helpful_reversed"): (1.102, 0.061),
        ("Gemma 4B Telugu", "icl_helpful_drop_nearest_replace_far"): (1.102, 0.017),
        ("Gemma 4B Telugu", "icl_helpful_drop_top2_replace_far"): (1.102, 0.003),
        ("Gemma 4B Telugu", "icl_corrupt"): (1.102, 0.186),
    }
    records = []
    for model in ["1b", "4b"]:
        payload = _read_json(
            PROJECT_ROOT / "research/results/prompt_composition_ablation_v1" / model / "aksharantar_tel_latin" / "seed42" / "nicl64" / "prompt_composition_ablation.json"
        )
        summary = payload["summary_by_condition"]
        model_name = "Gemma 1B Telugu" if model == "1b" else "Gemma 4B Telugu"
        for cond, vals in summary.items():
            if cond not in label_map:
                continue
            label = label_map[cond]
            text_x, text_y = label_pos[(model_name, cond)]
            records.append(
                {
                    "model": model_name,
                    "condition": cond,
                    "label": label,
                    "variant_order": int(order_map[cond]),
                    "first_entry": float(vals["mean_first_entry_correct"]),
                    "exact_bank": float(vals["exact_bank_copy_rate"]),
                    "fuzzy_bank": float(vals["fuzzy_bank_copy_rate"]),
                    "cer": float(vals["mean_akshara_cer"]),
                    "text_x": float(text_x),
                    "text_y": float(text_y),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_prompt_composition_tradeoff() -> None:
    df = _prepare_prompt_tradeoff_df()
    selected = ["helpful", "sim-front", "sim-back", "corrupt"]
    df = df[df["label"].isin(selected)].copy()
    x_scale = alt.Scale(domain=[0, 1.05])
    y_scale = alt.Scale(domain=[0, 0.5])
    shape_order = ["helpful", "sim-front", "sim-back", "corrupt"]
    shape_range = ["circle", "triangle-up", "triangle-down", "diamond"]

    point_base = alt.Chart(df).encode(
        x=alt.X("first_entry:Q", title="First-entry correctness", scale=x_scale),
        y=alt.Y("exact_bank:Q", title="Exact bank-copy rate", scale=y_scale),
        color=alt.Color(
            "cer:Q",
            title="Akshara CER",
            scale=alt.Scale(domain=[0.2, 1.0], range=[COLORS["pos"], COLORS["harm"]]),
            legend=alt.Legend(orient="top", direction="horizontal", gradientLength=120, titleOrient="left"),
        ),
        shape=alt.Shape(
            "label:N",
            scale=alt.Scale(domain=shape_order, range=shape_range),
            legend=alt.Legend(title="Shown variants", orient="bottom", direction="horizontal", columns=4, symbolSize=110),
        ),
        order=alt.Order("variant_order:Q", sort="ascending"),
    )
    points = point_base.mark_point(size=185, filled=True, stroke="white", strokeWidth=1.15, opacity=0.84)
    chart = points.properties(width=260, height=220).facet(
        column=alt.Column("model:N", sort=["Gemma 1B Telugu", "Gemma 4B Telugu"], title=None)
    )
    _save(_styled(chart), "fig_prompt_composition_tradeoff.png")


def main() -> int:
    plot_behavioral_regime_summary()
    plot_hindi_practical_patch()
    plot_telugu_bottleneck_summary()
    plot_cross_family_behavior()
    plot_prompt_composition_tradeoff()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
