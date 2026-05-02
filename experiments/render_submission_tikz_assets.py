#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT / "Paper Template and Paper" / "Paper"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def seq_color(value: float, lo: float, hi: float) -> tuple[int, int, int]:
    t = 0.0 if hi <= lo else clamp((value - lo) / (hi - lo), 0.0, 1.0)
    low = (246, 246, 249)
    high = (74, 132, 226)
    return tuple(lerp(l, h, t) for l, h in zip(low, high))


def div_color(value: float, lo: float, hi: float) -> tuple[int, int, int]:
    neg = (214, 82, 76)
    mid = (246, 246, 249)
    pos = (52, 199, 89)
    if value >= 0:
        t = 0.0 if hi <= 0 else clamp(value / hi, 0.0, 1.0)
        return tuple(lerp(m, p, t) for m, p in zip(mid, pos))
    t = 0.0 if lo >= 0 else clamp(value / lo, 0.0, 1.0)
    return tuple(lerp(m, n, t) for m, n in zip(mid, neg))


def fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def write_hindi_figure() -> None:
    patch = read_json(
        PROJECT_ROOT
        / "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
    )
    intervention = read_json(
        PROJECT_ROOT
        / "research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json"
    )

    selection_rows = patch["selection_rows"]
    patch_rows = {row["intervention"]: row for row in patch["summary_rows"]}
    intervention_rows = {row["intervention"]: row for row in intervention["summary_rows"]}

    selected_alpha = float(patch["selected_alpha"])
    top_metrics = [
        ("chosen_mean_shift", "Mean shift"),
        ("chosen_sign_flip", "Sign flip"),
        ("chosen_zero_ablate", "Zero ablate"),
    ]
    heldout_metric_specs = [
        (r"$\Delta$EM", "delta_exact_match", -0.05, 0.05),
        (r"$\Delta$CER", "delta_cer_improvement", -0.15, 0.18),
        (r"$\Delta$entry", "delta_first_entry_correct", -0.35, 0.35),
    ]
    lesion_specs = [
        ("zero_channel_5486", "Zero 5486"),
        ("zero_channel_2299", "Zero 2299"),
        ("zero_both_channels", "Zero both"),
        ("calibrated_mean_shift", "Mean shift"),
        ("calibrated_sign_flip", "Sign flip"),
    ]

    out: list[str] = [
        r"\begin{tikzpicture}[x=1cm,y=1cm, every node/.style={inner sep=0pt}]",
        r"  \tikzset{cell/.style={draw=black!8, line width=0.28pt, rounded corners=2.2pt}, head/.style={font=\bfseries\small}, rowlab/.style={font=\bfseries\scriptsize, text=black!88}, colab/.style={font=\bfseries\scriptsize}, tinycap/.style={font=\scriptsize, text=black!63}, num/.style={font=\bfseries\scriptsize}}",
        "",
        r"  \node[anchor=west, head] at (0.0,0.38) {A. Selection-split sweep};",
    ]

    cellw = 1.25
    cellh = 0.56
    x_alpha = 0.0
    x_gap = 1.55
    x_prob = 3.00
    y0 = -0.55
    out.append(rf"  \node[anchor=west, colab] at ({x_alpha:.2f},{y0+0.22:.2f}) {{alpha}};")
    out.append(rf"  \node[anchor=west, colab] at ({x_gap:.2f},{y0+0.22:.2f}) {{$\Delta$ gap}};")
    out.append(rf"  \node[anchor=west, colab] at ({x_prob:.2f},{y0+0.22:.2f}) {{$\Delta p_{{target}}$}};")
    for i, row in enumerate(selection_rows):
        y = y0 - (i + 1) * 0.68
        alpha = float(row["alpha"])
        gap = float(row["delta_mean_target_minus_latin_logit"])
        prob = float(row["delta_mean_target_prob"])
        gap_rgb = seq_color(gap, 0.0, 6.35)
        prob_rgb = seq_color(prob, 0.0, 0.33)
        gap_color = f"hpA_gap_{i}"
        prob_color = f"hpA_prob_{i}"
        out.append(rf"  \definecolor{{{gap_color}}}{{RGB}}{{{gap_rgb[0]},{gap_rgb[1]},{gap_rgb[2]}}}")
        out.append(rf"  \definecolor{{{prob_color}}}{{RGB}}{{{prob_rgb[0]},{prob_rgb[1]},{prob_rgb[2]}}}")
        outline = "appleOrange" if abs(alpha - selected_alpha) < 1e-9 else "black!8"
        width = "0.9pt" if abs(alpha - selected_alpha) < 1e-9 else "0.28pt"
        out.append(rf"  \path[fill=white, draw={outline}, line width={width}, rounded corners=2.2pt] ({x_alpha:.2f},{y:.2f}) rectangle ++(1.05,-{cellh:.2f});")
        out.append(rf"  \node[num] at ({x_alpha+0.525:.2f},{y-0.28:.2f}) {{{alpha:.2f}}};")
        out.append(rf"  \path[fill={gap_color}, draw={outline}, line width={width}, rounded corners=2.2pt] ({x_gap:.2f},{y:.2f}) rectangle ++({cellw:.2f},-{cellh:.2f});")
        out.append(rf"  \node[num] at ({x_gap+cellw/2:.2f},{y-0.28:.2f}) {{{gap:+.2f}}};")
        out.append(rf"  \path[fill={prob_color}, draw={outline}, line width={width}, rounded corners=2.2pt] ({x_prob:.2f},{y:.2f}) rectangle ++({cellw:.2f},-{cellh:.2f});")
        out.append(rf"  \node[num] at ({x_prob+cellw/2:.2f},{y-0.28:.2f}) {{{prob:+.2f}}};")

    bx = 5.25
    by = 0.38
    out.append(rf"  \node[anchor=west, head] at ({bx:.2f},{by:.2f}) {{B. Held-out deltas on 200 items}};")
    header_y = -0.55
    row_y0 = -1.18
    bcol_x = [bx, bx + 2.05, bx + 3.92, bx + 5.79]
    out.append(rf"  \node[anchor=west, colab] at ({bcol_x[0]:.2f},{header_y+0.22:.2f}) {{intervention}};")
    for j, (label, _k, _lo, _hi) in enumerate(heldout_metric_specs, start=1):
        out.append(rf"  \node[anchor=west, colab] at ({bcol_x[j]:.2f},{header_y+0.22:.2f}) {{{label}}};")
    for i, (raw, label) in enumerate(top_metrics):
        y = row_y0 - i * 0.72
        out.append(rf"  \node[anchor=west, rowlab] at ({bcol_x[0]:.2f},{y-0.26:.2f}) {{{label}}};")
        for j, (_mlabel, key, lo, hi) in enumerate(heldout_metric_specs, start=1):
            value = float(patch_rows[raw][key]["mean"])
            rgb = div_color(value, lo, hi)
            cname = f"hpB_{i}_{j}"
            out.append(rf"  \definecolor{{{cname}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
            out.append(rf"  \path[fill={cname}, cell] ({bcol_x[j]:.2f},{y:.2f}) rectangle ++(1.48,-0.56);")
            out.append(rf"  \node[num] at ({bcol_x[j]+0.74:.2f},{y-0.28:.2f}) {{{value:+.2f}}};")

    cx = 0.0
    cy = -6.15
    out.append(rf"  \node[anchor=west, head] at ({cx:.2f},{cy:.2f}) {{C. Signed steering vs. lesioning}};")
    cheader_y = cy - 0.93
    crow_y0 = cy - 1.56
    ccol_x = [cx, cx + 4.15, cx + 6.18]
    out.append(rf"  \node[anchor=west, colab] at ({ccol_x[0]:.2f},{cheader_y+0.22:.2f}) {{intervention}};")
    out.append(rf"  \node[anchor=west, colab] at ({ccol_x[1]:.2f},{cheader_y+0.22:.2f}) {{$\Delta$ CER}};")
    out.append(rf"  \node[anchor=west, colab] at ({ccol_x[2]:.2f},{cheader_y+0.22:.2f}) {{$\Delta$ EM}};")
    for i, (raw, label) in enumerate(lesion_specs):
        y = crow_y0 - i * 0.70
        cer = float(intervention_rows[raw]["delta_cer_improvement"]["mean"])
        em = float(intervention_rows[raw]["delta_exact_match"]["mean"])
        cer_rgb = div_color(cer, -0.15, 0.18)
        em_rgb = div_color(em, -0.05, 0.05)
        cer_color = f"hpC_cer_{i}"
        em_color = f"hpC_em_{i}"
        out.append(rf"  \definecolor{{{cer_color}}}{{RGB}}{{{cer_rgb[0]},{cer_rgb[1]},{cer_rgb[2]}}}")
        out.append(rf"  \definecolor{{{em_color}}}{{RGB}}{{{em_rgb[0]},{em_rgb[1]},{em_rgb[2]}}}")
        out.append(rf"  \node[anchor=west, rowlab] at ({ccol_x[0]:.2f},{y-0.26:.2f}) {{{label}}};")
        out.append(rf"  \path[fill={cer_color}, cell] ({ccol_x[1]:.2f},{y:.2f}) rectangle ++(1.48,-0.56);")
        out.append(rf"  \node[num] at ({ccol_x[1]+0.74:.2f},{y-0.28:.2f}) {{{cer:+.2f}}};")
        out.append(rf"  \path[fill={em_color}, cell] ({ccol_x[2]:.2f},{y:.2f}) rectangle ++(1.22,-0.56);")
        out.append(rf"  \node[num] at ({ccol_x[2]+0.61:.2f},{y-0.28:.2f}) {{{em:+.3f}}};")

    out.append(r"  \path[use as bounding box] (-0.15,0.62) rectangle (12.15,-10.05);")
    out.append(r"\end{tikzpicture}")
    out.append("")

    (FIG_DIR / "fig_hindi_practical_patch_tikz.tex").write_text("\n".join(out), encoding="utf-8")


def write_cross_family_figure() -> None:
    rows: list[dict[str, Any]] = []
    gemma_paths = {
        ("Gemma 1B", "Hindi", 8): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_hin_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Hindi", 64): PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Telugu", 8): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_tel_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 1B", "Telugu", 64): PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Hindi", 8): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/loop2_full/raw/4b/aksharantar_hin_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Hindi", 64): PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/4b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Telugu", 8): PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/loop2_full/raw/4b/aksharantar_tel_latin/nicl8/neutral_filler_recency_controls.json",
        ("Gemma 4B", "Telugu", 64): PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/4b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
    }
    for (model_name, language, k), path in gemma_paths.items():
        payload = read_json(path)["summary_by_condition"]
        helpful = payload["icl_helpful"]
        corrupt = payload["icl_corrupt"]
        cell = f"{'Hin' if language == 'Hindi' else 'Tel'}-{k}"
        rows.extend(
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
                path = PROJECT_ROOT / "research/results/autoresearch/cross_model_behavioral_v1" / model_dir / pair / "seed42" / f"nicl{k}" / "cross_model_behavioral.json"
                payload = read_json(path)["summary"]
                helpful = payload["icl_helpful"]
                corrupt = payload["icl_corrupt"]
                cell = f"{'Hin' if language == 'Hindi' else 'Tel'}-{k}"
                rows.extend(
                    [
                        {"model": model_name, "cell": cell, "metric": "Helpful exact match", "value": float(helpful["mean_exact_match"])},
                        {"model": model_name, "cell": cell, "metric": "Helpful first-entry", "value": float(helpful["mean_first_entry_correct"])},
                        {"model": model_name, "cell": cell, "metric": "CER gain vs corrupt", "value": float(corrupt["mean_akshara_cer"] - helpful["mean_akshara_cer"])},
                    ]
                )

    metric_specs = [
        ("Helpful exact match", (0.0, 0.50), "sequential", "Helpful exact match", "end-to-end success under helpful prompting"),
        ("Helpful first-entry", (0.0, 1.0), "sequential", "Helpful first-entry", "first akshara correct under helpful prompting"),
        ("CER gain vs corrupt", (-0.10, 0.45), "diverging", "CER gain vs corrupt", "positive means helpful beats corrupt on CER"),
    ]
    model_order = ["Gemma 1B", "Gemma 4B", "Qwen 2.5 1.5B", "Qwen 2.5 3B", "Llama 3.2 1B", "Llama 3.2 3B"]
    cell_order = ["Hin-8", "Hin-64", "Tel-8", "Tel-64"]
    lookup = {(r["metric"], r["model"], r["cell"]): r["value"] for r in rows}

    out = [
        "\\begin{tikzpicture}[x=1cm,y=1cm, every node/.style={inner sep=0pt}]",
        "  \\def\\cellw{1.15}",
        "  \\def\\cellh{0.82}",
        "  \\def\\panelgap{1.55}",
        "  \\tikzset{cell/.style={draw=black!8, line width=0.28pt, rounded corners=2.2pt}, head/.style={font=\\bfseries\\small}, rowlab/.style={font=\\bfseries\\scriptsize, text=black!88}, colab/.style={font=\\bfseries\\scriptsize}, tinycap/.style={font=\\scriptsize, text=black!63}, num/.style={font=\\bfseries\\scriptsize}}",
        "",
    ]

    x0 = 0.0
    y_title = 0.42
    row_y = [-0.35 - i * 0.92 for i in range(len(model_order))]
    col_x = [0.0 + j * 1.27 for j in range(len(cell_order))]
    panel_width = 4 * 1.27
    for p, (metric, limits, mode, title, subtitle) in enumerate(metric_specs):
        left = x0 + p * (panel_width + 1.55)
        out.append(f"  \\node[anchor=west, head] at ({left:.2f},{y_title:.2f}) {{{title}}};")
        out.append(f"  \\node[anchor=west, tinycap] at ({left:.2f},{y_title-0.30:.2f}) {{{subtitle}}};")
        for j, cell in enumerate(cell_order):
            cx = left + col_x[j] + 0.575
            out.append(f"  \\node[anchor=north, colab] at ({cx:.3f}, {-6.03:.2f}) {{{cell}}};")
        if p == 0:
            for i, model in enumerate(model_order):
                out.append(f"  \\node[anchor=east, rowlab] at ({left-0.18:.2f},{row_y[i]-0.41:.2f}) {{{model}}};")
        for i, model in enumerate(model_order):
            for j, cell in enumerate(cell_order):
                value = lookup[(metric, model, cell)]
                rgb = seq_color(value, *limits) if mode == "sequential" else div_color(value, *limits)
                color_name = f"cf{p}{i}{j}"
                out.append(f"  \\definecolor{{{color_name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
                x = left + col_x[j]
                y = row_y[i]
                out.append(f"  \\path[fill={color_name}, cell] ({x:.3f},{y:.3f}) rectangle ++(\\cellw,-\\cellh);")
                digits = 2
                out.append(f"  \\node[num] at ({x+0.575:.3f},{y-0.41:.3f}) {{$ {value:.2f} $}};")
        # horizontal group separators
        sep1y = row_y[1] - 0.87
        sep2y = row_y[3] - 0.87
        out.append(f"  \\draw[black!16, line width=0.32pt] ({left:.3f},{sep1y:.3f}) -- ++({panel_width:.3f},0);")
        out.append(f"  \\draw[black!16, line width=0.32pt] ({left:.3f},{sep2y:.3f}) -- ++({panel_width:.3f},0);")
        out.append(f"  \\draw[black!12, line width=0.32pt] ({left:.3f},{-6.15:.3f}) -- ++({panel_width:.3f},0);")
        out.append("")

    out.append("  \\node[anchor=west, tinycap] at (0.0,-6.75) {Rows are aligned across panels. Gemma is shown for reference beside Qwen 2.5 and Llama 3.2 at seed 42.};")
    out.append("\\end{tikzpicture}")
    out.append("")

    (FIG_DIR / "fig_cross_family_behavior_tikz.tex").write_text("\n".join(out), encoding="utf-8")


def write_appendix_table() -> None:
    rows = [
        (
            "High-shot ICL splits into entry and continuation regimes",
            "Obs.",
            "Breadth panel: 1B/4B $\\times$ \\{Hindi, Telugu, Bengali, Tamil\\} $\\times$ $k\\in\\{8,64\\}$, 3 seeds, 30 held-out items/cell; Marathi matched control at $k=64$ on seed 42.",
            "Seed aggregate and pooled confidence-interval memo available.",
            "Established",
        ),
        (
            "Same-script support helps entry more than late continuation",
            "Obs.",
            "Marathi high-shot control plus first-entry / continuation follow-ups at 1B and 4B (seed 42).",
            "No dedicated multi-seed CI panel; supported by targeted audits.",
            "Supported",
        ),
        (
            "Hindi entry failure localizes to a bounded L25 MLP bottleneck",
            "Intv.",
            "30-item audited mechanistic panel; 100-item selection split; held-out seed11 channel-repeat; dense-patch and random-channel controls.",
            "Bounded held-out selection$\\rightarrow$eval design; not exhaustive multiplicity control.",
            "Supported (bounded)",
        ),
        (
            "A fixed Hindi patch improves held-out generation",
            "Intv.",
            "Selection $n=200$, evaluation $n=200$, seed 42.",
            "Bootstrap CIs on EM, CER, first-entry, and first-token gap.",
            "Established",
        ),
        (
            "Telugu depends on a late residual bottleneck (L26 / L34)",
            "Intv.",
            "Focus-60 localization plus mediation and donor-style tests in both 1B and 4B on seed 42.",
            "Off-band and random-direction controls included; sufficiency-style transplant run completed.",
            "Established",
        ),
        (
            "Static Telugu patches do not transfer like the Hindi patch",
            "Intv.",
            "Original shared-prefix practical panel ($n=57$/model) plus larger held-out Telugu rerun ($n=191$, seed 42).",
            "Larger-$N$ negative rerun available; claim is contrastive rather than positive.",
            "Supported",
        ),
        (
            "The behavioral regime map extends beyond Gemma above a capability floor",
            "Obs.",
            "Qwen 2.5 1.5B/3B and Llama 3.2 1B/3B on Hindi+Telugu, $k\\in\\{8,64\\}$, $n=200$/cell, seed 42.",
            "Single-seed behavioral replication only; no cross-family mechanistic localization.",
            "Supported (provisional)",
        ),
    ]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{0.24\\linewidth}>{\\raggedright\\arraybackslash}p{0.07\\linewidth}>{\\raggedright\\arraybackslash}p{0.28\\linewidth}>{\\raggedright\\arraybackslash}p{0.23\\linewidth}>{\\raggedright\\arraybackslash}p{0.11\\linewidth}}",
        "\\toprule",
        "Claim & Evidence & Coverage / panel size & CI / hold-out status & Status \\\\",
        "\\midrule",
    ]
    for claim, evidence, coverage, ci, status in rows:
        lines.append(f"{claim} & {evidence} & {coverage} & {ci} & {status} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(
        [
            "\\end{tabularx}",
            "\\caption{Compact appendix coverage table for the paper's major claims. `Obs.` = observational / comparative evidence; `Intv.` = interventional evidence.}",
            "\\label{tab:claim_coverage}",
            "\\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "tab_claim_coverage_appendix.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    write_hindi_figure()
    write_cross_family_figure()
    write_appendix_table()
    print("Wrote submission-facing TikZ figures and appendix table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
