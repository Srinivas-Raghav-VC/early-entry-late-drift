from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_by_language(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lang = str(row.get("language", "unknown"))
        grouped.setdefault(lang, []).append(row)
    for lang in grouped:
        grouped[lang].sort(key=lambda r: int(r.get("N", 0)))
    return grouped


def plot_deterministic(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    prompt_template: str,
    icl_variant: str,
) -> None:
    grouped = _group_by_language(rows)
    langs = sorted(grouped.keys())
    if not langs:
        raise RuntimeError("No behavior_summary rows found")

    fig, axes = plt.subplots(len(langs), 1, figsize=(8, 3.6 * len(langs)), sharex=True)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        pts = grouped[lang]
        xs = [int(r["N"]) for r in pts]
        exact = [float(r["exact_match_rate"]) for r in pts]
        cer = [float(r["akshara_CER_mean"]) for r in pts]
        script = [float(r["script_validity_rate"]) for r in pts]

        ax.plot(xs, exact, marker="o", label="Exact match", color="#1f77b4")
        ax.plot(xs, script, marker="s", label="Script-valid", color="#2ca02c", linestyle="--")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{lang}: deterministic degradation ({prompt_template}/{icl_variant})"
        )

        ax2 = ax.twinx()
        ax2.plot(xs, cer, marker="^", label="Akshara CER", color="#d62728")
        ax2.set_ylabel("CER")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="best")

    axes[-1].set_xlabel("ICL N")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_judge_correspondence(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    prompt_template: str,
    icl_variant: str,
) -> None:
    grouped = _group_by_language(rows)
    langs = sorted(grouped.keys())
    if not langs:
        raise RuntimeError("No judge_curve rows found")

    fig, axes = plt.subplots(len(langs), 1, figsize=(8, 3.6 * len(langs)), sharex=True)
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        pts = grouped[lang]
        xs = [int(r["N"]) for r in pts]
        judge_acc = [
            float(r.get("judge_acceptable_rate"))
            if r.get("judge_acceptable_rate") is not None
            else float("nan")
            for r in pts
        ]
        det_exact = [float(r.get("det_exact_rate", 0.0)) for r in pts]
        det_cer = [float(r.get("det_cer_mean", 0.0)) for r in pts]

        ax.plot(xs, det_exact, marker="o", label="Deterministic exact", color="#1f77b4")
        ax.plot(xs, judge_acc, marker="s", label="Gemini acceptable", color="#9467bd")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{lang}: judge correspondence ({prompt_template}/{icl_variant})"
        )

        ax2 = ax.twinx()
        ax2.plot(xs, det_cer, marker="^", label="Deterministic CER", color="#d62728")
        ax2.set_ylabel("CER")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="best")

    axes[-1].set_xlabel("ICL N")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Phase 0A degradation and judge correspondence curves.")
    ap.add_argument(
        "--input",
        default="research/results/phase0/phase0a_packet_results.json",
        help="Path to Phase 0A packet JSON",
    )
    ap.add_argument("--out-dir", default="research/results/phase0")
    ap.add_argument("--prompt-template", default="canonical")
    ap.add_argument("--icl-variant", default="helpful")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = _load_json(Path(args.input))
    out_dir = Path(args.out_dir)

    behavior_rows_all = list(payload.get("behavior_summary", []))
    judge_rows_all = list(payload.get("judge_curve", []))

    behavior_rows = [
        row
        for row in behavior_rows_all
        if str(row.get("prompt_template", "canonical")) == str(args.prompt_template)
        and str(row.get("icl_variant", "helpful")) == str(args.icl_variant)
    ]
    judge_rows = [
        row
        for row in judge_rows_all
        if str(row.get("prompt_template", "canonical")) == str(args.prompt_template)
        and str(row.get("icl_variant", "helpful")) == str(args.icl_variant)
    ]

    fig1 = out_dir / f"fig_phase0_deterministic_degradation_{args.prompt_template}_{args.icl_variant}.png"
    plot_deterministic(
        behavior_rows,
        fig1,
        prompt_template=str(args.prompt_template),
        icl_variant=str(args.icl_variant),
    )
    print(str(fig1))

    if judge_rows:
        fig2 = out_dir / f"fig_phase0_judge_correspondence_{args.prompt_template}_{args.icl_variant}.png"
        plot_judge_correspondence(
            judge_rows,
            fig2,
            prompt_template=str(args.prompt_template),
            icl_variant=str(args.icl_variant),
        )
        print(str(fig2))
    else:
        print("judge_curve missing for selected template/variant; skip correspondence plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
