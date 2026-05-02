#!/usr/bin/env python3
from __future__ import annotations

"""Audit final reviewer-feedback polish targets.

This audit captures presentation/claim-calibration risks raised by a simulated
workshop review. It does not score scientific truth; it checks whether the final
paper exposes the narrow story clearly: stage-specific failure, bounded causal
handles, intervention outcomes, and why the result matters beyond a scalar score.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Paper Template and Paper" / "Paper" / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
PUBLIC_PAPER = ROOT / "paper" / "icml2026" / "submission.tex"
LEGACY_FIG = ROOT / "Paper Template and Paper" / "Paper" / "figures" / "fig_stage_intervention_overview_tikz.tex"
PUBLIC_FIG = ROOT / "paper" / "figures" / "fig_stage_intervention_overview_tikz.tex"
STAGE_AXIS_FIG = ROOT / "paper" / "figures" / "fig_stage_axis_map_tikz.tex"


def main() -> int:
    paper_text = PAPER.read_text(encoding="utf-8") if PAPER.exists() else ""
    public_text = PUBLIC_PAPER.read_text(encoding="utf-8") if PUBLIC_PAPER.exists() else ""
    issues: list[dict[str, str]] = []

    def issue(key: str, message: str) -> None:
        issues.append({"key": key, "message": message})

    if "Stage-Specific Failure and Intervention" not in paper_text:
        issue("title_not_intervention_specific", "Title should foreground stage-specific failure and intervention, not only stage names.")
    if "fig_stage_intervention_overview_tikz.tex" not in paper_text:
        issue("overview_lacks_stage_intervention_panel", "Figure 1 should include a compact panel linking behavior, causal handle, and outcome.")
    if not LEGACY_FIG.exists() or not PUBLIC_FIG.exists():
        issue("stage_intervention_figure_missing", "Stage/intervention overview TikZ should exist in both local and public figure trees.")
    if "failure should not be treated as a scalar performance drop" not in paper_text:
        issue("broader_lesson_missing", "Introduction should state the broader lesson beyond Indic transliteration.")
    bottleneck_count = paper_text.lower().count("bottleneck")
    if bottleneck_count > 4:
        issue("bottleneck_language_overused", f"Bottleneck appears {bottleneck_count} times; use causal handle/site language unless strongly justified.")
    if "95\\% CI" in _abstract(paper_text):
        issue("abstract_frontloads_ci", "Abstract should lead with conceptual contribution and avoid confidence-interval detail before setup.")
    if "Telugu: Favorable Oracle Diagnostic" not in paper_text:
        issue("telugu_negative_not_reframed", "Telugu section should frame the oracle diagnostic as an informative negative result, not only a failed patch.")
    axis_text = STAGE_AXIS_FIG.read_text(encoding="utf-8") if STAGE_AXIS_FIG.exists() else ""
    if "bottleneck" in axis_text.lower():
        issue("stage_axis_uses_bottleneck", "Figure 1b should use behavioral-stage language rather than bottleneck language.")
    if any(label in axis_text for label in ["{Bn}", "{Ta}", "{Mr", "{Te}"]):
        issue("stage_axis_overlabels_cells", "Figure 1b should avoid dense per-language labels; highlight case-study cells and summarize the rest.")
    if public_text and "Stage-Specific Failure and Intervention" not in public_text:
        issue("public_source_not_synced", "Public paper source should be synced after title/figure polish.")

    payload = {
        "reviewer_feedback_flags": len(issues),
        "bottleneck_count": bottleneck_count,
        "issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"METRIC reviewer_feedback_flags={len(issues)}")
    print(f"METRIC bottleneck_count={bottleneck_count}")
    return 0


def _abstract(text: str) -> str:
    start = text.find("\\begin{abstract}")
    end = text.find("\\end{abstract}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end]


if __name__ == "__main__":
    raise SystemExit(main())
