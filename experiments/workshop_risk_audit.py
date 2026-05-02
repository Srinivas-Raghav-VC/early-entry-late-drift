#!/usr/bin/env python3
from __future__ import annotations

"""Conservative audit for remaining ICML Mech Interp workshop reviewer risks.

This complements submission_readiness_audit.py. The readiness audit checks basic
submission affordances; this one tracks methodological/presentation risks that
external-style reviews repeatedly flagged. The point is not to game a score, but
to keep the loop focused on concrete reviewer objections.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Paper Template and Paper" / "Paper" / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
README_REPRO = ROOT / "README_REPRODUCE.md"
STAGE_FIG = ROOT / "Paper Template and Paper" / "Paper" / "figures" / "fig_stage_axis_map_tikz.tex"
SAME_LENGTH = ROOT / "research/results/autoresearch/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_causal_patch_panel.json"
CHANNEL_SUMMARY = ROOT / "research/submission/hindi_channel_interpretation_summary_2026-04-28.md"
CROSSOVER_SUMMARY = ROOT / "research/submission/mechanistic_crossover_summary_2026-04-28.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def main() -> int:
    paper = _read(PAPER)
    risks: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def check(key: str, ok: bool, message: str) -> None:
        row = {"key": key, "ok": bool(ok), "message": message}
        checks.append(row)
        if not ok:
            risks.append(row)

    check(
        "claim_ledger_present",
        "tab:claimledger" in paper,
        "Paper should include a claim-to-evidence ledger near the front.",
    )
    check(
        "data_driven_stage_teaser_present",
        "fig_stage_axis_map_tikz" in paper and STAGE_FIG.exists(),
        "Overview should contain a data-driven early-entry vs continuation-stability map.",
    )
    check(
        "same_length_control_summarized",
        "tab:hindi_same_length" in paper and SAME_LENGTH.exists(),
        "Hindi same-length helpful/corrupt patch control should be run and summarized.",
    )
    check(
        "channel_interpretation_summarized",
        "tab:hindi_channel_readout" in paper and CHANNEL_SUMMARY.exists(),
        "Hindi channels 5486/2299 need a minimal interpretation/readout-geometry table.",
    )
    check(
        "telugu_negative_bounded",
        "tested full-state mean-shift family" in paper and "not a proof" in paper,
        "Telugu negative result should be bounded to the tested edit family.",
    )
    check(
        "mechanistic_crossover_summarized",
        "tab:mechanistic_crossover" in paper and CROSSOVER_SUMMARY.exists(),
        "Paper should expose the bounded Telugu compact-channel crossover check.",
    )
    check(
        "repro_readme_present",
        README_REPRO.exists()
        and "anonymous" in _read(README_REPRO).lower()
        and "Figure" in _read(README_REPRO),
        "Top-level README_REPRODUCE.md should map figures/tables to commands and mention anonymization.",
    )

    payload = {
        "paper": str(PAPER.relative_to(ROOT)),
        "reviewer_risk_points": len(risks),
        "checks": checks,
        "risks": risks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"METRIC reviewer_risk_points={len(risks)}")
    print(f"METRIC passed_risk_checks={len(checks) - len(risks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
