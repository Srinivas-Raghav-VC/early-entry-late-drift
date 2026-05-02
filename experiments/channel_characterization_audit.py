#!/usr/bin/env python3
from __future__ import annotations

"""Audit that Hindi channel characterization goes beyond raw localization.

This is a narrow reviewer-risk check: it does not certify semantic
interpretability, but it verifies that the submission exposes the item-level
bounded characterization rather than stopping at channel IDs and readout dots.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
SUMMARY = ROOT / "research/submission/hindi_channel_interpretation_summary_2026-04-28.md"
SUMMARY_JSON = ROOT / "research/submission/hindi_channel_interpretation_summary_2026-04-28.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def main() -> int:
    paper = _read(PAPER)
    summary = _read(SUMMARY)
    risks: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def check(key: str, ok: bool, message: str) -> None:
        row = {"key": key, "ok": bool(ok), "message": message}
        checks.append(row)
        if not ok:
            risks.append(row)

    check(
        "summary_has_item_level_characterization",
        "Item-level characterization" in summary and "corr(value, Latin top-1)" in summary,
        "Channel summary should include item-level contexts/correlations, not only readout geometry.",
    )
    check(
        "paper_mentions_bounded_item_audit",
        "item-level audit" in paper and "not a clean monosemantic" in paper,
        "Paper should mention the bounded item-level channel audit and avoid monosemantic overclaiming.",
    )
    if SUMMARY_JSON.exists():
        payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        rows = payload.get("summary_rows", [])
        check(
            "json_has_context_correlations",
            all("corr_helpful_value_with_latin_top1" in row and "top_helpful_value_contexts" in row for row in rows),
            "Summary JSON should retain the correlation and top-context fields for auditability.",
        )
    else:
        check("json_has_context_correlations", False, "Summary JSON missing.")

    payload = {
        "channel_characterization_missing": len(risks),
        "checks": checks,
        "risks": risks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"METRIC channel_characterization_missing={len(risks)}")
    print(f"METRIC passed_channel_characterization_checks={len(checks) - len(risks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
