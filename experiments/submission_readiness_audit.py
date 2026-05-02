#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight submission-readiness audit for the ICML 2026 workshop paper.

This is intentionally conservative: it checks whether the repo contains the
reproducibility and submission-safety affordances that came up repeatedly in external-style reviews.
It does not certify scientific correctness; it flags missing artifacts that make
reviewer objections easy.
"""

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Paper Template and Paper" / "Paper" / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
V30 = ROOT / "Paper Template and Paper" / "Paper" / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_v30.tex"
README_REPRO = ROOT / "README_REPRODUCE.md"
STAGE_FIG = ROOT / "Paper Template and Paper" / "Paper" / "figures" / "fig_stage_axis_map_tikz.tex"
CAUSAL_PATCH = ROOT / "experiments" / "hindi_1b_causal_patch_panel.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _has_identity_leak(text: str) -> bool:
    # Keep this generic so the public review artifact does not itself expose
    # private usernames or repository handles. It catches the common leak
    # classes that matter for double-blind review: named GitHub URLs, raw
    # private VM addresses, and local home-directory paths.
    patterns = [
        r"github\.com/[^\s{}]+",
        r"[A-Za-z0-9._%+-]+@(?!vm\.example\.invalid)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(?:\d{1,3}\.){3}\d{1,3}",
        r"/home/(?!reviewer(?:/|$))[A-Za-z0-9._-]+(?:/|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).lower()
            if value in {"anon@anon.invalid", "anonymous@anonymous.invalid"}:
                continue
            return True
    return False


def main() -> int:
    paper = _read(PAPER)
    patch_script = _read(CAUSAL_PATCH)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def blocker(key: str, message: str) -> None:
        blockers.append({"key": key, "message": message})

    def warning(key: str, message: str) -> None:
        warnings.append({"key": key, "message": message})

    if not PAPER.exists():
        blocker("missing_final_tex", f"Missing canonical paper TeX: {PAPER}")
    if V30.exists() and PAPER.exists() and _read(V30) != paper:
        warning("v30_final_diverge", "v30.tex and final.tex differ; keep only one canonical submission target or sync them.")

    if "Claim ledger" not in paper and "Claim-to-evidence" not in paper:
        blocker("missing_claim_ledger", "No claim ledger / claim-to-evidence table in the main paper.")

    if "fig_stage_axis_map_tikz" not in paper or not STAGE_FIG.exists():
        blocker("missing_data_driven_stage_teaser", "The overview figure does not use the data-driven early-entry vs continuation-stability stage map.")

    if not README_REPRO.exists():
        blocker("missing_reproduce_readme", "Missing top-level README_REPRODUCE.md with figure/table-to-command mapping.")
    else:
        repro = _read(README_REPRO)
        for token in ["Figure", "Table", "GPU", "Expected", "Anonymous"]:
            if token.lower() not in repro.lower():
                warning(f"repro_missing_{token.lower()}", f"README_REPRODUCE.md may be missing a {token!r} section or note.")
        if _has_identity_leak(repro):
            blocker("identity_leak_repro", "README_REPRODUCE.md appears to contain identifying GitHub/user details.")

    if "--patch-pairs" not in patch_script or "icl_helpful:icl_corrupt" not in patch_script:
        blocker("missing_same_length_patch_support", "Hindi causal patch panel cannot yet run explicit same-length helpful/corrupt patch pairs from CLI.")

    if "same-length helpful" in paper and "lacks a definitive same-length" in paper:
        warning("paper_admits_missing_same_length", "Paper still states the same-length Hindi control is missing.")

    if _has_identity_leak(paper):
        blocker("identity_leak_paper", "Paper TeX contains likely identifying details.")

    # A tiny overclaim heuristic: flag if strong circuit language appears without bounded qualifiers nearby.
    for match in re.finditer(r"complete (?:circuit|mechanism)|full circuit|reverse-engineer", paper, flags=re.IGNORECASE):
        context = paper[max(0, match.start() - 120): match.end() + 120]
        if not re.search(r"not|do not|no|without|bounded", context, flags=re.IGNORECASE):
            warning("possible_overclaim", f"Potentially strong mechanistic phrase near: {match.group(0)!r}")

    payload = {
        "paper": str(PAPER.relative_to(ROOT)),
        "critical_submission_blockers": len(blockers),
        "warning_count": len(warnings),
        "blockers": blockers,
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"METRIC critical_submission_blockers={len(blockers)}")
    print(f"METRIC warning_count={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
