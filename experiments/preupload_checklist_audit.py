#!/usr/bin/env python3
from __future__ import annotations

"""Audit human-facing final-upload instructions.

This does not judge scientific correctness. It checks whether the repo tells a
future submitter to use the metadata-sanitizing package path, run the archive
surface audit, and preserve the current long-paper evidence ledger unless the
venue forces a shorter format.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_README = ROOT / "paper" / "README.md"
REPRO_README = ROOT / "README_REPRODUCE.md"

CHECKS = [
    (
        "paper_readme_uses_package_script",
        PAPER_README,
        ["package_submission_artifacts", "metadata-sanitized"],
        "paper/README.md should make the metadata-sanitizing package script the primary build path.",
    ),
    (
        "paper_readme_mentions_archive_audit",
        PAPER_README,
        ["public_archive_surface_audit", "final_submission_audit"],
        "paper/README.md should tell submitters to run both final PDF/package and archive-surface audits.",
    ),
    (
        "long_paper_decision_recorded",
        PAPER_README,
        ["long paper", "short paper"],
        "paper/README.md should record the long-vs-short submission decision to avoid last-minute evidence-cutting.",
    ),
    (
        "repro_has_archive_command_and_audit",
        REPRO_README,
        ["git archive", "public_archive_surface_audit", "final_submission_audit"],
        "README_REPRODUCE.md should include archive creation and both final audits.",
    ),
]


def main() -> int:
    issues: list[dict[str, str]] = []
    for key, path, required_phrases, message in CHECKS:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        missing = [phrase for phrase in required_phrases if phrase.lower() not in text.lower()]
        if missing:
            issues.append({"key": key, "path": str(path.relative_to(ROOT)), "missing": ", ".join(missing), "message": message})
    payload = {"preupload_instruction_issues": len(issues), "issues": issues}
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"METRIC preupload_instruction_issues={len(issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
