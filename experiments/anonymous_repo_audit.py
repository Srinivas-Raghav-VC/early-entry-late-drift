#!/usr/bin/env python3
from __future__ import annotations

"""Audit the tracked tree for double-blind review repo readiness.

This does not upload to anonymous.4open.science. It answers the local question a
reviewer/release manager cares about before upload: if we export the tracked
repo, does it contain the main reproduction affordances and avoid obvious author
or machine identifiers?
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRACKED = [
    "README.md",
    "README_REPRODUCE.md",
    "Draft_Results/core.py",
    "Draft_Results/paper2_fidelity_calibrated/run_neutral_filler_recency_controls.py",
    "experiments/hindi_1b_practical_patch_eval.py",
    "experiments/telugu_continuation_practical_patch_eval.py",
    "experiments/telugu_1b_mlp_channel_crossover.py",
    "experiments/modal_telugu_mlp_channel_crossover.py",
    "experiments/render_stage_axis_map.py",
    "research/submission/hindi_channel_interpretation_summary_2026-04-28.md",
    "research/submission/cross_model_behavioral_synthesis_2026-04-28.md",
]
DISALLOWED_TRACKED = [
    "api.txt",
    ".env",
    "modal.toml",
]
IDENTITY_PATTERNS = [
    r"sri" + r"nivas",
    r"academic" + r"techie2022",
    r"10\.10\.0\.215",
    r"github\.com/" + r"Sri" + r"nivas",
    r"Sri" + r"nivas" + r"-Raghav-VC",
]
PRIVATE_MACHINE_PATTERNS = [
    r"/home/(?!reviewer(?:/|$))[A-Za-z0-9._-]+(?:/|$)",
    r"[A-Za-z0-9._%+-]+@(?!anon\.invalid|anonymous\.invalid|vm\.example\.invalid)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
]

# Third-party ICML/LaTeX support files can contain package maintainer emails in
# comments. Those are not author/machine leaks; scan the paper source and repo
# code instead.
TEXT_SCAN_EXCLUDE = {
    f"paper/{venue}/{filename}"
    for venue in ("icml2026", "complearn2026")
    for filename in (
        "algorithm.sty",
        "algorithmic.sty",
        "fancyhdr.sty",
        "icml2026.bst",
        "icml2026.sty",
    )
}


def _git_ls_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _read_rel(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    tracked = set(_git_ls_files())
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def blocker(key: str, message: str) -> None:
        blockers.append({"key": key, "message": message})

    def warning(key: str, message: str) -> None:
        warnings.append({"key": key, "message": message})

    for rel in REQUIRED_TRACKED:
        if rel not in tracked:
            blocker("missing_required_tracked_file", f"Required review/reproduction file is not tracked: {rel}")
    for rel in DISALLOWED_TRACKED:
        if rel in tracked:
            blocker("secret_or_local_config_tracked", f"Do not include local secret/config file in anonymous repo: {rel}")

    leak_hits: list[dict[str, str]] = []
    for rel in sorted(tracked):
        if rel in TEXT_SCAN_EXCLUDE:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = _read_rel(rel)
        except UnicodeDecodeError:
            continue
        for pattern in IDENTITY_PATTERNS + PRIVATE_MACHINE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                leak_hits.append({"path": rel, "pattern": pattern})
                break
    if leak_hits:
        blocker("identity_or_machine_leak", f"Tracked files contain likely double-blind leaks: {leak_hits[:20]}")

    repro = _read_rel("README_REPRODUCE.md") if "README_REPRODUCE.md" in tracked else ""
    for phrase in ["Anonymous review note", "Figure", "Expected headline", "Modal", "Hugging Face"]:
        if phrase.lower() not in repro.lower():
            warning("repro_readme_missing_phrase", f"README_REPRODUCE.md missing reviewer-useful phrase: {phrase}")

    payload = {
        "anonymous_repo_blockers": len(blockers),
        "anonymous_repo_warnings": len(warnings),
        "tracked_file_count": len(tracked),
        "blockers": blockers,
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"METRIC anonymous_repo_blockers={len(blockers)}")
    print(f"METRIC anonymous_repo_warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
