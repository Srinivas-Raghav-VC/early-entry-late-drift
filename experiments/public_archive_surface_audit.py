#!/usr/bin/env python3
from __future__ import annotations

"""Audit the public `git archive` surface.

This is a packaging-quality check, not a scientific metric. The goal is to keep
anonymous review archives focused on paper source, code, tests, compact retained
artifacts, and reproduction docs while excluding internal agent/session logs and
setup notes that distract reviewers.
"""

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ARCHIVE_PATHS = [
    "README.md",
    "README_REPRODUCE.md",
    "autoresearch.sh",
    "paper/icml2026/submission.pdf",
    "paper/icml2026/submission.tex",
    "experiments/package_submission_artifacts.py",
    "experiments/final_submission_audit.py",
    "experiments/hindi_1b_practical_patch_eval.py",
    "experiments/telugu_continuation_practical_patch_eval.py",
    "research/submission/hindi_channel_interpretation_summary_2026-04-28.md",
    "research/submission/mechanistic_crossover_summary_2026-04-28.md",
    "research/submission/cross_model_behavioral_synthesis_2026-04-28.md",
    "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
]

# Internal local-agent/research-operating-system files. They can remain tracked
# in the development repo, but the anonymous review archive should not force
# reviewers to wade through them.
CLUTTER_PATHS = [
    "AGENTS.md",
    "CHANGELOG.md",
    "INTERPRETABILITY_RESEARCH_SETUP.md",
    "README_SETUP.md",
    "SESSION_SETUP_REFERENCE.md",
    "autoresearch.md",
    "research/ANSWER_EXPLAINER_CONTRACT.md",
    "research/CODE_QUALITY_CONTRACT.md",
    "research/EXAMPLES.md",
    "research/FINAL_PLANNER_PROMPT.xml",
    "research/MODEL_ROUTING.md",
    "research/PAPER_AND_TOOL_POLICY.md",
    "research/PHASE0A_RUNBOOK.md",
    "research/PROJECT_BRIEF.md",
    "research/RESEARCH_JOURNAL.md",
    "research/RETROSPECTIVE.md",
    "research/TASK_TEMPLATE.md",
    "research/VERIFICATION_AND_DEGRADATION.md",
    "research/VISUAL_DESIGN_REVIEW.md",
    "research/spec.md",
    "research/spec_journal.md",
]

CLUTTER_PREFIXES = [
    "research/agents/",
]

FORBIDDEN_PATH_PARTS = [
    ".git/",
    "api.txt",
    "autoresearch.jsonl",
    "feynman-session-",
    "Paper Template and Paper/",
    "outputs/",
    "notes/",
]


def _archive_names() -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "anonymous-review.tar.gz"
        subprocess.check_call(
            [
                "git",
                "archive",
                "--worktree-attributes",
                "--format=tar.gz",
                "--output",
                str(archive),
                "HEAD",
            ],
            cwd=ROOT,
        )
        with tarfile.open(archive, "r:gz") as tf:
            return sorted(tf.getnames())


def main() -> int:
    names = _archive_names()
    name_set = set(names)
    issues: list[dict[str, Any]] = []

    for rel in REQUIRED_ARCHIVE_PATHS:
        if rel not in name_set:
            issues.append({"kind": "missing_required", "path": rel})
    for rel in CLUTTER_PATHS:
        if rel in name_set:
            issues.append({"kind": "internal_clutter", "path": rel})
    for prefix in CLUTTER_PREFIXES:
        hits = [name for name in names if name.startswith(prefix)]
        if hits:
            issues.append({"kind": "internal_clutter_prefix", "path": prefix, "examples": hits[:5], "count": len(hits)})
    for part in FORBIDDEN_PATH_PARTS:
        hits = [name for name in names if part in name]
        if hits:
            issues.append({"kind": "forbidden_path", "path": part, "examples": hits[:5], "count": len(hits)})

    payload = {
        "archive_surface_issues": len(issues),
        "archive_file_count": len(names),
        "issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"METRIC archive_surface_issues={len(issues)}")
    print(f"METRIC archive_file_count={len(names)}")
    return 1 if any(issue["kind"] in {"missing_required", "forbidden_path"} for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
