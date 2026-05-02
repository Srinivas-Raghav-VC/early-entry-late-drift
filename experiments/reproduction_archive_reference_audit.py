#!/usr/bin/env python3
from __future__ import annotations

"""Audit README_REPRODUCE for references that break in the anonymous archive.

The compact review archive intentionally omits bulky raw sweeps, local output
summaries, and the local manuscript working tree. This audit catches reproduction
instructions that still look executable from those omitted paths without an
explicit compact-archive caveat.
"""

import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README_REPRODUCE.md"

OMITTED_PATH_PATTERNS = [
    re.compile(r"outputs/"),
    re.compile(r"Paper Template and Paper/"),
    re.compile(r"research/results/autoresearch/four_lang_thesis_panel"),
    re.compile(r"research/results/autoresearch/loop2_vm_controls"),
    re.compile(r"research/results/autoresearch/cross_model_behavioral_v1"),
]

REQUIRED_CAVEATS = [
    "Compact anonymous archive note",
    "Full raw-sweep regeneration",
    "included in the compact archive",
]


def _archive_names() -> set[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "anonymous.tar.gz"
        subprocess.check_call(
            ["git", "archive", "--worktree-attributes", "--format=tar.gz", "--output", str(archive), "HEAD"],
            cwd=ROOT,
        )
        with tarfile.open(archive, "r:gz") as tf:
            return set(tf.getnames())


def _backtick_paths(text: str) -> Iterable[str]:
    # Only inspect inline code spans. Fenced code blocks often contain shell
    # commands and language labels; treating the whole block as one path creates
    # noisy false positives.
    for match in re.finditer(r"`([^`\n]+)`", text):
        value = match.group(1).strip()
        if value.startswith("google/") or re.fullmatch(r"\d+\s*/\s*\d+", value):
            continue
        if "/" in value and not value.startswith(("http://", "https://")):
            yield value


def main() -> int:
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    archive_names = _archive_names()
    issues: list[dict[str, str]] = []

    for phrase in REQUIRED_CAVEATS:
        if phrase.lower() not in text.lower():
            issues.append({"kind": "missing_caveat", "value": phrase})

    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped or "not included" in lower or "does not include" in lower or "local working" in lower:
            continue
        for pattern in OMITTED_PATH_PATTERNS:
            if pattern.search(stripped):
                issues.append({"kind": "omitted_path_reference", "line": str(idx), "value": stripped})
                break

    missing_backticks: list[str] = []
    for path in _backtick_paths(text):
        if any(pattern.search(path) for pattern in OMITTED_PATH_PATTERNS):
            continue
        if any(ch in path for ch in "*[]"):
            continue
        if path.endswith(("/", ".invalid")):
            continue
        if path not in archive_names and not (ROOT / path).exists():
            missing_backticks.append(path)
    for path in sorted(set(missing_backticks))[:20]:
        issues.append({"kind": "missing_backtick_path", "value": path})

    payload = {
        "repro_archive_reference_issues": len(issues),
        "issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"METRIC repro_archive_reference_issues={len(issues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
