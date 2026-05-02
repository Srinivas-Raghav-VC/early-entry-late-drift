#!/usr/bin/env python3
from __future__ import annotations

"""Final packaging audit for anonymous workshop submission.

Checks the public paper directory, selected retained result artifacts, PDF
metadata/text, and the tracked-tree archive surface. This is deliberately a
packaging/reproducibility audit, not a scientific correctness proof.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TITLE = "Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL"

REQUIRED_PUBLIC_FILES = [
    "README.md",
    "README_REPRODUCE.md",
    "paper/README.md",
    "paper/icml2026/submission.tex",
    "paper/icml2026/submission.pdf",
    "paper/icml2026/icml2026.sty",
    "paper/figures/fig_stage_intervention_overview_tikz.tex",
    "paper/figures/fig_stage_axis_map_tikz.tex",
    "paper/figures/fig_behavioral_regime_summary_tikz_v19.tex",
    "paper/figures/fig_hindi_practical_patch_tikz_v18.tex",
    "research/submission/hindi_channel_interpretation_summary_2026-04-28.md",
    "research/submission/mechanistic_crossover_summary_2026-04-28.md",
    "research/submission/cross_model_behavioral_synthesis_2026-04-28.md",
]

# Selected raw-ish JSON artifacts that make the reproduction map useful without
# uploading every exploratory sweep or local log.
REQUIRED_RESULT_FILES = [
    "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
    "research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json",
    "research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json",
    "research/results/autoresearch/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_causal_patch_panel.json",
    "research/results/autoresearch/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json",
    "research/results/autoresearch/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json",
    "research/results/autoresearch/telugu_mlp_channel_crossover_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_1b_mlp_channel_crossover.json",
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

FORBIDDEN_TEXT_PATTERNS = [
    re.compile(r"academic" + r"techie2022", re.IGNORECASE),
    re.compile(r"/mnt/[cd]/", re.IGNORECASE),
    re.compile(r"/home/test", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9_\-]{20,}"),
    re.compile(r"github\.com/[A-Za-z0-9_.-]+/", re.IGNORECASE),
]


def _git_tracked() -> set[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return {line.strip() for line in output.splitlines() if line.strip()}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _audit_file_presence_and_tracking(errors: list[str], warnings: list[str]) -> None:
    tracked = _git_tracked()
    required = REQUIRED_PUBLIC_FILES + REQUIRED_RESULT_FILES
    for rel in required:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing_required_file:{rel}")
        elif rel not in tracked:
            errors.append(f"required_file_not_tracked:{rel}")
    for rel in REQUIRED_RESULT_FILES:
        path = ROOT / rel
        if path.exists():
            try:
                _read_json(path)
            except Exception as exc:  # pragma: no cover - diagnostics only
                errors.append(f"invalid_json:{rel}:{exc}")
    for rel in REQUIRED_PUBLIC_FILES:
        if rel.startswith("paper/") and (ROOT / rel).exists() and rel not in tracked:
            errors.append(f"paper_file_not_tracked:{rel}")
    if "api.txt" in tracked:
        errors.append("secret_file_tracked:api.txt")
    if "autoresearch.jsonl" in tracked:
        warnings.append("autoresearch_log_tracked")


def _audit_pdf(errors: list[str], warnings: list[str]) -> None:
    pdf_path = ROOT / "paper/icml2026/submission.pdf"
    tex_path = ROOT / "paper/icml2026/submission.tex"
    if not pdf_path.exists():
        errors.append("missing_pdf:paper/icml2026/submission.pdf")
        return
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()
    if title != EXPECTED_TITLE:
        errors.append(f"pdf_bad_title:{title!r}")
    if author != "Anonymous Authors":
        errors.append(f"pdf_bad_author:{author!r}")
    for key in ("creationDate", "modDate"):
        value = (meta.get(key) or "").strip()
        if value:
            warnings.append(f"pdf_metadata_has_{key}")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if "Anonymous Authors" not in text:
        errors.append("pdf_missing_anonymous_author_text")
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        if pattern.search(text):
            errors.append(f"pdf_forbidden_text:{pattern.pattern}")
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        if EXPECTED_TITLE not in tex:
            errors.append("tex_missing_expected_title")
        if "Copy, Retrieve, or Compose" in tex:
            errors.append("tex_contains_old_overbroad_title")
    if len(text) < 1000:
        warnings.append("pdf_text_extraction_short")


def _audit_archive_surface(errors: list[str], warnings: list[str]) -> None:
    # Use the index/cached file list rather than HEAD so this audit can run
    # before log_experiment creates the keep commit. After commit, the same list
    # is what `git archive HEAD` will expose.
    names = sorted(_git_tracked())
    for forbidden in FORBIDDEN_PATH_PARTS:
        matches = [name for name in names if forbidden in name]
        if matches:
            errors.append(f"archive_forbidden_path:{forbidden}:{matches[:3]}")
    for rel in REQUIRED_PUBLIC_FILES + REQUIRED_RESULT_FILES:
        if rel not in names:
            errors.append(f"archive_missing_required_file:{rel}")
    total_size = sum((ROOT / name).stat().st_size for name in names if (ROOT / name).exists())
    if total_size > 75 * 1024 * 1024:
        warnings.append(f"archive_large_bytes:{total_size}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    _audit_file_presence_and_tracking(errors, warnings)
    _audit_pdf(errors, warnings)
    _audit_archive_surface(errors, warnings)

    payload = {
        "final_submission_blockers": len(errors),
        "final_submission_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checked_required_files": len(REQUIRED_PUBLIC_FILES) + len(REQUIRED_RESULT_FILES),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"METRIC final_submission_blockers={len(errors)}")
    print(f"METRIC final_submission_warnings={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
