#!/usr/bin/env python3
from __future__ import annotations

"""Build the clean anonymous paper package under paper/.

The historical working manuscript lives in a local directory with spaces and
many old versions. This script mirrors only the final anonymous source, needed
style files, and needed figures into `paper/`, builds the public PDF, and strips
nonessential PDF metadata. It is intentionally narrow: it packages existing
artifacts, it does not alter experiment results.
"""

import shutil
import subprocess
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Paper Template and Paper" / "Paper"
LEGACY_TEX = SOURCE_DIR / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
PUBLIC_DIR = ROOT / "paper"
PUBLIC_TEX_DIR = PUBLIC_DIR / "icml2026"
PUBLIC_FIG_DIR = PUBLIC_DIR / "figures"
PUBLIC_TEX = PUBLIC_TEX_DIR / "submission.tex"
PUBLIC_PDF = PUBLIC_TEX_DIR / "submission.pdf"
TITLE = "Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL"

STYLE_FILES = [
    "icml2026.sty",
    "icml2026.bst",
    "algorithm.sty",
    "algorithmic.sty",
    "fancyhdr.sty",
]

FIGURE_FILES = [
    "fig_stage_intervention_overview_tikz.tex",
    "fig_stage_axis_map_tikz.tex",
    "fig_behavioral_regime_summary_tikz_v19.tex",
    "fig_hindi_practical_patch_tikz_v18.tex",
]


def copy_public_sources() -> None:
    PUBLIC_TEX_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_TEX, PUBLIC_TEX)
    for name in STYLE_FILES:
        shutil.copy2(SOURCE_DIR / "icml2026" / name, PUBLIC_TEX_DIR / name)
    for name in FIGURE_FILES:
        shutil.copy2(SOURCE_DIR / "figures" / name, PUBLIC_FIG_DIR / name)


def build_pdf() -> None:
    subprocess.check_call(["tectonic", "submission.tex"], cwd=PUBLIC_TEX_DIR)


def sanitize_pdf_metadata() -> None:
    if not PUBLIC_PDF.exists():
        raise FileNotFoundError(PUBLIC_PDF)
    doc = fitz.open(PUBLIC_PDF)
    doc.set_metadata(
        {
            "title": TITLE,
            "author": "Anonymous Authors",
            "subject": "",
            "keywords": "mechanistic interpretability; multilingual in-context learning; transliteration",
            "creator": "LaTeX with hyperref",
            "producer": "",
            "creationDate": "",
            "modDate": "",
        }
    )
    tmp = PUBLIC_PDF.with_name("submission_sanitized.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(PUBLIC_PDF)


def main() -> int:
    copy_public_sources()
    build_pdf()
    sanitize_pdf_metadata()
    print(f"Wrote {PUBLIC_TEX.relative_to(ROOT)}")
    print(f"Wrote {PUBLIC_PDF.relative_to(ROOT)}")
    print("METRIC package_failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
