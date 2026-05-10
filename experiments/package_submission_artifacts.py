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

CODE_URL = "https://anonymous.4open.science/r/early-entry-late-drift-7EBB/README.md"
CODE_PARAGRAPH = rf"""\paragraph{{Code.}} Anonymous code and reproducibility materials are
available via the clickable \href{{{CODE_URL}}}{{\texttt{{anonymous repository}}}}. The repository
contains experiment scripts, retained result artifacts, paper source, and
deterministic reproduction checks; full raw GPU reruns require gated model access
and A100-class compute."""

OLD_CODE_PARAGRAPH = rf"""\paragraph{{Code.}} Anonymous reproducibility materials are available at
\url{{{CODE_URL}}}.
The repository contains experiment scripts, retained result artifacts, paper source,
and deterministic reproduction checks; full raw GPU reruns require gated model
access and A100-class compute."""

CUSTOM_SPACING_BLOCK = r"""% Tighten float and display spacing so the dense workshop paper does not
% strand large blank bands around figures, tables, and short equations.
\AtBeginDocument{%
  \setlength{\textfloatsep}{8pt plus 2pt minus 2pt}%
  \setlength{\floatsep}{7pt plus 2pt minus 2pt}%
  \setlength{\intextsep}{7pt plus 2pt minus 2pt}%
  \setlength{\abovecaptionskip}{4pt plus 1pt minus 1pt}%
  \setlength{\belowcaptionskip}{3pt plus 1pt minus 1pt}%
  \setlength{\abovedisplayskip}{5pt plus 2pt minus 2pt}%
  \setlength{\belowdisplayskip}{5pt plus 2pt minus 2pt}%
  \setlength{\abovedisplayshortskip}{3pt plus 1pt minus 1pt}%
  \setlength{\belowdisplayshortskip}{4pt plus 1pt minus 1pt}%
}

"""

APPENDIX_SPACING_BLOCK = r"""\onecolumn
\setlength{\intextsep}{5pt plus 1pt minus 1pt}
\setlength{\abovecaptionskip}{4pt plus 1pt minus 1pt}
\setlength{\belowcaptionskip}{3pt plus 1pt minus 1pt}
"""


def make_template_strict(text: str) -> str:
    """Remove spacing hacks and keep the appendix in the ICML two-column format."""
    text = text.replace(CUSTOM_SPACING_BLOCK, "")
    text = text.replace(APPENDIX_SPACING_BLOCK, "")
    text = text.replace("\n\\enlargethispage{6\\baselineskip}\n", "\n")
    if "\\appendix" in text:
        head, appendix = text.split("\\appendix", 1)
        appendix = appendix.replace("\\begin{table}[H]", "\\begin{table*}[t]")
        appendix = appendix.replace("\\end{table}", "\\end{table*}")
        text = head + "\\appendix" + appendix
    return text


def add_code_link(text: str) -> str:
    """Insert the anonymous code link once, using compact clickable text."""
    if OLD_CODE_PARAGRAPH in text:
        return text.replace(OLD_CODE_PARAGRAPH, CODE_PARAGRAPH, 1)
    if CODE_URL in text:
        return text
    marker = "\\end{enumerate}\n\n\\begin{figure*}[t]"
    if marker not in text:
        raise ValueError("Could not find contribution-list marker for code-link insertion.")
    return text.replace(marker, f"\\end{{enumerate}}\n\n{CODE_PARAGRAPH}\n\n\\begin{{figure*}}[t]", 1)


def copy_public_sources() -> None:
    PUBLIC_TEX_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    source = LEGACY_TEX.read_text(encoding="utf-8")
    PUBLIC_TEX.write_text(add_code_link(make_template_strict(source)), encoding="utf-8")
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
