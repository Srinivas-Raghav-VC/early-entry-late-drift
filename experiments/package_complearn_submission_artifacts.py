#!/usr/bin/env python3
from __future__ import annotations

"""Build the anonymous CompLearn workshop variant under paper/complearn2026/.

The CompLearn variant keeps the same results and claim boundaries as the main
ICML-format submission, but foregrounds compositional transliteration, prompt-bank
attraction, and stage-specific generalization. It is generated from the final
working manuscript so figure/style synchronization remains repeatable.
"""

import shutil
import subprocess
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Paper Template and Paper" / "Paper"
LEGACY_TEX = SOURCE_DIR / "icml2026" / "gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
PUBLIC_DIR = ROOT / "paper"
PUBLIC_TEX_DIR = PUBLIC_DIR / "complearn2026"
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

ABSTRACT_MAIN = r"""In-context learning (ICL) failures are often reported as a single accuracy drop, but Latin-to-Indic transliteration makes two failure stages visible: early target-entry failure and late prompt-bank continuation drift. We operationalize this split with first-akshara accuracy, continuation-tail CER, bank-copy diagnostics, and activation patching across five languages, then ask whether different stages expose different internal-state edit affordances. In \texttt{1B} Hindi, high-shot prompts often lose the first target token to Latin/source-like competitors; prompt-final patching identifies the strongest tested rescue at an \texttt{L25} MLP state, and a fixed two-channel shift gives a held-out rescue. In Telugu, a deliberately favorable shared-prefix oracle diagnostic identifies high-layer residual sites, yet the tested full-state mean-shift and Hindi-style compact-channel edits do not rescue continuation. Together, these results give a regime map plus one compact editable early-stage handle and one informative negative continuation-stage case; the Hindi edit reduces held-out CER from \textbf{0.827 to 0.703} ($n{=}200$), but we do not claim matched interventions or complete circuits."""

ABSTRACT_COMPLEARN = r"""Compositional in-context learning (ICL) should build query-specific outputs from reusable exemplars, but aggregate accuracy hides where construction fails. Latin-to-Indic transliteration makes two stages visible: early target-entry failure and late prompt-bank continuation drift. We measure this split with first-akshara accuracy, continuation-tail CER, bank-copy diagnostics, and activation patching across five languages. In \texttt{1B} Hindi, high-shot prompts often lose the first target token to Latin/source-like competitors; prompt-final patching identifies the strongest tested rescue at an \texttt{L25} MLP state, and a fixed two-channel shift reduces held-out CER from \textbf{0.827 to 0.703} ($n{=}200$). In Telugu, a deliberately favorable shared-prefix oracle diagnostic identifies high-layer residual sites, yet the tested full-state mean-shift and Hindi-style compact-channel edits do not rescue continuation. The result is a stage-resolved account of compositional transliteration ICL: one compact editable early-stage handle, one informative negative continuation-stage case, and no claim of matched interventions or complete circuits."""

REPLACEMENTS = {
    r"\icmlkeywords{mechanistic interpretability, multilingual in-context learning, transliteration}":
        r"\icmlkeywords{compositional learning, mechanistic interpretability, multilingual in-context learning}",
    ABSTRACT_MAIN: ABSTRACT_COMPLEARN,
    "We use multilingual transliteration to separate these stages. Transliteration exposes failure modes that are easy to tell apart: outputting Latin instead of the target script, starting correctly and then drifting toward a prompt-bank continuation, or producing the correct native-script word. Throughout the paper, the early axis is operationalized by \\emph{first-akshara correctness}, not by a looser script-family check. This makes transliteration a useful setting for asking not only whether prompting helps, but which stage of the computation it helps or harms. The broader lesson is that ICL failure should not be treated as a scalar performance drop: different failure stages can expose different intervention affordances.":
        "We use multilingual transliteration as a compact compositional setting. A successful model must build a query-specific native-script sequence from reusable source-target exemplars, rather than merely copying a nearby prompt target or defaulting to the source script. The resulting failures are easy to tell apart: outputting Latin instead of the target script, starting correctly and then drifting toward a prompt-bank continuation, or producing the correct native-script word. Throughout the paper, the early axis is operationalized by \\emph{first-akshara correctness}, not by a looser script-family check. This makes transliteration useful for asking not only whether prompting helps, but which stage of the compositional computation it helps or harms.",
    "Our core empirical result is a behavioral regime map for 64-shot transliteration. The first axis is \\emph{early target start}; the second is \\emph{late query-specific continuation}. In Gemma~3 \\texttt{1B}, Hindi fails early: helpful demonstrations worsen first-step competition and push output toward Latin. Telugu fails later: the model usually begins with the correct first akshara but then drifts toward similar prompt-bank continuations. Marathi, a same-script but different-language probe sharing Devanagari with Hindi, improves much more at the first akshara than at the continuation stage, though we do not treat it as a script-only control. In the broader figure-level summary, \\texttt{4B} attenuates the clearest \\texttt{1B} failures.":
        "Our core empirical result is a behavioral regime map for 64-shot transliteration. The first axis is \\emph{early target start}; the second is \\emph{late query-specific continuation}. In Gemma~3 \\texttt{1B}, Hindi fails early: helpful demonstrations worsen first-step competition and push output toward Latin. Telugu fails later: the model usually begins with the correct first akshara but then drifts toward similar prompt-bank continuations, a failure of query-specific composition after a successful start. Marathi improves much more at the first akshara than at continuation; we use it as targeted diagnostic support rather than as a second causal case study. In the broader figure-level summary, \\texttt{4B} attenuates the clearest \\texttt{1B} failures.",
    r"\item A \textbf{behavioral regime map for this transliteration setting} showing that 64-shot ICL failures separate into early-start versus late-continuation regimes (\S\ref{sec:behavior}).":
        r"\item A \textbf{stage-resolved compositional transliteration evaluation} showing that 64-shot ICL failures separate into early-start versus late-continuation regimes (\S\ref{sec:behavior}).",
    r"\item \textbf{Causal diagnostics for two regimes}: Hindi points to a candidate compact near-output MLP handle, while Telugu points to broader high-layer residual handles, directly at the post-\texttt{L25} residual in \texttt{1B} and, in a bounded \texttt{4B} follow-up, most strongly at the post-\texttt{L33} residual (\S\ref{sec:mech}).":
        r"\item \textbf{Causal diagnostics for two compositional failure regimes}: Hindi points to a candidate compact near-output MLP handle, while Telugu points to broader high-layer residual handles, directly at the post-\texttt{L25} residual in \texttt{1B} and, in a bounded \texttt{4B} follow-up, most strongly at the post-\texttt{L33} residual (\S\ref{sec:mech}).",
    "We study Latin-to-native-script transliteration for five Indic languages: Hindi and Marathi (Devanagari), Telugu (Telugu script), Tamil (Tamil script), and Bengali (Bengali script), using Aksharantar \\citep{aksharantar}. Unless otherwise stated, prompts contain $k{=}64$ source-target demonstrations $((x_1,y_1),\\ldots,(x_k,y_k))$ followed by a held-out query $x_q$, and the model must generate the native-script target $y_q$. The query split is held out from the prompt bank. This task is useful because early target-start failure, late prompt-bank drift, and successful composition are easy to distinguish behaviorally.":
        "We study Latin-to-native-script transliteration for five Indic languages: Hindi and Marathi (Devanagari), Telugu (Telugu script), Tamil (Tamil script), and Bengali (Bengali script), using Aksharantar \\citep{aksharantar}. Unless otherwise stated, prompts contain $k{=}64$ source-target demonstrations $((x_1,y_1),\\ldots,(x_k,y_k))$ followed by a held-out query $x_q$, and the model must generate the native-script target $y_q$. The query split is held out from the prompt bank. This task is useful because a correct answer requires query-specific sequence construction, while early target-start failure, late prompt-bank drift, and successful composition are easy to distinguish behaviorally.",
    "\\textbf{Marathi}, our same-script different-language probe, improves much more at the first akshara than at continuation; we use it as targeted diagnostic support rather than as a script-only control or a second full causal case study.":
        "\\textbf{Marathi} improves more at the first akshara than at continuation; we use it as targeted diagnostic support rather than as a second causal case study.",
    "We build on two nearby lines of work. The first studies transliteration and romanization as tools for multilingual transfer \\citep{transliticl, romanlens, scriptbarrier}.":
        "We connect compositional learning with two nearby lines of mechanistic evidence. The first studies transliteration and romanization as tools for multilingual transfer \\citep{transliticl, romanlens, scriptbarrier}.",
    "This intervention has an important limitation: a small cross-task safety audit (Appendix~\\ref{app:safety}) suggests off-task degradation.":
        "A small cross-task safety audit (Appendix~\\ref{app:safety}) shows degradation outside transliteration.",
    "In this transliteration setting, multilingual high-shot ICL separates into at least two behaviorally legible failure regimes.":
        "In this compositional transliteration setting, multilingual high-shot ICL separates into at least two behaviorally legible failure regimes.",
    "Even with that limitation, the paper shows that transliteration is a useful setting for separating failure stage, localizing bounded causal handles, and asking when internal-state insight does or does not yield a useful intervention.":
        "Even with that limitation, the paper shows that compositional transliteration is a useful setting for separating failure stage, localizing bounded causal handles, and asking when internal-state insight does or does not yield a useful intervention.",
}


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


def transform_for_complearn(text: str) -> str:
    text = make_template_strict(text)
    for old, new in REPLACEMENTS.items():
        if old not in text:
            raise ValueError(f"Expected source text not found for replacement: {old[:120]!r}")
        text = text.replace(old, new, 1)
    return add_code_link(text)


def copy_public_sources() -> None:
    PUBLIC_TEX_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    source = LEGACY_TEX.read_text(encoding="utf-8")
    PUBLIC_TEX.write_text(transform_for_complearn(source), encoding="utf-8")
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
            "keywords": "compositional learning; mechanistic interpretability; multilingual in-context learning; transliteration",
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
    print("METRIC complearn_package_failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
