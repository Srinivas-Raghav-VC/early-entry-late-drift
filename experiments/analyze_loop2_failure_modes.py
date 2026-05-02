#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP2_ROOT = PROJECT_ROOT / "research" / "results" / "autoresearch" / "loop2_vm_controls"
DEFAULT_FULL_ROOT = LOOP2_ROOT / "loop2_full" / "raw"
DEFAULT_THRESHOLD_ROOT = LOOP2_ROOT / "threshold_1b_seed42" / "raw"
DEFAULT_MD_OUT = PROJECT_ROOT / "outputs" / "loop2_failure_modes_2026-03-29.md"
DEFAULT_JSON_OUT = PROJECT_ROOT / "outputs" / "loop2_failure_modes_2026-03-29.json"

CORE_CELLS: List[Tuple[str, Path, str, str, int]] = [
    ("core", DEFAULT_FULL_ROOT, "1b", "aksharantar_hin_latin", 8),
    ("core", DEFAULT_FULL_ROOT, "1b", "aksharantar_hin_latin", 64),
    ("core", DEFAULT_FULL_ROOT, "1b", "aksharantar_tel_latin", 8),
    ("core", DEFAULT_FULL_ROOT, "1b", "aksharantar_tel_latin", 64),
    ("core", DEFAULT_FULL_ROOT, "4b", "aksharantar_hin_latin", 64),
    ("core", DEFAULT_FULL_ROOT, "4b", "aksharantar_tel_latin", 64),
]

THRESHOLD_CELLS: List[Tuple[str, Path, str, str, int]] = [
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_hin_latin", 48),
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_hin_latin", 56),
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_hin_latin", 64),
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_tel_latin", 48),
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_tel_latin", 56),
    ("threshold", DEFAULT_THRESHOLD_ROOT, "1b", "aksharantar_tel_latin", 64),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value == value and value not in (float("inf"), -float("inf")):
            return value
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _is_latinish(text: str) -> bool:
    letters = [ch for ch in str(text) if ch.isalpha()]
    if not letters:
        return False
    return all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def _rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _f3(x: float) -> str:
    return f"{float(x):.3f}"


def _load_payload(root: Path, model: str, pair: str, n_icl: int) -> Dict[str, Any]:
    path = root / model / pair / f"nicl{n_icl}" / "neutral_filler_recency_controls.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_rows(payload: Dict[str, Any], condition: str) -> List[Dict[str, Any]]:
    return [row for row in payload.get("item_rows", []) if str(row.get("condition")) == str(condition)]


def _target_bank(payload: Dict[str, Any]) -> set[str]:
    meta_rows = payload.get("prompt_ordering_metadata_by_item") or []
    if not meta_rows:
        return set()
    desc_rows = meta_rows[0].get("helpful_similarity_desc") or []
    return {str(row.get("target", "")).strip() for row in desc_rows if str(row.get("target", "")).strip()}


def _copy_stats(rows: Iterable[Dict[str, Any]], bank_targets: set[str]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    source_copy = sum(1 for row in rows if str(row.get("prediction", "")).strip() == str(row.get("word_ood", "")).strip())
    latinish = sum(1 for row in rows if _is_latinish(str(row.get("prediction", "")).strip()))
    bank_copy = sum(1 for row in rows if str(row.get("prediction", "")).strip() in bank_targets)
    first_correct_wrong = [row for row in rows if float(row.get("first_entry_correct", 0.0)) == 1.0 and float(row.get("exact_match", 0.0)) == 0.0]
    first_correct_wrong_bank = sum(1 for row in first_correct_wrong if str(row.get("prediction", "")).strip() in bank_targets)
    return {
        "n_items": n,
        "source_copy_count": int(source_copy),
        "source_copy_rate": _rate(source_copy, n),
        "latinish_count": int(latinish),
        "latinish_rate": _rate(latinish, n),
        "bank_copy_count": int(bank_copy),
        "bank_copy_rate": _rate(bank_copy, n),
        "first_correct_but_wrong_count": int(len(first_correct_wrong)),
        "first_correct_but_wrong_rate": _rate(len(first_correct_wrong), n),
        "first_correct_but_wrong_bank_copy_count": int(first_correct_wrong_bank),
        "first_correct_but_wrong_bank_copy_rate": _rate(first_correct_wrong_bank, len(first_correct_wrong)),
        "examples": [
            {
                "word_ood": str(row.get("word_ood", "")),
                "gold": str(row.get("word_hindi", "")),
                "prediction": str(row.get("prediction", "")),
                "akshara_cer": float(row.get("akshara_cer", 0.0)),
                "continuation_fidelity": row.get("continuation_fidelity"),
                "bank_copy": bool(str(row.get("prediction", "")).strip() in bank_targets),
            }
            for row in first_correct_wrong[:8]
        ],
    }


def _cell_summary(run_type: str, root: Path, model: str, pair: str, n_icl: int) -> Dict[str, Any]:
    payload = _load_payload(root, model, pair, n_icl)
    summary = payload["summary_by_condition"]
    bank_targets = _target_bank(payload)
    helpful_rows = _condition_rows(payload, "icl_helpful")
    corrupt_rows = _condition_rows(payload, "icl_corrupt")
    zs = dict(summary["zs"])
    helpful = dict(summary["icl_helpful"])
    corrupt = dict(summary["icl_corrupt"])
    helpful_copy = _copy_stats(helpful_rows, bank_targets)
    corrupt_copy = _copy_stats(corrupt_rows, bank_targets)
    return {
        "run_type": run_type,
        "path": str(root / model / pair / f"nicl{n_icl}" / "neutral_filler_recency_controls.json"),
        "model": model,
        "pair": pair,
        "n_icl": int(n_icl),
        "zs": zs,
        "helpful": helpful,
        "corrupt": corrupt,
        "deltas": {
            "helpful_minus_zs_first_prob": float(helpful["mean_first_prob"] - zs["mean_first_prob"]),
            "helpful_minus_zs_first_entry_correct": float(helpful["mean_first_entry_correct"] - zs["mean_first_entry_correct"]),
            "helpful_minus_zs_exact_match": float(helpful["mean_exact_match"] - zs["mean_exact_match"]),
            "helpful_minus_zs_akshara_cer": float(helpful["mean_akshara_cer"] - zs["mean_akshara_cer"]),
            "helpful_minus_corrupt_first_prob": float(helpful["mean_first_prob"] - corrupt["mean_first_prob"]),
            "helpful_minus_corrupt_first_entry_correct": float(helpful["mean_first_entry_correct"] - corrupt["mean_first_entry_correct"]),
            "helpful_minus_corrupt_exact_match": float(helpful["mean_exact_match"] - corrupt["mean_exact_match"]),
            "helpful_minus_corrupt_akshara_cer": float(helpful["mean_akshara_cer"] - corrupt["mean_akshara_cer"]),
        },
        "helpful_copy_stats": helpful_copy,
        "corrupt_copy_stats": corrupt_copy,
    }


def _core_row_md(cell: Dict[str, Any]) -> str:
    return "| {model} | {pair} | {n_icl} | {hfprob} | {zfprob} | {dfprob} | {hfirst} | {zfirst} | {hexact} | {zexact} | {sourcecopy} | {bankcopy} |".format(
        model=cell["model"],
        pair=cell["pair"].replace("aksharantar_", "").replace("_latin", ""),
        n_icl=cell["n_icl"],
        hfprob=_f3(cell["helpful"]["mean_first_prob"]),
        zfprob=_f3(cell["zs"]["mean_first_prob"]),
        dfprob=_f3(cell["deltas"]["helpful_minus_zs_first_prob"]),
        hfirst=_f3(cell["helpful"]["mean_first_entry_correct"]),
        zfirst=_f3(cell["zs"]["mean_first_entry_correct"]),
        hexact=_f3(cell["helpful"]["mean_exact_match"]),
        zexact=_f3(cell["zs"]["mean_exact_match"]),
        sourcecopy=_pct(cell["helpful_copy_stats"]["source_copy_rate"]),
        bankcopy=_pct(cell["helpful_copy_stats"]["bank_copy_rate"]),
    )


def _threshold_row_md(cell: Dict[str, Any]) -> str:
    return "| {pair} | {n_icl} | {hfprob} | {zfprob} | {hfirst} | {zfirst} | {hexact} | {zexact} | {hcer} | {zcer} | {bankcopy} | {latinish} |".format(
        pair=cell["pair"].replace("aksharantar_", "").replace("_latin", ""),
        n_icl=cell["n_icl"],
        hfprob=_f3(cell["helpful"]["mean_first_prob"]),
        zfprob=_f3(cell["zs"]["mean_first_prob"]),
        hfirst=_f3(cell["helpful"]["mean_first_entry_correct"]),
        zfirst=_f3(cell["zs"]["mean_first_entry_correct"]),
        hexact=_f3(cell["helpful"]["mean_exact_match"]),
        zexact=_f3(cell["zs"]["mean_exact_match"]),
        hcer=_f3(cell["helpful"]["mean_akshara_cer"]),
        zcer=_f3(cell["zs"]["mean_akshara_cer"]),
        bankcopy=_pct(cell["helpful_copy_stats"]["bank_copy_rate"]),
        latinish=_pct(cell["helpful_copy_stats"]["latinish_rate"]),
    )


def _render_markdown(cells: List[Dict[str, Any]]) -> str:
    core_cells = [cell for cell in cells if cell["run_type"] == "core"]
    threshold_cells = [cell for cell in cells if cell["run_type"] == "threshold"]
    get = lambda m, p, n: next(cell for cell in cells if cell["model"] == m and cell["pair"] == p and cell["n_icl"] == n)

    c_1b_hin_64 = get("1b", "aksharantar_hin_latin", 64)
    c_1b_tel_64 = get("1b", "aksharantar_tel_latin", 64)
    c_4b_tel_64 = get("4b", "aksharantar_tel_latin", 64)
    c_4b_hin_64 = get("4b", "aksharantar_hin_latin", 64)

    lines: List[str] = []
    lines.append("# Loop 2 failure-mode follow-up (first token vs continuation)")
    lines.append("")
    lines.append("This memo decomposes the current Loop 2 behavior into two pieces using already-generated control artifacts:")
    lines.append("")
    lines.append("1. **early target selection** — `mean_first_prob`, `mean_first_entry_correct`")
    lines.append("2. **whole-word continuation** — `mean_exact_match`, `mean_akshara_cer`, plus copy-style error patterns")
    lines.append("")
    lines.append("## Key takeaways")
    lines.append("")
    lines.append(f"- **Established:** `1B × Hindi × n_icl=64` is already failing at the **first target token**. Helpful ICL is worse than zero-shot on first-token probability ({_f3(c_1b_hin_64['helpful']['mean_first_prob'])} vs {_f3(c_1b_hin_64['zs']['mean_first_prob'])}) and first-entry correctness ({_f3(c_1b_hin_64['helpful']['mean_first_entry_correct'])} vs {_f3(c_1b_hin_64['zs']['mean_first_entry_correct'])}).")
    lines.append(f"- **Established:** `1B × Telugu × n_icl=64` is **not** primarily a first-token failure. Helpful ICL drives first-token probability from {_f3(c_1b_tel_64['zs']['mean_first_prob'])} to {_f3(c_1b_tel_64['helpful']['mean_first_prob'])} and first-entry correctness from {_f3(c_1b_tel_64['zs']['mean_first_entry_correct'])} to {_f3(c_1b_tel_64['helpful']['mean_first_entry_correct'])}, yet exact match stays at {_f3(c_1b_tel_64['helpful']['mean_exact_match'])}.")
    lines.append(f"- **Established:** `1B × Telugu × n_icl=64` often emits an **ICL target string** rather than the query-specific answer: {_pct(c_1b_tel_64['helpful_copy_stats']['bank_copy_rate'])} of helpful predictions are exact copies of one of the prompt-bank targets, and {_pct(c_1b_tel_64['helpful_copy_stats']['first_correct_but_wrong_bank_copy_rate'])} of the `first-correct but exact-wrong` cases are bank copies.")
    lines.append(f"- **Established:** `1B × Hindi × n_icl=64` often falls back to **Latin/source-like outputs**: {_pct(c_1b_hin_64['helpful_copy_stats']['latinish_rate'])} of helpful predictions are Latin-script, and {_pct(c_1b_hin_64['helpful_copy_stats']['source_copy_rate'])} are exact source copies.")
    lines.append(f"- **Supported but provisional:** `4B × Telugu × n_icl=64` is a clean positive anchor because it gets both stages right: first-token probability rises to {_f3(c_4b_tel_64['helpful']['mean_first_prob'])}, first-entry correctness to {_f3(c_4b_tel_64['helpful']['mean_first_entry_correct'])}, and exact match to {_f3(c_4b_tel_64['helpful']['mean_exact_match'])}; its residual errors are mostly near-misses rather than prompt-bank copies ({_pct(c_4b_tel_64['helpful_copy_stats']['bank_copy_rate'])}).")
    lines.append(f"- **Supported but provisional:** `4B × Hindi × n_icl=64` is a strong-base-capability comparison cell, not a rescue-heavy cell: zero-shot is already strong (exact {_f3(c_4b_hin_64['zs']['mean_exact_match'])}), and helpful ICL mostly sharpens first-token confidence rather than creating a new regime.")
    lines.append("")
    lines.append("## Core panel table")
    lines.append("")
    lines.append("| model | pair | n_icl | helpful first_prob | zs first_prob | Δ first_prob | helpful first_entry | zs first_entry | helpful exact | zs exact | helpful source-copy | helpful bank-copy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in core_cells:
        lines.append(_core_row_md(cell))
    lines.append("")
    lines.append("## Threshold check on `1B` (`n_icl = 48/56/64`)")
    lines.append("")
    lines.append("| pair | n_icl | helpful first_prob | zs first_prob | helpful first_entry | zs first_entry | helpful exact | zs exact | helpful CER | zs CER | helpful bank-copy | helpful Latin-script |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in threshold_cells:
        lines.append(_threshold_row_md(cell))
    lines.append("")
    lines.append("## Failure-mode interpretation")
    lines.append("")
    lines.append("### 1B Hindi: early target-selection failure")
    lines.append("")
    lines.append(f"Observed at `n_icl=64`: helpful ICL is worse than zero-shot on first-token probability ({_f3(c_1b_hin_64['helpful']['mean_first_prob'])} vs {_f3(c_1b_hin_64['zs']['mean_first_prob'])}), worse on first-entry correctness ({_f3(c_1b_hin_64['helpful']['mean_first_entry_correct'])} vs {_f3(c_1b_hin_64['zs']['mean_first_entry_correct'])}), and worse on exact match ({_f3(c_1b_hin_64['helpful']['mean_exact_match'])} vs {_f3(c_1b_hin_64['zs']['mean_exact_match'])}).")
    lines.append("")
    lines.append(f"Helpful outputs are often not even in the right script: {_pct(c_1b_hin_64['helpful_copy_stats']['latinish_rate'])} Latin-script predictions, {_pct(c_1b_hin_64['helpful_copy_stats']['source_copy_rate'])} exact source copies.")
    lines.append("")
    lines.append("This points to a **routing / target-selection problem before whole-word continuation**, not just a later suffix-composition issue.")
    lines.append("")
    lines.append("Representative failures:")
    lines.append("")
    for ex in c_1b_hin_64["helpful_copy_stats"]["examples"][:5]:
        lines.append(f"- `{ex['word_ood']}` → gold `{ex['gold']}` but predicted `{ex['prediction']}` (bank copy: `{ex['bank_copy']}`)")
    lines.append("")
    lines.append("### 1B Telugu: continuation / retrieval-composition failure")
    lines.append("")
    lines.append(f"Observed at `n_icl=64`: helpful ICL greatly improves first-token probability ({_f3(c_1b_tel_64['zs']['mean_first_prob'])} → {_f3(c_1b_tel_64['helpful']['mean_first_prob'])}) and first-entry correctness ({_f3(c_1b_tel_64['zs']['mean_first_entry_correct'])} → {_f3(c_1b_tel_64['helpful']['mean_first_entry_correct'])}), but exact match stays at `{_f3(c_1b_tel_64['helpful']['mean_exact_match'])}`.")
    lines.append("")
    lines.append(f"The dominant error mode is prompt-bank copying: {_pct(c_1b_tel_64['helpful_copy_stats']['bank_copy_rate'])} of helpful predictions are exact copies of one of the 64 prompt-bank targets.")
    lines.append("")
    lines.append(f"Among the items where the model gets the first entry correct but still misses the whole word, {_pct(c_1b_tel_64['helpful_copy_stats']['first_correct_but_wrong_bank_copy_rate'])} are bank copies. This is much closer to **retrieving the wrong full answer from the prompt bank** than to failing to choose the target script.")
    lines.append("")
    lines.append("Representative failures:")
    lines.append("")
    for ex in c_1b_tel_64["helpful_copy_stats"]["examples"][:5]:
        lines.append(f"- `{ex['word_ood']}` → gold `{ex['gold']}` but predicted `{ex['prediction']}` (bank copy: `{ex['bank_copy']}`)")
    lines.append("")
    lines.append("### 4B Telugu: the clean anchor remains clean")
    lines.append("")
    lines.append(f"`4B × Telugu × n_icl=64` improves both early and late metrics. Helpful exact match reaches {_f3(c_4b_tel_64['helpful']['mean_exact_match'])}, while helpful bank-copy rate is only {_pct(c_4b_tel_64['helpful_copy_stats']['bank_copy_rate'])}. Residual errors are usually near-misses rather than verbatim prompt-bank retrieval.")
    lines.append("")
    lines.append("## Claim ledger")
    lines.append("")
    lines.append("- **Established:** `1B × Hindi` high-shot fragility includes an early target-selection failure; helpful ICL often pushes the model toward Latin/source-like or wrong-bank outputs rather than toward the correct Devanagari target.")
    lines.append("- **Established:** `1B × Telugu` high-shot fragility is mainly not a first-token problem; it is a continuation/retrieval-composition problem in which the model often emits a prompt-bank target string that matches the script and sometimes the first character, but not the query-specific transliteration.")
    lines.append("- **Supported but provisional:** the `1B` story is therefore **not unitary** across languages. Hindi and Telugu appear to fail at different stages of the computation.")
    lines.append("- **Supported but provisional:** visibility remains a contributing architectural factor, but these failure-mode differences show that visibility alone cannot explain the `1B` behavior.")
    lines.append("")
    lines.append("## Recommended next bounded experiments")
    lines.append("")
    lines.append("1. **1B Hindi — first-token competition audit.** Build a small teacher-forced probe that records the correct target token, the emitted top competitor token, and whether the competitor is Latin/source-like or a wrong Devanagari bank token. This directly tests the early-routing hypothesis.")
    lines.append("2. **1B Telugu — prompt-bank copy audit with nearest-neighbor controls.** Measure how often the model outputs an in-context target exactly, and whether those copies correspond to the most orthographically similar prompt example rather than the correct answer. This directly tests the retrieval-composition hypothesis.")
    lines.append("3. **Only after those two audits:** decide whether causal interventions should target early token selection (`1B Hindi`) or later query-specific continuation (`1B Telugu`).")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for cell in cells:
        lines.append(f"- file://{cell['path']}")
    lines.append("- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/score.json")
    lines.append("- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/score.json")
    lines.append("- file:///mnt/d/Research/Honors/research/results/autoresearch/token_visibility_v1/results/token_visibility_summary.csv")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Loop 2 control artifacts by decomposing first-token vs continuation failure modes.")
    ap.add_argument("--md-out", type=str, default=str(DEFAULT_MD_OUT))
    ap.add_argument("--json-out", type=str, default=str(DEFAULT_JSON_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cells = [_cell_summary(*cell) for cell in CORE_CELLS + THRESHOLD_CELLS]
    md = _render_markdown(cells)
    md_path = Path(args.md_out).resolve()
    json_path = Path(args.json_out).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    _write_json(json_path, {"cells": cells})
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
