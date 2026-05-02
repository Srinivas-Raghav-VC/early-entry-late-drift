#!/usr/bin/env python3
from __future__ import annotations

"""Summarize retained non-Gemma behavioral checks for paper support.

This is intentionally a synthesis script, not a new benchmark. It keeps the
claim bounded: cross-model rows test whether the behavioral stage split is a
Gemma-only artifact, but they do not establish shared circuits.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research/results/autoresearch/cross_model_behavioral_v1"
OUT_DIR = ROOT / "research/submission"
OUT_JSON = OUT_DIR / "cross_model_behavioral_synthesis_2026-04-28.json"
OUT_MD = OUT_DIR / "cross_model_behavioral_synthesis_2026-04-28.md"

MODEL_ORDER = ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b", "llama3.2-3b"]
PAIR_LABEL = {
    "aksharantar_hin_latin": "Hindi",
    "aksharantar_tel_latin": "Telugu",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(model: str, pair: str, n_icl: int = 64) -> Path:
    return RESULTS / model / pair / "seed42" / f"nicl{n_icl}" / "cross_model_behavioral.json"


def _row(model: str, pair: str) -> dict[str, Any]:
    path = _path(model, pair)
    payload = _read(path)
    summary = payload["summary"]
    z = summary["zs"]
    h = summary["icl_helpful"]
    c = summary["icl_corrupt"]
    return {
        "model": model,
        "hf_id": payload.get("hf_id"),
        "pair": pair,
        "language": PAIR_LABEL[pair],
        "n": int(h["n_items"]),
        "helpful_first_entry": float(h["mean_first_entry_correct"]),
        "zs_first_entry": float(z["mean_first_entry_correct"]),
        "helpful_minus_zs_first_entry": float(h["mean_first_entry_correct"] - z["mean_first_entry_correct"]),
        "helpful_em": float(h["mean_exact_match"]),
        "corrupt_em": float(c["mean_exact_match"]),
        "helpful_minus_corrupt_em": float(h["mean_exact_match"] - c["mean_exact_match"]),
        "helpful_cer": float(h["mean_akshara_cer"]),
        "corrupt_cer": float(c["mean_akshara_cer"]),
        "helpful_cer_improvement_over_corrupt": float(c["mean_akshara_cer"] - h["mean_akshara_cer"]),
        "helpful_fuzzy_bank": float(h["fuzzy_bank_copy_rate"]),
        "corrupt_fuzzy_bank": float(c["fuzzy_bank_copy_rate"]),
        "artifact": str(path.relative_to(ROOT)),
    }


def main() -> int:
    rows = [_row(model, pair) for model in MODEL_ORDER for pair in PAIR_LABEL]
    strong_rows = [r for r in rows if r["helpful_first_entry"] >= 0.9]
    telugu_rows = [r for r in rows if r["pair"] == "aksharantar_tel_latin"]
    payload = {
        "source_root": str(RESULTS.relative_to(ROOT)),
        "scope": "single seed-42, 200-item, 64-shot non-Gemma behavioral checks",
        "rows": rows,
        "aggregate": {
            "models": MODEL_ORDER,
            "n_rows": len(rows),
            "strong_first_entry_rows": len(strong_rows),
            "telugu_rows_with_fuzzy_bank": sum(1 for r in telugu_rows if r["helpful_fuzzy_bank"] > 0),
            "mean_helpful_minus_corrupt_em": sum(r["helpful_minus_corrupt_em"] for r in rows) / len(rows),
            "mean_helpful_cer_improvement_over_corrupt": sum(r["helpful_cer_improvement_over_corrupt"] for r in rows) / len(rows),
        },
        "claim_status": {
            "established": [
                "Several non-Gemma instruction models show large helpful-prompt improvements in first-entry on Hindi/Telugu.",
                "Llama 3.2 1B is a capability-floor counterexample rather than a replication of the Gemma 1B split.",
            ],
            "supported_but_provisional": [
                "The early-entry vs continuation-difficulty decomposition is not obviously Gemma-only.",
                "Telugu remains harder end-to-end than first-entry accuracy alone suggests in stronger non-Gemma models.",
            ],
            "not_claimed": [
                "No shared circuit or shared channel mechanism across model families.",
                "No multi-seed cross-family estimate; these are single-seed robustness checks.",
            ],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Cross-model behavioral synthesis (2026-04-28)\n\n")
    lines.append(f"Scope: {payload['scope']}. These checks are behavioral robustness evidence, not mechanistic sharing evidence.\n\n")
    lines.append("## Summary table\n\n")
    lines.append("| model | lang | first-entry zs→help | EM help−corrupt | CER corrupt−help | fuzzy bank help |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['language']} | {r['zs_first_entry']:.3f}→{r['helpful_first_entry']:.3f} | "
            f"{r['helpful_minus_corrupt_em']:+.3f} | {r['helpful_cer_improvement_over_corrupt']:+.3f} | {r['helpful_fuzzy_bank']:.3f} |\n"
        )
    lines.append("\n## Interpretation\n\n")
    lines.append("The non-Gemma rows support the behavioral decomposition but also bound it. Qwen 2.5 1.5B/3B and Llama 3.2 3B mostly solve target entry under helpful prompting, while Telugu remains materially harder end-to-end and retains nonzero fuzzy bank-copy. Llama 3.2 1B is a useful counterexample: it fails both languages, so the Gemma 1B Hindi/Telugu split should not be described as universal.\n\n")
    lines.append("## Claim boundaries\n\n")
    for key, vals in payload["claim_status"].items():
        lines.append(f"### {key.replace('_', ' ').title()}\n")
        for val in vals:
            lines.append(f"- {val}\n")
        lines.append("\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
