#!/usr/bin/env python3
from __future__ import annotations

"""Audit whether cross-model behavioral evidence is synthesized and bounded."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "research/submission/cross_model_behavioral_synthesis_2026-04-28.md"
SUMMARY_JSON = ROOT / "research/submission/cross_model_behavioral_synthesis_2026-04-28.json"
README = ROOT / "README_REPRODUCE.md"
PAPER = ROOT / "Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex"
RESULTS = ROOT / "research/results/autoresearch/cross_model_behavioral_v1"
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b", "llama3.2-3b"]
PAIRS = ["aksharantar_hin_latin", "aksharantar_tel_latin"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def main() -> int:
    risks: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def check(key: str, ok: bool, message: str) -> None:
        row = {"key": key, "ok": bool(ok), "message": message}
        checks.append(row)
        if not ok:
            risks.append(row)

    missing = []
    for model in MODELS:
        for pair in PAIRS:
            path = RESULTS / model / pair / "seed42/nicl64/cross_model_behavioral.json"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
    check("artifact_coverage", not missing, f"Expected 4-model x 2-language n=200 artifacts. Missing: {missing}")

    summary_text = _read(SUMMARY)
    check("bounded_summary_present", SUMMARY.exists() and "not mechanistic sharing evidence" in summary_text, "Cross-model synthesis should exist and explicitly bound claims.")
    check("counterexample_noted", "Llama 3.2 1B" in summary_text and "counterexample" in summary_text, "Synthesis should mention the Llama 3.2 1B counterexample.")
    check("repro_mentions_cross_model", "cross_model_behavioral_synthesis" in _read(README), "README_REPRODUCE should point to cross-model synthesis/regeneration.")
    paper = _read(PAPER)
    check("paper_bounds_cross_family", "bounded behavioral support" in paper and "shared circuits" in paper, "Paper should bound cross-family checks as behavioral, not shared-circuit, evidence.")

    if SUMMARY_JSON.exists():
        payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        check("summary_json_has_aggregate", "mean_helpful_minus_corrupt_em" in aggregate, "Summary JSON should include aggregate cross-model metrics.")
    else:
        check("summary_json_has_aggregate", False, "Summary JSON missing.")

    payload = {
        "cross_model_synthesis_risks": len(risks),
        "checks": checks,
        "risks": risks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"METRIC cross_model_synthesis_risks={len(risks)}")
    print(f"METRIC passed_cross_model_checks={len(checks) - len(risks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
