#!/usr/bin/env python3
from __future__ import annotations

"""Summarize matched-intervention crossover evidence for the submission.

The goal is claim calibration, not score chasing. The summary
includes positive, negative, and pending cells so the paper can say exactly which
edit-family comparisons have actually been run.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HINDI_SPARSE = ROOT / "research/results/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json"
TELUGU_RESIDUAL = ROOT / "research/results/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"
TELUGU_SPARSE = ROOT / "research/results/telugu_mlp_channel_crossover_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_1b_mlp_channel_crossover.json"
OUT_DIR = ROOT / "research/submission"
OUT_JSON = OUT_DIR / "mechanistic_crossover_summary_2026-04-28.json"
OUT_MD = OUT_DIR / "mechanistic_crossover_summary_2026-04-28.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["intervention"]): row for row in payload.get("summary_rows", [])}


def _ci(row: dict[str, Any], key: str) -> str:
    obj = row.get(key, {})
    return f"{float(obj.get('mean', float('nan'))):+.3f} [{float(obj.get('ci_low', float('nan'))):+.3f},{float(obj.get('ci_high', float('nan'))):+.3f}]"


def _mean(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, {}).get("mean", float("nan")))


def main() -> int:
    cells: list[dict[str, Any]] = []

    hindi = _read(HINDI_SPARSE)
    hmap = _summary_map(hindi)
    hbase = hmap["baseline_no_patch"]
    hchosen = hmap["chosen_mean_shift"]
    cells.append(
        {
            "language": "Hindi",
            "failure_stage": "early target entry",
            "edit_family": "fixed sparse MLP-channel shift",
            "artifact": str(HINDI_SPARSE.relative_to(ROOT)),
            "status": "works",
            "n_items": int(hchosen["n_items"]),
            "baseline_primary": _mean(hbase, "akshara_cer"),
            "patched_primary": _mean(hchosen, "akshara_cer"),
            "primary_metric": "full akshara CER (lower better)",
            "delta": _mean(hchosen, "delta_cer_improvement"),
            "delta_ci_text": _ci(hchosen, "delta_cer_improvement"),
            "note": "Held-out fixed two-channel shift improves full-word CER and first-entry metrics; sign flip harms in the source artifact.",
        }
    )

    tel_res = _read(TELUGU_RESIDUAL)
    trmap = _summary_map(tel_res)
    trbase = trmap["baseline_no_patch"]
    trchosen = trmap["chosen_mean_shift"]
    cells.append(
        {
            "language": "Telugu",
            "failure_stage": "late continuation",
            "edit_family": "shared-prefix full-residual mean shift",
            "artifact": str(TELUGU_RESIDUAL.relative_to(ROOT)),
            "status": "no rescue",
            "n_items": int(trchosen["n_items"]),
            "baseline_primary": _mean(trbase, "continuation_akshara_cer"),
            "patched_primary": _mean(trchosen, "continuation_akshara_cer"),
            "primary_metric": "continuation akshara CER (lower better)",
            "delta": _mean(trchosen, "delta_continuation_cer_improvement"),
            "delta_ci_text": _ci(trchosen, "delta_continuation_cer_improvement"),
            "note": "Oracle-conditioned static residual shift does not improve continuation generation under this tested edit family.",
        }
    )

    if TELUGU_SPARSE.exists():
        tel_sparse = _read(TELUGU_SPARSE)
        tsmap = _summary_map(tel_sparse)
        tsbase = tsmap["baseline_no_patch"]
        tschosen = tsmap["chosen_channel_shift"]
        delta = _mean(tschosen, "delta_continuation_cer_improvement")
        cells.append(
            {
                "language": "Telugu",
                "failure_stage": "late continuation",
                "edit_family": "fixed sparse MLP-channel shift",
                "artifact": str(TELUGU_SPARSE.relative_to(ROOT)),
                "status": "no rescue" if delta <= 0.0 else "weak/mixed",
                "n_items": int(tschosen["n_items"]),
                "baseline_primary": _mean(tsbase, "continuation_akshara_cer"),
                "patched_primary": _mean(tschosen, "continuation_akshara_cer"),
                "primary_metric": "continuation akshara CER (lower better)",
                "delta": delta,
                "delta_ci_text": _ci(tschosen, "delta_continuation_cer_improvement"),
                "selected": tel_sparse.get("selected"),
                "random_aggregate": tel_sparse.get("random_aggregate"),
                "note": "The selected Hindi-style compact channel shift does not rescue Telugu continuation; the selected split gain was near zero and held-out continuation CER slightly worsened.",
            }
        )
    else:
        cells.append(
            {
                "language": "Telugu",
                "failure_stage": "late continuation",
                "edit_family": "fixed sparse MLP-channel shift",
                "artifact": str(TELUGU_SPARSE.relative_to(ROOT)),
                "status": "pending",
                "note": "Modal detached run not fetched yet.",
            }
        )

    payload = {
        "claim_status": "bounded crossover evidence",
        "cells": cells,
        "interpretation": (
            "The crossover matrix is meant to reduce the apples-to-oranges objection. "
            "It does not prove a universal early-vs-late law; it records which edit families rescue or fail under held-out evaluation."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Mechanistic crossover summary (2026-04-28)\n\n")
    lines.append(payload["interpretation"] + "\n\n")
    lines.append("| language | stage | edit family | n | status | primary before→after | delta |\n")
    lines.append("|---|---|---|---:|---|---:|---:|\n")
    for cell in cells:
        if cell["status"] == "pending":
            lines.append(
                f"| {cell['language']} | {cell['failure_stage']} | {cell['edit_family']} | — | pending | — | — |\n"
            )
            continue
        lines.append(
            f"| {cell['language']} | {cell['failure_stage']} | {cell['edit_family']} | {cell['n_items']} | {cell['status']} | "
            f"{cell['baseline_primary']:.3f}→{cell['patched_primary']:.3f} | {cell['delta_ci_text']} |\n"
        )
    lines.append("\nNotes:\n")
    for cell in cells:
        lines.append(f"- {cell['language']} / {cell['edit_family']}: {cell['note']}\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
