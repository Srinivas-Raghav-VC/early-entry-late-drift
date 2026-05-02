from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.modules.data.row_schema import get_target_text

from .metrics import exact_match, normalize_text, script_valid

LEGACY_REQUIRED_KEYS = {
    "pair",
    "source_language",
    "input_script_name",
    "output_script_name",
    "icl_examples",
    "eval_rows",
}
PHASE0A_REQUIRED_KEYS = {
    "pair",
    "source_language",
    "input_script_name",
    "output_script_name",
    "eval_rows",
    "icl_presets",
}
REQUIRED_CORE_ROW_KEYS = {"english", "ood"}


def load_snapshot(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot must be a JSON object: {path}")
    return payload


def _row_id(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(row.get("english", "")),
        normalize_text(row.get("ood", "")),
        get_target_text(row),
    )


def _snapshot_mode(snapshot: dict[str, Any]) -> str:
    if "icl_examples" in snapshot:
        return "legacy"
    if "candidate_pool" in snapshot and "icl_presets" in snapshot:
        return "phase0a"
    return "unknown"


def _split_map(snapshot: dict[str, Any]) -> dict[str, list[Any]]:
    mode = _snapshot_mode(snapshot)
    if mode == "legacy":
        return {
            "icl_examples": list(snapshot.get("icl_examples", [])),
            "eval_rows": list(snapshot.get("eval_rows", [])),
        }

    if mode == "phase0a":
        bank = snapshot.get("icl_bank", snapshot.get("candidate_pool", []))
        splits: dict[str, list[Any]] = {
            "icl_bank": list(bank),
            "eval_rows": list(snapshot.get("eval_rows", [])),
        }
        presets = snapshot.get("icl_presets", {})
        if isinstance(presets, dict):
            for key, rows in presets.items():
                splits[f"icl_presets[{key}]"] = list(rows if isinstance(rows, list) else [])
        return splits

    return {}


def validate_snapshot_schema(snapshot: dict[str, Any]) -> dict[str, Any]:
    mode = _snapshot_mode(snapshot)
    if mode == "legacy":
        missing_keys = sorted(LEGACY_REQUIRED_KEYS - set(snapshot.keys()))
    elif mode == "phase0a":
        missing_keys = sorted(PHASE0A_REQUIRED_KEYS - set(snapshot.keys()))
        if "candidate_pool" not in snapshot and "icl_bank" not in snapshot:
            missing_keys.append("candidate_pool_or_icl_bank")
    else:
        missing_keys = ["unknown_snapshot_schema"]

    row_errors: list[str] = []
    for split_name, rows in _split_map(snapshot).items():
        if not isinstance(rows, list):
            row_errors.append(f"{split_name} is not a list")
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                row_errors.append(f"{split_name}[{idx}] is not an object")
                continue
            missing_row = REQUIRED_CORE_ROW_KEYS - set(row.keys())
            if missing_row:
                row_errors.append(
                    f"{split_name}[{idx}] missing keys: {sorted(missing_row)}"
                )
            if not get_target_text(row):
                row_errors.append(
                    f"{split_name}[{idx}] missing non-empty target (accepted aliases: target/hindi)"
                )

    # Additional integrity checks for phase0a snapshots.
    if mode == "phase0a":
        candidate_source = snapshot.get("icl_bank", snapshot.get("candidate_pool", []))
        candidate_rows = [r for r in candidate_source if isinstance(r, dict)]
        candidate_ids = {_row_id(r) for r in candidate_rows}
        icl_presets = snapshot.get("icl_presets", {})
        if not isinstance(icl_presets, dict):
            row_errors.append("icl_presets must be a dict")
        else:
            for k, rows in icl_presets.items():
                if not isinstance(rows, list):
                    row_errors.append(f"icl_presets[{k}] is not a list")
                    continue
                try:
                    expected_n = int(k)
                except ValueError:
                    expected_n = -1
                if expected_n >= 0 and len(rows) != expected_n:
                    row_errors.append(
                        f"icl_presets[{k}] length {len(rows)} != expected {expected_n}"
                    )
                for idx, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    if _row_id(row) not in candidate_ids:
                        row_errors.append(
                            f"icl_presets[{k}][{idx}] not found in candidate_pool"
                        )

    return {
        "ok": (not missing_keys) and (not row_errors),
        "mode": mode,
        "missing_keys": missing_keys,
        "row_errors": row_errors,
    }


def validate_snapshot_disjointness(snapshot: dict[str, Any]) -> dict[str, Any]:
    mode = _snapshot_mode(snapshot)
    if mode == "legacy":
        icl_rows = [r for r in snapshot.get("icl_examples", []) if isinstance(r, dict)]
    elif mode == "phase0a":
        candidate_source = snapshot.get("icl_bank", snapshot.get("candidate_pool", []))
        icl_rows = [r for r in candidate_source if isinstance(r, dict)]
    else:
        icl_rows = []

    eval_rows = [r for r in snapshot.get("eval_rows", []) if isinstance(r, dict)]

    icl_ids = {_row_id(r) for r in icl_rows}
    eval_ids = {_row_id(r) for r in eval_rows}
    overlap = sorted(icl_ids.intersection(eval_ids))

    return {
        "ok": len(overlap) == 0,
        "mode": mode,
        "icl_count": len(icl_rows),
        "eval_count": len(eval_rows),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:5],
    }


def validate_snapshot_script(snapshot: dict[str, Any]) -> dict[str, Any]:
    target_script = str(snapshot.get("output_script_name", "")).strip()
    issues: list[dict[str, Any]] = []
    for split_name, rows in _split_map(snapshot).items():
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            tgt = get_target_text(row)
            if script_valid(tgt, target_script) < 1.0:
                issues.append(
                    {
                        "split": split_name,
                        "index": idx,
                        "english": row.get("english", ""),
                        "ood": row.get("ood", ""),
                        "target": tgt,
                    }
                )

    return {
        "ok": len(issues) == 0,
        "target_script": target_script,
        "invalid_count": len(issues),
        "invalid_examples": issues[:10],
    }


def validate_snapshot(path: Path | str) -> dict[str, Any]:
    payload = load_snapshot(path)
    schema = validate_snapshot_schema(payload)
    disjoint = validate_snapshot_disjointness(payload)
    script = validate_snapshot_script(payload)
    return {
        "path": str(path),
        "pair": payload.get("pair", ""),
        "schema": schema,
        "disjointness": disjoint,
        "script_validity": script,
        "ok": schema["ok"] and disjoint["ok"] and script["ok"],
    }


def deterministic_metric_sanity_cases() -> list[dict[str, str]]:
    # Small fixed sanity set intentionally spans all current Phase 0A scripts.
    return [
        {
            "case_id": "dev_exact_hin",
            "language": "Hindi",
            "script": "Devanagari",
            "source": "namaste",
            "reference": "नमस्ते",
            "output": "नमस्ते",
            "expected_exact": "1",
            "expected_script_valid": "1",
        },
        {
            "case_id": "dev_wrong_script_hin",
            "language": "Hindi",
            "script": "Devanagari",
            "source": "namaste",
            "reference": "नमस्ते",
            "output": "namaste",
            "expected_exact": "0",
            "expected_script_valid": "0",
        },
        {
            "case_id": "dev_exact_mar",
            "language": "Marathi",
            "script": "Devanagari",
            "source": "aai",
            "reference": "आई",
            "output": "आई",
            "expected_exact": "1",
            "expected_script_valid": "1",
        },
        {
            "case_id": "tel_exact",
            "language": "Telugu",
            "script": "Telugu",
            "source": "amma",
            "reference": "అమ్మ",
            "output": "అమ్మ",
            "expected_exact": "1",
            "expected_script_valid": "1",
        },
        {
            "case_id": "tel_wrong_word",
            "language": "Telugu",
            "script": "Telugu",
            "source": "amma",
            "reference": "అమ్మ",
            "output": "నాన్న",
            "expected_exact": "0",
            "expected_script_valid": "1",
        },
        {
            "case_id": "ben_exact",
            "language": "Bengali",
            "script": "Bengali",
            "source": "namaskar",
            "reference": "নমস্কার",
            "output": "নমস্কার",
            "expected_exact": "1",
            "expected_script_valid": "1",
        },
        {
            "case_id": "ben_wrong_script",
            "language": "Bengali",
            "script": "Bengali",
            "source": "namaskar",
            "reference": "নমস্কার",
            "output": "नमस्कार",
            "expected_exact": "0",
            "expected_script_valid": "0",
        },
        {
            "case_id": "tam_exact",
            "language": "Tamil",
            "script": "Tamil",
            "source": "amma",
            "reference": "அம்மா",
            "output": "அம்மா",
            "expected_exact": "1",
            "expected_script_valid": "1",
        },
        {
            "case_id": "tam_wrong_script",
            "language": "Tamil",
            "script": "Tamil",
            "source": "amma",
            "reference": "அம்மா",
            "output": "అమ్మా",
            "expected_exact": "0",
            "expected_script_valid": "0",
        },
    ]


def run_deterministic_metric_sanity() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in deterministic_metric_sanity_cases():
        expected_exact = float(case["expected_exact"])
        expected_script = float(case["expected_script_valid"])
        script_name = normalize_text(case.get("script", "")) or (
            "Telugu" if case["language"] == "Telugu" else "Devanagari"
        )
        got_exact = exact_match(case["output"], case["reference"])
        got_script = script_valid(case["output"], script_name)
        rows.append(
            {
                "case_id": case["case_id"],
                "got_exact": got_exact,
                "expected_exact": expected_exact,
                "got_script_valid": got_script,
                "expected_script_valid": expected_script,
                "agree": int(got_exact == expected_exact and got_script == expected_script),
            }
        )

    n = len(rows)
    agreed = sum(r["agree"] for r in rows)
    return {
        "n_cases": n,
        "accuracy": float(agreed / n) if n else 0.0,
        "rows": rows,
        "ok": n > 0 and agreed == n,
    }
