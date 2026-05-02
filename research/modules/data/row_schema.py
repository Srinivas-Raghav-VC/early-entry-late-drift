from __future__ import annotations

from typing import Any

from research.modules.eval.metrics import normalize_text

TARGET_FIELD_ALIASES: tuple[str, ...] = ("target", "hindi")
SOURCE_FIELD = "ood"
ID_FIELD = "english"


def get_row_id(row: dict[str, Any]) -> str:
    return normalize_text(str(row.get(ID_FIELD, "")))


def get_source_text(row: dict[str, Any]) -> str:
    return normalize_text(str(row.get(SOURCE_FIELD, "")))


def get_target_text(row: dict[str, Any]) -> str:
    for key in TARGET_FIELD_ALIASES:
        value = row.get(key)
        if value is None:
            continue
        text = normalize_text(str(value))
        if text:
            return text
    return ""


def with_target_text(
    row: dict[str, Any],
    target: str,
    *,
    keep_legacy_alias: bool = True,
) -> dict[str, str]:
    out = {k: str(v) for k, v in row.items()}
    target_norm = normalize_text(target)
    out["target"] = target_norm
    if keep_legacy_alias:
        out["hindi"] = target_norm
    elif "hindi" in out:
        del out["hindi"]
    return out


def canonicalize_row(
    *,
    english: str,
    ood: str,
    target: str,
    keep_legacy_alias: bool = True,
) -> dict[str, str]:
    row: dict[str, str] = {
        "english": normalize_text(english),
        "ood": normalize_text(ood),
        "target": normalize_text(target),
    }
    if keep_legacy_alias:
        row["hindi"] = row["target"]
    return row
