from __future__ import annotations

import random
from typing import Any

from research.modules.data.row_schema import canonicalize_row
from research.modules.eval.metrics import normalize_text, script_valid

AKSHARANTAR_DATASET = "ai4bharat/Aksharantar"

CONFIG_CANDIDATES: dict[str, list[str]] = {
    "hin": ["hin", "hi", "hindi"],
    "tel": ["tel", "te", "telugu"],
    "ben": ["ben", "bn", "bengali"],
    "tam": ["tam", "ta", "tamil"],
    "mar": ["mar", "mr", "marathi"],
}


def _pick_field(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = normalize_text(str(value))
        if text:
            return text
    return ""


def _resolve_config(pair_code: str, configs: list[str]) -> str:
    candidates = CONFIG_CANDIDATES.get(pair_code, [pair_code])
    for cand in candidates:
        if cand in configs:
            return cand
    hits = sorted({cfg for cfg in configs if any(cand in cfg or cfg in cand for cand in candidates)})
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"Unable to resolve Aksharantar config for {pair_code}. candidates={candidates}, hits={hits[:10]}"
    )


def load_aksharantar_rows(
    *,
    pair_code: str,
    target_script: str,
    seed: int = 42,
    min_rows: int = 300,
) -> list[dict[str, str]]:
    """Load and normalize Aksharantar rows into canonical shape with target aliasing."""
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:  # pragma: no cover - dependency/runtime specific
        raise RuntimeError(
            "datasets is required for Aksharantar loading. Install datasets in the runtime environment."
        ) from exc

    configs = get_dataset_config_names(AKSHARANTAR_DATASET)
    config_name = _resolve_config(pair_code, configs)
    ds = load_dataset(AKSHARANTAR_DATASET, config_name, split="train")

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(ds):
        src = _pick_field(
            row,
            [
                "english word",
                "english",
                "source",
                "src",
                "roman",
                "latin",
                "input",
            ],
        )
        tgt = _pick_field(
            row,
            [
                "native word",
                "target",
                "tgt",
                "output",
                "word",
                "transliteration",
            ],
        )
        if not src or not tgt:
            continue
        if script_valid(tgt, target_script) < 1.0:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        english_id = _pick_field(row, ["id", "english id", "english", "english word"]) or f"{pair_code}{idx}"
        rows.append(
            canonicalize_row(
                english=english_id,
                ood=src,
                target=tgt,
                keep_legacy_alias=True,
            )
        )

    if len(rows) < min_rows:
        raise RuntimeError(
            f"Aksharantar {pair_code} yielded too few rows ({len(rows)} < {min_rows})."
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows
