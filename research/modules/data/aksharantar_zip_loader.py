from __future__ import annotations

import json
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from research.modules.data.row_schema import canonicalize_row, get_target_text
from research.modules.eval.metrics import normalize_text, script_valid

AKSHARANTAR_ZIP_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/{code}.zip"

LANG_SPECS: dict[str, dict[str, str]] = {
    "hin": {"language": "Hindi", "script": "Devanagari"},
    "tel": {"language": "Telugu", "script": "Telugu"},
    "ben": {"language": "Bengali", "script": "Bengali"},
    "tam": {"language": "Tamil", "script": "Tamil"},
    "mar": {"language": "Marathi", "script": "Devanagari"},
}

_ALLOWED_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def _download_zip(code: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{code}.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = AKSHARANTAR_ZIP_URL.format(code=code)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with out_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return out_path


def _iter_jsonl_rows(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            with archive.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    return rows


def _is_latin_word(text: str) -> bool:
    if not text:
        return False
    if not _ALLOWED_SOURCE_RE.match(text):
        return False
    for ch in text:
        if ch in "'-":
            continue
        if not ch.isalpha():
            return False
        if "LATIN" not in unicodedata.name(ch, ""):
            return False
    return True


def load_unique_aksharantar_rows(
    *,
    code: str,
    cache_dir: Path | str,
    min_source_len: int = 3,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if code not in LANG_SPECS:
        raise ValueError(f"Unsupported Aksharantar code: {code}")

    spec = LANG_SPECS[code]
    script = spec["script"]

    zip_path = _download_zip(code, Path(cache_dir))
    raw_rows = _iter_jsonl_rows(zip_path)

    counters: dict[str, int] = defaultdict(int)
    counters["raw_rows"] = len(raw_rows)

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in raw_rows:
        counters["seen"] += 1
        uid = normalize_text(str(row.get("unique_identifier", "")))
        source = normalize_text(str(row.get("english word", ""))).lower()
        target = normalize_text(str(row.get("native word", "")))

        if not source or not target:
            counters["drop_empty"] += 1
            continue
        if len(source) < int(min_source_len):
            counters["drop_short_source"] += 1
            continue
        if " " in source:
            counters["drop_space_source"] += 1
            continue
        if not _is_latin_word(source):
            counters["drop_non_latin_source"] += 1
            continue
        if script_valid(target, script) < 1.0:
            counters["drop_script_mismatch"] += 1
            continue

        by_source[source].append(
            canonicalize_row(
                english=uid or f"{code}_{source}",
                ood=source,
                target=target,
                keep_legacy_alias=True,
            )
        )

    unique_rows: list[dict[str, str]] = []
    for source, rows in by_source.items():
        targets = {get_target_text(r) for r in rows}
        if len(targets) != 1:
            counters["drop_ambiguous_source"] += 1
            continue
        first = rows[0]
        first["english"] = first["english"] or f"{code}_{source}"
        unique_rows.append(first)

    unique_rows.sort(key=lambda r: (r["ood"], get_target_text(r)))
    counters["usable_unique_rows"] = len(unique_rows)

    report = {
        "language_code": code,
        "source_language": spec["language"],
        "target_script": script,
        "zip_path": str(zip_path),
        "counts": dict(counters),
    }
    return unique_rows, report


def build_unique_snapshot_from_rows(
    *,
    code: str,
    rows: list[dict[str, str]],
    quality_report: dict[str, Any],
    split_seed: int = 42,
    n_candidate: int = 300,
    n_eval: int = 50,
    n_values: list[int] | None = None,
) -> dict[str, Any]:
    n_values = sorted(n_values or [2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256])

    need = n_candidate + n_eval
    if len(rows) < need:
        raise RuntimeError(f"{code}: not enough unique rows ({len(rows)} < {need})")

    rng = random.Random(split_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    candidate_pool = shuffled[:n_candidate]
    eval_rows = shuffled[n_candidate : n_candidate + n_eval]

    icl_presets: dict[str, list[dict[str, str]]] = {}
    for n in n_values:
        if n > len(candidate_pool):
            raise RuntimeError(f"{code}: candidate pool {len(candidate_pool)} < required N={n}")
        icl_presets[str(n)] = candidate_pool[:n]

    spec = LANG_SPECS[code]
    return {
        "pair": f"aksharantar_{code}_latin",
        "source_language": spec["language"],
        "input_script_name": "Latin",
        "output_script_name": spec["script"],
        "split_seed": split_seed,
        "dataset_source": f"{AKSHARANTAR_ZIP_URL.format(code=code)}",
        "quality_report": quality_report,
        # Backward-compatible field name.
        "candidate_pool": candidate_pool,
        # Preferred explicit name for analysis semantics.
        "icl_bank": candidate_pool,
        "eval_rows": eval_rows,
        "icl_presets": icl_presets,
    }


def build_unique_snapshot(
    *,
    code: str,
    split_seed: int = 42,
    n_candidate: int = 300,
    n_eval: int = 50,
    n_values: list[int] | None = None,
    cache_dir: Path | str = ".cache/aksharantar",
) -> dict[str, Any]:
    rows, report = load_unique_aksharantar_rows(code=code, cache_dir=cache_dir)
    return build_unique_snapshot_from_rows(
        code=code,
        rows=rows,
        quality_report=report,
        split_seed=split_seed,
        n_candidate=n_candidate,
        n_eval=n_eval,
        n_values=n_values,
    )
