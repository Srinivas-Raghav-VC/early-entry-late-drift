from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config_multiscript import SCRIPT_PAIRS
from rescue_research.data_pipeline.manifest import SourceDescriptor
from rescue_research.data_pipeline.loaders import load_flores200_en_te


_PAIR_FIELD_ALIASES = {
    "english": ("english", "lemma", "id", "key", "source_latin"),
    "source": ("source", "src", "native", "source_word", "input"),
    "target": ("target", "tgt", "output", "target_word", "transliteration"),
}

FLORES_EN_TE_PAIR_ID = "english_telugu_translation"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def _pair_registry_path() -> Path:
    return _configs_dir() / "pair_registry.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_json_line(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _safe_yaml_load(path: Path) -> Dict:
    default = {
        "schema_version": "v1",
        "external_data_root_env": "RESCUE_DATA_ROOT",
        "pair_sources": {},
    }
    if not path.exists():
        return default
    try:
        import yaml  # type: ignore
    except ImportError:
        return default
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return default
    if not isinstance(payload, dict):
        return default
    merged = dict(default)
    merged.update(payload)
    if not isinstance(merged.get("pair_sources"), dict):
        merged["pair_sources"] = {}
    return merged


def _load_pair_registry() -> Dict:
    payload = _read_json(_pair_registry_path())
    if not isinstance(payload, dict):
        return {"schema_version": "v1", "pairs": {}, "recommended_workshop_pairs": []}
    pairs = payload.get("pairs", {})
    if not isinstance(pairs, dict):
        pairs = {}
    recommended = payload.get("recommended_workshop_pairs", [])
    if not isinstance(recommended, list):
        recommended = []
    return {
        "schema_version": str(payload.get("schema_version", "v1")),
        "pairs": pairs,
        "recommended_workshop_pairs": [str(x).strip() for x in recommended if str(x).strip()],
    }


def _registered_pair_ids() -> List[str]:
    return sorted(_load_pair_registry().get("pairs", {}).keys())


def _registered_pair_spec(pair_id: str) -> Dict:
    registry = _load_pair_registry().get("pairs", {})
    spec = registry.get(str(pair_id).strip(), {})
    return spec if isinstance(spec, dict) else {}


def _pick_field(row: Dict, aliases: Iterable[str]) -> str:
    for name in aliases:
        if name in row:
            return str(row.get(name, "")).strip()
    return ""


def _normalize_record(row: Dict) -> Dict[str, str] | None:
    english = _pick_field(row, _PAIR_FIELD_ALIASES["english"])
    source = _pick_field(row, _PAIR_FIELD_ALIASES["source"])
    target = _pick_field(row, _PAIR_FIELD_ALIASES["target"])
    if not english or not source or not target:
        return None
    return {"english": english, "source": source, "target": target}


def _parse_rows_from_json(path: Path) -> Tuple[List[Dict[str, str]], Dict]:
    payload = _read_json(path)
    if payload is None:
        return [], {}
    rows_raw = []
    meta = {}
    if isinstance(payload, list):
        rows_raw = payload
    elif isinstance(payload, dict):
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        candidate = payload.get("rows", payload.get("data", []))
        if isinstance(candidate, list):
            rows_raw = candidate
    rows: List[Dict[str, str]] = []
    for row in rows_raw:
        if isinstance(row, dict):
            rec = _normalize_record(row)
            if rec is not None:
                rows.append(rec)
    return rows, meta


def _load_sidecar_meta(path: Path) -> Dict:
    """
    Load metadata for row-based formats (jsonl/csv) from a sidecar file.

    Convention:
      <datafile>.<suffix>.meta.json
      e.g. `hindi_telugu.jsonl.meta.json`

    This keeps row files schema-stable while still recording explicit provenance
    (dataset name/url/license/version) for publication-credible runs.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    payload = _read_json(meta_path)
    return payload if isinstance(payload, dict) else {}


def _parse_rows_from_jsonl(path: Path) -> Tuple[List[Dict[str, str]], Dict]:
    rows: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        row = _read_json_line(s)
        if isinstance(row, dict):
            rec = _normalize_record(row)
            if rec is not None:
                rows.append(rec)
    return rows, _load_sidecar_meta(path)


def _parse_rows_from_csv(path: Path) -> Tuple[List[Dict[str, str]], Dict]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rec = _normalize_record(row)
            if rec is not None:
                rows.append(rec)
    return rows, _load_sidecar_meta(path)


def _parse_rows_from_file(path: Path) -> Tuple[List[Dict[str, str]], Dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_rows_from_json(path)
    if suffix == ".jsonl":
        return _parse_rows_from_jsonl(path)
    if suffix == ".csv":
        return _parse_rows_from_csv(path)
    return [], {}


def _dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        english = str(row.get("english", "")).strip()
        # Split integrity is defined over the source/target pair, not the
        # upstream row identifier. Benchmarks like Aksharantar can legitimately
        # repeat the same lexical mapping under multiple ids, and keeping those
        # duplicates causes Stage 0 to fail closed during split validation.
        key = (source, target)
        if not source or not target or not english:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "english": english,
                "source": source,
                "target": target,
            }
        )
    return out


def _builtin_rows(pair_id: str) -> List[Dict[str, str]]:
    if pair_id not in SCRIPT_PAIRS:
        raise ValueError(f"Unknown pair_id: {pair_id!r}")
    pair = SCRIPT_PAIRS[pair_id]
    rows: List[Dict[str, str]] = []
    for word in pair.words:
        rec = _normalize_record(dict(word))
        if rec is not None:
            rows.append(rec)
    return rows


def _configured_file_specs(pair_id: str) -> List[Dict]:
    cfg = _safe_yaml_load(_configs_dir() / "datasets.yaml")
    pair_sources = cfg.get("pair_sources", {})
    specs = pair_sources.get(pair_id, [])
    if not isinstance(specs, list):
        return []
    out: List[Dict] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if str(spec.get("type", "")).strip().lower() != "file":
            continue
        out.append(spec)
    return out


def _auto_discover_files(pair_id: str) -> List[Path]:
    cfg = _safe_yaml_load(_configs_dir() / "datasets.yaml")
    env_key = str(cfg.get("external_data_root_env", "RESCUE_DATA_ROOT")).strip() or "RESCUE_DATA_ROOT"
    roots: List[Path] = []
    env_root = str(os.environ.get(env_key, "")).strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(_project_root() / "data" / "transliteration")

    seen = set()
    out: List[Path] = []
    patterns = [
        f"{pair_id}.jsonl",
        f"{pair_id}.json",
        f"{pair_id}.csv",
        f"*{pair_id}*.jsonl",
        f"*{pair_id}*.json",
        f"*{pair_id}*.csv",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for p in sorted(root.rglob(pat)):
                rp = p.resolve()
                if not p.is_file() or rp in seen:
                    continue
                seen.add(rp)
                out.append(p)
    return out


def _resolve_spec_paths(spec: Dict) -> List[Path]:
    paths_raw = spec.get("paths", [])
    if isinstance(paths_raw, str):
        paths_raw = [paths_raw]
    if not isinstance(paths_raw, list):
        return []
    root = _project_root()
    out: List[Path] = []
    for item in paths_raw:
        path_like = str(item).strip()
        if not path_like:
            continue
        p = Path(path_like)
        if not p.is_absolute():
            p = (root / p).resolve()
        if any(ch in path_like for ch in ("*", "?", "[")):
            out.extend(sorted(root.glob(path_like)))
        else:
            out.append(p)
    uniq: List[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return [p for p in uniq if p.exists() and p.is_file()]


@dataclass(frozen=True)
class PairRecordsBundle:
    pair_id: str
    rows: List[Dict[str, str]]
    sources: List[SourceDescriptor]
    source_counts: Dict[str, int]


def list_available_pair_ids() -> List[str]:
    ids = sorted(set(SCRIPT_PAIRS.keys()) | set(_registered_pair_ids()))
    if FLORES_EN_TE_PAIR_ID not in ids:
        ids.append(FLORES_EN_TE_PAIR_ID)
    return ids


def get_pair_prompt_metadata(pair_id: str) -> Dict[str, str]:
    if pair_id == FLORES_EN_TE_PAIR_ID:
        return {
            "pair_id": pair_id,
            "source_language": "English",
            "source_script": "Latin",
            "target_script": "Telugu",
        }
    registered = _registered_pair_spec(pair_id)
    if registered:
        return {
            "pair_id": str(pair_id),
            "source_language": str(registered.get("source_language", "")),
            "source_script": str(registered.get("source_script", "")),
            "target_script": str(registered.get("target_script", "")),
        }
    if pair_id not in SCRIPT_PAIRS:
        raise ValueError(f"Unknown pair_id: {pair_id!r}")
    pair = SCRIPT_PAIRS[pair_id]
    return {
        "pair_id": pair_id,
        "source_language": str(getattr(pair, "source_language", "")),
        "source_script": str(getattr(pair, "source_script", "")),
        "target_script": str(getattr(pair, "target_script", "")),
    }


def load_pair_records_bundle(pair_id: str, *, include_builtin: bool = True) -> PairRecordsBundle:
    rows_all: List[Dict[str, str]] = []
    source_counts: Dict[str, int] = {}
    sources: List[SourceDescriptor] = []
    has_builtin = pair_id in SCRIPT_PAIRS

    if pair_id == FLORES_EN_TE_PAIR_ID:
        # Prefer a local/configured file when present (avoids hard dependency on
        # `datasets` at runtime). Fall back to direct HF loader otherwise.
        seen_files = set()

        for spec in _configured_file_specs(pair_id):
            for path in _resolve_spec_paths(spec):
                rp = path.resolve()
                if rp in seen_files:
                    continue
                seen_files.add(rp)
                rows, meta = _parse_rows_from_file(path)
                if rows:
                    rows_all.extend(rows)
                name = str(meta.get("name", spec.get("name", path.stem))).strip() or path.stem
                source_counts[name] = source_counts.get(name, 0) + len(rows)
                sources.append(
                    SourceDescriptor(
                        name=name,
                        url=str(meta.get("url", spec.get("url", f"file://{path}"))),
                        license=str(meta.get("license", spec.get("license", "unknown"))),
                        checksum=str(meta.get("checksum", _sha256(path))),
                        version_date=str(meta.get("version_date", spec.get("version_date", ""))),
                    )
                )

        for path in _auto_discover_files(pair_id):
            rp = path.resolve()
            if rp in seen_files:
                continue
            seen_files.add(rp)
            rows, meta = _parse_rows_from_file(path)
            if rows:
                rows_all.extend(rows)
            name = str(meta.get("name", path.stem)).strip() or path.stem
            source_counts[name] = source_counts.get(name, 0) + len(rows)
            sources.append(
                SourceDescriptor(
                    name=name,
                    url=str(meta.get("url", f"file://{path}")),
                    license=str(meta.get("license", "unknown")),
                    checksum=str(meta.get("checksum", _sha256(path))),
                    version_date=str(meta.get("version_date", "")),
                )
            )

        rows = _dedupe_rows(rows_all)
        if rows:
            return PairRecordsBundle(
                pair_id=pair_id,
                rows=rows,
                sources=sources,
                source_counts=source_counts,
            )

        ds = load_flores200_en_te(sample_size=600, seed=42)
        rows = _dedupe_rows(ds.rows)
        if not rows:
            raise ValueError(f"Pair {pair_id!r} has no usable records.")
        return PairRecordsBundle(
            pair_id=pair_id,
            rows=rows,
            sources=[
                SourceDescriptor(
                    name=ds.name,
                    url="hf://facebook/flores",
                    license="CC-BY-SA-4.0",
                    checksum="",
                    version_date="",
                )
            ],
            source_counts={ds.name: len(rows)},
        )

    if bool(include_builtin) and has_builtin:
        builtin = _builtin_rows(pair_id)
        rows_all.extend(builtin)
        source_counts["config_multiscript"] = len(builtin)
        sources.append(
            SourceDescriptor(
                name="config_multiscript",
                url="local://config_multiscript.py",
                license="project-local",
                checksum="",
                version_date="",
            )
        )

    seen_files = set()
    for spec in _configured_file_specs(pair_id):
        for path in _resolve_spec_paths(spec):
            rp = path.resolve()
            if rp in seen_files:
                continue
            seen_files.add(rp)
            rows, meta = _parse_rows_from_file(path)
            if rows:
                rows_all.extend(rows)
            name = str(meta.get("name", spec.get("name", path.stem))).strip() or path.stem
            source_counts[name] = source_counts.get(name, 0) + len(rows)
            sources.append(
                SourceDescriptor(
                    name=name,
                    url=str(meta.get("url", spec.get("url", f"file://{path}"))),
                    license=str(meta.get("license", spec.get("license", "unknown"))),
                    checksum=str(meta.get("checksum", _sha256(path))),
                    version_date=str(meta.get("version_date", spec.get("version_date", ""))),
                )
            )

    for path in _auto_discover_files(pair_id):
        rp = path.resolve()
        if rp in seen_files:
            continue
        seen_files.add(rp)
        rows, meta = _parse_rows_from_file(path)
        if rows:
            rows_all.extend(rows)
        name = str(meta.get("name", path.stem)).strip() or path.stem
        source_counts[name] = source_counts.get(name, 0) + len(rows)
        sources.append(
            SourceDescriptor(
                name=name,
                url=str(meta.get("url", f"file://{path}")),
                license=str(meta.get("license", "unknown")),
                checksum=str(meta.get("checksum", _sha256(path))),
                version_date=str(meta.get("version_date", "")),
            )
        )

    rows = _dedupe_rows(rows_all)
    if not rows:
        if has_builtin or _registered_pair_spec(pair_id):
            raise ValueError(f"Pair {pair_id!r} has no usable records.")
        raise ValueError(f"Unknown pair_id: {pair_id!r}")

    return PairRecordsBundle(
        pair_id=pair_id,
        rows=rows,
        sources=sources,
        source_counts=source_counts,
    )


def load_pair_records(pair_id: str) -> List[Dict[str, str]]:
    return list(load_pair_records_bundle(pair_id).rows)


def load_pair_words_for_experiment(pair_id: str) -> List[Dict[str, str]]:
    rows = load_pair_records(pair_id)
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "english": str(row["english"]),
                # Historical key names are preserved for compatibility, but the
                # semantics are always source -> target.
                "hindi": str(row["target"]),
                "ood": str(row["source"]),
            }
        )
    return out
