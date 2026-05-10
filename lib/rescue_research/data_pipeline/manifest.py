from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class SourceDescriptor:
    name: str
    url: str
    license: str
    checksum: str = ""
    version_date: str = ""


@dataclass
class PairManifest:
    pair_id: str
    source_language: str
    source_script: str
    target_script: str
    backups: List[str]
    sources: List[SourceDescriptor] = field(default_factory=list)
    min_pool_size: int = 0
    ambiguity_rate: float = 0.0
    notes: str = ""


@dataclass
class DatasetManifest:
    schema_version: str
    frozen_at: str
    pair_manifests: Dict[str, PairManifest]
    substitutions_used: List[str] = field(default_factory=list)
    blind_slice_sealed: bool = True

    def to_json_dict(self) -> Dict:
        payload = asdict(self)
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "DatasetManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        pair_manifests = {}
        for pair_id, data in raw.get("pair_manifests", {}).items():
            sources = [SourceDescriptor(**s) for s in data.get("sources", [])]
            pair_manifests[pair_id] = PairManifest(
                pair_id=data["pair_id"],
                source_language=data["source_language"],
                source_script=data["source_script"],
                target_script=data["target_script"],
                backups=list(data.get("backups", [])),
                sources=sources,
                min_pool_size=int(data.get("min_pool_size", 0)),
                ambiguity_rate=float(data.get("ambiguity_rate", 0.0)),
                notes=str(data.get("notes", "")),
            )
        return cls(
            schema_version=str(raw["schema_version"]),
            frozen_at=str(raw["frozen_at"]),
            pair_manifests=pair_manifests,
            substitutions_used=list(raw.get("substitutions_used", [])),
            blind_slice_sealed=bool(raw.get("blind_slice_sealed", True)),
        )

