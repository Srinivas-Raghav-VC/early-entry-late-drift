from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from research.modules.behavior.icl_variants import parse_variant_csv
from research.modules.prompts.prompt_templates import parse_prompt_template_csv

DEFAULT_N_VALUES = [0, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
DEFAULT_PROMPT_TEMPLATES = ["canonical", "output_only", "task_tagged"]
DEFAULT_ICL_VARIANTS = ["helpful", "random", "shuffled_targets", "corrupted_targets"]
DEFAULT_LANGUAGE_CODES = ["hin", "tel"]
DEFAULT_SEEDS = [11, 42, 101]


@dataclass
class JudgeConfig:
    enabled: bool = False
    model: str = "gemini-2.0-flash-lite"
    probe_per_condition: int = 0


@dataclass
class PathsConfig:
    snapshots_dir: str = "research/results/phase0/snapshots"
    output_dir: str = "research/results/phase0"


@dataclass
class Phase0ARunConfig:
    name: str = "phase0a-paper-rigor"
    seeds: list[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    language_codes: list[str] = field(default_factory=lambda: list(DEFAULT_LANGUAGE_CODES))
    n_values: list[int] = field(default_factory=lambda: list(DEFAULT_N_VALUES))
    n_candidate: int = 300
    n_eval: int = 50
    max_new_tokens: int = 32
    max_eval_rows: int = 0
    batch_size: int = 16
    prompt_templates: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPT_TEMPLATES))
    icl_variants: list[str] = field(default_factory=lambda: list(DEFAULT_ICL_VARIANTS))
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    aggregate_after_run: bool = True
    stats_after_run: bool = True
    plots_after_run: bool = True
    cross_seed_disjoint: bool = True
    baseline_prompt_template: str = "canonical"
    baseline_icl_variant: str = "helpful"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def to_modal_kwargs(self, *, seed: int, google_api_key: str = "", hf_token: str = "") -> dict[str, Any]:
        return {
            "hf_token": hf_token,
            "google_api_key": google_api_key,
            "max_new_tokens": int(self.max_new_tokens),
            "max_eval_rows": int(self.max_eval_rows),
            "batch_size": int(self.batch_size),
            "split_seed": int(seed),
            "n_candidate": int(self.n_candidate),
            "n_eval": int(self.n_eval),
            "n_values_csv": ",".join(str(v) for v in self.n_values),
            "judge_enabled": bool(self.judge.enabled),
            "judge_probe_per_condition": int(self.judge.probe_per_condition),
            "judge_model": str(self.judge.model),
            "snapshots_dir": str(self.paths.snapshots_dir),
            "language_codes_csv": ",".join(self.language_codes),
            "prompt_templates_csv": ",".join(self.prompt_templates),
            "icl_variants_csv": ",".join(self.icl_variants),
        }


def _load_yaml_if_available(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"YAML config requested ({path}) but PyYAML is not installed. "
            "Install pyyaml or use JSON config."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON/YAML object: {path}")
    return payload


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_if_available(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [int(p) for p in parts]
    if isinstance(value, list):
        return [int(v) for v in value]
    raise TypeError(f"Expected int-list or csv string, got: {type(value)}")


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raise TypeError(f"Expected str-list or csv string, got: {type(value)}")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_phase0a_run_config(path: Path | str) -> Phase0ARunConfig:
    cfg_path = Path(path)
    payload = _load_raw_config(cfg_path)

    judge_payload = payload.get("judge", {})
    if not isinstance(judge_payload, dict):
        raise ValueError("judge config must be an object")

    paths_payload = payload.get("paths", {})
    if not isinstance(paths_payload, dict):
        raise ValueError("paths config must be an object")

    n_values = sorted(set(_as_int_list(payload.get("n_values"), DEFAULT_N_VALUES)))
    if 0 not in n_values:
        n_values = [0, *n_values]

    prompt_templates = parse_prompt_template_csv(
        ",".join(_as_str_list(payload.get("prompt_templates"), DEFAULT_PROMPT_TEMPLATES))
    )
    if "canonical" not in prompt_templates:
        prompt_templates = ["canonical", *prompt_templates]

    icl_variants = parse_variant_csv(
        ",".join(_as_str_list(payload.get("icl_variants"), DEFAULT_ICL_VARIANTS))
    )
    if "helpful" not in icl_variants:
        icl_variants = ["helpful", *icl_variants]

    language_codes = _as_str_list(payload.get("language_codes"), DEFAULT_LANGUAGE_CODES)

    cfg = Phase0ARunConfig(
        name=str(payload.get("name", "phase0a-paper-rigor")),
        seeds=_as_int_list(payload.get("seeds"), DEFAULT_SEEDS),
        language_codes=language_codes,
        n_values=n_values,
        n_candidate=int(payload.get("n_candidate", 300)),
        n_eval=int(payload.get("n_eval", 50)),
        max_new_tokens=int(payload.get("max_new_tokens", 32)),
        max_eval_rows=int(payload.get("max_eval_rows", 0)),
        batch_size=int(payload.get("batch_size", 16)),
        prompt_templates=prompt_templates,
        icl_variants=icl_variants,
        judge=JudgeConfig(
            enabled=_as_bool(judge_payload.get("enabled"), False),
            model=str(judge_payload.get("model", "gemini-2.0-flash-lite")),
            probe_per_condition=int(judge_payload.get("probe_per_condition", 0)),
        ),
        paths=PathsConfig(
            snapshots_dir=str(paths_payload.get("snapshots_dir", "research/results/phase0/snapshots")),
            output_dir=str(paths_payload.get("output_dir", "research/results/phase0")),
        ),
        aggregate_after_run=_as_bool(payload.get("aggregate_after_run"), True),
        stats_after_run=_as_bool(payload.get("stats_after_run"), True),
        plots_after_run=_as_bool(payload.get("plots_after_run"), True),
        cross_seed_disjoint=_as_bool(payload.get("cross_seed_disjoint"), True),
        baseline_prompt_template=str(payload.get("baseline_prompt_template", "canonical")),
        baseline_icl_variant=str(payload.get("baseline_icl_variant", "helpful")),
    )

    if not cfg.seeds:
        raise ValueError("Config must specify at least one seed")
    if cfg.n_candidate <= 0 or cfg.n_eval <= 0:
        raise ValueError("n_candidate and n_eval must be positive")

    return cfg


def write_default_phase0a_run_config(path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Phase0ARunConfig()
    out_path.write_text(json.dumps(cfg.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
