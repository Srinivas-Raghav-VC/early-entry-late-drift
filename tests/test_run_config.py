from pathlib import Path

from research.modules.infra.phase0a_run_config import load_phase0a_run_config


def test_default_phase0a_run_config_loads() -> None:
    cfg = load_phase0a_run_config(Path("research/config/phase0a_run_config.json"))
    assert cfg.seeds == [11, 42, 101]
    assert cfg.judge.enabled is False
    assert cfg.cross_seed_disjoint is True
    assert cfg.batch_size == 16
    assert cfg.n_values[-1] == 256
    assert "canonical" in cfg.prompt_templates
    assert "helpful" in cfg.icl_variants


def test_to_modal_kwargs_contains_manifest_fields() -> None:
    cfg = load_phase0a_run_config(Path("research/config/phase0a_run_config.json"))
    kwargs = cfg.to_modal_kwargs(seed=42, google_api_key="", hf_token="")
    assert kwargs["split_seed"] == 42
    assert kwargs["judge_enabled"] is False
    assert kwargs["n_values_csv"].startswith("0,4,8")
    assert kwargs["language_codes_csv"].startswith("hin")
