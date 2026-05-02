from research.modules.behavior.icl_variants import materialize_icl_variant
from research.modules.data.row_schema import get_target_text
from research.modules.data.token_count_probe import canonical_prompt
from research.modules.prompts.prompt_templates import build_prompt, list_prompt_templates


def _candidate_pool() -> list[dict[str, str]]:
    return [
        {"english": "e1", "ood": "amma", "target": "अम्मा", "hindi": "अम्मा"},
        {"english": "e2", "ood": "namaste", "target": "नमस्ते", "hindi": "नमस्ते"},
        {"english": "e3", "ood": "pani", "target": "पानी", "hindi": "पानी"},
        {"english": "e4", "ood": "ghar", "target": "घर", "hindi": "घर"},
    ]


def test_prompt_templates_render() -> None:
    templates = list_prompt_templates()
    assert "canonical" in templates
    assert "output_only" in templates
    assert "task_tagged" in templates

    p = build_prompt(
        prompt_template="output_only",
        query="amma",
        examples=[{"ood": "namaste", "target": "नमस्ते"}],
        script_name="Devanagari",
        separator="->",
    )
    assert "Examples:" in p
    assert "namaste -> नमस्ते" in p
    assert p.strip().endswith("amma ->")


def test_canonical_prompt_back_compat() -> None:
    p = canonical_prompt(
        query="amma",
        examples=[{"ood": "namaste", "target": "नमस्ते"}],
        script_name="Devanagari",
    )
    assert "Transliterate the following English word" in p
    assert "Do not use Latin letters" in p
    assert p.strip().endswith("amma ->")


def test_icl_variant_materialization() -> None:
    pool = _candidate_pool()
    helpful = pool[:3]

    random_rows = materialize_icl_variant(
        variant="random",
        n=3,
        helpful_examples=helpful,
        candidate_pool=pool,
        rng_seed=7,
    )
    assert len(random_rows) == 3

    shuffled_rows = materialize_icl_variant(
        variant="shuffled_targets",
        n=3,
        helpful_examples=helpful,
        candidate_pool=pool,
        rng_seed=7,
    )
    assert len(shuffled_rows) == 3
    assert any(get_target_text(a) != get_target_text(b) for a, b in zip(helpful, shuffled_rows))

    corrupted_rows = materialize_icl_variant(
        variant="corrupted_targets",
        n=3,
        helpful_examples=helpful,
        candidate_pool=pool,
        rng_seed=7,
    )
    assert len(corrupted_rows) == 3
