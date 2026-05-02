from research.modules.eval.judge_wrapper import deterministic_label
from research.modules.eval.metrics import akshara_cer, exact_match, script_valid


def test_exact_match_and_akshara_cer_basic() -> None:
    assert exact_match("नमस्ते", "नमस्ते") == 1.0
    assert exact_match("नमस्ते", "नमस्त") == 0.0
    assert akshara_cer("నమస్కారం", "నమస్కారం") == 0.0


def test_script_validity_basic() -> None:
    assert script_valid("नमस्ते", "Devanagari") == 1.0
    assert script_valid("namaste", "Devanagari") == 0.0
    assert script_valid("అమ్మ", "Telugu") == 1.0


def test_judge_guardrails() -> None:
    exact = deterministic_label("namaste", "नमस्ते", "नमस्ते")
    assert exact is not None
    assert exact["label"] == "exact"

    invalid = deterministic_label("amma", "అమ్మ", "amma")
    assert invalid is not None
    assert invalid["label"] == "invalid_or_non_answer"
