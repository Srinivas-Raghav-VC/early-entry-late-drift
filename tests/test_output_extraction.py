from research.modules.eval.output_extraction import (
    analyze_generation_text,
    extract_transliteration_candidate,
    resolve_generation_stop_ids,
)


class _DummyTokenizer:
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False):
        if text == "\n":
            return {"input_ids": [13]}
        return {"input_ids": [1, 2]}


def test_extract_candidate_prefers_target_script_token() -> None:
    raw = "Answer: the transliteration is बुविंग"
    got = extract_transliteration_candidate(raw, script_name="Devanagari")
    assert got == "बुविंग"


def test_extract_candidate_rejects_wrong_script() -> None:
    raw = "Answer: boving"
    got = extract_transliteration_candidate(raw, script_name="Devanagari")
    assert got == ""


def test_extract_candidate_telugu() -> None:
    raw = "Output: ఇది అమ్మ"
    got = extract_transliteration_candidate(raw, script_name="Telugu")
    assert got == "అమ్మ"


def test_extract_candidate_bengali_and_tamil() -> None:
    assert extract_transliteration_candidate(
        "Answer: নমস্কার",
        script_name="Bengali",
    ) == "নমস্কার"
    assert extract_transliteration_candidate(
        "Result: அம்மா",
        script_name="Tamil",
    ) == "அம்மா"


def test_analyze_generation_text_flags_boilerplate() -> None:
    audit = analyze_generation_text("Answer: নমস্কার", "নমস্কার")
    assert audit["has_leading_text"] is True
    assert audit["strict_word_only"] is False


def test_resolve_generation_stop_ids_includes_newline() -> None:
    ids = resolve_generation_stop_ids(_DummyTokenizer())
    assert isinstance(ids, list)
    assert 2 in ids and 13 in ids
