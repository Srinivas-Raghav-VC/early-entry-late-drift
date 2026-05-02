from research.modules.data.token_count_probe import canonical_prompt
from research.modules.modal.modal_phase0_packet import N_VALUES, _render_model_prompts


def test_phase0a_n_values() -> None:
    assert N_VALUES == [0, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]


def test_canonical_prompt_format() -> None:
    prompt = canonical_prompt(
        query="namaste",
        examples=[{"ood": "amma", "target": "अम्मा"}],
        script_name="Devanagari",
    )
    assert "Reply with exactly one word in Devanagari only." in prompt
    assert "Do not use Latin letters" in prompt
    assert "amma -> अम्मा" in prompt
    assert prompt.strip().endswith("namaste ->")


class _ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[-1]["content"]
        if isinstance(content, list):
            content = content[0]["text"]
        return f"<chat>{content}</chat>"


def test_render_model_prompts_uses_chat_template_when_available() -> None:
    rendered, used = _render_model_prompts(_ChatTokenizer(), ["hello"])
    assert used is True
    assert rendered == ["<chat>hello</chat>"]
