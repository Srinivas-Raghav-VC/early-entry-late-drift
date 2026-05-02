from __future__ import annotations

from typing import Mapping

from rescue_research.prompts.templates import (
    CONFIRMATORY_SYSTEM_PROMPT,
    confirmatory_user_prompt,
)


def build_confirmatory_messages(
    *,
    user_text: str,
    system_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Build canonical chat messages for confirmatory prompts.

    We always include the confirmatory system prompt unless explicitly disabled
    by passing an empty string.
    """
    resolved_system = (
        CONFIRMATORY_SYSTEM_PROMPT if system_text is None else str(system_text)
    ).strip()
    messages: list[dict[str, str]] = []
    if resolved_system:
        messages.append({"role": "system", "content": resolved_system})
    messages.append({"role": "user", "content": str(user_text)})
    return messages


def apply_confirmatory_chat_template(
    tokenizer,
    *,
    user_text: str,
    system_text: str | None = None,
) -> str:
    """
    Convert user/system text into the exact model input string.

    - Chat models: use tokenizer.apply_chat_template with canonical roles.
    - Some multimodal chat tokenizers expect content blocks (list[dict]) instead
      of plain strings; we retry in that format if needed.
    - Non-chat models: prepend system text as plain context to keep intent
      equivalent across model families.
    """
    messages = build_confirmatory_messages(
        user_text=user_text,
        system_text=system_text,
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Gemma-3 multimodal style: content is a list of typed blocks.
            rich_messages = [
                {
                    "role": str(m["role"]),
                    "content": [{"type": "text", "text": str(m["content"])}],
                }
                for m in messages
            ]
            try:
                return tokenizer.apply_chat_template(
                    rich_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
    if len(messages) == 1:
        return messages[0]["content"]
    return f"{messages[0]['content']}\n\n{messages[1]['content']}"


def render_prompt(
    *,
    query_token: str,
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    icl_examples: list[Mapping[str, str]],
) -> dict[str, str]:
    """
    Render a two-part prompt payload.

    Returning both system and user content lets callers keep strict prompt
    immutability across all conditions.
    """
    return {
        "system": CONFIRMATORY_SYSTEM_PROMPT,
        "user": confirmatory_user_prompt(
            query_token=query_token,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            icl_examples=icl_examples,
        ),
    }

