from __future__ import annotations

from typing import Iterable, Mapping


PROMPT_FORMAT_VARIANTS = ("canonical", "compact", "tagged")


CONFIRMATORY_SYSTEM_PROMPT = (
    "You are a transliteration assistant. Convert only the input token from the "
    "input script to the output script. Return only the transliterated token "
    "without explanation."
)


def confirmatory_user_prompt(
    *,
    query_token: str,
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    icl_examples: Iterable[Mapping[str, str]] = (),
    variant: str = "canonical",
) -> str:
    """Render user prompt for confirmatory transliteration tasks.

    Variants are meaning-preserving format alternatives used for robustness checks.
    """
    v = str(variant or "canonical").strip().lower()
    if v not in PROMPT_FORMAT_VARIANTS:
        v = "canonical"

    examples = list(icl_examples)

    if v == "compact":
        lines = [
            f"Task: Transliterate {source_language} written in {input_script_name} into {output_script_name}.",
            "Output only the transliterated token.",
        ]
        if examples:
            ex = " ; ".join(
                [
                    f"{str(e.get('input', ''))} -> {str(e.get('output', ''))}"
                    for e in examples
                ]
            )
            lines.append(f"Examples: {ex}")
        lines.append(f"Now transliterate: {query_token} ->")
        return "\n".join(lines)

    if v == "tagged":
        lines = [
            f"Instruction: Transliterate {source_language} from {input_script_name} to {output_script_name}.",
            "Constraint: return only one transliterated token.",
        ]
        if examples:
            lines.append("Examples:")
            for i, ex in enumerate(examples, start=1):
                lines.append(f"[{i}] Input: {str(ex.get('input', ''))}")
                lines.append(f"[{i}] Output: {str(ex.get('output', ''))}")
        lines.append("Query:")
        lines.append(f"Input: {query_token}")
        lines.append("Output:")
        return "\n".join(lines)

    # Canonical format (locked default)
    lines = [
        f"Task: Transliterate {source_language} written in {input_script_name} into {output_script_name}.",
        "Output only the transliterated token.",
    ]
    if examples:
        lines.append("Examples:")
        for ex in examples:
            lines.append(
                f"{str(ex.get('input', ''))} -> {str(ex.get('output', ''))}"
            )
    lines.append("Now transliterate:")
    lines.append(f"{query_token} ->")
    return "\n".join(lines)

