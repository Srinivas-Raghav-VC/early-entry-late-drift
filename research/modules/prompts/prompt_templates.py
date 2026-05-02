from __future__ import annotations

from typing import Any

from research.modules.data.row_schema import get_target_text

PROMPT_TEMPLATE_SPECS: dict[str, dict[str, str]] = {
    "canonical": {
        "instruction": "Transliterate the following English word into {script_name}. Reply with exactly one word in {script_name} only. Do not use Latin letters, explanations, quotes, punctuation, code fences, or extra text.",
        "examples_header": "",
        "query_header": "",
    },
    "output_only": {
        "instruction": "Transliterate each English word into {script_name}. Output exactly one word in {script_name} only. No Latin letters, explanations, punctuation, or extra text.",
        "examples_header": "Examples:",
        "query_header": "Input:",
    },
    "task_tagged": {
        "instruction": "Task: English to {script_name} transliteration. Return exactly one final word in {script_name} only. No explanation, no Latin letters, and no extra text.",
        "examples_header": "Few-shot pairs:",
        "query_header": "Now transliterate:",
    },
}


def list_prompt_templates() -> list[str]:
    return sorted(PROMPT_TEMPLATE_SPECS.keys())


def parse_prompt_template_csv(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        _require_template(part)
        out.append(part)

    deduped: list[str] = []
    for name in out:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _require_template(prompt_template: str) -> dict[str, str]:
    key = str(prompt_template).strip()
    if key not in PROMPT_TEMPLATE_SPECS:
        raise ValueError(
            f"Unknown prompt template {prompt_template!r}; expected one of {list_prompt_templates()}"
        )
    return PROMPT_TEMPLATE_SPECS[key]


def build_prompt(
    *,
    prompt_template: str,
    query: str,
    examples: list[dict[str, str]],
    script_name: str,
    separator: str = "->",
) -> str:
    spec = _require_template(prompt_template)
    instruction = spec["instruction"].format(script_name=script_name)
    lines: list[str] = [instruction, ""]

    examples_header = spec.get("examples_header", "").strip()
    if examples_header:
        lines.append(examples_header)

    for ex in examples:
        lines.append(f"{ex['ood']} {separator} {get_target_text(ex)}")

    query_header = spec.get("query_header", "").strip()
    if query_header:
        lines.append(query_header)

    lines.append(f"{query} {separator}")
    return "\n".join(lines)


def render_prompt(
    *,
    prompt_template: str,
    query: str,
    examples: list[dict[str, str]],
    script_name: str,
    separator: str = "->",
) -> dict[str, Any]:
    prompt = build_prompt(
        prompt_template=prompt_template,
        query=query,
        examples=examples,
        script_name=script_name,
        separator=separator,
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "prompt_separator": separator,
    }
