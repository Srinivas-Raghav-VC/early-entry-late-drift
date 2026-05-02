from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.modules.data.row_schema import get_target_text
from research.modules.prompts.prompt_templates import build_prompt


@dataclass
class VisibilityRow:
    n: int
    total_tokens: int
    instruction_tokens: int
    examples_tokens: int
    query_tokens: int
    local_window: int
    exceeds_window: bool
    fully_visible_examples: int
    partially_visible_examples: int


def canonical_prompt(
    *,
    query: str,
    examples: list[dict[str, str]],
    script_name: str,
    separator: str = "->",
) -> str:
    return build_prompt(
        prompt_template="canonical",
        query=query,
        examples=examples,
        script_name=script_name,
        separator=separator,
    )


def _char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    hits = [
        idx
        for idx, (s, e) in enumerate(offsets)
        if e > s and not (e <= char_start or s >= char_end)
    ]
    if not hits:
        return None
    return (hits[0], hits[-1] + 1)


def _spans_from_offsets(
    *,
    prompt: str,
    tokenizer: Any,
    examples: list[dict[str, str]],
    query: str,
    separator: str,
) -> tuple[list[tuple[int, int] | None], tuple[int, int] | None]:
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

    example_spans: list[tuple[int, int] | None] = []
    cursor = 0
    for ex in examples:
        line = f"{ex['ood']} {separator} {get_target_text(ex)}"
        start = prompt.find(line, cursor)
        if start < 0:
            example_spans.append(None)
            continue
        end = start + len(line)
        cursor = end
        example_spans.append(_char_span_to_token_span(offsets, start, end))

    query_line = f"{query} {separator}"
    q_start = prompt.rfind(query_line)
    query_span = None
    if q_start >= 0:
        query_span = _char_span_to_token_span(offsets, q_start, q_start + len(query))

    return example_spans, query_span


def probe_prompt_visibility(
    *,
    tokenizer: Any,
    query: str,
    examples: list[dict[str, str]],
    script_name: str,
    local_window: int,
    prompt_template: str = "canonical",
    separator: str = "->",
) -> dict[str, int | bool]:
    prompt = build_prompt(
        prompt_template=prompt_template,
        query=query,
        examples=examples,
        script_name=script_name,
        separator=separator,
    )

    tokenized = tokenizer(prompt, add_special_tokens=True)
    total_tokens = int(len(tokenized["input_ids"]))

    query_line = f"{query} {separator}"
    first_example_line = ""
    if examples:
        first_example_line = f"{examples[0]['ood']} {separator} {get_target_text(examples[0])}"

    if first_example_line and first_example_line in prompt:
        instruction_text = prompt[: prompt.find(first_example_line)]
    else:
        marker = prompt.rfind(query_line)
        instruction_text = prompt[:marker] if marker >= 0 else ""

    instruction_tokens = int(len(tokenizer(instruction_text, add_special_tokens=False)["input_ids"]))
    query_tokens = int(len(tokenizer(query_line, add_special_tokens=False)["input_ids"]))
    examples_tokens = max(0, total_tokens - instruction_tokens - query_tokens)

    example_spans: list[tuple[int, int] | None]
    query_span: tuple[int, int] | None
    try:
        example_spans, query_span = _spans_from_offsets(
            prompt=prompt,
            tokenizer=tokenizer,
            examples=examples,
            query=query,
            separator=separator,
        )
    except Exception:
        # Fallback when offset mapping is unavailable (rare with non-fast tokenizers)
        query_span = None
        example_spans = [None] * len(examples)

    if query_span is None:
        source_last_subtoken = total_tokens - 1
    else:
        source_last_subtoken = max(0, query_span[1] - 1)

    visible_start = max(0, source_last_subtoken + 1 - int(local_window))
    visible_end = source_last_subtoken + 1

    fully_visible = 0
    partially_visible = 0
    for span in example_spans:
        if span is None:
            continue
        s, e = span
        if s >= visible_start and e <= visible_end:
            fully_visible += 1
        elif max(s, visible_start) < min(e, visible_end):
            partially_visible += 1

    return {
        "prompt_template": prompt_template,
        "total_tokens": total_tokens,
        "instruction_tokens": instruction_tokens,
        "examples_tokens": examples_tokens,
        "query_tokens": query_tokens,
        "local_window": int(local_window),
        "exceeds_window": bool(total_tokens > int(local_window)),
        "fully_visible_examples": fully_visible,
        "partially_visible_examples": partially_visible,
    }
