#!/usr/bin/env python3
"""
Phase 0B token visibility audit for the frozen CFOM protocol.

This is a measurement pass, not a behavioral or mechanistic experiment. It uses
the exact prompt rendering path, frozen primary ICL bank, and real tokenizer
for each model to quantify what is visible to local-attention layers at the
source-side query position and at the first target decoding position.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _min(values: Iterable[int]) -> int:
    vals = [int(v) for v in values]
    return min(vals) if vals else 0


def _max(values: Iterable[int]) -> int:
    vals = [int(v) for v in values]
    return max(vals) if vals else 0


def _csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


@lru_cache(maxsize=1)
def _project_imports() -> Dict[str, Any]:
    from config import get_model_config
    from core import (
        _find_query_span_in_rendered_prompt,
        apply_chat_template,
        build_task_prompt,
        split_data_three_way,
    )
    from paper2_fidelity_calibrated.protocol_utils import prompt_fingerprint, prompt_template_fingerprint
    from paper2_fidelity_calibrated.run import _load_words, _prompt_naming
    from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata

    return {
        "get_model_config": get_model_config,
        "_find_query_span_in_rendered_prompt": _find_query_span_in_rendered_prompt,
        "apply_chat_template": apply_chat_template,
        "build_task_prompt": build_task_prompt,
        "split_data_three_way": split_data_three_way,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_template_fingerprint": prompt_template_fingerprint,
        "_load_words": _load_words,
        "_prompt_naming": _prompt_naming,
        "get_pair_prompt_metadata": get_pair_prompt_metadata,
    }


def _language_label(pair_id: str) -> str:
    try:
        meta = _project_imports()["get_pair_prompt_metadata"](pair_id)
        label = str(meta.get("source_language", "")).strip().lower()
        if label:
            return label
    except Exception:
        pass
    parts = str(pair_id).split("_")
    if len(parts) >= 2 and parts[1]:
        return str(parts[1]).strip().lower()
    return str(pair_id)


def _load_tokenizer_and_config(model_key: str) -> Tuple[Any, Any, Mapping[str, Any]]:
    try:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Phase 0B token visibility audit requires transformers. "
            "Install requirements.txt or run inside the project research environment."
        ) from e

    cfg = _project_imports()["get_model_config"](str(model_key))
    config = AutoConfig.from_pretrained(cfg.hf_id, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)

    tokenizer = None
    tokenizer_errors: List[str] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
    except Exception as e:
        tokenizer_errors.append(f"AutoTokenizer: {e}")

    if tokenizer is None:
        try:
            processor = AutoProcessor.from_pretrained(cfg.hf_id, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", None) or processor
        except Exception as e:
            tokenizer_errors.append(f"AutoProcessor: {e}")

    if tokenizer is None:
        err_blob = " | ".join(tokenizer_errors) if tokenizer_errors else "unknown error"
        raise RuntimeError(f"Failed loading tokenizer for {cfg.hf_id}: {err_blob}")

    pattern = int(getattr(text_config, "sliding_window_pattern", 0) or 0)
    n_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    global_layers = [i for i in range(n_layers) if pattern > 0 and (i % pattern) == (pattern - 1)]
    local_layers = [i for i in range(n_layers) if i not in global_layers]

    arch = {
        "model_key": str(model_key),
        "hf_id": str(cfg.hf_id),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("revision", "") or "")
        if hasattr(tokenizer, "init_kwargs")
        else "",
        "num_hidden_layers": int(getattr(text_config, "num_hidden_layers", 0) or 0),
        "num_attention_heads": int(getattr(text_config, "num_attention_heads", 0) or 0),
        "num_key_value_heads": int(getattr(text_config, "num_key_value_heads", 0) or 0),
        "hidden_size": int(getattr(text_config, "hidden_size", 0) or 0),
        "head_dim": int(getattr(text_config, "head_dim", 0) or 0),
        "sliding_window": int(getattr(text_config, "sliding_window", 0) or 0),
        "sliding_window_pattern": int(pattern),
        "global_layers": global_layers,
        "local_layers": local_layers,
        "max_position_embeddings": int(getattr(text_config, "max_position_embeddings", 0) or 0),
        "vocab_size": int(getattr(text_config, "vocab_size", 0) or 0),
    }
    return tokenizer, text_config, arch


def _find_last_text_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    start = text.rfind(needle)
    if start < 0:
        return None
    return (start, start + len(needle))


def _find_sequential_text_spans(text: str, needles: Sequence[str]) -> List[Optional[Tuple[int, int]]]:
    spans: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for needle in needles:
        if not needle:
            spans.append(None)
            continue
        start = text.find(needle, cursor)
        if start < 0:
            spans.append(None)
            continue
        end = start + len(needle)
        spans.append((start, end))
        cursor = end
    return spans


def _build_prompt_region_spans(
    *,
    raw_prompt: str,
    query_token: str,
    icl_examples: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    lines = raw_prompt.splitlines()
    if len(lines) < 4:
        raise ValueError("Unexpected prompt shape; expected at least instruction and query lines.")

    instruction_text = "\n".join(lines[:2])
    instruction_span = (0, len(instruction_text))

    examples_block_span: Optional[Tuple[int, int]] = None
    example_line_spans: List[Optional[Tuple[int, int]]] = []
    example_lines: List[str] = []
    if icl_examples:
        if "Examples:\n" not in raw_prompt or "\nNow transliterate:" not in raw_prompt:
            raise ValueError("Expected canonical prompt to contain example block markers.")
        start = raw_prompt.index("Examples:")
        end = raw_prompt.index("\nNow transliterate:")
        examples_block_span = (start, end)
        example_lines = [
            f"{str(ex.get('ood', ex.get('input', '')))} -> {str(ex.get('hindi', ex.get('output', '')))}"
            for ex in icl_examples
        ]
        example_line_spans = _find_sequential_text_spans(raw_prompt, example_lines)

    query_line = f"{query_token} ->"
    q_line_span = _find_last_text_span(raw_prompt, query_line)
    if q_line_span is None:
        raise ValueError(f"Could not locate query line {query_line!r} in prompt.")
    query_token_span = (q_line_span[0], q_line_span[0] + len(query_token))

    return {
        "instruction_span": instruction_span,
        "examples_block_span": examples_block_span,
        "example_line_spans": example_line_spans,
        "example_lines": example_lines,
        "query_token_span": query_token_span,
    }


def _locate_user_text(rendered_prompt: str, user_text: str) -> int:
    idx = rendered_prompt.rfind(user_text)
    if idx < 0:
        raise ValueError("Could not locate raw user text inside rendered prompt.")
    return idx


def _char_span_to_token_span(
    offsets: Sequence[Tuple[int, int]],
    char_span: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if char_span is None:
        return None
    start, end = int(char_span[0]), int(char_span[1])
    hits = [
        idx
        for idx, (s, e) in enumerate(offsets)
        if int(e) > int(s) and not (int(e) <= start or int(s) >= end)
    ]
    if not hits:
        return None
    return (int(hits[0]), int(hits[-1]) + 1)


def _half_open_length(span: Optional[Tuple[int, int]]) -> int:
    if span is None:
        return 0
    return max(0, int(span[1]) - int(span[0]))


def _visible_window_for_prompt(
    *,
    prompt_len: int,
    local_window: int,
    source_last_subtoken: int,
) -> Dict[str, Tuple[int, int]]:
    source_end = int(source_last_subtoken) + 1
    source_start = max(0, source_end - int(local_window))
    target_end = int(prompt_len)
    target_start = max(0, target_end - int(local_window))
    return {
        "source_last_subtoken": (source_start, source_end),
        "target_pos1_teacher_forced": (target_start, target_end),
    }


def _span_intersection_len(
    left: Optional[Tuple[int, int]],
    right: Tuple[int, int],
) -> int:
    if left is None:
        return 0
    s = max(int(left[0]), int(right[0]))
    e = min(int(left[1]), int(right[1]))
    return max(0, e - s)


def _classify_visibility(
    token_span: Optional[Tuple[int, int]],
    visible_span: Tuple[int, int],
) -> str:
    if token_span is None:
        return "missing"
    if _span_intersection_len(token_span, visible_span) == 0:
        return "fully_invisible"
    if token_span[0] >= visible_span[0] and token_span[1] <= visible_span[1]:
        return "fully_visible"
    return "partially_visible"


def _rendered_region_token_spans(
    *,
    tokenizer: Any,
    raw_prompt: str,
    rendered_prompt: str,
    query_token: str,
    icl_examples: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    packet = _build_prompt_region_spans(
        raw_prompt=raw_prompt,
        query_token=query_token,
        icl_examples=icl_examples,
    )
    user_start = _locate_user_text(rendered_prompt, raw_prompt)

    encoded = tokenizer(
        rendered_prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"][0].tolist()]

    instruction_span = _char_span_to_token_span(
        offsets,
        (user_start + packet["instruction_span"][0], user_start + packet["instruction_span"][1]),
    )
    examples_block_raw = packet["examples_block_span"]
    examples_block_span = _char_span_to_token_span(
        offsets,
        None
        if examples_block_raw is None
        else (user_start + examples_block_raw[0], user_start + examples_block_raw[1]),
    )
    example_line_spans = [
        _char_span_to_token_span(
            offsets,
            None if span is None else (user_start + span[0], user_start + span[1]),
        )
        for span in packet["example_line_spans"]
    ]
    query_char_span = (
        user_start + packet["query_token_span"][0],
        user_start + packet["query_token_span"][1],
    )
    query_span = _project_imports()["_find_query_span_in_rendered_prompt"](
        tokenizer, rendered_prompt, query_token
    )
    query_span = query_span or _char_span_to_token_span(offsets, query_char_span)

    return {
        "prompt_ids": encoded["input_ids"][0].tolist(),
        "instruction_span": instruction_span,
        "examples_block_span": examples_block_span,
        "example_line_spans": example_line_spans,
        "query_span": query_span,
    }


def _item_visibility_record(
    *,
    word: Mapping[str, str],
    pair_id: str,
    condition: str,
    prompt_variant: str,
    raw_prompt: str,
    rendered_prompt: str,
    tokenizer: Any,
    local_window: int,
    icl_examples: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    project = _project_imports()
    regions = _rendered_region_token_spans(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        rendered_prompt=rendered_prompt,
        query_token=str(word["ood"]),
        icl_examples=icl_examples,
    )
    query_span = regions["query_span"]
    if query_span is None:
        raise ValueError(f"Fail-closed query span localization failed for {pair_id} / {condition}.")
    prompt_ids = list(regions["prompt_ids"])
    prompt_len = int(len(prompt_ids))
    visible_windows = _visible_window_for_prompt(
        prompt_len=prompt_len,
        local_window=int(local_window),
        source_last_subtoken=int(query_span[1] - 1),
    )
    examples_block_span = regions["examples_block_span"]
    example_line_spans = list(regions["example_line_spans"])

    loci: Dict[str, Any] = {}
    for locus_name, visible_span in visible_windows.items():
        visible_examples: List[int] = []
        partial_examples: List[int] = []
        invisible_examples: List[int] = []
        missing_examples: List[int] = []
        for idx, ex_span in enumerate(example_line_spans):
            status = _classify_visibility(ex_span, visible_span)
            if status == "fully_visible":
                visible_examples.append(idx)
            elif status == "partially_visible":
                partial_examples.append(idx)
            elif status == "fully_invisible":
                invisible_examples.append(idx)
            else:
                missing_examples.append(idx)

        icl_tokens_total = _half_open_length(examples_block_span)
        icl_tokens_visible = _span_intersection_len(examples_block_span, visible_span)
        icl_tokens_outside = max(0, icl_tokens_total - icl_tokens_visible)
        loci[locus_name] = {
            "visible_prompt_token_span": [int(visible_span[0]), int(visible_span[1])],
            "visible_prompt_token_count": int(visible_span[1] - visible_span[0]),
            "icl_block_tokens_total": int(icl_tokens_total),
            "icl_block_tokens_inside_local_window": int(icl_tokens_visible),
            "icl_block_tokens_outside_local_window": int(icl_tokens_outside),
            "icl_block_fraction_outside_local_window": (
                float(icl_tokens_outside / icl_tokens_total) if icl_tokens_total > 0 else 0.0
            ),
            "fully_visible_example_indices": visible_examples,
            "partially_visible_example_indices": partial_examples,
            "fully_invisible_example_indices": invisible_examples,
            "missing_example_indices": missing_examples,
            "fully_visible_example_count": len(visible_examples),
            "partially_visible_example_count": len(partial_examples),
            "fully_invisible_example_count": len(invisible_examples),
            "source_last_subtoken_index": int(query_span[1] - 1),
            "target_pos1_teacher_forced_index": int(prompt_len),
        }

    return {
        "pair": str(pair_id),
        "word_id": str(word["english"]),
        "source_romanized": str(word["ood"]),
        "target_native": str(word["hindi"]),
        "condition": str(condition),
        "prompt_variant": str(prompt_variant),
        "rendered_prompt_fingerprint": project["prompt_fingerprint"](
            raw_prompt=raw_prompt,
            rendered_prompt=rendered_prompt,
        ),
        "total_prompt_tokens": int(prompt_len),
        "instruction_tokens": int(_half_open_length(regions["instruction_span"])),
        "icl_block_tokens": int(_half_open_length(examples_block_span)),
        "query_source_span_tokens": int(_half_open_length(query_span)),
        "local_window": int(local_window),
        "local_window_exceeded": bool(prompt_len > int(local_window)),
        "loci": loci,
    }


def _aggregate_locus_rows(
    items: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    pair_id: str,
    condition: str,
    locus_name: str,
) -> Dict[str, Any]:
    prompt_tokens = [int(item["total_prompt_tokens"]) for item in items]
    instruction_tokens = [int(item["instruction_tokens"]) for item in items]
    icl_tokens = [int(item["icl_block_tokens"]) for item in items]
    query_tokens = [int(item["query_source_span_tokens"]) for item in items]
    local_exceeded = [1 if bool(item["local_window_exceeded"]) else 0 for item in items]

    locus_rows = [dict(item["loci"][locus_name]) for item in items]
    example_vis = Counter()
    example_partial = Counter()
    example_invis = Counter()
    n_items = int(len(items))
    for row in locus_rows:
        for idx in row["fully_visible_example_indices"]:
            example_vis[int(idx)] += 1
        for idx in row["partially_visible_example_indices"]:
            example_partial[int(idx)] += 1
        for idx in row["fully_invisible_example_indices"]:
            example_invis[int(idx)] += 1

    max_example_index = -1
    for c in (example_vis, example_partial, example_invis):
        if c:
            max_example_index = max(max_example_index, max(c.keys()))

    example_frequency = []
    for idx in range(max_example_index + 1):
        example_frequency.append(
            {
                "example_index": int(idx),
                "fully_visible_rate": float(example_vis[idx] / n_items) if n_items else float("nan"),
                "partially_visible_rate": float(example_partial[idx] / n_items) if n_items else float("nan"),
                "fully_invisible_rate": float(example_invis[idx] / n_items) if n_items else float("nan"),
            }
        )

    return {
        "model_key": str(model_key),
        "pair": str(pair_id),
        "language": _language_label(pair_id),
        "condition": str(condition),
        "locus": str(locus_name),
        "n_items": n_items,
        "total_prompt_tokens_mean": _mean(prompt_tokens),
        "total_prompt_tokens_min": _min(prompt_tokens),
        "total_prompt_tokens_max": _max(prompt_tokens),
        "instruction_tokens_mean": _mean(instruction_tokens),
        "icl_block_tokens_mean": _mean(icl_tokens),
        "query_source_span_tokens_mean": _mean(query_tokens),
        "local_window": int(locus_rows[0]["visible_prompt_token_count"]) if locus_rows else 0,
        "local_window_exceeded_rate": _mean(local_exceeded),
        "icl_block_tokens_inside_local_window_mean": _mean(
            row["icl_block_tokens_inside_local_window"] for row in locus_rows
        ),
        "icl_block_tokens_outside_local_window_mean": _mean(
            row["icl_block_tokens_outside_local_window"] for row in locus_rows
        ),
        "icl_block_fraction_outside_local_window_mean": _mean(
            row["icl_block_fraction_outside_local_window"] for row in locus_rows
        ),
        "fully_visible_example_count_mean": _mean(
            row["fully_visible_example_count"] for row in locus_rows
        ),
        "partially_visible_example_count_mean": _mean(
            row["partially_visible_example_count"] for row in locus_rows
        ),
        "fully_invisible_example_count_mean": _mean(
            row["fully_invisible_example_count"] for row in locus_rows
        ),
        "example_visibility_frequency": example_frequency,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        for key, value in list(flat.items()):
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False)
        flat_rows.append(flat)
    fieldnames: List[str] = []
    for row in flat_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 0B token visibility audit.")
    ap.add_argument("--models", type=str, default="1b,4b")
    ap.add_argument("--pairs", type=str, default="aksharantar_hin_latin,aksharantar_tel_latin")
    ap.add_argument("--conditions", type=str, default="explicit_zs,icl8,icl64")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap for quick debug runs.")
    ap.add_argument("--out-root", type=str, default="artifacts/phase0")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = (PROJECT_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    project = _project_imports()

    summary_rows: List[Dict[str, Any]] = []

    for model_key in _csv(args.models):
        tokenizer, _, architecture = _load_tokenizer_and_config(model_key)
        template_fp = project["prompt_template_fingerprint"](tokenizer)

        for pair_id in _csv(args.pairs):
            words, provenance = project["_load_words"](
                pair_id,
                external_only=bool(args.external_only),
                require_external_sources=bool(args.require_external_sources),
                min_pool_size=int(args.min_pool_size),
            )
            icl_examples, _, eval_samples = project["split_data_three_way"](
                words=words,
                n_icl=int(args.n_icl),
                n_select=int(args.n_select),
                n_eval=int(args.n_eval),
                seed=int(args.seed),
            )
            if int(args.max_items) > 0:
                eval_samples = list(eval_samples[: int(args.max_items)])

            prompt_meta = project["get_pair_prompt_metadata"](pair_id)
            source_language, input_script_name, output_script_name = project["_prompt_naming"](prompt_meta)
            conditions = {
                "explicit_zs": [],
                "icl8": list(icl_examples[:8]),
                "icl64": list(icl_examples),
            }

            for condition in _csv(args.conditions):
                if condition not in conditions:
                    raise ValueError(f"Unsupported condition {condition!r}.")
                cond_examples = conditions[condition]
                item_rows: List[Dict[str, Any]] = []

                for word in eval_samples:
                    raw_prompt = project["build_task_prompt"](
                        str(word["ood"]),
                        cond_examples or None,
                        input_script_name=input_script_name,
                        source_language=source_language,
                        output_script_name=output_script_name,
                        prompt_variant=str(args.prompt_variant),
                    )
                    rendered_prompt = project["apply_chat_template"](tokenizer, raw_prompt)
                    item_rows.append(
                        _item_visibility_record(
                            word=word,
                            pair_id=pair_id,
                            condition=condition,
                            prompt_variant=str(args.prompt_variant),
                            raw_prompt=raw_prompt,
                            rendered_prompt=rendered_prompt,
                            tokenizer=tokenizer,
                            local_window=int(architecture["sliding_window"]),
                            icl_examples=cond_examples,
                        )
                    )

                loci_summary = {
                    locus: _aggregate_locus_rows(
                        item_rows,
                        model_key=model_key,
                        pair_id=pair_id,
                        condition=condition,
                        locus_name=locus,
                    )
                    for locus in ("source_last_subtoken", "target_pos1_teacher_forced")
                }

                payload = {
                    "created_at_utc": _now_utc(),
                    "phase": "0B_token_visibility_audit",
                    "model_key": str(model_key),
                    "pair": str(pair_id),
                    "language": _language_label(pair_id),
                    "condition": str(condition),
                    "prompt_variant": str(args.prompt_variant),
                    "seed": int(args.seed),
                    "split_sizes": {
                        "icl": int(len(icl_examples)),
                        "selection": int(args.n_select),
                        "eval": int(len(eval_samples)),
                    },
                    "architecture": architecture,
                    "prompt_template": template_fp,
                    "provenance": provenance,
                    "summary": loci_summary,
                    "items": item_rows,
                }

                out_json = out_root / f"token_visibility_{model_key}_{_language_label(pair_id)}_{condition}.json"
                out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                for row in loci_summary.values():
                    summary_rows.append(row)

    summary_json = out_root / "token_visibility_summary.json"
    summary_csv = out_root / "token_visibility_summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, summary_rows)
    print(str(summary_json))
    print(str(summary_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
