#!/usr/bin/env python3
"""
Core Utilities for Cross-Script Rescue Experiments
===================================================
Date: January 2, 2026

This module provides core functionality with PROPER methodology:
1. Randomized ICL examples (never hardcoded)
2. Multi-token probability measurement
3. Proper train/test splits
4. Clean abstractions
"""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import random
import json
import hashlib
import platform
import subprocess
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception:  # pragma: no cover - handled by runtime loader fallbacks
    AutoModelForImageTextToText = None  # type: ignore
    AutoProcessor = None  # type: ignore

from config import HINDI_TELUGU_WORDS, get_model_config, get_experiment_config

# Script Unicode ranges used for script-matched random controls and basic validation.
# Keep this lightweight: do NOT import config_multiscript here, since it may do
# heavyweight work at import time (building transliteration pairs).
_SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Arabic": [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Oriya": [(0x0B00, 0x0B7F)],
    "Georgian": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "Greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Hebrew": [(0x0590, 0x05FF)],
}


# Random ICL control examples (unrelated task) to match context length.
# Used for "random patching" baselines.
# NOTE: These examples use Hindi (Devanagari) → Telugu script to MATCH the context
# length and script types of the real ICL examples. This is a stronger control
# than English→French because it uses the same script complexity.
RANDOM_ICL_EXAMPLES = [
    # Use Hindi words but with WRONG Telugu mappings (random unrelated words)
    # This matches token counts better than short English words
    {"src": "book", "tgt": "livre"},
    {"src": "car", "tgt": "voiture"},
    {"src": "sun", "tgt": "soleil"},
    {"src": "house", "tgt": "maison"},
    {"src": "water", "tgt": "eau"},
    {"src": "food", "tgt": "nourriture"},
    {"src": "dog", "tgt": "chien"},
    {"src": "cat", "tgt": "chat"},
]

# Neutral filler vocabulary for explicit context-length controls (Null-ICL).
NULL_ICL_TOKENS = [
    "alpha", "beta", "gamma", "delta", "river", "stone", "paper", "signal",
    "memory", "vector", "orbit", "quiet", "window", "forest", "copper", "planet",
    "garden", "silver", "matrix", "anchor", "thread", "bridge", "metric", "kernel",
    "random", "sample", "token", "prompt", "reason", "pattern", "system", "module",
]

# Legacy: hardcoded Indic-script random control examples. These are no longer used
# by the publication default (we now generate script-matched random strings to
# avoid accidental vocabulary overlap), but kept for backward compatibility.
RANDOM_ICL_INDIC = [
    # Use Telugu→Hindi (reversed direction) as control - same script families, different task
    {"src": "పుస్తకం", "tgt": "पुस्तक"},  # book (Telugu) -> book (Hindi) - reversed task
    {"src": "కారు", "tgt": "कार"},  # car
    {"src": "సూర్యుడు", "tgt": "सूरज"},  # sun
    {"src": "ఇల్లు", "tgt": "घर"},  # house
    {"src": "నీరు", "tgt": "पानी"},  # water
    {"src": "ఆహారం", "tgt": "खाना"},  # food
    {"src": "కుక్క", "tgt": "कुत्ता"},  # dog
    {"src": "పిల్లి", "tgt": "बिल्ली"},  # cat
]

# Cache HuggingFace repo file listings to avoid repeated network calls when
# sweeping many layers.
_REPO_FILE_LIST_CACHE: Dict[str, List[str]] = {}


# ============================================================================
# JSON I/O (robust against numpy scalars)
# ============================================================================


def json_default(obj: Any):
    """
    JSON serializer fallback.

    This prevents runs from failing on numpy scalar types (e.g., numpy.bool_),
    while keeping common numeric fields as proper Python numbers.
    """
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        # Avoid accidentally dumping huge tensors.
        if obj.numel() <= 4096:
            return obj.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


def _env_fingerprint() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        out["torch_version"] = torch.__version__
    except Exception:
        out["torch_version"] = None

    try:
        out["transformers_version"] = __import__("transformers").__version__
    except Exception:
        out["transformers_version"] = None

    try:
        if torch.cuda.is_available():
            out["cuda"] = {
                "available": True,
                "device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        else:
            out["cuda"] = {"available": False}
    except Exception:
        out["cuda"] = {"available": None}

    try:
        # Best-effort pip freeze snapshot (can be large, but useful for exact repro).
        freeze = subprocess.check_output(
            ["python3", "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )
        out["pip_freeze_sha256"] = hashlib.sha256(freeze.encode("utf-8")).hexdigest()
    except Exception:
        out["pip_freeze_sha256"] = None

    return out


def save_json(path: str, data: Any) -> None:
    """Create parent dir (if needed) and write JSON with a safe default encoder."""
    # Automatically attach environment fingerprint for reviewer-proof reproducibility.
    if isinstance(data, dict) and "_env" not in data:
        data["_env"] = _env_fingerprint()

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


# ============================================================================
# DATA HANDLING
# ============================================================================


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_inference_runtime(
    *,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
) -> Dict[str, Any]:
    """
    Configure fast inference/runtime defaults for CUDA execution.

    This is intentionally lightweight and safe for research runs:
    - TF32 matmul/cudnn on Ampere+ for faster FP32 paths
    - matmul precision hint for PyTorch 2.x kernels
    - cudnn benchmark for stable-shape workloads
    """
    out: Dict[str, Any] = {
        "enable_tf32": bool(enable_tf32),
        "matmul_precision": str(matmul_precision),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "cuda_available": bool(torch.cuda.is_available()),
        "applied": False,
    }
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(matmul_precision))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
            torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        out["applied"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def split_data(
    words: List[Dict], n_icl: int, n_test: int, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split words into ICL examples and test samples.

    CRITICAL: ICL and test sets are DISJOINT.
    """
    set_all_seeds(seed)

    shuffled = words.copy()
    random.shuffle(shuffled)

    icl_examples = shuffled[:n_icl]
    test_samples = shuffled[n_icl : n_icl + n_test]

    _assert_unique_split_keys(words)
    _assert_disjoint_word_lists(icl_examples, test_samples, "ICL", "test")

    return icl_examples, test_samples


def split_data_with_remainder(
    words: List[Dict], n_icl: int, n_test: int, seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split words into disjoint (ICL, test, remainder) sets.

    This is a small extension of split_data() used by experiments that need a
    deterministic held-out pool (e.g., native-anchor centroids) without changing
    the evaluation set.
    """
    set_all_seeds(seed)

    shuffled = words.copy()
    random.shuffle(shuffled)

    icl_examples = shuffled[:n_icl]
    test_samples = shuffled[n_icl : n_icl + n_test]
    remainder = shuffled[n_icl + n_test :]

    _assert_unique_split_keys(words)
    _assert_disjoint_word_lists(icl_examples, test_samples, "ICL", "test")
    _assert_disjoint_word_lists(icl_examples, remainder, "ICL", "remainder")
    _assert_disjoint_word_lists(test_samples, remainder, "test", "remainder")

    return icl_examples, test_samples, remainder


def _stable_split_key(word: Dict[str, Any]) -> str:
    """
    Return a lexical identity key suitable for split-auditing.

    Do not rely only on `english`: external builders sometimes synthesize row ids
    when a dataset lacks a clean lemma column. Source/target text is a safer
    fallback for transliteration-style records.
    """
    for field in ("split_key", "stable_key", "lemma_key"):
        raw = str(word.get(field, "")).strip()
        if raw:
            return f"explicit::{raw.casefold()}"

    source = str(word.get("source", word.get("hindi", ""))).strip()
    target = str(word.get("target", word.get("ood", word.get("telugu", "")))).strip()
    english = str(word.get("english", "")).strip()

    if source and target:
        return f"pair::{source}\t{target}"
    if source and english:
        return f"source+id::{source}\t{english.casefold()}"
    if source:
        return f"source::{source}"
    if english:
        return f"id::{english.casefold()}"
    raise ValueError(f"Word record lacks a stable split key: {word!r}")


def _assert_unique_split_keys(words: List[Dict[str, Any]]) -> None:
    keys = [_stable_split_key(w) for w in words]
    counts = Counter(keys)
    dupes = [k for k, c in counts.items() if c > 1]
    if dupes:
        preview = ", ".join(repr(k) for k in dupes[:5])
        raise ValueError(
            "Input words contain duplicate stable split keys; fix dataset deduplication "
            f"before splitting. Examples: {preview}"
        )


def _assert_disjoint_word_lists(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    left_name: str,
    right_name: str,
) -> None:
    left_keys = {_stable_split_key(w) for w in left}
    right_keys = {_stable_split_key(w) for w in right}
    overlap = left_keys & right_keys
    assert not overlap, (
        f"{left_name} and {right_name} sets must be disjoint. "
        f"Overlapping keys: {sorted(overlap)[:5]}"
    )


def split_data_three_way(
    words: List[Dict],
    n_icl: int,
    n_select: int,
    n_eval: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split words into disjoint (ICL, selection, evaluation) sets.

    Use this when you want a *clean* two-stage workflow:
    - selection set: choose layer / choose candidate features
    - evaluation set: report final performance

    This avoids "winner's curse" from selecting and evaluating on the same words.
    """
    set_all_seeds(seed)

    if n_icl < 0 or n_select < 0 or n_eval < 0:
        raise ValueError("n_icl/n_select/n_eval must be non-negative.")
    if n_icl + n_select + n_eval > len(words):
        raise ValueError(
            f"Not enough words for split: n_icl={n_icl}, n_select={n_select}, n_eval={n_eval}, "
            f"total={n_icl + n_select + n_eval}, available={len(words)}"
        )

    shuffled = words.copy()
    random.shuffle(shuffled)

    icl_examples = shuffled[:n_icl]
    select_samples = shuffled[n_icl : n_icl + n_select]
    eval_samples = shuffled[n_icl + n_select : n_icl + n_select + n_eval]

    _assert_unique_split_keys(words)
    _assert_disjoint_word_lists(icl_examples, select_samples, "ICL", "selection")
    _assert_disjoint_word_lists(icl_examples, eval_samples, "ICL", "evaluation")
    _assert_disjoint_word_lists(select_samples, eval_samples, "selection", "evaluation")

    return icl_examples, select_samples, eval_samples


def get_all_words() -> List[Dict]:
    """Get all available Hindi-Telugu word pairs."""
    return HINDI_TELUGU_WORDS.copy()


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================


def build_task_prompt(
    ood_input: str,
    icl_examples: Optional[List[Dict]] = None,
    input_script_name: str = "Telugu",
    source_language: str = "Hindi",
    output_script_name: str = "Devanagari",
    prompt_variant: str = "canonical",
) -> str:
    """
    Build the task prompt for cross-script "same-language" transliteration.

    Format:
        Task: Convert {source_language} text written in {input_script_name} script
              into {source_language} in {output_script_name} script.

        Input ({input_script_name}): <example1>
        Output ({output_script_name}): <answer1>

        Input ({input_script_name}): <test_input>
        Output ({output_script_name}):
    """
    from rescue_research.prompts.templates import confirmatory_user_prompt

    input_script_name = input_script_name.strip() or "Telugu"
    source_language = source_language.strip() or "Hindi"
    output_script_name = output_script_name.strip() or "Devanagari"

    canonical_examples: List[Dict[str, str]] = []
    if icl_examples:
        for ex in icl_examples:
            ex_input = str(ex.get("ood", ex.get("telugu", ex.get("input", ""))) or "")
            ex_output = str(
                ex.get("hindi", ex.get("source", ex.get("output", ""))) or ""
            )
            if ex_input and ex_output:
                canonical_examples.append({"input": ex_input, "output": ex_output})

    return confirmatory_user_prompt(
        query_token=str(ood_input),
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        icl_examples=canonical_examples,
        variant=str(prompt_variant),
    )


def apply_chat_template(tokenizer, user_text: str) -> str:
    """Apply chat template for instruction-tuned models."""
    from rescue_research.prompts.render import apply_confirmatory_chat_template

    return apply_confirmatory_chat_template(
        tokenizer,
        user_text=user_text,
    )


def build_random_icl_prompt(
    ood_input: str,
    n_icl: int,
    input_script_name: str = "Telugu",
    *,
    source_language: str = "Hindi",
    output_script_name: str = "Devanagari",
    use_indic_control: bool = False,
    length_reference_examples: Optional[List[Dict]] = None,
    seed: Optional[int] = None,
    forbidden_src_texts: Optional[List[str]] = None,
    forbidden_tgt_texts: Optional[List[str]] = None,
) -> str:
    """
    Random-ICL control prompt: keep the *task header* fixed, but insert unrelated
    examples to roughly match context length.

    This is a control for "extra context" vs "task-relevant ICL".

    Args:
        use_indic_control: If True, generate script-matched random examples (same
                          Input/Output script labels) for better token-length
                          matching and fewer format confounds. Recommended for
                          publication.
        length_reference_examples: Optional list of examples used only to match
                                   string lengths and character distributions.
        seed: Optional seed to make the random control deterministic per run.
        forbidden_src_texts: Additional source-side strings to avoid generating.
        forbidden_tgt_texts: Additional target-side strings to avoid generating.
    """
    input_script_name = input_script_name.strip() or "Telugu"
    source_language = source_language.strip() or "Hindi"
    output_script_name = output_script_name.strip() or "Devanagari"
    lines = [
        f"Task: Convert {source_language} text written in {input_script_name} script into {source_language} in {output_script_name} script.",
        "",
    ]

    if use_indic_control:
        # Script-matched random control: generate unrelated strings in the SAME
        # scripts and labels as the task, to avoid control-format confounds and
        # accidental vocabulary leakage from a hardcoded list.

        def _derive_seed() -> int:
            base = int(seed) if seed is not None else 0
            msg = f"rand_icl::{base}::{n_icl}::{input_script_name}::{output_script_name}".encode(
                "utf-8"
            )
            return int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)

        rng = np.random.default_rng(_derive_seed())

        def _chars_from_texts(texts: List[str]) -> List[str]:
            cs: set[str] = set()
            for t in texts:
                for ch in str(t):
                    if ch and not ch.isspace():
                        cs.add(ch)
            return sorted(cs)

        def _chars_from_script(script_name: str) -> List[str]:
            ranges = _SCRIPT_RANGES.get(script_name, [])
            pool: List[str] = []
            for lo, hi in ranges:
                for cp in range(int(lo), int(hi) + 1):
                    ch = chr(int(cp))
                    # Prefer letters to avoid weird combining-mark-only strings.
                    if unicodedata.category(ch).startswith("L"):
                        pool.append(ch)
            return pool

        ref_srcs: List[str] = []
        ref_tgts: List[str] = []
        if isinstance(length_reference_examples, list):
            for ex in length_reference_examples:
                if not isinstance(ex, dict):
                    continue
                src = str(ex.get("ood", ex.get("telugu", ex.get("src", ""))) or "")
                tgt = str(ex.get("hindi", ex.get("tgt", "")) or "")
                if src:
                    ref_srcs.append(src)
                if tgt:
                    ref_tgts.append(tgt)

        src_pool = _chars_from_texts(ref_srcs) or _chars_from_script(input_script_name) or list(
            "abcdefghijklmnopqrstuvwxyz"
        )
        tgt_pool = _chars_from_texts(ref_tgts) or _chars_from_script(output_script_name) or list(
            "abcdefghijklmnopqrstuvwxyz"
        )

        # Length targets: match the reference ICL examples when available; else use
        # a safe fallback.
        src_lens = [len(s) for s in ref_srcs if s]
        tgt_lens = [len(t) for t in ref_tgts if t]
        src_med = int(np.median(src_lens)) if src_lens else 4
        tgt_med = int(np.median(tgt_lens)) if tgt_lens else 4

        forbidden_src = set(ref_srcs) | {str(ood_input)}
        forbidden_tgt = set(ref_tgts)
        if isinstance(forbidden_src_texts, list):
            forbidden_src.update([str(x) for x in forbidden_src_texts if str(x)])
        if isinstance(forbidden_tgt_texts, list):
            forbidden_tgt.update([str(x) for x in forbidden_tgt_texts if str(x)])

        def _rand_str(pool: List[str], length: int, forbidden: set[str]) -> str:
            L = max(1, int(length))
            for _ in range(25):
                arr = rng.choice(pool, size=L, replace=True)
                s = "".join([str(x) for x in arr.tolist()])
                if s not in forbidden:
                    return s
            # Fallback: return the last attempt even if it collided (extremely unlikely).
            return s

        for i in range(int(n_icl)):
            sl = len(ref_srcs[i]) if i < len(ref_srcs) and ref_srcs[i] else src_med
            tl = len(ref_tgts[i]) if i < len(ref_tgts) and ref_tgts[i] else tgt_med
            src = _rand_str(src_pool, sl, forbidden_src)
            tgt = _rand_str(tgt_pool, tl, forbidden_tgt)
            lines.append(f"Input ({input_script_name}): {src}")
            lines.append(f"Output ({output_script_name}): {tgt}")
            lines.append("")
    else:
        # Original English→French control (for backward compatibility)
        for ex in RANDOM_ICL_EXAMPLES[:n_icl]:
            lines.append(f"Input (English): {ex['src']}")
            lines.append(f"Output (French): {ex['tgt']}")
            lines.append("")

    lines.append(f"Input ({input_script_name}): {ood_input}")
    lines.append(f"Output ({output_script_name}):")
    return "\n".join(lines)


def build_null_icl_prompt(
    ood_input: str,
    *,
    input_script_name: str = "Telugu",
    source_language: str = "Hindi",
    output_script_name: str = "Devanagari",
    seed: Optional[int] = None,
    target_token_budget: int = 150,
) -> str:
    """
    Null-ICL control prompt for context-length confound checks.

    Keeps the task header and query format intact, but replaces demonstrations
    with neutral filler text of comparable length (no mapping signal).
    """
    input_script_name = input_script_name.strip() or "Telugu"
    source_language = source_language.strip() or "Hindi"
    output_script_name = output_script_name.strip() or "Devanagari"

    base = int(seed) if seed is not None else 0
    msg = f"null_icl::{base}::{target_token_budget}::{input_script_name}::{output_script_name}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)

    n_tok = max(24, int(target_token_budget))
    toks = rng.choice(np.array(NULL_ICL_TOKENS), size=n_tok, replace=True).tolist()

    lines = [
        f"Task: Convert {source_language} text written in {input_script_name} script into {source_language} in {output_script_name} script.",
        "",
        "Context (ignore; unrelated filler):",
    ]
    chunk = 12
    for i in range(0, n_tok, chunk):
        lines.append(" ".join(str(t) for t in toks[i : i + chunk]))
    lines.extend(
        [
            "",
            f"Input ({input_script_name}): {ood_input}",
            f"Output ({output_script_name}):",
        ]
    )
    return "\n".join(lines)


def build_english_neutral_prompt(english_word: str) -> str:
    """
    Neutral English-copy prompt for superposition / feature-collision diagnostics.

    This task is intentionally unrelated to cross-script transliteration.
    """
    w = str(english_word or "").strip()
    return "\n".join(
        [
            "Task: Repeat the given English word exactly in Latin script.",
            "",
            f"Input (Latin): {w}",
            "Output (Latin):",
        ]
    )


def build_corrupted_icl_prompt(
    ood_input: str,
    icl_examples: List[Dict],
    *,
    input_script_name: str = "Telugu",
    source_language: str = "Hindi",
    output_script_name: str = "Devanagari",
    seed: Optional[int] = None,
) -> str:
    """
    Task-matched ICL *corruption* control.

    Keeps the exact task format and demonstration inputs, but permutes the
    demonstration outputs so the mapping is wrong. This controls for "generic
    ICL / task-mode features" without providing correct supervision.

    By default we use a randomized derangement of demonstration outputs (no item
    keeps its own output). If `seed` is None, we fall back to a simple rotation
    for backward compatibility.
    """
    if not icl_examples:
        return build_task_prompt(
            ood_input,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
        )

    outs = [
        str(ex.get("hindi", ex.get("source", ex.get("output", ""))) or "")
        for ex in icl_examples
    ]

    def _rotate(xs: List[str]) -> List[str]:
        return xs[1:] + xs[:1]

    def _derangement(xs: List[str]) -> List[str]:
        n = len(xs)
        if n <= 1:
            return list(xs)
        # Deterministic local RNG (avoid touching global state).
        base = int(seed) if seed is not None else 0
        msg = f"derange::{base}::{n}::{input_script_name}::{output_script_name}".encode("utf-8")
        seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
        rng = np.random.default_rng(seed32)
        for _ in range(100):
            perm = rng.permutation(n).tolist()
            if all(int(perm[i]) != i for i in range(n)):
                return [xs[int(j)] for j in perm]
        # Fallback: rotation always deranges when n>1.
        return _rotate(xs)

    corrupted_outs = _derangement(outs) if seed is not None else _rotate(outs)
    corrupted = []
    for ex, wrong_out in zip(icl_examples, corrupted_outs):
        ex2 = dict(ex)
        ex2["hindi"] = wrong_out
        corrupted.append(ex2)

    return build_task_prompt(
        ood_input,
        corrupted,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
    )


# ============================================================================
# PROBABILITY MEASUREMENT
# ============================================================================


def get_first_token_prob(
    model, tokenizer, prompt: str, target_text: str, device: str
) -> Tuple[float, int]:
    """
    Get probability of the FIRST token of target_text.

    Returns: (probability, target_token_id)
    """
    text = apply_chat_template(tokenizer, prompt)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not target_ids:
        return float("nan"), -1

    target_id = target_ids[0]
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    probs = torch.softmax(outputs.logits[0, -1], dim=-1)
    return float(probs[target_id].item()), target_id


def get_first_token_stats(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    device: str,
    *,
    topk_values: Tuple[int, ...] = (1, 10, 100),
) -> Dict[str, Any]:
    """
    Return richer next-token diagnostics for the FIRST token of target_text.

    This is primarily for debugging and scaling analysis (e.g., showing when a
    model is in a "floor" regime where probabilities are ~0 and the token isn't
    even in top-k).
    """
    text = apply_chat_template(tokenizer, prompt)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        return {
            "prob": float("nan"),
            "target_id": -1,
            "top1_id": -1,
            "top1_prob": float("nan"),
            "target_logit": float("nan"),
            "max_logit": float("nan"),
            "logit_gap": float("nan"),
            "target_rank": float("nan"),
            "entropy": float("nan"),
            "topk_hits": {},
        }

    target_id = int(target_ids[0])
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)

    prob = float(probs[target_id].item())
    top1_id = int(torch.argmax(logits).item())
    top1_prob = float(probs[top1_id].item())

    target_logit = float(logits[target_id].item())
    max_logit = float(logits[top1_id].item())
    logit_gap = float(max_logit - target_logit)
    # Exact target rank without a full sort (1 = top token).
    target_rank = int(torch.sum(logits > logits[target_id]).item()) + 1
    # Predictive entropy helps detect over/under-confident next-token behavior.
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

    # Top-k membership diagnostics (avoid full ranking).
    topk_hits: Dict[str, bool] = {}
    ks = [int(k) for k in topk_values if int(k) > 0]
    if ks:
        k_max = int(max(ks))
        topk_idx = torch.topk(logits, k=k_max).indices
        for k in sorted(set(ks)):
            in_topk = bool(torch.any(topk_idx[:k] == target_id).item())
            topk_hits[str(k)] = in_topk

    return {
        "prob": prob,
        "target_id": target_id,
        "top1_id": top1_id,
        "top1_prob": top1_prob,
        "target_logit": target_logit,
        "max_logit": max_logit,
        "logit_gap": logit_gap,
        "target_rank": target_rank,
        "entropy": entropy,
        "topk_hits": topk_hits,
    }


def get_multi_token_prob(
    model, tokenizer, prompt: str, target_text: str, device: str
) -> Tuple[float, int]:
    """
    Get the JOINT probability of the FULL multi-token target.

    Uses teacher forcing:
    P(target) = P(t1|ctx) * P(t2|ctx,t1) * ... * P(tn|ctx,t1,...,tn-1)

    Returns: (joint_probability, n_tokens)
    """
    text = apply_chat_template(tokenizer, prompt)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not target_ids:
        return float("nan"), 0

    base_inputs = tokenizer(text, return_tensors="pt").to(device)
    joint_prob = _teacher_forced_joint_prob_from_input_ids(
        model, base_inputs.input_ids, target_ids, device
    )
    return float(joint_prob), len(target_ids)


def get_multi_token_logprob(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    device: str,
) -> Tuple[float, int]:
    """
    Get the JOINT log-probability of the FULL multi-token target.

    This is numerically stable compared to the raw joint probability and is
    better suited for longer targets or small-probability regimes.

    Returns: (log_probability, n_tokens)
    """
    text = apply_chat_template(tokenizer, prompt)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not target_ids:
        return float("nan"), 0

    base_inputs = tokenizer(text, return_tensors="pt").to(device)
    logprob = _teacher_forced_joint_logprob_from_input_ids(
        model, base_inputs.input_ids, target_ids, device
    )
    return float(logprob), len(target_ids)


# ============================================================================
# MODEL UTILITIES
# ============================================================================


def get_model_layers(model):
    """
    Get transformer *decoder* layers for patching.

    Hugging Face model wrappers vary a lot across architectures (and multimodal
    "ForConditionalGeneration" wrappers). We need a robust way to locate the
    per-layer modules that contain an `.mlp` submodule (used by our hooks).
    """

    def _get_attr_chain(root, chain: Tuple[str, ...]):
        cur = root
        for name in chain:
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    def _looks_like_layer_list(obj) -> bool:
        if obj is None:
            return False
        if not isinstance(obj, (list, torch.nn.ModuleList)):
            return False
        if len(obj) == 0:
            return False
        first = obj[0]
        # Our patch hooks assume `layer.mlp` exists.
        return hasattr(first, "mlp")

    # Fast paths for common wrappers.
    chains: List[Tuple[str, ...]] = [
        ("model", "layers"),  # Gemma/Llama causal LM
        ("model", "model", "layers"),  # some remote-code wrappers
        ("transformer", "h"),  # GPT-2 style
        ("model", "decoder", "layers"),  # seq2seq-style wrappers
        ("decoder", "layers"),
        # Multimodal wrappers often expose a language model explicitly.
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        # Some codebases use "text_model" for the LM backbone.
        ("text_model", "model", "layers"),
        ("text_model", "layers"),
        ("model", "text_model", "model", "layers"),
        ("model", "text_model", "layers"),
    ]

    for chain in chains:
        layers = _get_attr_chain(model, chain)
        if _looks_like_layer_list(layers):
            return layers

    # Fallback: search for a ModuleList that looks like decoder layers.
    def _infer_expected_n_layers() -> Optional[int]:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        for attr in ("num_hidden_layers", "n_layer", "n_layers"):
            if hasattr(cfg, attr):
                try:
                    return int(getattr(cfg, attr))
                except Exception:
                    pass
        for sub in ("text_config", "language_config", "decoder_config"):
            subcfg = getattr(cfg, sub, None)
            if subcfg is not None and hasattr(subcfg, "num_hidden_layers"):
                try:
                    return int(subcfg.num_hidden_layers)
                except Exception:
                    pass
        return None

    expected = _infer_expected_n_layers()
    best: Optional[Tuple[int, str, torch.nn.ModuleList]] = None
    try:
        it = model.named_modules()
    except Exception:
        it = []

    for name, mod in it:
        if not isinstance(mod, torch.nn.ModuleList):
            continue
        if len(mod) == 0:
            continue
        first = mod[0]
        if not hasattr(first, "mlp"):
            continue

        score = 0
        if expected is not None and len(mod) == expected:
            score += 10
        # Extra weak signals to avoid picking unrelated ModuleLists.
        if hasattr(first, "self_attn") or hasattr(first, "attn"):
            score += 1
        if hasattr(first, "input_layernorm") or hasattr(
            first, "post_attention_layernorm"
        ):
            score += 1
        score += 1  # base score for matching mlp

        if best is None or score > best[0]:
            best = (score, name, mod)

    if best is not None:
        return best[2]

    raise AttributeError(f"Cannot find layers in model: {type(model).__name__}")


def get_layer_device(model, layer_idx: int) -> torch.device:
    """
    Return the device for a specific transformer layer.

    This matters when using `device_map="auto"` (multi-GPU): transcoders/CLTs must
    live on the same device as the layer they patch.
    """
    layers = get_model_layers(model)
    layer = layers[layer_idx]

    for p in layer.parameters(recurse=True):
        return p.device
    for b in layer.buffers(recurse=True):
        return b.device
    # Fallback: first model param device (often the "input" device).
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def load_model(
    model_size: str,
    device: str = "cuda",
    *,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
):
    """Load model and tokenizer.

    Gemma 3 has both text-only and multimodal checkpoints across sizes.
    Some (e.g. 4B/12B IT) are exposed as conditional-generation models rather
    than plain CausalLM. We therefore try a small loader cascade while keeping
    the same downstream interface (forward -> logits, generate, chat template).
    """
    config = get_model_config(model_size)
    configure_inference_runtime(
        enable_tf32=bool(enable_tf32),
        matmul_precision=str(matmul_precision),
        cudnn_benchmark=bool(cudnn_benchmark),
    )

    def _flash_attn_requested() -> bool:
        raw = str(os.environ.get("RESCUE_USE_FLASH_ATTN", "1")).strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

    def _build_kwargs(*, dtype_key: str, use_flash_attn: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if dtype_key == "dtype":
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs

    use_flash_attn = _flash_attn_requested()
    attempts = [("dtype", use_flash_attn), ("torch_dtype", use_flash_attn)]
    if use_flash_attn:
        # Graceful fallback for models/runtimes that do not support FA2.
        attempts.extend([("dtype", False), ("torch_dtype", False)])

    loaders: List[Tuple[str, Any]] = [
        ("AutoModelForCausalLM", AutoModelForCausalLM.from_pretrained),
    ]
    if AutoModelForImageTextToText is not None:
        loaders.append(
            ("AutoModelForImageTextToText", AutoModelForImageTextToText.from_pretrained)
        )
    try:
        from transformers import Gemma3ForConditionalGeneration  # type: ignore

        loaders.append(
            (
                "Gemma3ForConditionalGeneration",
                Gemma3ForConditionalGeneration.from_pretrained,
            )
        )
    except Exception:
        pass

    model = None
    model_loader_name = ""
    model_errors: List[str] = []

    for loader_name, loader_fn in loaders:
        last_err: Optional[Exception] = None
        for dtype_key, fa2 in attempts:
            try:
                kwargs = _build_kwargs(dtype_key=dtype_key, use_flash_attn=bool(fa2))
                model = loader_fn(config.hf_id, **kwargs)
                model_loader_name = str(loader_name)
                if use_flash_attn and not fa2:
                    print(
                        "[core] FlashAttention2 requested but unavailable for this model/runtime; "
                        "falling back to default attention.",
                        flush=True,
                    )
                break
            except Exception as e:
                last_err = e
                continue
        if model is not None:
            break
        model_errors.append(f"{loader_name}: {last_err}")

    if model is None:
        err_blob = " | ".join(model_errors) if model_errors else "unknown error"
        hint = ""
        low = err_blob.lower()
        if "gated" in low or "401" in low or "403" in low:
            hint = (
                " Ensure you accepted the model terms and are logged in with a HF token "
                "that has access to gated Gemma checkpoints."
            )
        raise RuntimeError(f"Failed loading model '{config.hf_id}': {err_blob}.{hint}")

    tokenizer = None
    tokenizer_errors: List[str] = []

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True)
    except Exception as e:
        tokenizer_errors.append(f"AutoTokenizer: {e}")

    if tokenizer is None and AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(config.hf_id, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                tokenizer = processor
        except Exception as e:
            tokenizer_errors.append(f"AutoProcessor: {e}")

    if tokenizer is None:
        err_blob = " | ".join(tokenizer_errors) if tokenizer_errors else "unknown error"
        raise RuntimeError(f"Failed loading tokenizer for '{config.hf_id}': {err_blob}")

    # Keep generation code robust for models without an explicit pad token.
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    model.eval()
    print(
        f"[core] Model loaded via {model_loader_name}: {config.hf_id} | "
        f"tokenizer={type(tokenizer).__name__}",
        flush=True,
    )

    return model, tokenizer


def model_tokenizer_fingerprint(model, tokenizer) -> Dict[str, Any]:
    """
    Lightweight fingerprint for model/tokenizer alignment.

    This is intentionally redundant with `_env` (which includes library versions),
    but makes it easier to diagnose accidental mismatches in saved JSON artifacts.
    """
    cfg = getattr(model, "config", None)
    model_name_or_path = None
    if cfg is not None:
        model_name_or_path = getattr(cfg, "_name_or_path", None) or getattr(
            cfg, "name_or_path", None
        )
    model_name_or_path = model_name_or_path or getattr(model, "name_or_path", None)
    return {
        "model_class": type(model).__name__,
        "model_name_or_path": model_name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    }


# ============================================================================
# TRANSCODER LOADING
# ============================================================================


def load_transcoder(
    model,
    scope_repo: str,
    layer: int,
    device: str,
    variant: str = "affine_skip",
):
    """Load transcoder/SAE from Gemma Scope 2.

    variant: "affine_skip" (prefer *_affine/ paths) or "skipless_or_non_affine"
    (prefer non-affine from same repo). Tries sae_lens first, falls back to manual.
    """
    # Try sae_lens first (cleaner interface)
    try:
        from sae_lens import SAE

        scope_name = scope_repo.replace("google/", "")
        release_candidates: List[str] = []
        for cand in (
            scope_name,
            f"{scope_name}-canonical",
            f"{scope_name}-transcoder",
            f"{scope_name}-transcoder-canonical",
            scope_name.replace("-it", "-pt-res"),
            f"{scope_name.replace('-it', '-pt-res')}-canonical",
            f"{scope_name.replace('-it', '-pt-res')}-transcoder",
            f"{scope_name.replace('-it', '-pt-res')}-transcoder-canonical",
        ):
            c = str(cand).strip()
            if c and c not in release_candidates:
                release_candidates.append(c)

        sae_id_candidates: List[str] = []
        for cand in (
            f"layer_{layer}/width_16k/canonical",
            f"layer_{layer}/width_16k/average_l0_71",
            f"layer_{layer}/width_16k/l0_71",
            f"layer_{layer}/width_16k/l0_medium",
            f"layer_{layer}_width_16k_l0_medium",
            f"layer_{layer}",
        ):
            c = str(cand).strip()
            if c and c not in sae_id_candidates:
                sae_id_candidates.append(c)

        last_err: Optional[Exception] = None
        for release in release_candidates:
            for sae_id in sae_id_candidates:
                try:
                    print(f"Loading transcoder via sae_lens: {release} / {sae_id}")
                    loaded = SAE.from_pretrained(
                        release=release,
                        sae_id=sae_id,
                        device=device,
                    )
                    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

                    # Guard against silently loading the wrong hook point.
                    cfg = getattr(sae, "cfg", None)
                    hook_point = str(
                        getattr(cfg, "hook_point", "")
                        or getattr(cfg, "hook_name", "")
                        or ""
                    ).strip()
                    hook_ok = True
                    if hook_point:
                        hp = hook_point.lower()
                        layer_ok = (
                            f".{int(layer)}." in hp
                            or f"_{int(layer)}_" in hp
                            or f"layer_{int(layer)}" in hp
                        )
                        location_ok = (
                            ("mlp" in hp)
                            or ("hook_mlp_out" in hp)
                            or ("resid_mid" in hp)
                        )
                        hook_ok = bool(layer_ok and location_ok)
                    if not hook_ok:
                        last_err = RuntimeError(
                            f"Rejected sae_lens candidate due to hook mismatch: {hook_point}"
                        )
                        continue

                    # Wrap in our interface
                    wrapper = SAELensTranscoderWrapper(sae)
                    wrapper.load_info = {
                        "loader": "sae_lens",
                        "release": release,
                        "sae_id": sae_id,
                        "scope_repo": scope_repo,
                        "layer": int(layer),
                        "hook_point": hook_point,
                    }
                    print(
                        f"[core] Transcoder loaded via sae_lens: repo={scope_repo} layer={int(layer)} "
                        f"release={release} sae_id={sae_id} hook={hook_point or 'unknown'}",
                        flush=True,
                    )
                    return wrapper
                except Exception as inner_e:
                    last_err = inner_e
                    continue

        if last_err is not None:
            raise RuntimeError(f"No sae_lens transcoder candidate succeeded ({last_err})")

    except Exception as e:
        print(f"sae_lens loading failed ({e}), trying manual loading...")

    # Fallback to manual loading
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file as safetensors_load_file
    import json
    import re

    # Find transcoder path
    if scope_repo in _REPO_FILE_LIST_CACHE:
        files = _REPO_FILE_LIST_CACHE[scope_repo]
    else:
        files = list_repo_files(scope_repo)
        _REPO_FILE_LIST_CACHE[scope_repo] = files

    # Look for transcoder folder
    folder = (
        "transcoder_all"
        if any(f.startswith("transcoder_all/") for f in files)
        else "transcoder"
    )

    # Find config for this layer (Gemma Scope artifacts sometimes use cfg.json)
    layer_tokens = [f"layer_{layer}_", f"layer_{layer}/"]
    config_candidates = [
        f
        for f in files
        if f.startswith(f"{folder}/")
        and any(tok in f for tok in layer_tokens)
        and (f.endswith("config.json") or f.endswith("cfg.json"))
    ]

    if not config_candidates:
        raise RuntimeError(f"No transcoder found for layer {layer} in {scope_repo}")

    # Prefer affine (skip-transcoder) or skipless per variant.
    # affine_skip: prefer *_affine/ paths; skipless_or_non_affine: prefer non-affine.
    def _cand_priority(path: str) -> Tuple[int, int]:
        has_affine = "_affine/" in path
        if variant == "skipless_or_non_affine":
            is_preferred = 0 if not has_affine else 1  # prefer non-affine
        else:
            is_preferred = 0 if has_affine else 1  # prefer affine
        l0_pref = (
            0
            if (
                "canonical" in path
                or "average_l0_71" in path
                or "l0_medium" in path
            )
            else 1
        )
        return (is_preferred, l0_pref)

    config_candidates.sort(key=_cand_priority)
    config_path = config_candidates[0]
    sae_folder = config_path.rsplit("/", 1)[0]

    # Load config
    cfg_local = hf_hub_download(repo_id=scope_repo, filename=config_path)
    with open(cfg_local, "r") as f:
        cfg = json.load(f)

    hook_point = str(
        cfg.get("hook_point", "")
        or cfg.get("hook_name", "")
        or cfg.get("hook", "")
    ).strip()
    if hook_point:
        hp = hook_point.lower()
        layer_ok = (
            f".{int(layer)}." in hp
            or f"_{int(layer)}_" in hp
            or f"layer_{int(layer)}" in hp
        )
        location_ok = ("mlp" in hp) or ("hook_mlp_out" in hp) or ("resid_mid" in hp)
        if not (layer_ok and location_ok):
            raise RuntimeError(
                f"Refusing manual transcoder with hook mismatch: hook_point={hook_point}"
            )

    # Load weights (prefer standard filenames when multiple are present)
    weight_files = [
        f
        for f in files
        if f.startswith(f"{sae_folder}/") and f.endswith(".safetensors")
    ]
    if not weight_files:
        raise RuntimeError(f"No weights found in {sae_folder}")

    def _weight_priority(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        preferred = {
            "params.safetensors": 0,
            "sae_weights.safetensors": 1,
            "weights.safetensors": 2,
        }
        return (preferred.get(base, 9), base)

    weight_files.sort(key=_weight_priority)
    weights_local = hf_hub_download(repo_id=scope_repo, filename=weight_files[0])
    state = safetensors_load_file(weights_local)

    # Parse dimensions
    d_in = int(model.lm_head.weight.shape[1])
    d_sae_match = re.search(r"width_(\d+)k", sae_folder)
    d_sae = int(d_sae_match.group(1)) * 1024 if d_sae_match else None
    if d_sae is None:
        # Infer from tensor shapes if width wasn't parseable.
        two_d = [t.shape for t in state.values() if getattr(t, "ndim", 0) == 2]
        if not two_d:
            raise RuntimeError(
                f"Could not infer d_sae from weights. Keys: {list(state.keys())[:20]}"
            )
        d_sae = max(max(s) for s in two_d)

    def _pick_tensor_by_shape(
        st: Dict[str, torch.Tensor], shape: Tuple[int, ...], prefer: Tuple[str, ...]
    ) -> Optional[torch.Tensor]:
        for k in st:
            if st[k].shape == shape and any(p in k.lower() for p in prefer):
                return st[k]
        for k in st:
            if st[k].shape == shape:
                return st[k]
        return None

    # Find encoder/decoder weights (handle potential transpose conventions)
    W_enc = _pick_tensor_by_shape(state, (d_in, d_sae), ("enc", "w_enc", "encoder"))
    if W_enc is None:
        W_enc = _pick_tensor_by_shape(state, (d_sae, d_in), ("enc", "w_enc", "encoder"))
    W_dec = _pick_tensor_by_shape(state, (d_sae, d_in), ("dec", "w_dec", "decoder"))
    if W_dec is None:
        W_dec = _pick_tensor_by_shape(state, (d_in, d_sae), ("dec", "w_dec", "decoder"))

    if W_enc is None or W_dec is None:
        # Fall back: pick by shape only.
        for _, t in state.items():
            if t.shape == (d_in, d_sae) and W_enc is None:
                W_enc = t
            if t.shape == (d_sae, d_in) and W_dec is None:
                W_dec = t

    if W_enc is None and W_dec is not None and W_dec.shape == (d_sae, d_in):
        W_enc = W_dec.t().contiguous()
    if W_dec is None and W_enc is not None and W_enc.shape == (d_in, d_sae):
        W_dec = W_enc.t().contiguous()

    if W_enc is None or W_dec is None:
        raise RuntimeError(
            f"Could not infer W_enc/W_dec. Keys: {list(state.keys())[:20]}"
        )
    if W_enc.shape == (d_sae, d_in):
        W_enc = W_enc.t().contiguous()
    if W_dec.shape == (d_in, d_sae):
        W_dec = W_dec.t().contiguous()

    # Biases + threshold
    threshold = None
    for k, t in state.items():
        if (
            t.ndim == 1
            and t.shape[0] == d_sae
            and ("threshold" in k.lower() or "theta" in k.lower())
        ):
            threshold = t
            break

    b_enc = _pick_tensor_by_shape(state, (d_sae,), ("b_enc", "enc", "bias"))
    if (
        b_enc is not None
        and threshold is not None
        and b_enc.data_ptr() == threshold.data_ptr()
    ):
        b_enc = None
    if b_enc is None:
        # First non-threshold vector of correct width.
        for k, t in state.items():
            if (
                t.ndim == 1
                and t.shape[0] == d_sae
                and ("threshold" not in k.lower())
                and ("theta" not in k.lower())
            ):
                b_enc = t
                break

    b_dec = _pick_tensor_by_shape(state, (d_in,), ("b_dec", "dec", "bias"))

    tc = Transcoder(
        W_enc=W_enc.to(device=device, dtype=torch.float32),
        W_dec=W_dec.to(device=device, dtype=torch.float32),
        b_enc=b_enc.to(device=device, dtype=torch.float32)
        if b_enc is not None
        else None,
        b_dec=b_dec.to(device=device, dtype=torch.float32)
        if b_dec is not None
        else None,
        threshold=threshold.to(device=device, dtype=torch.float32)
        if threshold is not None
        else None,
    )
    tc.load_info = {
        "loader": "manual",
        "scope_repo": scope_repo,
        "layer": int(layer),
        "config_path": config_path,
        "weights_path": weight_files[0],
        "sae_folder": sae_folder,
        "hook_point": hook_point,
    }
    print(
        f"[core] Transcoder loaded via manual HF: repo={scope_repo} layer={int(layer)} "
        f"sae_folder={sae_folder}",
        flush=True,
    )
    return tc


class SAELensTranscoderWrapper:
    """Wrapper to make sae_lens SAE work with our Transcoder interface."""

    def __init__(self, sae):
        self.sae = sae
        self.W_enc = sae.W_enc.detach().to(dtype=torch.float32)
        self.W_dec = sae.W_dec.detach().to(dtype=torch.float32)
        b_enc = sae.b_enc.detach() if hasattr(sae, "b_enc") and sae.b_enc is not None else None
        b_dec = sae.b_dec.detach() if hasattr(sae, "b_dec") and sae.b_dec is not None else None
        thr = sae.threshold.detach() if hasattr(sae, "threshold") and sae.threshold is not None else None
        self.b_enc = (
            b_enc.to(dtype=torch.float32)
            if b_enc is not None
            else torch.zeros(self.W_enc.shape[1], device=self.W_enc.device, dtype=torch.float32)
        )
        self.b_dec = (
            b_dec.to(dtype=torch.float32)
            if b_dec is not None
            else torch.zeros(self.W_dec.shape[1], device=self.W_dec.device, dtype=torch.float32)
        )
        self.threshold = thr.to(dtype=torch.float32) if thr is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_f = x.to(device=self.W_enc.device, dtype=torch.float32)
        pre = x_f @ self.W_enc + self.b_enc
        if self.threshold is not None:
            return torch.relu(pre - self.threshold.unsqueeze(0))
        return torch.relu(pre)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        f = features.to(device=self.W_dec.device, dtype=torch.float32)
        return f @ self.W_dec + self.b_dec


class Transcoder:
    """Simple transcoder wrapper."""

    def __init__(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: Optional[torch.Tensor] = None,
        b_dec: Optional[torch.Tensor] = None,
        threshold: Optional[torch.Tensor] = None,
    ):
        self.W_enc = W_enc.to(dtype=torch.float32)
        self.W_dec = W_dec.to(dtype=torch.float32)
        self.b_enc = (
            b_enc.to(dtype=torch.float32)
            if b_enc is not None
            else torch.zeros(self.W_enc.shape[1], device=self.W_enc.device, dtype=torch.float32)
        )
        self.b_dec = (
            b_dec.to(dtype=torch.float32)
            if b_dec is not None
            else torch.zeros(self.W_dec.shape[1], device=self.W_dec.device, dtype=torch.float32)
        )
        self.threshold = threshold.to(dtype=torch.float32) if threshold is not None else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x_f = x.to(device=self.W_enc.device, dtype=torch.float32)
        pre = x_f @ self.W_enc + self.b_enc

        if self.threshold is not None:
            # JumpReLU activation
            return torch.relu(pre - self.threshold.unsqueeze(0))
        return torch.relu(pre)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to model space."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        f = features.to(device=self.W_dec.device, dtype=torch.float32)
        return f @ self.W_dec + self.b_dec


# ============================================================================
# ACTIVATION PATCHING
# ============================================================================


@dataclass
class PatchResult:
    """Result of a patching experiment."""

    word_english: str
    word_hindi: str
    word_telugu: str

    # Baseline probabilities
    prob_zs_first: float
    prob_zs_multi: float
    prob_icl_first: float
    prob_icl_multi: float

    logit_zs_first: float
    logit_icl_first: float
    prob_competitor_zs_first: float
    prob_competitor_icl_first: float
    logit_competitor_zs_first: float
    logit_competitor_icl_first: float

    # Patched probabilities
    prob_patched_first: float
    prob_patched_multi: float

    logit_patched_first: float
    prob_competitor_patched_first: float
    logit_competitor_patched_first: float

    # Controls / causal tests
    prob_rand_patched_first: float
    prob_rand_patched_multi: float
    prob_corrupt_patched_first: float
    prob_corrupt_patched_multi: float
    prob_null_patched_first: float
    prob_null_patched_multi: float
    prob_mean_pool_patched_first: float
    prob_mean_pool_patched_multi: float
    prob_ablated_first: float
    prob_ablated_multi: float

    # Derived metrics
    icl_lift_first: float
    icl_lift_multi: float
    pe_first: float
    pe_multi: float
    pe_random_first: float
    pe_random_multi: float
    pe_corrupt_first: float
    pe_corrupt_multi: float
    pe_null_first: float
    pe_null_multi: float
    pe_mean_pool_first: float
    pe_mean_pool_multi: float
    pe_logit_first: float
    ae_first: float  # Ablation effect (ICL→ablated); negative if necessary
    ae_multi: float
    icl_lift_logit_first: float
    delta_competitor_prob_patch: float = float("nan")
    delta_competitor_logit_patch: float = float("nan")
    pe_decoupled_first: float = float("nan")
    pe_decoupled_multi: float = float("nan")

    # New advanced controls
    prob_auto_scale_patched_first: float = float("nan")
    prob_auto_scale_patched_multi: float = float("nan")
    pe_auto_scale_first: float = float("nan")
    pe_auto_scale_multi: float = float("nan")

    prob_auto_shift_patched_first: float = float("nan")
    prob_auto_shift_patched_multi: float = float("nan")
    pe_auto_shift_first: float = float("nan")
    pe_auto_shift_multi: float = float("nan")

    prob_mean_residual_patched_first: float = float("nan")
    prob_mean_residual_patched_multi: float = float("nan")
    pe_mean_residual_first: float = float("nan")
    pe_mean_residual_multi: float = float("nan")

    prob_empty_prompt_patched_first: float = float("nan")
    prob_empty_prompt_patched_multi: float = float("nan")
    pe_empty_prompt_first: float = float("nan")
    pe_empty_prompt_multi: float = float("nan")

    prob_cross_task_patched_first: float = float("nan")
    prob_cross_task_patched_multi: float = float("nan")
    pe_cross_task_first: float = float("nan")
    pe_cross_task_multi: float = float("nan")

    prob_attn_head_ablated_first: float = float("nan")
    prob_attn_head_ablated_multi: float = float("nan")
    pe_attn_head_ablation_first: float = float("nan")
    pe_attn_head_ablation_multi: float = float("nan")
    attn_head_indices: str = ""
    attn_sink_heads_excluded: int = 0
    attn_sink_threshold: float = 0.80

    # Metadata
    n_target_tokens: int = 0
    layer: int = 0
    topk: int = 0
    seed: int = 0

    prob_decoupled_patched_first: float = float("nan")
    prob_decoupled_patched_multi: float = float("nan")

    # Rescue necessity test: patch ZS with features, then ablate those same features
    # If PE drops significantly, the specific features are necessary for rescue
    prob_patch_then_ablate_first: float = float("nan")
    prob_patch_then_ablate_multi: float = float("nan")
    pe_necessity_first: float = float(
        "nan"
    )  # prob_patched - prob_patch_then_ablate; positive = features necessary
    pe_necessity_multi: float = float("nan")
    patch_style: str = "sparse"
    feature_selection: str = "topk_abs_delta"
    feature_pooling: str = "last_token"
    patch_geometry: str = "raw"
    patch_geometry_clip_cap_used: float = float("nan")
    patch_geometry_sign_scale_used: float = float("nan")
    patch_geometry_fraction_clipped: float = float("nan")
    patch_position_mode: str = "source_last_subtoken"
    selected_feature_indices: str = ""
    selected_feature_magnitudes_raw: List[float] = field(default_factory=list)
    control_mode_requested: str = "default"
    control_mode_resolved: str = "default"
    control_mode_notes: str = ""
    # Additional controls (optional; may be absent in older result JSONs)
    prob_shuffle_patched_first: float = float("nan")
    prob_shuffle_patched_multi: float = float("nan")
    pe_shuffle_first: float = float("nan")
    pe_shuffle_multi: float = float("nan")
    # Gaussian noise control (optional): patch the same indices with N(mu,sigma)
    # noise matched to the selected ICL feature values.
    prob_gauss_patched_first: float = float("nan")
    prob_gauss_patched_multi: float = float("nan")
    pe_gauss_first: float = float("nan")
    pe_gauss_multi: float = float("nan")
    # Attention-only control (optional): add a matched-magnitude random vector
    # to the attention output at the same layer/position used for patching.
    prob_attention_patched_first: float = float("nan")
    prob_attention_patched_multi: float = float("nan")
    pe_attention_first: float = float("nan")
    pe_attention_multi: float = float("nan")
    # Structured attention-site comparison: add the decoded sparse patch vector
    # at attention output instead of MLP output.
    prob_attention_structured_first: float = float("nan")
    prob_attention_structured_multi: float = float("nan")
    pe_attention_structured_first: float = float("nan")
    pe_attention_structured_multi: float = float("nan")
    # Random-basis control (SRHT): select top-k coefficients in a randomized
    # orthogonal basis, then project back to feature space before patching.
    prob_basis_patched_first: float = float("nan")
    prob_basis_patched_multi: float = float("nan")
    pe_basis_first: float = float("nan")
    pe_basis_multi: float = float("nan")
    basis_control_method: str = "srht_topk"

    # Transcoder fidelity diagnostics
    active_features_icl: float = float("nan")
    active_features_zs: float = float("nan")
    feature_coverage_ratio: float = float("nan")
    selected_feature_fraction: float = float("nan")
    max_selected_feature_zscore: float = float("nan")
    selected_outlier_gt5sigma: bool = False
    reconstruction_cosine_icl: float = float("nan")
    reconstruction_rel_error_icl: float = float("nan")
    reconstruction_mse_icl: float = float("nan")
    feature_cosine_zs_icl: float = float("nan")
    feature_identity_jaccard_zs_icl: float = float("nan")
    decoded_patch_norm_ratio: float = float("nan")
    decoded_patch_norm: float = float("nan")
    latent_patch_norm_pre_geometry: float = float("nan")
    latent_patch_norm_post_geometry: float = float("nan")
    decoded_patch_norm_pre_geometry: float = float("nan")
    decoded_patch_norm_post_geometry: float = float("nan")
    mlp_in_icl_norm: float = float("nan")
    mlp_out_icl_norm: float = float("nan")
    mean_selected_feature_dla_target: float = float("nan")
    top_selected_feature_dla_target: float = float("nan")
    mean_selected_feature_dla_competitor: float = float("nan")
    dla_target_minus_competitor: float = float("nan")

    # Multi-token log-probability / NLL diagnostics (optional; may be absent in older JSONs)
    logprob_zs: float = float("nan")
    logprob_icl: float = float("nan")
    logprob_patched: float = float("nan")
    logprob_rand_patched: float = float("nan")
    logprob_corrupt_patched: float = float("nan")
    logprob_null_patched: float = float("nan")
    logprob_mean_pool_patched: float = float("nan")
    logprob_shuffle_patched: float = float("nan")
    logprob_gauss_patched: float = float("nan")
    logprob_attention_patched: float = float("nan")
    logprob_basis_patched: float = float("nan")
    logprob_ablated: float = float("nan")
    logprob_decoupled_patched: float = float("nan")
    logprob_auto_scale_patched: float = float("nan")
    logprob_auto_shift_patched: float = float("nan")
    logprob_mean_residual_patched: float = float("nan")
    logprob_empty_prompt_patched: float = float("nan")
    logprob_cross_task_patched: float = float("nan")
    logprob_attn_head_ablated: float = float("nan")

    nll_per_token_zs: float = float("nan")
    nll_per_token_icl: float = float("nan")
    nll_per_token_patched: float = float("nan")
    nll_per_token_rand_patched: float = float("nan")
    nll_per_token_corrupt_patched: float = float("nan")
    nll_per_token_null_patched: float = float("nan")
    nll_per_token_mean_pool_patched: float = float("nan")
    nll_per_token_shuffle_patched: float = float("nan")
    nll_per_token_gauss_patched: float = float("nan")
    nll_per_token_attention_patched: float = float("nan")
    nll_per_token_attention_structured: float = float("nan")
    nll_per_token_basis_patched: float = float("nan")
    nll_per_token_ablated: float = float("nan")
    nll_per_token_decoupled_patched: float = float("nan")
    nll_per_token_auto_scale_patched: float = float("nan")
    nll_per_token_auto_shift_patched: float = float("nan")
    nll_per_token_mean_residual_patched: float = float("nan")
    nll_per_token_empty_prompt_patched: float = float("nan")
    nll_per_token_cross_task_patched: float = float("nan")
    nll_per_token_attn_head_ablated: float = float("nan")
    nll_per_token_english_neutral_zs: float = float("nan")
    nll_per_token_english_neutral_patched: float = float("nan")
    nll_harm_english_neutral_patch: float = float("nan")

    # Tokens / Sweeping Info
    n_input_tokens_ood: int = 1
    icl_feature_position: int = -1
    zs_patch_position: int = -1
    rope_position_gap: int = 0
    bos_attention_zs_next_layer: float = float("nan")
    bos_attention_patched_next_layer: float = float("nan")
    delta_bos_attention_next_layer_patch: float = float("nan")

    # Decoupled control secondary metric
    prob_decoupled_target_first: float = float("nan")

    # Generation metrics
    gen_zs: str = ""
    gen_icl: str = ""
    gen_patched: str = ""
    script_zs: str = ""
    script_icl: str = ""
    script_patched: str = ""
    selector_reference_mode: str = "zs"
    query_span_found_zs: bool = False
    query_span_found_icl: bool = False
    query_span_found_selector_reference: bool = False
    local_window_exceeded_zs: bool = False
    local_window_exceeded_icl: bool = False
    local_window_exceeded_selector_reference: bool = False
    nll_pos1_zs: float = float("nan")
    nll_pos2_zs: float = float("nan")
    nll_pos3_zs: float = float("nan")
    nll_pos1_icl: float = float("nan")
    nll_pos2_icl: float = float("nan")
    nll_pos3_icl: float = float("nan")
    nll_pos1_patched: float = float("nan")
    nll_pos2_patched: float = float("nan")
    nll_pos3_patched: float = float("nan")

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_mlp_io_last_token(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract MLP (input, output) vectors at the last token position for a layer."""
    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    layers = get_model_layers(model)
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs_tuple, output):
        captured["mlp_in"] = inputs_tuple[0].detach()
        y = output[0] if isinstance(output, tuple) else output
        captured["mlp_out"] = y.detach()

    handle = layers[layer].mlp.register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    handle.remove()

    if "mlp_in" not in captured or "mlp_out" not in captured:
        raise RuntimeError("Failed to capture MLP input/output.")

    last_pos = int(captured["mlp_in"].shape[1] - 1)
    attn_mask = inputs.get("attention_mask", None) if hasattr(inputs, "get") else None
    if attn_mask is not None and torch.is_tensor(attn_mask):
        try:
            last_pos = int(torch.clamp(attn_mask[0].sum() - 1, min=0).item())
        except Exception:
            last_pos = int(captured["mlp_in"].shape[1] - 1)

    mlp_in = captured["mlp_in"][0, last_pos, :]
    mlp_out = captured["mlp_out"][0, last_pos, :]
    return mlp_in, mlp_out


def _extract_mlp_in_last_token(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: str,
) -> torch.Tensor:
    """Extract MLP input vector at the last token position for a given layer."""
    mlp_in, _ = _extract_mlp_io_last_token(model, tokenizer, prompt, layer, device)
    return mlp_in


def get_transcoder_features_last_token(
    model,
    tokenizer,
    transcoder: Transcoder,
    prompt: str,
    *,
    layer: int,
    device: str,
) -> torch.Tensor:
    mlp_in = _extract_mlp_in_last_token(model, tokenizer, prompt, layer, device)
    return transcoder.encode(mlp_in.unsqueeze(0)).squeeze(0)


def _find_last_subsequence(haystack: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    """Find the last contiguous occurrence of needle in haystack."""
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start : start + n] == needle:
            return start, start + n
    return None


def _find_query_span_in_rendered_prompt(
    tokenizer,
    rendered_prompt: str,
    query_token: str,
) -> Optional[Tuple[int, int]]:
    """
    Locate the last contiguous token span for the query token inside a rendered prompt.

    This mirrors the main patching path: query-span discovery happens after chat-template
    rendering and uses exact token-subsequence matching.
    """
    query_ids = tokenizer.encode(query_token, add_special_tokens=False)
    if not query_ids:
        return None
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()
    return _find_last_subsequence(prompt_ids, query_ids)


def _extract_mlp_in_positions_from_input_ids(
    model,
    input_ids: torch.Tensor,
    layer: int,
    positions: List[int],
) -> torch.Tensor:
    """
    Extract MLP input vectors at specific token positions for a layer.

    Returns shape [len(positions), d_model] in model/device dtype.
    """
    if not positions:
        raise ValueError("positions must be non-empty")

    layers = get_model_layers(model)
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs_tuple, output):
        captured["mlp_in"] = inputs_tuple[0].detach()

    handle = layers[layer].mlp.register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()

    if "mlp_in" not in captured:
        raise RuntimeError("Failed to capture MLP input sequence.")

    seq = captured["mlp_in"][0]  # [seq, d]
    seq_len = int(seq.shape[0])
    pos = [int(p) for p in positions if 0 <= int(p) < seq_len]
    if not pos:
        raise ValueError("No valid positions in sequence for MLP extraction.")
    idx = torch.tensor(pos, device=seq.device, dtype=torch.long)
    return torch.index_select(seq, 0, idx)


def _extract_mlp_io_at_position_from_input_ids(
    model,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (MLP input, MLP output) vectors at an absolute token position.
    """
    layers = get_model_layers(model)
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs_tuple, output):
        captured["mlp_in"] = inputs_tuple[0].detach()
        y = output[0] if isinstance(output, tuple) else output
        captured["mlp_out"] = y.detach()

    handle = layers[layer].mlp.register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()

    if "mlp_in" not in captured or "mlp_out" not in captured:
        raise RuntimeError("Failed to capture MLP input/output sequence.")

    seq_len = int(captured["mlp_in"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["mlp_in"][0, pos, :], captured["mlp_out"][0, pos, :]


def _topk_indices(feats: torch.Tensor, k: int) -> torch.Tensor:
    if feats.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    k = min(k, feats.numel())
    return torch.topk(torch.abs(feats), k).indices


def _bottomk_indices_abs(feats: torch.Tensor, k: int) -> torch.Tensor:
    if feats.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    scores = torch.abs(feats).detach()
    # Prefer non-zero entries so bottom-k isn't dominated by all-zero inactive features.
    nz = torch.nonzero(scores > 0, as_tuple=False).flatten()
    pool = (
        nz
        if nz.numel() >= int(k)
        else torch.arange(scores.numel(), device=scores.device)
    )

    k = max(0, min(int(k), int(pool.numel())))
    if k == 0:
        return torch.tensor([], dtype=torch.long)

    pool_scores = scores[pool]
    # Largest of (-score) == smallest score.
    local_idx = torch.topk(-pool_scores, k).indices
    return pool[local_idx].to(dtype=torch.long)


def _build_patch_vector(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    patch_feats = torch.zeros_like(feats)
    if idx.numel() > 0:
        patch_feats[idx] = feats[idx]
    return patch_feats


def _build_ablation_vector(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    ablated = feats.clone()
    if idx.numel() > 0:
        ablated[idx] = 0.0
    return ablated


def _apply_patch_geometry(
    patch_feats: torch.Tensor,
    idx: torch.Tensor,
    *,
    mode: str,
    clip_cap: Optional[float] = None,
    sign_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply a geometry transform to the selected latent coordinates only.

    Modes:
      - raw: keep selected coordinates unchanged
      - clipped: clamp selected coordinates to a robust amplitude cap
      - normalized_sign / sign_normalized: discard per-coordinate magnitude,
        keep sign, and restore a shared bounded scale
    """
    resolved = str(mode or "raw").strip().lower()
    if resolved == "normalized_sign":
        resolved = "sign_normalized"
    if resolved == "raw" or idx.numel() == 0:
        return patch_feats

    if resolved not in {"clipped", "sign_normalized"}:
        raise ValueError(
            f"Unknown patch_geometry='{mode}'. Expected one of: raw, clipped, normalized_sign."
        )

    out = patch_feats.clone()
    vals = out[idx].detach().float()
    if vals.numel() == 0:
        return out

    abs_vals = torch.abs(vals)
    scale = float(torch.mean(abs_vals).item()) if abs_vals.numel() else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    if resolved == "clipped":
        cap = float(clip_cap) if clip_cap is not None else float(torch.median(abs_vals).item()) if abs_vals.numel() else scale
        if not np.isfinite(cap) or cap <= 0.0:
            cap = scale
        out[idx] = torch.clamp(out[idx], min=-cap, max=cap)
        return out

    # sign_normalized
    if sign_scale is not None and np.isfinite(float(sign_scale)) and float(sign_scale) > 0.0:
        scale = float(sign_scale)
    signs = torch.sign(out[idx])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    out[idx] = signs * scale
    return out


def estimate_patch_geometry_params(
    model,
    tokenizer,
    transcoder,
    layer: int,
    selection_words: List[Dict[str, Any]],
    *,
    icl_examples: List[Dict[str, Any]],
    topk: int,
    device: str,
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    patch_style: str = "sparse",
    feature_selection: str = "topk_abs_delta",
    selector_reference_mode: str = "corrupt_icl",
    prompt_variant: str = "canonical",
    patch_position_mode: str = "source_last_subtoken",
) -> Dict[str, float]:
    """
    Estimate geometry calibration stats from the selection split only.

    This mirrors the existing selector semantics but stops before patching.
    The returned values are used by Stage A.5 for bounded geometry variants.
    """

    def _resolve_prompt_position(
        prompt_text: str,
        query_token: str,
        *,
        target_ids: Optional[List[int]] = None,
    ) -> Tuple[str, torch.Tensor, int]:
        rendered = apply_chat_template(tokenizer, prompt_text)
        encoded = tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        prompt_len = int(input_ids.shape[1])
        if patch_position_mode == "target_pos1":
            if not target_ids:
                raise ValueError("target_pos1 geometry estimation requires non-empty target_ids")
            tf_ids = torch.tensor([target_ids], device=device, dtype=torch.long)
            full_input_ids = torch.cat([input_ids, tf_ids], dim=1)
            return rendered, full_input_ids, prompt_len

        query_ids = tokenizer.encode(query_token, add_special_tokens=False)
        span = _find_query_span_in_rendered_prompt(tokenizer, rendered, query_token)
        if span is None:
            raise ValueError("Selection-set geometry estimation requires a valid query span")
        patch_pos = int(span[1] - 1)
        return rendered, input_ids, patch_pos

    feature_selection_base = str(feature_selection or "topk_abs_delta").strip().lower()
    selector_reference_mode = str(selector_reference_mode or "corrupt_icl").strip().lower()

    selected_abs: List[float] = []
    for word in selection_words:
        query_token = str(word["ood"])
        target_text = str(word["hindi"])
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)

        zs_prompt = build_task_prompt(
            query_token,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        )
        icl_prompt = build_task_prompt(
            query_token,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        )
        corrupt_prompt = build_corrupted_icl_prompt(
            query_token,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            seed=0,
        )

        _zs_text, zs_ids, zs_pos = _resolve_prompt_position(zs_prompt, query_token, target_ids=target_ids)
        _icl_text, icl_ids, icl_pos = _resolve_prompt_position(icl_prompt, query_token, target_ids=target_ids)
        mlp_in_icl, _ = _extract_mlp_io_at_position_from_input_ids(model, icl_ids, layer, icl_pos)
        icl_feats = transcoder.encode(mlp_in_icl.unsqueeze(0)).squeeze(0)

        zs_feats = None
        if "_delta" in feature_selection_base or patch_style == "substitute":
            mlp_in_zs, _ = _extract_mlp_io_at_position_from_input_ids(model, zs_ids, layer, zs_pos)
            zs_feats = transcoder.encode(mlp_in_zs.unsqueeze(0)).squeeze(0)

        selector_ref_feats = zs_feats
        if selector_reference_mode == "corrupt_icl":
            _corr_text, corr_ids, corr_pos = _resolve_prompt_position(corrupt_prompt, query_token, target_ids=target_ids)
            mlp_in_corr, _ = _extract_mlp_io_at_position_from_input_ids(model, corr_ids, layer, corr_pos)
            selector_ref_feats = transcoder.encode(mlp_in_corr.unsqueeze(0)).squeeze(0)

        src_feats = icl_feats
        if feature_selection_base in {"topk_abs_delta", "bottomk_abs_delta"}:
            if selector_ref_feats is None:
                raise RuntimeError("Delta selection requires a selector reference feature vector")
            score_src = torch.abs(src_feats - selector_ref_feats)
        elif feature_selection_base in {"topk_abs_icl", "bottomk_abs_icl"}:
            score_src = torch.abs(src_feats)
        else:
            raise RuntimeError(f"Unhandled feature_selection='{feature_selection_base}'")

        descending = feature_selection_base.startswith("topk")
        idx = torch.argsort(score_src, descending=descending)[: int(topk)]
        if idx.numel() == 0:
            continue
        selected_abs.extend(torch.abs(src_feats[idx].detach().float()).cpu().tolist())

    arr = np.asarray(selected_abs, dtype=np.float64)
    if arr.size == 0:
        return {"clip_cap": 1.0, "sign_scale": 1.0, "n_selected_values": 0}
    clip_cap = float(np.nanpercentile(arr, 99))
    if not np.isfinite(clip_cap) or clip_cap <= 0.0:
        clip_cap = float(np.nanmedian(arr))
    if not np.isfinite(clip_cap) or clip_cap <= 0.0:
        clip_cap = 1.0
    sign_scale = float(np.nanmedian(np.clip(arr, a_min=None, a_max=clip_cap)))
    if not np.isfinite(sign_scale) or sign_scale <= 0.0:
        sign_scale = clip_cap
    return {
        "clip_cap": float(clip_cap),
        "sign_scale": float(sign_scale),
        "n_selected_values": int(arr.size),
    }


def _get_unembedding_weight(model) -> Optional[torch.Tensor]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        return lm_head.weight
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        emb = get_out()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    return None


def _compute_selected_feature_dla(
    *,
    transcoder,
    model,
    selected_indices: List[int],
    target_id: int,
    competitor_id: int = -1,
) -> Dict[str, float]:
    out = {
        "mean_selected_feature_dla_target": float("nan"),
        "top_selected_feature_dla_target": float("nan"),
        "mean_selected_feature_dla_competitor": float("nan"),
        "dla_target_minus_competitor": float("nan"),
    }
    if int(target_id) < 0 or not selected_indices:
        return out

    W_dec = getattr(transcoder, "W_dec", None)
    W_u = _get_unembedding_weight(model)
    if W_dec is None or W_u is None:
        return out

    try:
        dec = W_dec.detach().float()
        unemb = W_u.detach().float()
        if dec.ndim != 2 or unemb.ndim != 2:
            return out

        d_model = int(unemb.shape[1])
        if int(dec.shape[1]) != d_model and int(dec.shape[0]) == d_model:
            dec = dec.t().contiguous()
        if int(dec.shape[1]) != d_model:
            return out

        idx_t = torch.tensor(selected_indices, device=dec.device, dtype=torch.long)
        idx_t = idx_t[(idx_t >= 0) & (idx_t < int(dec.shape[0]))]
        if idx_t.numel() <= 0:
            return out

        feature_dirs = torch.index_select(dec, 0, idx_t)
        target_dir = unemb[int(target_id)].to(device=dec.device, dtype=torch.float32)
        dla_target = torch.mv(feature_dirs, target_dir)

        out["mean_selected_feature_dla_target"] = float(torch.mean(dla_target).item())
        out["top_selected_feature_dla_target"] = float(torch.max(dla_target).item())

        if int(competitor_id) >= 0 and int(competitor_id) < int(unemb.shape[0]):
            competitor_dir = unemb[int(competitor_id)].to(
                device=dec.device, dtype=torch.float32
            )
            dla_comp = torch.mv(feature_dirs, competitor_dir)
            out["mean_selected_feature_dla_competitor"] = float(
                torch.mean(dla_comp).item()
            )
            out["dla_target_minus_competitor"] = float(
                out["mean_selected_feature_dla_target"]
                - out["mean_selected_feature_dla_competitor"]
            )
    except Exception:
        return out

    return out


def register_transcoder_feature_patch_hook(
    model,
    transcoder: Transcoder,
    layer: int,
    patch_feats: torch.Tensor,
    *,
    patch_position: Optional[int] = None,
    residual_override: Optional[torch.Tensor] = None,
    target_output_norm: Optional[float] = None,
):
    """
    Patch by swapping *feature contribution* while preserving the residual part
    of the MLP output that is not captured by the transcoder reconstruction.
    If residual_override is provided, replace the residual as well.

    Args:
        model: The language model.
        transcoder: Transcoder/SAE for the target layer.
        layer: Layer index to patch.
        patch_feats: Feature vector to patch in.
        patch_position: If specified, patch at this absolute position instead of -1.
                       This is CRITICAL for teacher-forced multi-token computation
                       where context length grows but we want to patch the original
                       prompt's last token position.
        residual_override: Optional tensor to replace the computed residual.
        target_output_norm: Optional L2 norm target for the patched MLP output
                           vector (RMSNorm-stability control).
    """
    patch_feats = patch_feats.detach()
    if residual_override is not None:
        residual_override = residual_override.detach()
    layers = get_model_layers(model)

    def hook(module, inputs_tuple, output):
        mlp_in = inputs_tuple[0]
        y = output[0] if isinstance(output, tuple) else output

        # Determine which position to patch
        seq_len = mlp_in.shape[1]
        if patch_position is not None:
            # Fixed position mode: only patch if we're at or past the target position
            if patch_position >= seq_len:
                # Position not yet in sequence, don't patch
                return output
            pos = patch_position
        else:
            # Default: patch the last position (backward compatible)
            pos = seq_len - 1

        x_pos = mlp_in[:, pos, :]
        y_pos = y[:, pos, :]

        # Keep transcoder math in FP32 to reduce BF16 rounding drift.
        x_pos_f = x_pos.float()
        y_pos_f = y_pos.float()
        cur_feats = transcoder.encode(x_pos_f)
        cur_contrib = transcoder.decode(cur_feats).float()

        if residual_override is not None:
            residual = residual_override.to(device=y.device, dtype=torch.float32)
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)
        else:
            residual = y_pos_f - cur_contrib

        pf = patch_feats.to(device=y.device, dtype=torch.float32).unsqueeze(0)
        new_contrib = transcoder.decode(pf).float()

        patched_pos = residual + new_contrib
        try:
            target_norm = float(target_output_norm) if target_output_norm is not None else float("nan")
        except Exception:
            target_norm = float("nan")
        if target_norm == target_norm and target_norm > 0.0:
            cur_norm = torch.norm(patched_pos, dim=1, keepdim=True)
            scale = float(target_norm) / torch.clamp(cur_norm, min=1e-8)
            patched_pos = patched_pos * scale

        y_new = y.clone()
        y_new[:, pos, :] = patched_pos.to(dtype=y.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[layer].mlp.register_forward_hook(hook)


def register_transcoder_feature_ablation_hook(
    model,
    transcoder: Transcoder,
    layer: int,
    ablate_idx: torch.Tensor,
    *,
    ablate_position: Optional[int] = None,
):
    """
    Ablate (set to zero) a subset of transcoder features *in the current forward pass*,
    while preserving the residual part of the MLP output that is not captured by the
    transcoder reconstruction.

    This is useful for "support layer" tests where you want to see whether some downstream
    layer's features are necessary for an upstream patch effect.

    Args:
        model: The language model.
        transcoder: Transcoder/SAE for the target layer.
        layer: Layer index to ablate.
        ablate_idx: Tensor of feature indices to zero out.
        ablate_position: If specified, ablate at this absolute position instead of -1.
                        Critical for teacher-forced multi-token computation.
    """
    layers = get_model_layers(model)
    ablate_idx = ablate_idx.detach().to(dtype=torch.long)

    def hook(module, inputs_tuple, output):
        mlp_in = inputs_tuple[0]
        y = output[0] if isinstance(output, tuple) else output

        # Determine which position to ablate
        seq_len = mlp_in.shape[1]
        if ablate_position is not None:
            if ablate_position >= seq_len:
                return output
            pos = ablate_position
        else:
            pos = seq_len - 1

        x_pos = mlp_in[:, pos, :]
        y_pos = y[:, pos, :]

        x_pos_f = x_pos.float()
        y_pos_f = y_pos.float()
        cur_feats = transcoder.encode(x_pos_f)
        cur_contrib = transcoder.decode(cur_feats).float()
        residual = y_pos_f - cur_contrib

        idx = ablate_idx.to(device=cur_feats.device)
        cur_feats_mod = cur_feats.clone()
        if idx.numel() > 0:
            cur_feats_mod[:, idx] = 0.0

        new_contrib = transcoder.decode(cur_feats_mod).float()

        y_new = y.clone()
        y_new[:, pos, :] = (residual + new_contrib).to(dtype=y.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[layer].mlp.register_forward_hook(hook)


def _make_magnitude_matched_attention_vector(
    reference_vector: torch.Tensor,
    *,
    seed: int,
    layer: int,
    topk: int,
    word_key: str,
) -> torch.Tensor:
    """
    Build a deterministic random vector with the same L2 norm as reference_vector.

    This is the control used for attention-only interventions: matched magnitude,
    random direction.
    """
    ref = reference_vector.detach()
    if ref.ndim != 1:
        ref = ref.reshape(-1)
    ref_norm = float(torch.norm(ref).item())
    if not np.isfinite(ref_norm) or ref_norm <= 0:
        return torch.zeros_like(ref)

    msg = f"attn::{seed}::{layer}::{topk}::{word_key}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    rand = rng.normal(loc=0.0, scale=1.0, size=(int(ref.numel()),)).astype(np.float32)
    rand_t = torch.tensor(rand, dtype=ref.dtype, device=ref.device)
    rand_norm = float(torch.norm(rand_t).item())
    if not np.isfinite(rand_norm) or rand_norm <= 0:
        return torch.zeros_like(ref)
    return rand_t * (ref_norm / rand_norm)


def _next_power_of_two(n: int) -> int:
    n_int = max(1, int(n))
    return 1 << (n_int - 1).bit_length()


def _fwht_1d(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform for 1D tensors (length must be power-of-two)."""
    if x.ndim != 1:
        raise ValueError("_fwht_1d expects a 1D tensor")
    n = int(x.numel())
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"_fwht_1d length must be power-of-two, got {n}")

    y = x.clone()
    h = 1
    while h < n:
        y2 = y.view(-1, h * 2)
        a = y2[:, :h].clone()
        b = y2[:, h : 2 * h].clone()
        y2[:, :h] = a + b
        y2[:, h : 2 * h] = a - b
        y = y2.reshape(n)
        h *= 2
    return y


def _build_srht_basis_patch_feats(
    src_feats: torch.Tensor,
    *,
    topk: int,
    seed: int,
    layer: int,
    word_key: str,
) -> torch.Tensor:
    """
    Build a random-basis top-k control vector via a structured random Hadamard map.

    This preserves total dimensionality while selecting top-k coefficients in a
    random orthogonal basis, then projects back to original feature coordinates.
    """
    d = int(src_feats.numel())
    if d <= 0:
        return torch.zeros_like(src_feats)

    n = _next_power_of_two(d)
    work = src_feats.detach().float()
    x = torch.zeros(n, dtype=work.dtype, device=work.device)
    x[:d] = work

    msg = f"basis_srht::{seed}::{layer}::{topk}::{word_key}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)

    signs_np = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    perm_np = rng.permutation(n).astype(np.int64)
    signs = torch.tensor(signs_np, dtype=x.dtype, device=x.device)
    perm = torch.tensor(perm_np, dtype=torch.long, device=x.device)

    # Randomized orthogonal pre-transform: v = D * P * x
    v = signs * x[perm]

    # Orthonormal Hadamard coefficients.
    y = _fwht_1d(v) / float(np.sqrt(float(n)))

    k = max(1, min(int(topk), int(y.numel())))
    idx = torch.topk(torch.abs(y), k).indices
    y_sparse = torch.zeros_like(y)
    y_sparse[idx] = y[idx]

    # Inverse orthonormal transform + inverse randomization.
    v_hat = _fwht_1d(y_sparse) / float(np.sqrt(float(n)))
    x_hat = torch.zeros_like(v_hat)
    x_hat[perm] = signs * v_hat

    return x_hat[:d].to(dtype=src_feats.dtype)


def register_attention_output_patch_hook(
    model,
    layer: int,
    patch_vector: torch.Tensor,
    *,
    patch_position: Optional[int] = None,
):
    """
    Add a vector intervention to the attention output at one sequence position.

    This is used as an attention-only matched-magnitude control.
    """
    layers = get_model_layers(model)
    patch_vector = patch_vector.detach()

    # Gemma-style decoder layers expose self_attn. Keep a tiny fallback to
    # support naming differences in adjacent architectures.
    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None:
        raise AttributeError(
            f"Layer {layer} has no self-attention module for attention control."
        )

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output

        seq_len = int(y.shape[1])
        if patch_position is not None:
            if patch_position >= seq_len:
                return output
            pos = int(patch_position)
        else:
            pos = seq_len - 1

        pv = patch_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if pv.shape[1] != y.shape[2]:
            return output

        y_new = y.clone()
        y_new[:, pos, :] = y_new[:, pos, :] + pv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return attn_module.register_forward_hook(hook)


def register_attention_output_replace_hook(
    model,
    layer: int,
    patch_vector: torch.Tensor,
    *,
    patch_position: Optional[int] = None,
):
    """
    Replace the full attention output vector at one sequence position.

    This is the exact donor-replacement counterpart to
    `register_dense_mlp_output_patch_hook`, used for component-localization
    experiments where attention and MLP outputs must use matched semantics.
    """
    layers = get_model_layers(model)
    patch_vector = patch_vector.detach()

    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None:
        raise AttributeError(
            f"Layer {layer} has no self-attention module for attention replacement."
        )

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output

        seq_len = int(y.shape[1])
        if patch_position is not None:
            if patch_position >= seq_len:
                return output
            pos = int(patch_position)
        else:
            pos = seq_len - 1

        pv = patch_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if pv.shape[1] != y.shape[2]:
            return output

        y_new = y.clone()
        y_new[:, pos, :] = pv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return attn_module.register_forward_hook(hook)



def register_dense_mlp_output_patch_hook(
    model,
    layer: int,
    patch_vector: torch.Tensor,
    *,
    patch_position: Optional[int] = None,
):
    """
    Replace the full MLP output vector at one sequence position.

    This is the dense upper-bound check for any sparse/transcoder-based
    intervention at the same layer/site family. Unlike the sparse patch hook,
    this does not preserve a residual decomposition; it swaps the complete
    `mlp_out` vector at the chosen position with a dense reference vector.
    """
    layers = get_model_layers(model)
    patch_vector = patch_vector.detach()

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output

        seq_len = int(y.shape[1])
        if patch_position is not None:
            if patch_position >= seq_len:
                return output
            pos = int(patch_position)
        else:
            pos = seq_len - 1

        pv = patch_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if pv.shape[1] != y.shape[2]:
            return output

        y_new = y.clone()
        y_new[:, pos, :] = pv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[layer].mlp.register_forward_hook(hook)


def _extract_attention_output_at_position_from_input_ids(
    model,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    """
    Extract the self-attention module output vector at an absolute token position.
    """
    layers = get_model_layers(model)
    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None:
        raise AttributeError(
            f"Layer {layer} has no self-attention module for attention extraction."
        )

    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["attn_out"] = y.detach()

    handle = attn_module.register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()

    if "attn_out" not in captured:
        raise RuntimeError("Failed to capture attention output sequence.")

    seq_len = int(captured["attn_out"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["attn_out"][0, pos, :]



def register_layer_output_replace_hook(
    model,
    layer: int,
    patch_vector: torch.Tensor,
    *,
    patch_position: Optional[int] = None,
):
    """
    Replace the full decoder-layer output vector at one sequence position.

    This is the joint-state bridge between pre-hook input interventions and
    component-specific output interventions.
    """
    layers = get_model_layers(model)
    patch_vector = patch_vector.detach()

    def hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(y) or y.ndim != 3:
            return output

        seq_len = int(y.shape[1])
        if patch_position is not None:
            if patch_position >= seq_len:
                return output
            pos = int(patch_position)
        else:
            pos = seq_len - 1

        pv = patch_vector.to(device=y.device, dtype=y.dtype).view(1, -1)
        if pv.shape[1] != y.shape[2]:
            return output

        y_new = y.clone()
        y_new[:, pos, :] = pv
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[layer].register_forward_hook(hook)



def _extract_layer_output_at_position_from_input_ids(
    model,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    """
    Extract the decoder-layer output vector at an absolute token position.
    """
    layers = get_model_layers(model)
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs_tuple, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["layer_out"] = y.detach()

    handle = layers[layer].register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()

    if "layer_out" not in captured:
        raise RuntimeError("Failed to capture layer output sequence.")

    seq_len = int(captured["layer_out"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["layer_out"][0, pos, :]


def _extract_attn_o_proj_in_last_token(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: str,
) -> torch.Tensor:
    """Capture self-attention o_proj input at the last token for one layer."""
    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    layers = get_model_layers(model)
    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None or not hasattr(attn_module, "o_proj"):
        raise AttributeError(f"Layer {layer} attention has no o_proj for head extraction.")

    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(module, inputs_tuple):
        x = inputs_tuple[0]
        if torch.is_tensor(x) and x.ndim == 3:
            captured["attn_o_in"] = x.detach()

    handle = attn_module.o_proj.register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    handle.remove()

    if "attn_o_in" not in captured:
        raise RuntimeError("Failed to capture attention o_proj input.")

    return captured["attn_o_in"][0, -1, :].to(device)


def _extract_attention_bos_fraction_last_query(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    device: str,
) -> Optional[torch.Tensor]:
    """Return per-head attention mass on BOS for the last query token."""
    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False, output_attentions=True)
    except Exception:
        return None

    attentions = getattr(out, "attentions", None)
    if attentions is None and isinstance(out, tuple):
        for item in out:
            if isinstance(item, (list, tuple)) and item and torch.is_tensor(item[0]) and item[0].ndim == 4:
                attentions = item
                break
    if attentions is None or layer >= len(attentions):
        return None

    attn = attentions[layer]
    if not torch.is_tensor(attn) or attn.ndim != 4:
        return None
    # [batch, heads, q, k] -> last query, BOS key index 0
    try:
        bos = attn[0, :, -1, 0].detach().float()
    except Exception:
        return None
    return bos


def _infer_attention_num_heads(model, attn_module) -> int:
    candidates = [
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(getattr(attn_module, "config", None), "num_attention_heads", None),
        getattr(getattr(model, "config", None), "num_attention_heads", None),
    ]
    for candidate in candidates:
        try:
            out = int(candidate)
        except Exception:
            out = 0
        if out > 0:
            return out
    return 0


def _select_top_attention_heads_by_delta(
    model,
    tokenizer,
    layer: int,
    icl_prompt: str,
    zs_prompt: str,
    device: str,
    topk_heads: int = 3,
    sink_bos_threshold: float = 0.80,
) -> Tuple[List[int], int]:
    """Select top delta heads and exclude BOS-sink heads when possible."""
    layers = get_model_layers(model)
    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None:
        return [], 0

    num_heads = _infer_attention_num_heads(model, attn_module)
    if num_heads <= 0:
        return [], 0

    v_icl = _extract_attn_o_proj_in_last_token(model, tokenizer, icl_prompt, layer, device)
    v_zs = _extract_attn_o_proj_in_last_token(model, tokenizer, zs_prompt, layer, device)
    d = int(v_icl.numel())
    if d <= 0 or d % num_heads != 0:
        return [], 0

    head_dim = d // num_heads
    delta = (v_icl - v_zs).view(num_heads, head_dim)
    scores = torch.mean(torch.abs(delta), dim=1)
    ordered = torch.argsort(scores, descending=True).detach().cpu().tolist()

    sink_mask: Optional[np.ndarray] = None
    n_sink_excluded = 0
    if sink_bos_threshold > 0.0:
        bos = _extract_attention_bos_fraction_last_query(
            model=model,
            tokenizer=tokenizer,
            prompt=icl_prompt,
            layer=layer,
            device=device,
        )
        if bos is not None and int(bos.numel()) == int(num_heads):
            sink_mask = (bos.detach().cpu().numpy() < float(sink_bos_threshold))
            n_sink_excluded = int(np.sum(~sink_mask))

    k = max(1, min(int(topk_heads), int(num_heads)))
    selected: List[int] = []
    for h in ordered:
        hi = int(h)
        if sink_mask is not None and (hi < 0 or hi >= len(sink_mask) or not bool(sink_mask[hi])):
            continue
        selected.append(hi)
        if len(selected) >= k:
            break

    if not selected:
        # Fallback if sink filtering removed everything or was unavailable.
        selected = [int(i) for i in ordered[:k]]
    return selected, n_sink_excluded


def register_attention_head_ablation_hook(
    model,
    layer: int,
    head_indices: List[int],
    *,
    ablate_position: Optional[int] = None,
):
    """
    Zero selected attention-head channels at o_proj input for a given sequence position.

    This approximates per-head ablation while keeping the rest of attention intact.
    """
    layers = get_model_layers(model)
    attn_module = getattr(layers[layer], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[layer], "self_attention", None)
    if attn_module is None or not hasattr(attn_module, "o_proj"):
        raise AttributeError(f"Layer {layer} attention has no o_proj for head ablation.")

    heads = [int(h) for h in head_indices if int(h) >= 0]
    if not heads:
        class _Noop:
            def remove(self):
                return None
        return _Noop()

    num_heads = _infer_attention_num_heads(model, attn_module)
    if num_heads <= 0:
        raise ValueError("Could not infer num_heads for attention ablation.")

    def pre_hook(module, inputs_tuple):
        if not inputs_tuple:
            return None
        x = inputs_tuple[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None

        seq_len = int(x.shape[1])
        if ablate_position is not None:
            if ablate_position >= seq_len:
                return None
            pos = int(ablate_position)
        else:
            pos = seq_len - 1

        d = int(x.shape[2])
        if d % num_heads != 0:
            return None
        head_dim = d // num_heads

        x_new = x.clone()
        for h in heads:
            if h >= num_heads:
                continue
            s = h * head_dim
            e = (h + 1) * head_dim
            x_new[:, pos, s:e] = 0.0

        if len(inputs_tuple) == 1:
            return (x_new,)
        return (x_new,) + tuple(inputs_tuple[1:])

    return attn_module.o_proj.register_forward_pre_hook(pre_hook)


def _teacher_forced_joint_prob_from_input_ids(
    model,
    input_ids: torch.Tensor,
    target_ids: List[int],
    device: str,
) -> float:
    """Compute teacher-forced joint probability for target_ids given input_ids."""
    metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=input_ids,
        target_ids=target_ids,
        target_id=-1,
        device=device,
    )
    logprob = float(metrics["joint_logprob"])
    if not np.isfinite(logprob):
        return float("nan")
    return float(np.exp(logprob))


def _teacher_forced_joint_logprob_from_input_ids(
    model,
    input_ids: torch.Tensor,
    target_ids: List[int],
    device: str,
) -> float:
    """Compute teacher-forced joint log-probability for target_ids given input_ids."""
    metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=input_ids,
        target_ids=target_ids,
        target_id=-1,
        device=device,
    )
    return float(metrics["joint_logprob"])


def _teacher_forced_metrics_from_input_ids(
    model,
    input_ids: torch.Tensor,
    target_ids: List[int],
    target_id: int,
    device: str,
    competitor_id: int = -1,
) -> Dict[str, float]:
    """
    Compute first-token and multi-token teacher-forced metrics in ONE forward pass.

    Returns:
      - first_prob, first_logit, first_logit_std (NaN if target_id < 0)
      - joint_logprob
      - n_tokens
      - target_pos{1,2,3}_nll for early target-position diagnostics
    """
    if not target_ids:
        return {
            "first_prob": float("nan"),
            "first_logit": float("nan"),
            "first_logit_std": float("nan"),
            "competitor_first_prob": float("nan"),
            "competitor_first_logit": float("nan"),
            "joint_logprob": float("nan"),
            "n_tokens": 0.0,
            "target_pos1_nll": float("nan"),
            "target_pos2_nll": float("nan"),
            "target_pos3_nll": float("nan"),
        }

    tgt = torch.tensor(target_ids, device=device, dtype=input_ids.dtype).unsqueeze(0)
    full_input_ids = torch.cat([input_ids, tgt], dim=1)
    start = int(input_ids.shape[1] - 1)
    end = int(start + len(target_ids))

    with torch.inference_mode():
        outputs = model(input_ids=full_input_ids, use_cache=False)

    logits = outputs.logits[0, start:end, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    tgt_idx = tgt[0].to(dtype=torch.long)
    row_idx = torch.arange(tgt_idx.shape[0], device=logits.device)
    token_logprobs = log_probs[row_idx, tgt_idx]
    joint_logprob = float(token_logprobs.sum().item())
    token_nlls = (-token_logprobs).detach().cpu().tolist()

    first_prob = float("nan")
    first_logit = float("nan")
    first_logit_std = float("nan")
    competitor_first_prob = float("nan")
    competitor_first_logit = float("nan")
    first_logits = logits[0]
    if int(target_id) >= 0:
        first_logit = float(first_logits[int(target_id)].item())
        first_prob = float(torch.softmax(first_logits, dim=-1)[int(target_id)].item())
        first_logit_std = float(torch.std(first_logits, unbiased=False).item())
    if int(competitor_id) >= 0:
        competitor_first_logit = float(first_logits[int(competitor_id)].item())
        competitor_first_prob = float(
            torch.softmax(first_logits, dim=-1)[int(competitor_id)].item()
        )

    return {
        "first_prob": first_prob,
        "first_logit": first_logit,
        "first_logit_std": first_logit_std,
        "competitor_first_prob": competitor_first_prob,
        "competitor_first_logit": competitor_first_logit,
        "joint_logprob": joint_logprob,
        "n_tokens": float(len(target_ids)),
        "target_pos1_nll": float(token_nlls[0]) if len(token_nlls) >= 1 else float("nan"),
        "target_pos2_nll": float(token_nlls[1]) if len(token_nlls) >= 2 else float("nan"),
        "target_pos3_nll": float(token_nlls[2]) if len(token_nlls) >= 3 else float("nan"),
    }


def _classify_script(text: str) -> str:
    """Classify the dominant script of a given text based on Unicode blocks."""
    if not text:
        return "Unknown"
    
    script_counts = {}
    
    # Mapping of basic unicode ranges to scripts (simplified for speed)
    blocks = [
        ("Latin", 0x0000, 0x024F),
        ("Devanagari", 0x0900, 0x097F),
        ("Bengali", 0x0980, 0x09FF),
        ("Gurmukhi", 0x0A00, 0x0A7F),
        ("Gujarati", 0x0A80, 0x0AFF),
        ("Oriya", 0x0B00, 0x0B7F),
        ("Tamil", 0x0B80, 0x0BFF),
        ("Telugu", 0x0C00, 0x0C7F),
        ("Kannada", 0x0C80, 0x0CFF),
        ("Malayalam", 0x0D00, 0x0D7F),
        ("Sinhala", 0x0D80, 0x0DFF),
        ("Thai", 0x0E00, 0x0E7F),
        ("Arabic", 0x0600, 0x06FF),
        ("Cyrillic", 0x0400, 0x04FF),
        ("Hebrew", 0x0590, 0x05FF),
        ("Georgian", 0x10A0, 0x10FF),
    ]
    
    for char in text:
        if char.isspace() or char.isascii() and not char.isalpha():
            continue
        cp = ord(char)
        found = False
        for name, start, end in blocks:
            if start <= cp <= end:
                script_counts[name] = script_counts.get(name, 0) + 1
                found = True
                break
        if not found:
            script_counts["Other"] = script_counts.get("Other", 0) + 1
            
    if not script_counts:
        return "Unknown"

    return max(script_counts.items(), key=lambda x: x[1])[0]


def _resolve_control_mode(control_mode: str) -> tuple[str, List[str]]:
    """
    Resolve control-mode aliases to a canonical primary mode.

    Behavior:
    - default: keep standard ICL feature patch as primary measurement.
    - single explicit mode (e.g. "null_icl"): use that control as primary.
    - comma-separated multi-mode labels without explicit primary: keep default
      and record a warning note to avoid silent reinterpretation.

    Returns:
      (resolved_mode, notes)
    """
    raw = str(control_mode or "default").strip().lower()
    if not raw:
        return "default", []

    alias = {
        "default": "default",
        "icl": "default",
        "icl_patch": "default",
        "main": "default",
        "null_icl": "null_icl",
        "null": "null_icl",
        "random_icl": "random_icl",
        "rand": "random_icl",
        "corrupt_icl": "corrupt_icl",
        "corrupt": "corrupt_icl",
        "auto_scale": "auto_scale_zs",
        "auto_scale_zs": "auto_scale_zs",
        "auto_shift": "auto_shift_zs",
        "auto_shift_zs": "auto_shift_zs",
        "attention": "attention_only",
        "attention_only": "attention_only",
        "attention_structured": "attention_structured",
        "basis": "basis_random",
        "basis_random": "basis_random",
        "shuffle": "shuffle_random",
        "shuffle_random": "shuffle_random",
        "gauss": "gaussian_noise",
        "gaussian": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "mean_pool": "mean_pool_patch",
        "meanpool": "mean_pool_patch",
        "mean_pool_patch": "mean_pool_patch",
        "attn_head_ablation": "attn_head_ablation",
    }

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return "default", []

    notes: List[str] = []
    explicit_primary: Optional[str] = None
    normalized: List[str] = []

    for tok in tokens:
        if tok.startswith("primary="):
            explicit_primary = tok.split("=", 1)[1].strip()
            continue
        if tok.startswith("primary:"):
            explicit_primary = tok.split(":", 1)[1].strip()
            continue
        normalized.append(alias.get(tok, tok))

    if explicit_primary is not None:
        resolved = alias.get(explicit_primary, explicit_primary)
        return resolved if resolved else "default", notes

    actionable = [m for m in normalized if m and m != "default"]
    if len(tokens) == 1 and len(actionable) == 1:
        return actionable[0], notes

    if len(actionable) > 1:
        notes.append(
            "multiple control labels provided without explicit primary; "
            "keeping default primary patch"
        )

    return "default", notes


def run_patching_experiment(
    model,
    tokenizer,
    transcoder: Transcoder,
    layer: int,
    test_word: Dict,
    icl_examples: List[Dict],
    topk: int,
    device: str,
    seed: int,
    input_script_name: str = "Telugu",
    source_language: str = "Hindi",
    output_script_name: str = "Devanagari",
    patch_style: str = "sparse",
    feature_selection: str = "topk_abs_delta",
    idx_override: Optional[torch.Tensor] = None,
    prompt_variant: str = "canonical",
    enable_basis_control: bool = True,
    decoupled_word: Optional[Dict] = None,
    cross_task_word: Optional[Dict] = None,
    patch_position: Optional[int] = None,
    control_mode: str = "default",
    eval_generation: bool = False,
    max_new_tokens: int = 32,
    selector_reference_mode: str = "zs",
    require_query_span_match: bool = False,
    use_norm_matching: bool = True,
    patch_position_mode: str = "source_last_subtoken",
    patch_geometry: str = "raw",
    patch_geometry_clip_cap: Optional[float] = None,
    patch_geometry_sign_scale: Optional[float] = None,
) -> PatchResult:
    """
    Run a single patching experiment on one word.

    1. Get zero-shot probability
    2. Get ICL probability
    3. Extract ICL features from transcoder
    4. Patch top-k features into zero-shot
    5. Get patched probability
    """
    hindi = test_word["hindi"]
    telugu = test_word.get("ood", test_word.get("telugu"))
    english = test_word["english"]

    control_mode_resolved, control_mode_notes = _resolve_control_mode(control_mode)

    # Build prompts
    zs_prompt = build_task_prompt(
        telugu,
        None,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    icl_prompt = build_task_prompt(
        telugu,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        prompt_variant=prompt_variant,
    )
    rand_prompt = build_random_icl_prompt(
        telugu,
        n_icl=len(icl_examples),
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        # Publication default: keep the task header fixed but use a script-
        # matched (Indic) random ICL control instead of an unrelated
        # English→French mapping. This reduces "control mismatch" confounds
        # when interpreting PE-minus-rand.
        use_indic_control=True,
        length_reference_examples=icl_examples,
        seed=int(seed),
        forbidden_tgt_texts=[str(hindi)],
    )
    null_prompt = build_null_icl_prompt(
        telugu,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        seed=int(seed),
        target_token_budget=150,
    )
    corrupt_prompt = build_corrupted_icl_prompt(
        telugu,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        seed=int(seed),
    )

    selector_reference_mode = str(selector_reference_mode or "zs").strip().lower()
    if selector_reference_mode not in {"zs", "corrupt_icl"}:
        raise ValueError(
            f"Unknown selector_reference_mode='{selector_reference_mode}'. "
            "Expected one of: zs, corrupt_icl."
        )
    patch_position_mode = str(patch_position_mode or "source_last_subtoken").strip().lower()
    if patch_position_mode == "target_pos1_teacher_forced":
        patch_position_mode = "target_pos1"
    if patch_position_mode not in {"source_last_subtoken", "target_pos1"}:
        raise ValueError(
            f"Unknown patch_position_mode='{patch_position_mode}'. "
            "Expected one of: source_last_subtoken, target_pos1_teacher_forced."
        )
    patch_geometry = str(patch_geometry or "raw").strip().lower()
    if patch_geometry == "normalized_sign":
        patch_geometry = "sign_normalized"
    if patch_geometry not in {"raw", "clipped", "sign_normalized"}:
        raise ValueError(
            f"Unknown patch_geometry='{patch_geometry}'. "
            "Expected one of: raw, clipped, normalized_sign."
        )

    # Prepare tokenized prompt ids once (shared by all controls/interventions).
    zs_text = apply_chat_template(tokenizer, zs_prompt)
    zs_input_ids = tokenizer(zs_text, return_tensors="pt").to(device).input_ids
    zs_prompt_len = int(zs_input_ids.shape[1])

    icl_text = apply_chat_template(tokenizer, icl_prompt)
    icl_input_ids = tokenizer(icl_text, return_tensors="pt").to(device).input_ids
    icl_prompt_len = int(icl_input_ids.shape[1])

    corrupt_text = apply_chat_template(tokenizer, corrupt_prompt)
    corrupt_input_ids = tokenizer(corrupt_text, return_tensors="pt").to(device).input_ids
    corrupt_prompt_len = int(corrupt_input_ids.shape[1])

    target_ids = tokenizer.encode(hindi, add_special_tokens=False)
    target_id = int(target_ids[0]) if target_ids else -1
    n_tokens = int(len(target_ids))
    target_ids_tensor = (
        torch.tensor(target_ids, device=device, dtype=zs_input_ids.dtype).unsqueeze(0)
        if target_ids
        else None
    )
    zs_tf_input_ids = (
        torch.cat([zs_input_ids, target_ids_tensor], dim=1)
        if target_ids_tensor is not None
        else zs_input_ids
    )
    icl_tf_input_ids = (
        torch.cat([icl_input_ids, target_ids_tensor], dim=1)
        if target_ids_tensor is not None
        else icl_input_ids
    )
    corrupt_tf_input_ids = (
        torch.cat([corrupt_input_ids, target_ids_tensor], dim=1)
        if target_ids_tensor is not None
        else corrupt_input_ids
    )

    ood_ids = tokenizer.encode(telugu, add_special_tokens=False)
    n_input_tokens_ood = len(ood_ids) if ood_ids else 1
    competitor_id = int(ood_ids[0]) if ood_ids else -1

    span_zs = _find_last_subsequence(
        zs_input_ids[0].detach().to("cpu", dtype=torch.long).tolist(),
        [int(x) for x in ood_ids],
    )
    span_icl = _find_last_subsequence(
        icl_input_ids[0].detach().to("cpu", dtype=torch.long).tolist(),
        [int(x) for x in ood_ids],
    )
    span_corrupt = _find_last_subsequence(
        corrupt_input_ids[0].detach().to("cpu", dtype=torch.long).tolist(),
        [int(x) for x in ood_ids],
    )

    if require_query_span_match and patch_position_mode == "source_last_subtoken":
        required_spans = {
            "zero_shot": span_zs,
            "icl": span_icl,
        }
        if selector_reference_mode == "corrupt_icl":
            required_spans["selector_reference"] = span_corrupt
        missing = [name for name, span in required_spans.items() if span is None]
        if missing:
            raise ValueError(
                "Query span localization failed for confirmatory patching: "
                + ", ".join(missing)
            )

    # Get baselines in one forward per context (first-token + joint logprob).
    zs_metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=zs_input_ids,
        target_ids=target_ids,
        target_id=target_id,
        device=device,
        competitor_id=competitor_id,
    )
    prob_zs_first = float(zs_metrics["first_prob"])
    logit_zs_first = float(zs_metrics["first_logit"])
    prob_competitor_zs_first = float(zs_metrics.get("competitor_first_prob", float("nan")))
    logit_competitor_zs_first = float(zs_metrics.get("competitor_first_logit", float("nan")))
    logprob_zs = float(zs_metrics["joint_logprob"])
    prob_zs_multi = float(np.exp(logprob_zs)) if np.isfinite(logprob_zs) else float("nan")
    nll_zs = float(-logprob_zs / n_tokens) if (n_tokens and np.isfinite(logprob_zs)) else float("nan")
    nll_pos1_zs = float(zs_metrics.get("target_pos1_nll", float("nan")))
    nll_pos2_zs = float(zs_metrics.get("target_pos2_nll", float("nan")))
    nll_pos3_zs = float(zs_metrics.get("target_pos3_nll", float("nan")))

    icl_metrics = _teacher_forced_metrics_from_input_ids(
        model=model,
        input_ids=icl_input_ids,
        target_ids=target_ids,
        target_id=target_id,
        device=device,
        competitor_id=competitor_id,
    )
    prob_icl_first = float(icl_metrics["first_prob"])
    logit_icl_first = float(icl_metrics["first_logit"])
    prob_competitor_icl_first = float(icl_metrics.get("competitor_first_prob", float("nan")))
    logit_competitor_icl_first = float(icl_metrics.get("competitor_first_logit", float("nan")))
    logprob_icl = float(icl_metrics["joint_logprob"])
    prob_icl_multi = float(np.exp(logprob_icl)) if np.isfinite(logprob_icl) else float("nan")
    nll_icl = float(-logprob_icl / n_tokens) if (n_tokens and np.isfinite(logprob_icl)) else float("nan")
    nll_pos1_icl = float(icl_metrics.get("target_pos1_nll", float("nan")))
    nll_pos2_icl = float(icl_metrics.get("target_pos2_nll", float("nan")))
    nll_pos3_icl = float(icl_metrics.get("target_pos3_nll", float("nan")))

    # Determine the absolute position for patching in the ZS prompt.
    # Prefer the last token of the query word span when detectable.
    feature_icl_input_ids = icl_input_ids
    feature_selector_input_ids = zs_input_ids
    if patch_position is not None:
        if patch_position < 0:
            patch_pos = max(0, zs_prompt_len + int(patch_position))
        else:
            patch_pos = int(patch_position)
        icl_feature_position = int(span_icl[1] - 1) if span_icl is not None else int(icl_prompt_len - 1)
    else:
        if patch_position_mode == "target_pos1":
            if target_ids_tensor is None:
                raise ValueError("target_pos1 patching requires non-empty target_ids.")
            patch_pos = int(zs_prompt_len)
            icl_feature_position = int(icl_prompt_len)
            feature_icl_input_ids = icl_tf_input_ids
            feature_selector_input_ids = zs_tf_input_ids
        else:
            if span_zs is not None:
                patch_pos = int(span_zs[1] - 1)
            else:
                patch_pos = zs_prompt_len - 1
            icl_feature_position = int(span_icl[1] - 1) if span_icl is not None else int(icl_prompt_len - 1)
    zs_patch_position = int(patch_pos)
    rope_position_gap = int(icl_feature_position - zs_patch_position)

    # Extract features (ICL, random-ICL)
    mlp_in_icl, mlp_out_icl = _extract_mlp_io_at_position_from_input_ids(
        model=model,
        input_ids=feature_icl_input_ids,
        layer=layer,
        position=icl_feature_position,
    )
    icl_feats = transcoder.encode(mlp_in_icl.unsqueeze(0)).squeeze(0)
    rec_icl = transcoder.decode(icl_feats.unsqueeze(0)).squeeze(0).detach()
    icl_abs = icl_feats.detach().abs()
    active_feats_icl = int(torch.count_nonzero(icl_abs > 1e-8).item())
    active_feats_zs = float("nan")
    d_sae = max(1, int(icl_feats.numel()))
    feature_coverage_ratio = float(active_feats_icl / d_sae)
    mlp_in_icl_norm = float(torch.norm(mlp_in_icl).item())
    mlp_out_icl_norm = float(torch.norm(mlp_out_icl).item())
    rec_icl_norm = float(torch.norm(rec_icl).item())
    if mlp_out_icl_norm > 0.0 and rec_icl_norm > 0.0:
        reconstruction_cosine_icl = float(
            torch.dot(rec_icl.float(), mlp_out_icl.float()).item()
            / max(1e-12, rec_icl_norm * mlp_out_icl_norm)
        )
    else:
        reconstruction_cosine_icl = float("nan")
    reconstruction_rel_error_icl = float(
        torch.norm((rec_icl - mlp_out_icl).float()).item() / max(1e-12, mlp_out_icl_norm)
    )
    reconstruction_mse_icl = float(
        torch.mean((rec_icl.float() - mlp_out_icl.float()) ** 2).item()
    )
    patch_norm_target = (
        float(mlp_out_icl_norm)
        if bool(use_norm_matching)
        and np.isfinite(mlp_out_icl_norm)
        and float(mlp_out_icl_norm) > 0.0
        else None
    )
    patch_style = str(patch_style or "sparse").strip().lower()
    feature_selection = str(feature_selection or "topk_abs_delta").strip().lower()
    feature_selection_base = feature_selection
    if feature_selection_base.endswith("_global"):
        feature_selection_base = feature_selection_base[: -len("_global")]

    if patch_style not in {"sparse", "substitute"}:
        raise ValueError(
            f"Unknown patch_style='{patch_style}'. Expected one of: sparse, substitute."
        )

    allowed = {"topk_abs_icl", "topk_abs_delta", "bottomk_abs_icl", "bottomk_abs_delta"}
    if feature_selection_base not in allowed:
        raise ValueError(
            f"Unknown feature_selection='{feature_selection}'. Expected one of: "
            "topk_abs_icl, topk_abs_delta, bottomk_abs_icl, bottomk_abs_delta (optionally with _global)."
        )

    zs_feats = None
    selector_ref_feats = None
    selector_ref_position = int(zs_patch_position)
    selector_ref_prompt_len = int(zs_prompt_len)
    selector_ref_input_ids = feature_selector_input_ids
    feature_cosine_zs_icl = float("nan")
    feature_identity_jaccard_zs_icl = float("nan")
    need_reference_feats = (patch_style == "substitute") or (
        "_delta" in feature_selection_base
    )
    if need_reference_feats:
        mlp_in_zs, _ = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=feature_selector_input_ids,
            layer=layer,
            position=zs_patch_position,
        )
        zs_feats = transcoder.encode(mlp_in_zs.unsqueeze(0)).squeeze(0)
        selector_ref_feats = zs_feats
        active_feats_zs = float(torch.count_nonzero(torch.abs(zs_feats.detach()) > 1e-8).item())
        zs_norm = float(torch.norm(zs_feats).item())
        icl_norm = float(torch.norm(icl_feats).item())
        if zs_norm > 0.0 and icl_norm > 0.0:
            feature_cosine_zs_icl = float(
                torch.dot(zs_feats.float(), icl_feats.float()).item()
                / max(1e-12, zs_norm * icl_norm)
            )

        # Feature identity overlap: are the top-k active features the same?
        idx_icl_abs = _topk_indices(icl_feats, topk)
        idx_zs_abs = _topk_indices(zs_feats, topk)
        set_icl = set(int(i) for i in idx_icl_abs.detach().cpu().tolist())
        set_zs = set(int(i) for i in idx_zs_abs.detach().cpu().tolist())
        union = len(set_icl | set_zs)
        if union > 0:
            feature_identity_jaccard_zs_icl = float(len(set_icl & set_zs) / union)

        if selector_reference_mode == "corrupt_icl" and "_delta" in feature_selection_base:
            selector_ref_position = int(corrupt_prompt_len) if patch_position_mode == "target_pos1" else (
                int(span_corrupt[1] - 1)
                if span_corrupt is not None
                else int(corrupt_prompt_len - 1)
            )
            selector_ref_prompt_len = int(corrupt_prompt_len)
            selector_ref_input_ids = (
                corrupt_tf_input_ids if patch_position_mode == "target_pos1" else corrupt_input_ids
            )
            mlp_in_selector_ref, _ = _extract_mlp_io_at_position_from_input_ids(
                model=model,
                input_ids=selector_ref_input_ids,
                layer=layer,
                position=selector_ref_position,
            )
            selector_ref_feats = transcoder.encode(
                mlp_in_selector_ref.unsqueeze(0)
            ).squeeze(0)

    # For validity diagnostics, try to log ZS active-feature count even when ZS
    # features are not required by the selected patch style/selector.
    if active_feats_zs != active_feats_zs:
        try:
            mlp_in_zs_l0, _ = _extract_mlp_io_at_position_from_input_ids(
                model=model,
                input_ids=feature_selector_input_ids,
                layer=layer,
                position=zs_patch_position,
            )
            zs_feats_l0 = transcoder.encode(mlp_in_zs_l0.unsqueeze(0)).squeeze(0)
            active_feats_zs = float(torch.count_nonzero(torch.abs(zs_feats_l0.detach()) > 1e-8).item())
        except Exception:
            active_feats_zs = float("nan")

    def _select_idx(
        *,
        src_feats: torch.Tensor,
        ref_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feature_selection_base == "topk_abs_icl":
            return _topk_indices(src_feats, topk)
        if feature_selection_base == "bottomk_abs_icl":
            return _bottomk_indices_abs(src_feats, topk)
        assert ref_feats is not None
        delta = src_feats - ref_feats
        if feature_selection_base == "topk_abs_delta":
            return _topk_indices(delta, topk)
        if feature_selection_base == "bottomk_abs_delta":
            return _bottomk_indices_abs(delta, topk)
        raise RuntimeError(
            f"Unhandled feature_selection_base='{feature_selection_base}'"
        )

    if idx_override is not None:
        idx = idx_override.detach().to(device=icl_feats.device, dtype=torch.long)
    else:
        idx = _select_idx(src_feats=icl_feats, ref_feats=selector_ref_feats)

    selected_feature_indices = [
        int(i) for i in idx.detach().to("cpu", dtype=torch.long).tolist()
    ]
    dla_diag = _compute_selected_feature_dla(
        transcoder=transcoder,
        model=model,
        selected_indices=selected_feature_indices,
        target_id=target_id,
        competitor_id=competitor_id,
    )
    max_selected_feature_zscore = float("nan")
    n_selected_outlier_gt5sigma = 0
    selected_outlier_gt5sigma = False

    score_src = icl_feats
    if "_delta" in feature_selection_base and zs_feats is not None:
        score_src = icl_feats - zs_feats
    score_abs = torch.abs(score_src.detach().float())
    if score_abs.numel() > 0 and idx.numel() > 0:
        mu = float(torch.mean(score_abs).item())
        sigma = float(torch.std(score_abs, unbiased=False).item())
        if np.isfinite(sigma) and sigma > 1e-8:
            sel_scores = score_abs[idx]
            z = (sel_scores - mu) / sigma
            max_selected_feature_zscore = float(torch.max(z).item())
            n_selected_outlier_gt5sigma = int(torch.sum(z > 5.0).item())
            selected_outlier_gt5sigma = bool(n_selected_outlier_gt5sigma > 0)

    def _build_patch(
        src_feats: torch.Tensor,
        idx: torch.Tensor,
        zs_ref: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if patch_style == "sparse":
            return _build_patch_vector(src_feats, idx)
        assert zs_ref is not None
        patch = zs_ref.clone()
        if idx.numel() > 0:
            patch[idx] = src_feats[idx]
        return patch

    base_patch_feats = _build_patch(icl_feats, idx, zs_feats)
    selected_raw_vals = (
        base_patch_feats[idx].detach().float().to("cpu")
        if idx.numel() > 0
        else torch.empty(0, dtype=torch.float32)
    )
    clip_cap_used = float("nan")
    if patch_geometry_clip_cap is not None and np.isfinite(float(patch_geometry_clip_cap)):
        clip_cap_used = float(patch_geometry_clip_cap)
    sign_scale_used = float("nan")
    if patch_geometry_sign_scale is not None and np.isfinite(float(patch_geometry_sign_scale)):
        sign_scale_used = float(patch_geometry_sign_scale)
    patch_feats = _apply_patch_geometry(
        base_patch_feats,
        idx,
        mode=patch_geometry,
        clip_cap=patch_geometry_clip_cap,
        sign_scale=patch_geometry_sign_scale,
    )
    selected_feature_fraction = float(idx.numel() / d_sae)
    ablate_feats = _build_ablation_vector(icl_feats, idx)
    # Matched-magnitude attention control vector: same norm as the real decoded
    # patch contribution, randomized direction.
    latent_patch_norm_pre_geometry = float(torch.norm(base_patch_feats).item())
    latent_patch_norm_post_geometry = float(torch.norm(patch_feats).item())
    decoded_patch_pre = transcoder.decode(base_patch_feats.unsqueeze(0)).squeeze(0).detach()
    decoded_patch = transcoder.decode(patch_feats.unsqueeze(0)).squeeze(0).detach()
    decoded_patch_norm_pre_geometry = float(torch.norm(decoded_patch_pre).item())
    decoded_patch_norm_post_geometry = float(torch.norm(decoded_patch).item())
    decoded_patch_norm = float(torch.norm(decoded_patch).item())
    decoded_patch_norm_ratio = float(decoded_patch_norm / max(1e-12, mlp_out_icl_norm))
    patch_geometry_fraction_clipped = float("nan")
    if patch_geometry == "clipped" and selected_raw_vals.numel() > 0 and np.isfinite(clip_cap_used):
        patch_geometry_fraction_clipped = float(
            torch.mean((torch.abs(selected_raw_vals) > clip_cap_used).float()).item()
        )
    elif selected_raw_vals.numel() > 0:
        patch_geometry_fraction_clipped = 0.0
    attention_patch_vector = _make_magnitude_matched_attention_vector(
        decoded_patch,
        seed=int(seed),
        layer=int(layer),
        topk=int(topk),
        word_key=str(english),
    )

    # Random-basis top-k control (SRHT): choose top-k coefficients in a random
    # orthogonal basis and project back to original feature space.
    basis_patch_feats: Optional[torch.Tensor] = None
    if bool(enable_basis_control):
        try:
            if patch_style == "substitute" and zs_feats is not None:
                basis_delta = _build_srht_basis_patch_feats(
                    icl_feats - zs_feats,
                    topk=int(topk),
                    seed=int(seed),
                    layer=int(layer),
                    word_key=str(english),
                )
                basis_patch_feats = zs_feats + basis_delta
            else:
                basis_patch_feats = _build_srht_basis_patch_feats(
                    icl_feats,
                    topk=int(topk),
                    seed=int(seed),
                    layer=int(layer),
                    word_key=str(english),
                )
            basis_patch_feats = _apply_patch_geometry(
                basis_patch_feats,
                idx,
                mode=patch_geometry,
                clip_cap=patch_geometry_clip_cap,
                sign_scale=patch_geometry_sign_scale,
            )
        except Exception:
            basis_patch_feats = None

    # Stronger control than random-context: keep the same top-k values but place
    # them at randomly chosen *different* feature indices. This tests whether
    # feature identity matters beyond simply injecting a sparse vector.
    shuffle_patch_feats: Optional[torch.Tensor] = None
    if idx.numel() > 0:
        n_dim = int(icl_feats.numel())
        idx_cpu = idx.detach().to("cpu", dtype=torch.long).numpy()
        pool = np.setdiff1d(np.arange(n_dim), idx_cpu)
        if pool.shape[0] >= idx_cpu.shape[0]:
            # Local RNG: deterministic given (seed, layer, topk, word).
            import hashlib

            msg = f"shuffle::{seed}::{layer}::{topk}::{english}".encode("utf-8")
            seed32 = int.from_bytes(
                hashlib.sha256(msg).digest()[:4], "little", signed=False
            )
            rng = np.random.default_rng(seed32)

            shuf_idx = rng.choice(pool, size=idx_cpu.shape[0], replace=False).astype(
                np.int64
            )
            # Permute values too (optional but avoids any accidental alignment).
            vals = icl_feats[idx].detach()
            perm = rng.permutation(vals.shape[0]).astype(np.int64)
            vals = vals[torch.tensor(perm, device=vals.device, dtype=torch.long)]

            shuf_idx_t = torch.tensor(
                shuf_idx, device=icl_feats.device, dtype=torch.long
            )
            if patch_style == "sparse":
                shuffle_patch_feats = torch.zeros_like(icl_feats)
                shuffle_patch_feats[shuf_idx_t] = vals
            else:
                assert zs_feats is not None
                shuffle_patch_feats = zs_feats.clone()
                shuffle_patch_feats[shuf_idx_t] = vals
            shuffle_patch_feats = _apply_patch_geometry(
                shuffle_patch_feats,
                shuf_idx_t,
                mode=patch_geometry,
                clip_cap=patch_geometry_clip_cap,
                sign_scale=patch_geometry_sign_scale,
            )

    # Gaussian-noise control: use the *same* selected indices as the real patch,
    # but replace the feature values with N(mu, sigma) noise matched to the
    # selected ICL feature values. This tests whether "any activation energy"
    # at the chosen indices can produce rescue.
    gauss_patch_feats: Optional[torch.Tensor] = None
    if idx.numel() > 0:
        vals = icl_feats[idx].detach()
        mu = float(vals.mean().item()) if vals.numel() else 0.0
        sigma = float(vals.std(unbiased=False).item()) if vals.numel() else 0.0
        if not np.isfinite(sigma) or sigma <= 0.0:
            # If variance is 0 (common for tiny top-k), fall back to a scale
            # based on mean absolute value.
            sigma = float(torch.mean(torch.abs(vals)).item()) if vals.numel() else 0.0

        import hashlib

        msg = f"gauss::{seed}::{layer}::{topk}::{english}".encode("utf-8")
        seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
        rng = np.random.default_rng(seed32)

        noise = rng.normal(loc=mu, scale=max(1e-8, sigma), size=(int(idx.numel()),)).astype(np.float32)
        noise_t = torch.tensor(noise, device=icl_feats.device, dtype=icl_feats.dtype)
        if patch_style == "sparse":
            gauss_patch_feats = torch.zeros_like(icl_feats)
            gauss_patch_feats[idx] = noise_t
        else:
            assert zs_feats is not None
            gauss_patch_feats = zs_feats.clone()
            gauss_patch_feats[idx] = noise_t
        gauss_patch_feats = _apply_patch_geometry(
            gauss_patch_feats,
            idx,
            mode=patch_geometry,
            clip_cap=patch_geometry_clip_cap,
            sign_scale=patch_geometry_sign_scale,
        )

    def _extract_prompt_feats_at_query(
        prompt_text: str,
        query_token_ids: Optional[List[int]] = None,
        *,
        require_span_match: bool = False,
        prompt_label: str = "prompt",
        teacher_forced_target_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        pt = apply_chat_template(tokenizer, prompt_text)
        ids = tokenizer(pt, return_tensors="pt").to(device).input_ids
        if patch_position_mode == "target_pos1":
            tf_target_ids = list(teacher_forced_target_ids or target_ids)
            if not tf_target_ids:
                raise ValueError(
                    f"Target-position patching requires non-empty target ids for {prompt_label}."
                )
            tf_tgt = torch.tensor(tf_target_ids, device=device, dtype=ids.dtype).unsqueeze(0)
            ids = torch.cat([ids, tf_tgt], dim=1)
            pos = int(ids.shape[1] - tf_tgt.shape[1])
        else:
            pos = int(ids.shape[1] - 1)
            q_ids = [int(x) for x in (query_token_ids if query_token_ids is not None else ood_ids)]
            if q_ids:
                span = _find_last_subsequence(
                    ids[0].detach().to("cpu", dtype=torch.long).tolist(),
                    q_ids,
                )
                if span is not None:
                    pos = int(span[1] - 1)
                elif require_span_match:
                    raise ValueError(
                        f"Query span localization failed for {prompt_label} under fail-closed mode."
                    )
        mlp_in_q, _ = _extract_mlp_io_at_position_from_input_ids(
            model=model,
            input_ids=ids,
            layer=layer,
            position=pos,
        )
        return transcoder.encode(mlp_in_q.unsqueeze(0)).squeeze(0)

    def _extract_control_patch_feats(
        prompt_text: str,
        *,
        prompt_label: str,
        query_token_ids: Optional[List[int]] = None,
        teacher_forced_target_ids: Optional[List[int]] = None,
    ) -> Optional[torch.Tensor]:
        try:
            return _extract_prompt_feats_at_query(
                prompt_text,
                query_token_ids=query_token_ids,
                require_span_match=bool(require_query_span_match),
                prompt_label=prompt_label,
                teacher_forced_target_ids=teacher_forced_target_ids,
            )
        except ValueError as exc:
            control_mode_notes.append(f"{prompt_label}_unavailable: {exc}")
            return None

    rand_feats = _extract_control_patch_feats(
        rand_prompt,
        prompt_label="random_icl",
    )
    rand_patch_feats: Optional[torch.Tensor] = None
    if rand_feats is not None:
        rand_idx = (
            idx if idx_override is not None else _select_idx(src_feats=rand_feats, ref_feats=zs_feats)
        )
        rand_patch_feats = _apply_patch_geometry(
            _build_patch(rand_feats, rand_idx, zs_feats),
            rand_idx,
            mode=patch_geometry,
            clip_cap=patch_geometry_clip_cap,
            sign_scale=patch_geometry_sign_scale,
        )

    # Null-ICL control (context length only, no demonstrations)
    null_feats = _extract_control_patch_feats(
        null_prompt,
        prompt_label="null_icl",
    )
    null_patch_feats: Optional[torch.Tensor] = None
    if null_feats is not None:
        null_idx = (
            idx if idx_override is not None else _select_idx(src_feats=null_feats, ref_feats=zs_feats)
        )
        null_patch_feats = _apply_patch_geometry(
            _build_patch(null_feats, null_idx, zs_feats),
            null_idx,
            mode=patch_geometry,
            clip_cap=patch_geometry_clip_cap,
            sign_scale=patch_geometry_sign_scale,
        )

    # Task-matched corrupted ICL control (same format, wrong mapping)
    corrupt_feats = _extract_control_patch_feats(
        corrupt_prompt,
        prompt_label="corrupt_icl",
    )
    corrupt_patch_feats: Optional[torch.Tensor] = None
    if corrupt_feats is not None:
        corrupt_idx = (
            idx
            if idx_override is not None
            else _select_idx(src_feats=corrupt_feats, ref_feats=zs_feats)
        )
        corrupt_patch_feats = _apply_patch_geometry(
            _build_patch(corrupt_feats, corrupt_idx, zs_feats),
            corrupt_idx,
            mode=patch_geometry,
            clip_cap=patch_geometry_clip_cap,
            sign_scale=patch_geometry_sign_scale,
        )

    # Patched (sufficiency)
    prob_patched_first = float("nan")
    logit_patched_first = float("nan")
    prob_patched_multi = float("nan")
    logprob_patched = float("nan")
    nll_patched = float("nan")
    nll_pos1_patched = float("nan")
    nll_pos2_patched = float("nan")
    nll_pos3_patched = float("nan")
    prob_competitor_patched_first = float("nan")
    logit_competitor_patched_first = float("nan")
    if target_id >= 0:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        patched_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
            competitor_id=competitor_id,
        )
        hook.remove()
        prob_patched_first = float(patched_metrics["first_prob"])
        logit_patched_first = float(patched_metrics["first_logit"])
        prob_competitor_patched_first = float(
            patched_metrics.get("competitor_first_prob", float("nan"))
        )
        logit_competitor_patched_first = float(
            patched_metrics.get("competitor_first_logit", float("nan"))
        )
        logprob_patched = float(patched_metrics["joint_logprob"])
        prob_patched_multi = float(np.exp(logprob_patched)) if np.isfinite(logprob_patched) else float("nan")
        nll_patched = float(-logprob_patched / n_tokens) if (n_tokens and np.isfinite(logprob_patched)) else float("nan")
        nll_pos1_patched = float(patched_metrics.get("target_pos1_nll", float("nan")))
        nll_pos2_patched = float(patched_metrics.get("target_pos2_nll", float("nan")))
        nll_pos3_patched = float(patched_metrics.get("target_pos3_nll", float("nan")))

    # Mean-pool control: average features over all OOD sub-token positions.
    prob_mean_pool_patched_first = float("nan")
    prob_mean_pool_patched_multi = float("nan")
    logprob_mean_pool_patched = float("nan")
    nll_mean_pool_patched = float("nan")
    if (
        target_id >= 0
        and span_icl is not None
        and span_zs is not None
        and (span_icl[1] - span_icl[0]) > 1
        and (span_zs[1] - span_zs[0]) > 1
    ):
        try:
            icl_positions = list(range(int(span_icl[0]), int(span_icl[1])))
            zs_positions = list(range(int(span_zs[0]), int(span_zs[1])))

            icl_mlp_seq = _extract_mlp_in_positions_from_input_ids(
                model=model,
                input_ids=icl_input_ids,
                layer=layer,
                positions=icl_positions,
            )
            icl_mlp_pool = icl_mlp_seq.float().mean(dim=0)
            icl_feats_pool = transcoder.encode(icl_mlp_pool.unsqueeze(0)).squeeze(0)

            zs_feats_pool: Optional[torch.Tensor] = None
            if need_zs_feats:
                zs_mlp_seq = _extract_mlp_in_positions_from_input_ids(
                    model=model,
                    input_ids=zs_input_ids,
                    layer=layer,
                    positions=zs_positions,
                )
                zs_mlp_pool = zs_mlp_seq.float().mean(dim=0)
                zs_feats_pool = transcoder.encode(zs_mlp_pool.unsqueeze(0)).squeeze(0)

            idx_pool = (
                idx
                if idx_override is not None
                else _select_idx(src_feats=icl_feats_pool, ref_feats=zs_feats_pool)
            )
            patch_feats_pool = _build_patch(icl_feats_pool, idx_pool, zs_feats_pool)

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                patch_feats_pool,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            pool_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=zs_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
                competitor_id=competitor_id,
            )
            hook.remove()
            prob_mean_pool_patched_first = float(pool_metrics["first_prob"])
            logprob_mean_pool_patched = float(pool_metrics["joint_logprob"])
            prob_mean_pool_patched_multi = (
                float(np.exp(logprob_mean_pool_patched))
                if np.isfinite(logprob_mean_pool_patched)
                else float("nan")
            )
            nll_mean_pool_patched = (
                float(-logprob_mean_pool_patched / n_tokens)
                if (n_tokens and np.isfinite(logprob_mean_pool_patched))
                else float("nan")
            )
        except Exception:
            pass

    # Superposition / feature-collision diagnostic on an unrelated English task.
    nll_english_neutral_zs = float("nan")
    nll_english_neutral_patched = float("nan")
    nll_harm_english_neutral_patch = float("nan")
    try:
        english_prompt = build_english_neutral_prompt(english)
        english_text = apply_chat_template(tokenizer, english_prompt)
        english_input_ids = tokenizer(english_text, return_tensors="pt").to(device).input_ids
        english_target_ids = tokenizer.encode(str(english), add_special_tokens=False)
        english_target_id = int(english_target_ids[0]) if english_target_ids else -1
        english_n_tokens = int(len(english_target_ids))

        if english_target_id >= 0 and english_n_tokens > 0:
            english_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=english_input_ids,
                target_ids=english_target_ids,
                target_id=english_target_id,
                device=device,
            )
            english_logprob = float(english_metrics["joint_logprob"])
            if np.isfinite(english_logprob):
                nll_english_neutral_zs = float(-english_logprob / english_n_tokens)

            english_span = _find_last_subsequence(
                english_input_ids[0].detach().to("cpu", dtype=torch.long).tolist(),
                [int(x) for x in english_target_ids],
            )
            if patch_position_mode == "target_pos1":
                english_patch_pos = int(english_input_ids.shape[1])
            else:
                english_patch_pos = (
                    int(english_span[1] - 1)
                    if english_span is not None
                    else int(english_input_ids.shape[1] - 1)
                )

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                patch_feats,
                patch_position=english_patch_pos,
                target_output_norm=patch_norm_target,
            )
            english_patched_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=english_input_ids,
                target_ids=english_target_ids,
                target_id=english_target_id,
                device=device,
            )
            hook.remove()
            english_patched_logprob = float(english_patched_metrics["joint_logprob"])
            if np.isfinite(english_patched_logprob):
                nll_english_neutral_patched = float(
                    -english_patched_logprob / english_n_tokens
                )

            if np.isfinite(nll_english_neutral_zs) and np.isfinite(
                nll_english_neutral_patched
            ):
                nll_harm_english_neutral_patch = float(
                    nll_english_neutral_patched - nll_english_neutral_zs
                )
    except Exception:
        pass

    # KV-cache identity confound diagnostic: does the patch induce BOS-seeking
    # attention in the next layer even without demonstrations in context?
    bos_attention_zs_next_layer = float("nan")
    bos_attention_patched_next_layer = float("nan")
    delta_bos_attention_next_layer_patch = float("nan")
    try:
        n_layers = len(get_model_layers(model))
        next_layer = int(min(max(0, layer + 1), max(0, n_layers - 1)))
        if next_layer >= 0:
            bos_base = _extract_attention_bos_fraction_last_query(
                model=model,
                tokenizer=tokenizer,
                prompt=zs_prompt,
                layer=next_layer,
                device=device,
            )
            if bos_base is not None and int(bos_base.numel()) > 0:
                bos_attention_zs_next_layer = float(
                    torch.mean(bos_base.detach().float()).item()
                )

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                patch_feats,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            bos_patch = _extract_attention_bos_fraction_last_query(
                model=model,
                tokenizer=tokenizer,
                prompt=zs_prompt,
                layer=next_layer,
                device=device,
            )
            hook.remove()
            if bos_patch is not None and int(bos_patch.numel()) > 0:
                bos_attention_patched_next_layer = float(
                    torch.mean(bos_patch.detach().float()).item()
                )

            if np.isfinite(bos_attention_zs_next_layer) and np.isfinite(
                bos_attention_patched_next_layer
            ):
                delta_bos_attention_next_layer_patch = float(
                    bos_attention_patched_next_layer - bos_attention_zs_next_layer
                )
    except Exception:
        pass

    # Semantic Decoupling Control
    prob_decoupled_patched_first = float("nan")
    prob_decoupled_patched_multi = float("nan")
    logprob_decoupled_patched = float("nan")
    nll_decoupled_patched = float("nan")
    prob_decoupled_target_first = float("nan")
    
    if decoupled_word is not None and target_id >= 0:
        decoupled_telugu = decoupled_word.get("ood", decoupled_word.get("telugu"))
        decoupled_prompt = build_task_prompt(
            decoupled_telugu,
            icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        )
        decoupled_ids = tokenizer.encode(str(decoupled_telugu), add_special_tokens=False)
        decoupled_feats = _extract_control_patch_feats(
            decoupled_prompt,
            query_token_ids=[int(x) for x in decoupled_ids],
            prompt_label="cross_target_mismatch",
            teacher_forced_target_ids=tokenizer.encode(
                str(decoupled_word.get("hindi", "")),
                add_special_tokens=False,
            ),
        )
        if decoupled_feats is not None:
            # Use same idx selection strategy for the decoupled word
            decoupled_idx = (
                _select_idx(src_feats=decoupled_feats, ref_feats=zs_feats)
                if idx_override is None
                else idx
            )
            decoupled_patch_feats = _apply_patch_geometry(
                _build_patch(decoupled_feats, decoupled_idx, zs_feats),
                decoupled_idx,
                mode=patch_geometry,
                clip_cap=patch_geometry_clip_cap,
                sign_scale=patch_geometry_sign_scale,
            )

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                decoupled_patch_feats,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            decoupled_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=zs_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
            )
            hook.remove()
            prob_decoupled_patched_first = float(decoupled_metrics["first_prob"])
            logprob_decoupled_patched = float(decoupled_metrics["joint_logprob"])
            prob_decoupled_patched_multi = float(np.exp(logprob_decoupled_patched)) if np.isfinite(logprob_decoupled_patched) else float("nan")
            nll_decoupled_patched = float(-logprob_decoupled_patched / n_tokens) if (n_tokens and np.isfinite(logprob_decoupled_patched)) else float("nan")

            # Evaluate if the patched model outputs the decoupled word's answer instead of the query word's answer
            decoupled_hindi = str(decoupled_word.get("hindi", ""))
            if decoupled_hindi:
                decoupled_target_ids = tokenizer.encode(decoupled_hindi, add_special_tokens=False)
                decoupled_target_id = int(decoupled_target_ids[0]) if decoupled_target_ids else -1
                if decoupled_target_id >= 0:
                    hook = register_transcoder_feature_patch_hook(
                        model,
                        transcoder,
                        layer,
                        decoupled_patch_feats,
                        patch_position=patch_pos,
                        target_output_norm=patch_norm_target,
                    )
                    decoupled_target_metrics = _teacher_forced_metrics_from_input_ids(
                        model=model,
                        input_ids=zs_input_ids,
                        target_ids=decoupled_target_ids,
                        target_id=decoupled_target_id,
                        device=device,
                    )
                    hook.remove()
                    prob_decoupled_target_first = float(decoupled_target_metrics["first_prob"])

    # Random patch control
    prob_rand_patched_first = float("nan")
    prob_rand_patched_multi = float("nan")
    logprob_rand_patched = float("nan")
    nll_rand_patched = float("nan")
    if target_id >= 0 and rand_patch_feats is not None:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            rand_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        rand_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_rand_patched_first = float(rand_metrics["first_prob"])
        logprob_rand_patched = float(rand_metrics["joint_logprob"])
        prob_rand_patched_multi = float(np.exp(logprob_rand_patched)) if np.isfinite(logprob_rand_patched) else float("nan")
        nll_rand_patched = float(-logprob_rand_patched / n_tokens) if (n_tokens and np.isfinite(logprob_rand_patched)) else float("nan")

    # Null-ICL patch control (context-length only)
    prob_null_patched_first = float("nan")
    prob_null_patched_multi = float("nan")
    logprob_null_patched = float("nan")
    nll_null_patched = float("nan")
    if target_id >= 0 and null_patch_feats is not None:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            null_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        null_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_null_patched_first = float(null_metrics["first_prob"])
        logprob_null_patched = float(null_metrics["joint_logprob"])
        prob_null_patched_multi = (
            float(np.exp(logprob_null_patched)) if np.isfinite(logprob_null_patched) else float("nan")
        )
        nll_null_patched = (
            float(-logprob_null_patched / n_tokens)
            if (n_tokens and np.isfinite(logprob_null_patched))
            else float("nan")
        )

    # Ablation in ICL (necessity test)
    # This tests: if we remove the top-k features from ICL, does performance drop?
    # A drop indicates these features are NECESSARY for ICL's benefit.
    prob_ablated_first = float("nan")
    prob_ablated_multi = float("nan")
    logprob_ablated = float("nan")
    nll_ablated = float("nan")
    if target_id >= 0 and corrupt_patch_feats is not None:
        hook = register_transcoder_feature_ablation_hook(
            model,
            transcoder,
            layer,
            idx,
            ablate_position=icl_feature_position,
        )
        ablated_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=icl_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_ablated_first = float(ablated_metrics["first_prob"])
        logprob_ablated = float(ablated_metrics["joint_logprob"])
        prob_ablated_multi = float(np.exp(logprob_ablated)) if np.isfinite(logprob_ablated) else float("nan")
        nll_ablated = float(-logprob_ablated / n_tokens) if (n_tokens and np.isfinite(logprob_ablated)) else float("nan")

    # Attention-head ablation on ICL (skeptic control)
    prob_attn_head_ablated_first = float("nan")
    prob_attn_head_ablated_multi = float("nan")
    logprob_attn_head_ablated = float("nan")
    nll_attn_head_ablated = float("nan")
    attn_head_indices: List[int] = []
    attn_sink_heads_excluded = 0
    attn_sink_threshold = 0.80
    if target_id >= 0:
        try:
            attn_head_indices, attn_sink_heads_excluded = _select_top_attention_heads_by_delta(
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                icl_prompt=icl_prompt,
                zs_prompt=zs_prompt,
                device=device,
                topk_heads=3,
                sink_bos_threshold=attn_sink_threshold,
            )
            if attn_head_indices:
                hook = register_attention_head_ablation_hook(
                    model,
                    layer,
                    attn_head_indices,
                    ablate_position=icl_feature_position,
                )
                attn_head_abl_metrics = _teacher_forced_metrics_from_input_ids(
                    model=model,
                    input_ids=icl_input_ids,
                    target_ids=target_ids,
                    target_id=target_id,
                    device=device,
                )
                hook.remove()
                prob_attn_head_ablated_first = float(attn_head_abl_metrics["first_prob"])
                logprob_attn_head_ablated = float(attn_head_abl_metrics["joint_logprob"])
                prob_attn_head_ablated_multi = (
                    float(np.exp(logprob_attn_head_ablated))
                    if np.isfinite(logprob_attn_head_ablated)
                    else float("nan")
                )
                nll_attn_head_ablated = (
                    float(-logprob_attn_head_ablated / n_tokens)
                    if (n_tokens and np.isfinite(logprob_attn_head_ablated))
                    else float("nan")
                )
        except Exception:
            pass

    # Corrupted-ICL patch control (task-matched; wrong mapping)
    prob_corrupt_patched_first = float("nan")
    prob_corrupt_patched_multi = float("nan")
    logprob_corrupt_patched = float("nan")
    nll_corrupt_patched = float("nan")
    if target_id >= 0:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            corrupt_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        corrupt_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_corrupt_patched_first = float(corrupt_metrics["first_prob"])
        logprob_corrupt_patched = float(corrupt_metrics["joint_logprob"])
        prob_corrupt_patched_multi = float(np.exp(logprob_corrupt_patched)) if np.isfinite(logprob_corrupt_patched) else float("nan")
        nll_corrupt_patched = float(-logprob_corrupt_patched / n_tokens) if (n_tokens and np.isfinite(logprob_corrupt_patched)) else float("nan")

    # Shuffle-index control (optional)
    prob_shuffle_patched_first = float("nan")
    prob_shuffle_patched_multi = float("nan")
    logprob_shuffle_patched = float("nan")
    nll_shuffle_patched = float("nan")
    if target_id >= 0 and shuffle_patch_feats is not None:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            shuffle_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        shuffle_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_shuffle_patched_first = float(shuffle_metrics["first_prob"])
        logprob_shuffle_patched = float(shuffle_metrics["joint_logprob"])
        prob_shuffle_patched_multi = float(np.exp(logprob_shuffle_patched)) if np.isfinite(logprob_shuffle_patched) else float("nan")
        nll_shuffle_patched = float(-logprob_shuffle_patched / n_tokens) if (n_tokens and np.isfinite(logprob_shuffle_patched)) else float("nan")

    # Gaussian-noise control (optional)
    prob_gauss_patched_first = float("nan")
    prob_gauss_patched_multi = float("nan")
    logprob_gauss_patched = float("nan")
    nll_gauss_patched = float("nan")
    if target_id >= 0 and gauss_patch_feats is not None:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            gauss_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        gauss_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_gauss_patched_first = float(gauss_metrics["first_prob"])
        logprob_gauss_patched = float(gauss_metrics["joint_logprob"])
        prob_gauss_patched_multi = float(np.exp(logprob_gauss_patched)) if np.isfinite(logprob_gauss_patched) else float("nan")
        nll_gauss_patched = float(-logprob_gauss_patched / n_tokens) if (n_tokens and np.isfinite(logprob_gauss_patched)) else float("nan")

    # Attention-only matched-magnitude control
    prob_attention_patched_first = float("nan")
    prob_attention_patched_multi = float("nan")
    logprob_attention_patched = float("nan")
    nll_attention_patched = float("nan")
    prob_attention_structured_first = float("nan")
    prob_attention_structured_multi = float("nan")
    logprob_attention_structured = float("nan")
    nll_attention_structured = float("nan")
    if target_id >= 0 and attention_patch_vector.numel() > 0:
        try:
            hook = register_attention_output_patch_hook(
                model,
                layer,
                attention_patch_vector,
                patch_position=patch_pos,
            )
            attention_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=zs_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
            )
            hook.remove()
            prob_attention_patched_first = float(attention_metrics["first_prob"])
            logprob_attention_patched = float(attention_metrics["joint_logprob"])
            prob_attention_patched_multi = float(np.exp(logprob_attention_patched)) if np.isfinite(logprob_attention_patched) else float("nan")
            nll_attention_patched = float(-logprob_attention_patched / n_tokens) if (n_tokens and np.isfinite(logprob_attention_patched)) else float("nan")
        except AttributeError:
            # Some architectures may expose attention modules under different names.
            # Keep the control as NaN rather than failing the full run.
            pass

    if target_id >= 0 and decoded_patch.numel() > 0:
        try:
            hook = register_attention_output_patch_hook(
                model,
                layer,
                decoded_patch,
                patch_position=patch_pos,
            )
            attention_structured_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=zs_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
            )
            hook.remove()
            prob_attention_structured_first = float(attention_structured_metrics["first_prob"])
            logprob_attention_structured = float(attention_structured_metrics["joint_logprob"])
            prob_attention_structured_multi = (
                float(np.exp(logprob_attention_structured))
                if np.isfinite(logprob_attention_structured)
                else float("nan")
            )
            nll_attention_structured = (
                float(-logprob_attention_structured / n_tokens)
                if (n_tokens and np.isfinite(logprob_attention_structured))
                else float("nan")
            )
        except AttributeError:
            pass

    # Random-basis control (SRHT top-k projected back to feature space)
    prob_basis_patched_first = float("nan")
    prob_basis_patched_multi = float("nan")
    logprob_basis_patched = float("nan")
    nll_basis_patched = float("nan")
    if bool(enable_basis_control) and target_id >= 0 and basis_patch_feats is not None:
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            basis_patch_feats,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        basis_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_basis_patched_first = float(basis_metrics["first_prob"])
        logprob_basis_patched = float(basis_metrics["joint_logprob"])
        prob_basis_patched_multi = float(np.exp(logprob_basis_patched)) if np.isfinite(logprob_basis_patched) else float("nan")
        nll_basis_patched = float(-logprob_basis_patched / n_tokens) if (n_tokens and np.isfinite(logprob_basis_patched)) else float("nan")

    # Rescue necessity test: Patch ZS with features that have the selected indices
    # REMOVED (zeroed out). If the rescue effect vanishes, those specific features
    # are necessary for the rescue.
    #
    # IMPLEMENTATION NOTE (Feb 2026): Previously used two hooks (patch + ablate)
    # on the same module, which created an undefined interaction because the
    # ablation hook's residual calculation used the already-patched output.
    # Now uses a single hook with a modified patch vector instead.
    prob_patch_then_ablate_first = float("nan")
    prob_patch_then_ablate_multi = float("nan")
    if target_id >= 0 and idx.numel() > 0:
        # Build patch with selected features removed
        if patch_style == "sparse":
            # Sparse mode: the patch IS only the selected features. Removing them
            # gives an empty patch (all zeros), which is equivalent to no patch.
            # This makes the test trivially pe_first, so use substitute-style
            # for this specific diagnostic regardless of main patch_style.
            if zs_feats is not None:
                patch_minus_selected = zs_feats.clone()  # restore ZS values at idx
            else:
                # Can't do substitute without ZS features; fall back to zero patch.
                patch_minus_selected = torch.zeros_like(patch_feats)
        else:
            # Substitute mode: start from the full patch, restore ZS values at
            # selected indices (undoing the substitution).
            patch_minus_selected = patch_feats.clone()
            if zs_feats is not None:
                patch_minus_selected[idx] = zs_feats[idx]
            else:
                patch_minus_selected[idx] = 0.0

        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            patch_minus_selected,
            patch_position=patch_pos,
            target_output_norm=patch_norm_target,
        )
        patch_ablate_metrics = _teacher_forced_metrics_from_input_ids(
            model=model,
            input_ids=zs_input_ids,
            target_ids=target_ids,
            target_id=target_id,
            device=device,
        )
        hook.remove()
        prob_patch_then_ablate_first = float(patch_ablate_metrics["first_prob"])
        logprob_patch_ablate = float(patch_ablate_metrics["joint_logprob"])
        prob_patch_then_ablate_multi = float(np.exp(logprob_patch_ablate)) if np.isfinite(logprob_patch_ablate) else float("nan")

    # Compute necessity metrics
    pe_necessity_first = (
        prob_patched_first - prob_patch_then_ablate_first
        if not (np.isnan(prob_patched_first) or np.isnan(prob_patch_then_ablate_first))
        else float("nan")
    )
    pe_necessity_multi = (
        prob_patched_multi - prob_patch_then_ablate_multi
        if not (np.isnan(prob_patched_multi) or np.isnan(prob_patch_then_ablate_multi))
        else float("nan")
    )

    # === Generation evaluation (optional) ===
    # When enabled, generates free-form text for ZS, ICL, and patched conditions
    # and detects the dominant Unicode script of each output.

    # New Advanced Controls
    prob_auto_scale_patched_first = float("nan")
    prob_auto_scale_patched_multi = float("nan")
    logprob_auto_scale = float("nan")
    nll_auto_scale = float("nan")

    prob_auto_shift_patched_first = float("nan")
    prob_auto_shift_patched_multi = float("nan")
    logprob_auto_shift = float("nan")
    nll_auto_shift = float("nan")

    prob_mean_residual_patched_first = float("nan")
    prob_mean_residual_patched_multi = float("nan")
    logprob_mean_residual = float("nan")
    nll_mean_residual = float("nan")

    prob_empty_prompt_patched_first = float("nan")
    prob_empty_prompt_patched_multi = float("nan")
    logprob_empty_prompt = float("nan")
    nll_empty_prompt = float("nan")

    prob_cross_task_patched_first = float("nan")
    prob_cross_task_patched_multi = float("nan")
    logprob_cross_task = float("nan")
    nll_cross_task = float("nan")

    if target_id >= 0:
        # 1. Auto-Scale ZS Control
        if zs_feats is not None:
            # Scale ZS features to match ICL magnitude
            icl_norm = float(torch.norm(icl_feats).item())
            zs_norm = float(torch.norm(zs_feats).item())
            scale_factor = icl_norm / max(1e-6, zs_norm)
            auto_scale_feats = zs_feats * scale_factor
            
            # For sparse patch style, we zero out everything except the selected top-k
            # but using the scaled ZS values instead of ICL values.
            if patch_style == "sparse":
                auto_scale_patch = torch.zeros_like(zs_feats)
                auto_scale_patch[idx] = auto_scale_feats[idx]
            else:
                # Substitute style: just scale everything (or scale top-k? Let's follow substitute logic)
                auto_scale_patch = zs_feats.clone()
                auto_scale_patch[idx] = auto_scale_feats[idx]
                
            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                auto_scale_patch,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            auto_scale_metrics = _teacher_forced_metrics_from_input_ids(
                model=model, input_ids=zs_input_ids, target_ids=target_ids, target_id=target_id, device=device
            )
            hook.remove()
            prob_auto_scale_patched_first = float(auto_scale_metrics["first_prob"])
            logprob_auto_scale = float(auto_scale_metrics["joint_logprob"])
            prob_auto_scale_patched_multi = float(np.exp(logprob_auto_scale)) if np.isfinite(logprob_auto_scale) else float("nan")
            nll_auto_scale = float(-logprob_auto_scale / n_tokens) if (n_tokens and np.isfinite(logprob_auto_scale)) else float("nan")

            # 1b. Additive shift control (threshold/non-linearity confound check)
            add_shift_patch = torch.zeros_like(zs_feats) if patch_style == "sparse" else zs_feats.clone()
            if idx.numel() > 0:
                shift = float(torch.mean((icl_feats[idx] - zs_feats[idx]).float()).item())
                add_vals = zs_feats[idx] + shift
                add_shift_patch[idx] = add_vals
            elif patch_style != "sparse":
                add_shift_patch = zs_feats.clone()

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                add_shift_patch,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            auto_shift_metrics = _teacher_forced_metrics_from_input_ids(
                model=model,
                input_ids=zs_input_ids,
                target_ids=target_ids,
                target_id=target_id,
                device=device,
            )
            hook.remove()
            prob_auto_shift_patched_first = float(auto_shift_metrics["first_prob"])
            logprob_auto_shift = float(auto_shift_metrics["joint_logprob"])
            prob_auto_shift_patched_multi = (
                float(np.exp(logprob_auto_shift)) if np.isfinite(logprob_auto_shift) else float("nan")
            )
            nll_auto_shift = (
                float(-logprob_auto_shift / n_tokens)
                if (n_tokens and np.isfinite(logprob_auto_shift))
                else float("nan")
            )

        # 2. Mean Residual Patch (ICL Residual into ZS pass)
        # Using mlp_in_icl and decoding icl_feats gives us the ICL residual.
        icl_contrib = transcoder.decode(icl_feats.unsqueeze(0)).squeeze(0)
        icl_residual = mlp_out_icl - icl_contrib
        
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            patch_feats,
            patch_position=patch_pos,
            residual_override=icl_residual,
            target_output_norm=patch_norm_target,
        )
        mean_res_metrics = _teacher_forced_metrics_from_input_ids(
            model=model, input_ids=zs_input_ids, target_ids=target_ids, target_id=target_id, device=device
        )
        hook.remove()
        prob_mean_residual_patched_first = float(mean_res_metrics["first_prob"])
        logprob_mean_residual = float(mean_res_metrics["joint_logprob"])
        prob_mean_residual_patched_multi = float(np.exp(logprob_mean_residual)) if np.isfinite(logprob_mean_residual) else float("nan")
        nll_mean_residual = float(-logprob_mean_residual / n_tokens) if (n_tokens and np.isfinite(logprob_mean_residual)) else float("nan")

        # 3. Empty Prompt Patch (Is the feature an output switch or input-conditioned?)
        empty_prompt = build_task_prompt(
            "",
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant=prompt_variant,
        )
        empty_text = apply_chat_template(tokenizer, empty_prompt)
        empty_input_ids = tokenizer(empty_text, return_tensors="pt").to(device).input_ids
        empty_prompt_len = int(empty_input_ids.shape[1])
        
        hook = register_transcoder_feature_patch_hook(
            model,
            transcoder,
            layer,
            patch_feats,
            patch_position=(
                int(empty_prompt_len)
                if patch_position_mode == "target_pos1"
                else int(empty_prompt_len - 1)
            ),
            target_output_norm=patch_norm_target,
        )
        empty_prompt_metrics = _teacher_forced_metrics_from_input_ids(
            model=model, input_ids=empty_input_ids, target_ids=target_ids, target_id=target_id, device=device
        )
        hook.remove()
        prob_empty_prompt_patched_first = float(empty_prompt_metrics["first_prob"])
        logprob_empty_prompt = float(empty_prompt_metrics["joint_logprob"])
        prob_empty_prompt_patched_multi = float(np.exp(logprob_empty_prompt)) if np.isfinite(logprob_empty_prompt) else float("nan")
        nll_empty_prompt = float(-logprob_empty_prompt / n_tokens) if (n_tokens and np.isfinite(logprob_empty_prompt)) else float("nan")

        # 4. Cross-Task Same-Script Control
        if cross_task_word is not None:
            # We evaluate the patch on a translation task (e.g. English -> Telugu language)
            ct_hindi = cross_task_word["hindi"]
            ct_target = cross_task_word.get("ood", cross_task_word.get("telugu", cross_task_word.get("hindi", "")))
            ct_english = cross_task_word["english"]
            # For translation, the input is English, output is target language in target script
            ct_prompt = build_task_prompt(
                ct_english,
                None,
                input_script_name="Latin",
                source_language="English",
                output_script_name=output_script_name,
                prompt_variant=prompt_variant,
            )
            ct_text = apply_chat_template(tokenizer, ct_prompt)
            ct_input_ids = tokenizer(ct_text, return_tensors="pt").to(device).input_ids
            ct_prompt_len = int(ct_input_ids.shape[1])
            
            ct_target_ids = tokenizer.encode(ct_target, add_special_tokens=False)
            ct_target_id = int(ct_target_ids[0]) if ct_target_ids else -1
            ct_n_tokens = len(ct_target_ids)
            
            if ct_target_id >= 0:
                hook = register_transcoder_feature_patch_hook(
                    model,
                    transcoder,
                    layer,
                    patch_feats,
                    patch_position=(
                        int(ct_prompt_len)
                        if patch_position_mode == "target_pos1"
                        else int(ct_prompt_len - 1)
                    ),
                    target_output_norm=patch_norm_target,
                )
                ct_metrics = _teacher_forced_metrics_from_input_ids(
                    model=model, input_ids=ct_input_ids, target_ids=ct_target_ids, target_id=ct_target_id, device=device
                )
                hook.remove()
                prob_cross_task_patched_first = float(ct_metrics["first_prob"])
                logprob_cross_task = float(ct_metrics["joint_logprob"])
                prob_cross_task_patched_multi = float(np.exp(logprob_cross_task)) if np.isfinite(logprob_cross_task) else float("nan")
                nll_cross_task = float(-logprob_cross_task / ct_n_tokens) if (ct_n_tokens and np.isfinite(logprob_cross_task)) else float("nan")

    gen_zs = ""
    gen_icl = ""
    gen_patched = ""
    script_zs = ""
    script_icl = ""
    script_patched = ""

    if eval_generation and target_id >= 0:
        try:
            def _generate(input_ids: torch.Tensor, hook=None) -> str:
                """Greedy-decode up to max_new_tokens from input_ids."""
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(tokenizer, "pad_token_id", None)
                        or getattr(tokenizer, "eos_token_id", 0),
                    )
                # Decode only the new tokens (after the prompt).
                new_tokens = out[0, input_ids.shape[1] :]
                return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            def _dominant_script(text: str) -> str:
                """Return the dominant Unicode script name in text."""
                from collections import Counter
                scripts = Counter()
                for ch in text:
                    if ch.isspace() or ch in ".,;:!?()-\"'":
                        continue
                    try:
                        name = unicodedata.name(ch, "")
                        # Unicode character names start with script block name
                        if name:
                            script = name.split()[0]
                            scripts[script] += 1
                    except ValueError:
                        pass
                if not scripts:
                    return ""
                return scripts.most_common(1)[0][0]

            gen_zs = _generate(zs_input_ids)
            script_zs = _dominant_script(gen_zs)

            gen_icl = _generate(icl_input_ids)
            script_icl = _dominant_script(gen_icl)

            hook = register_transcoder_feature_patch_hook(
                model,
                transcoder,
                layer,
                patch_feats,
                patch_position=patch_pos,
                target_output_norm=patch_norm_target,
            )
            gen_patched = _generate(zs_input_ids)
            hook.remove()
            script_patched = _dominant_script(gen_patched)
        except Exception:
            pass  # Generation eval is best-effort; don't fail the run.

    primary_prob_first = prob_patched_first
    primary_prob_multi = prob_patched_multi
    primary_logit_first = logit_patched_first
    primary_logprob = logprob_patched
    primary_nll = nll_patched

    control_primary_map = {
        "default": (
            prob_patched_first,
            prob_patched_multi,
            logit_patched_first,
            logprob_patched,
            nll_patched,
        ),
        "null_icl": (
            prob_null_patched_first,
            prob_null_patched_multi,
            float("nan"),
            logprob_null_patched,
            nll_null_patched,
        ),
        "random_icl": (
            prob_rand_patched_first,
            prob_rand_patched_multi,
            float("nan"),
            logprob_rand_patched,
            nll_rand_patched,
        ),
        "corrupt_icl": (
            prob_corrupt_patched_first,
            prob_corrupt_patched_multi,
            float("nan"),
            logprob_corrupt_patched,
            nll_corrupt_patched,
        ),
        "auto_scale_zs": (
            prob_auto_scale_patched_first,
            prob_auto_scale_patched_multi,
            float("nan"),
            logprob_auto_scale,
            nll_auto_scale,
        ),
        "auto_shift_zs": (
            prob_auto_shift_patched_first,
            prob_auto_shift_patched_multi,
            float("nan"),
            logprob_auto_shift,
            nll_auto_shift,
        ),
        "attention_only": (
            prob_attention_patched_first,
            prob_attention_patched_multi,
            float("nan"),
            logprob_attention_patched,
            nll_attention_patched,
        ),
        "attention_structured": (
            prob_attention_structured_first,
            prob_attention_structured_multi,
            float("nan"),
            logprob_attention_structured,
            nll_attention_structured,
        ),
        "basis_random": (
            prob_basis_patched_first,
            prob_basis_patched_multi,
            float("nan"),
            logprob_basis_patched,
            nll_basis_patched,
        ),
        "shuffle_random": (
            prob_shuffle_patched_first,
            prob_shuffle_patched_multi,
            float("nan"),
            logprob_shuffle_patched,
            nll_shuffle_patched,
        ),
        "gaussian_noise": (
            prob_gauss_patched_first,
            prob_gauss_patched_multi,
            float("nan"),
            logprob_gauss_patched,
            nll_gauss_patched,
        ),
        "mean_pool_patch": (
            prob_mean_pool_patched_first,
            prob_mean_pool_patched_multi,
            float("nan"),
            logprob_mean_pool_patched,
            nll_mean_pool_patched,
        ),
        "attn_head_ablation": (
            prob_attn_head_ablated_first,
            prob_attn_head_ablated_multi,
            float("nan"),
            logprob_attn_head_ablated,
            nll_attn_head_ablated,
        ),
    }

    if control_mode_resolved in control_primary_map:
        (
            primary_prob_first,
            primary_prob_multi,
            primary_logit_first,
            primary_logprob,
            primary_nll,
        ) = control_primary_map[control_mode_resolved]
    elif control_mode_resolved != "default":
        control_mode_notes.append(
            f"unknown control mode '{control_mode_resolved}'; default primary patch retained"
        )
        control_mode_resolved = "default"

    pe_first_primary = (
        primary_prob_first - prob_zs_first
        if np.isfinite(primary_prob_first) and np.isfinite(prob_zs_first)
        else float("nan")
    )
    pe_multi_primary = (
        primary_prob_multi - prob_zs_multi
        if np.isfinite(primary_prob_multi) and np.isfinite(prob_zs_multi)
        else float("nan")
    )
    pe_logit_primary = (
        primary_logit_first - logit_zs_first
        if np.isfinite(primary_logit_first) and np.isfinite(logit_zs_first)
        else float("nan")
    )
    delta_competitor_prob_patch = (
        prob_competitor_patched_first - prob_competitor_zs_first
        if np.isfinite(prob_competitor_patched_first)
        and np.isfinite(prob_competitor_zs_first)
        else float("nan")
    )
    delta_competitor_logit_patch = (
        logit_competitor_patched_first - logit_competitor_zs_first
        if np.isfinite(logit_competitor_patched_first)
        and np.isfinite(logit_competitor_zs_first)
        else float("nan")
    )

    return PatchResult(
        word_english=english,
        word_hindi=hindi,
        word_telugu=telugu,
        prob_zs_first=prob_zs_first,
        prob_zs_multi=prob_zs_multi,
        prob_icl_first=prob_icl_first,
        prob_icl_multi=prob_icl_multi,
        logit_zs_first=logit_zs_first,
        logit_icl_first=logit_icl_first,
        prob_competitor_zs_first=prob_competitor_zs_first,
        prob_competitor_icl_first=prob_competitor_icl_first,
        logit_competitor_zs_first=logit_competitor_zs_first,
        logit_competitor_icl_first=logit_competitor_icl_first,
        prob_patched_first=primary_prob_first,
        prob_patched_multi=primary_prob_multi,
        logit_patched_first=primary_logit_first,
        prob_competitor_patched_first=prob_competitor_patched_first,
        logit_competitor_patched_first=logit_competitor_patched_first,
        prob_rand_patched_first=prob_rand_patched_first,
        prob_rand_patched_multi=prob_rand_patched_multi,
        prob_corrupt_patched_first=prob_corrupt_patched_first,
        prob_corrupt_patched_multi=prob_corrupt_patched_multi,
        prob_null_patched_first=prob_null_patched_first,
        prob_null_patched_multi=prob_null_patched_multi,
        prob_mean_pool_patched_first=prob_mean_pool_patched_first,
        prob_mean_pool_patched_multi=prob_mean_pool_patched_multi,
        prob_shuffle_patched_first=prob_shuffle_patched_first,
        prob_shuffle_patched_multi=prob_shuffle_patched_multi,
        prob_gauss_patched_first=prob_gauss_patched_first,
        prob_gauss_patched_multi=prob_gauss_patched_multi,
        prob_ablated_first=prob_ablated_first,
        prob_ablated_multi=prob_ablated_multi,
        icl_lift_first=prob_icl_first - prob_zs_first,
        icl_lift_multi=prob_icl_multi - prob_zs_multi,
        pe_first=pe_first_primary,
        pe_multi=pe_multi_primary,
        pe_random_first=prob_rand_patched_first - prob_zs_first,
        pe_random_multi=prob_rand_patched_multi - prob_zs_multi,
        pe_corrupt_first=prob_corrupt_patched_first - prob_zs_first,
        pe_corrupt_multi=prob_corrupt_patched_multi - prob_zs_multi,
        pe_null_first=prob_null_patched_first - prob_zs_first,
        pe_null_multi=prob_null_patched_multi - prob_zs_multi,
        pe_mean_pool_first=prob_mean_pool_patched_first - prob_zs_first,
        pe_mean_pool_multi=prob_mean_pool_patched_multi - prob_zs_multi,
        pe_decoupled_first=prob_decoupled_patched_first - prob_zs_first,
        pe_decoupled_multi=prob_decoupled_patched_multi - prob_zs_multi,
        prob_auto_scale_patched_first=prob_auto_scale_patched_first,
        prob_auto_scale_patched_multi=prob_auto_scale_patched_multi,
        pe_auto_scale_first=prob_auto_scale_patched_first - prob_zs_first,
        pe_auto_scale_multi=prob_auto_scale_patched_multi - prob_zs_multi,
        prob_auto_shift_patched_first=prob_auto_shift_patched_first,
        prob_auto_shift_patched_multi=prob_auto_shift_patched_multi,
        pe_auto_shift_first=prob_auto_shift_patched_first - prob_zs_first,
        pe_auto_shift_multi=prob_auto_shift_patched_multi - prob_zs_multi,
        prob_mean_residual_patched_first=prob_mean_residual_patched_first,
        prob_mean_residual_patched_multi=prob_mean_residual_patched_multi,
        pe_mean_residual_first=prob_mean_residual_patched_first - prob_zs_first,
        pe_mean_residual_multi=prob_mean_residual_patched_multi - prob_zs_multi,
        prob_empty_prompt_patched_first=prob_empty_prompt_patched_first,
        prob_empty_prompt_patched_multi=prob_empty_prompt_patched_multi,
        pe_empty_prompt_first=prob_empty_prompt_patched_first - prob_zs_first,
        pe_empty_prompt_multi=prob_empty_prompt_patched_multi - prob_zs_multi,
        prob_cross_task_patched_first=prob_cross_task_patched_first,
        prob_cross_task_patched_multi=prob_cross_task_patched_multi,
        pe_cross_task_first=prob_cross_task_patched_first - prob_zs_first,
        pe_cross_task_multi=prob_cross_task_patched_multi - prob_zs_multi,
        prob_attn_head_ablated_first=prob_attn_head_ablated_first,
        prob_attn_head_ablated_multi=prob_attn_head_ablated_multi,
        pe_attn_head_ablation_first=prob_attn_head_ablated_first - prob_icl_first,
        pe_attn_head_ablation_multi=prob_attn_head_ablated_multi - prob_icl_multi,
        attn_head_indices=",".join(str(int(h)) for h in attn_head_indices),
        attn_sink_heads_excluded=int(attn_sink_heads_excluded),
        attn_sink_threshold=float(attn_sink_threshold),
        pe_logit_first=pe_logit_primary,
        pe_shuffle_first=prob_shuffle_patched_first - prob_zs_first,
        pe_shuffle_multi=prob_shuffle_patched_multi - prob_zs_multi,
        pe_gauss_first=prob_gauss_patched_first - prob_zs_first,
        pe_gauss_multi=prob_gauss_patched_multi - prob_zs_multi,
        prob_attention_patched_first=prob_attention_patched_first,
        prob_attention_patched_multi=prob_attention_patched_multi,
        pe_attention_first=prob_attention_patched_first - prob_zs_first,
        pe_attention_multi=prob_attention_patched_multi - prob_zs_multi,
        prob_attention_structured_first=prob_attention_structured_first,
        prob_attention_structured_multi=prob_attention_structured_multi,
        pe_attention_structured_first=prob_attention_structured_first - prob_zs_first,
        pe_attention_structured_multi=prob_attention_structured_multi - prob_zs_multi,
        prob_basis_patched_first=prob_basis_patched_first,
        prob_basis_patched_multi=prob_basis_patched_multi,
        pe_basis_first=prob_basis_patched_first - prob_zs_first,
        pe_basis_multi=prob_basis_patched_multi - prob_zs_multi,
        basis_control_method="srht_topk",
        ae_first=prob_ablated_first - prob_icl_first,
        ae_multi=prob_ablated_multi - prob_icl_multi,
        icl_lift_logit_first=logit_icl_first - logit_zs_first,
        delta_competitor_prob_patch=delta_competitor_prob_patch,
        delta_competitor_logit_patch=delta_competitor_logit_patch,
        prob_patch_then_ablate_first=prob_patch_then_ablate_first,
        prob_patch_then_ablate_multi=prob_patch_then_ablate_multi,
        pe_necessity_first=pe_necessity_first,
        pe_necessity_multi=pe_necessity_multi,
        active_features_icl=float(active_feats_icl),
        active_features_zs=float(active_feats_zs),
        feature_coverage_ratio=feature_coverage_ratio,
        selected_feature_fraction=selected_feature_fraction,
        reconstruction_cosine_icl=reconstruction_cosine_icl,
        reconstruction_rel_error_icl=reconstruction_rel_error_icl,
        reconstruction_mse_icl=reconstruction_mse_icl,
        feature_cosine_zs_icl=feature_cosine_zs_icl,
        feature_identity_jaccard_zs_icl=feature_identity_jaccard_zs_icl,
        decoded_patch_norm_ratio=decoded_patch_norm_ratio,
        decoded_patch_norm=decoded_patch_norm,
        latent_patch_norm_pre_geometry=latent_patch_norm_pre_geometry,
        latent_patch_norm_post_geometry=latent_patch_norm_post_geometry,
        decoded_patch_norm_pre_geometry=decoded_patch_norm_pre_geometry,
        decoded_patch_norm_post_geometry=decoded_patch_norm_post_geometry,
        mlp_in_icl_norm=mlp_in_icl_norm,
        mlp_out_icl_norm=mlp_out_icl_norm,
        mean_selected_feature_dla_target=float(
            dla_diag.get("mean_selected_feature_dla_target", float("nan"))
        ),
        top_selected_feature_dla_target=float(
            dla_diag.get("top_selected_feature_dla_target", float("nan"))
        ),
        mean_selected_feature_dla_competitor=float(
            dla_diag.get("mean_selected_feature_dla_competitor", float("nan"))
        ),
        dla_target_minus_competitor=float(
            dla_diag.get("dla_target_minus_competitor", float("nan"))
        ),
        n_target_tokens=n_tokens,
        logprob_zs=float(logprob_zs),
        logprob_icl=float(logprob_icl),
        logprob_patched=float(primary_logprob),
        logprob_rand_patched=float(logprob_rand_patched),
        logprob_corrupt_patched=float(logprob_corrupt_patched),
        logprob_null_patched=float(logprob_null_patched),
        logprob_mean_pool_patched=float(logprob_mean_pool_patched),
        logprob_shuffle_patched=float(logprob_shuffle_patched),
        logprob_gauss_patched=float(logprob_gauss_patched),
        logprob_attention_patched=float(logprob_attention_patched),
        logprob_basis_patched=float(logprob_basis_patched),
        logprob_ablated=float(logprob_ablated),
        logprob_decoupled_patched=float(logprob_decoupled_patched),
        logprob_auto_scale_patched=float(logprob_auto_scale),
        logprob_auto_shift_patched=float(logprob_auto_shift),
        logprob_mean_residual_patched=float(logprob_mean_residual),
        logprob_empty_prompt_patched=float(logprob_empty_prompt),
        logprob_cross_task_patched=float(logprob_cross_task),
        logprob_attn_head_ablated=float(logprob_attn_head_ablated),
        nll_per_token_zs=float(nll_zs),
        nll_per_token_icl=float(nll_icl),
        nll_per_token_patched=float(primary_nll),
        nll_per_token_rand_patched=float(nll_rand_patched),
        nll_per_token_corrupt_patched=float(nll_corrupt_patched),
        nll_per_token_null_patched=float(nll_null_patched),
        nll_per_token_mean_pool_patched=float(nll_mean_pool_patched),
        nll_per_token_shuffle_patched=float(nll_shuffle_patched),
        nll_per_token_gauss_patched=float(nll_gauss_patched),
        nll_per_token_attention_patched=float(nll_attention_patched),
        nll_per_token_attention_structured=float(nll_attention_structured),
        nll_per_token_basis_patched=float(nll_basis_patched),
        nll_per_token_ablated=float(nll_ablated),
        nll_per_token_decoupled_patched=float(nll_decoupled_patched),
        nll_per_token_auto_scale_patched=float(nll_auto_scale),
        nll_per_token_auto_shift_patched=float(nll_auto_shift),
        nll_per_token_mean_residual_patched=float(nll_mean_residual),
        nll_per_token_empty_prompt_patched=float(nll_empty_prompt),
        nll_per_token_cross_task_patched=float(nll_cross_task),
        nll_per_token_attn_head_ablated=float(nll_attn_head_ablated),
        nll_per_token_english_neutral_zs=float(nll_english_neutral_zs),
        nll_per_token_english_neutral_patched=float(nll_english_neutral_patched),
        nll_harm_english_neutral_patch=float(nll_harm_english_neutral_patch),
        n_input_tokens_ood=int(n_input_tokens_ood),
        icl_feature_position=int(icl_feature_position),
        zs_patch_position=int(zs_patch_position),
        rope_position_gap=int(rope_position_gap),
        bos_attention_zs_next_layer=float(bos_attention_zs_next_layer),
        bos_attention_patched_next_layer=float(bos_attention_patched_next_layer),
        delta_bos_attention_next_layer_patch=float(delta_bos_attention_next_layer_patch),
        prob_decoupled_target_first=float(prob_decoupled_target_first),
        layer=layer,
        topk=topk,
        seed=seed,
        patch_style=patch_style,
        feature_selection=feature_selection,
        feature_pooling=("mean_ood_tokens" if np.isfinite(prob_mean_pool_patched_first) else "last_token"),
        patch_geometry=str(patch_geometry),
        patch_geometry_clip_cap_used=float(clip_cap_used),
        patch_geometry_sign_scale_used=float(sign_scale_used),
        patch_geometry_fraction_clipped=float(patch_geometry_fraction_clipped),
        patch_position_mode=str(patch_position_mode),
        selected_feature_indices=",".join(str(int(i)) for i in selected_feature_indices),
        selected_feature_magnitudes_raw=[float(v) for v in selected_raw_vals.tolist()],
        max_selected_feature_zscore=float(max_selected_feature_zscore),
        selected_outlier_gt5sigma=bool(selected_outlier_gt5sigma),
        control_mode_requested=str(control_mode),
        control_mode_resolved=str(control_mode_resolved),
        control_mode_notes="; ".join(str(n) for n in control_mode_notes),
        gen_zs=gen_zs,
        gen_icl=gen_icl,
        gen_patched=gen_patched,
        script_zs=script_zs,
        script_icl=script_icl,
        script_patched=script_patched,
        selector_reference_mode=str(selector_reference_mode),
        query_span_found_zs=bool(span_zs is not None),
        query_span_found_icl=bool(span_icl is not None),
        query_span_found_selector_reference=bool(
            span_corrupt is not None if selector_reference_mode == "corrupt_icl" else span_zs is not None
        ),
        local_window_exceeded_zs=bool(zs_prompt_len > 1024),
        local_window_exceeded_icl=bool(icl_prompt_len > 1024),
        local_window_exceeded_selector_reference=bool(
            (corrupt_prompt_len if selector_reference_mode == "corrupt_icl" else zs_prompt_len) > 1024
        ),
        nll_pos1_zs=nll_pos1_zs,
        nll_pos2_zs=nll_pos2_zs,
        nll_pos3_zs=nll_pos3_zs,
        nll_pos1_icl=nll_pos1_icl,
        nll_pos2_icl=nll_pos2_icl,
        nll_pos3_icl=nll_pos3_icl,
        nll_pos1_patched=nll_pos1_patched,
        nll_pos2_patched=nll_pos2_patched,
        nll_pos3_patched=nll_pos3_patched,
    )


# ============================================================================
# STATISTICS
# ============================================================================


def compute_statistics(results: List[PatchResult]) -> Dict[str, Any]:
    """Compute aggregate statistics from patching results."""
    from scipy import stats
    try:
        from rescue_research.analysis.stats import (
            paired_permutation_pvalue as _paired_permutation_pvalue,
        )
    except Exception:
        _paired_permutation_pvalue = None

    def _bootstrap_mean_ci(
        values: np.ndarray, *, seed: int = 0, n_resamples: int = 2000
    ) -> Tuple[float, float]:
        v = values[np.isfinite(values)]
        if v.shape[0] < 2:
            return float("nan"), float("nan")
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, v.shape[0], size=(int(n_resamples), v.shape[0]))
        means = np.mean(v[idx], axis=1)
        lo = float(np.quantile(means, 0.025))
        hi = float(np.quantile(means, 0.975))
        return lo, hi

    def _paired_perm_pvalue(a: np.ndarray, b: np.ndarray, *, seed: int) -> float:
        if _paired_permutation_pvalue is None:
            return float("nan")
        if a.shape[0] != b.shape[0] or a.shape[0] < 2:
            return float("nan")
        try:
            return float(
                _paired_permutation_pvalue(
                    a.tolist(),
                    b.tolist(),
                    n_permutations=2000,
                    seed=int(seed),
                )
            )
        except Exception:
            return float("nan")

    pe_values = np.array([r.pe_first for r in results], dtype=np.float64)
    pe_random_values = np.array([r.pe_random_first for r in results], dtype=np.float64)
    pe_corrupt_values = np.array(
        [r.pe_corrupt_first for r in results], dtype=np.float64
    )
    pe_null_values = np.array([r.pe_null_first for r in results], dtype=np.float64)
    pe_mean_pool_values = np.array([r.pe_mean_pool_first for r in results], dtype=np.float64)
    pe_shuffle_values = np.array(
        [r.pe_shuffle_first for r in results], dtype=np.float64
    )
    pe_gauss_values = np.array([r.pe_gauss_first for r in results], dtype=np.float64)
    pe_attention_values = np.array(
        [r.pe_attention_first for r in results], dtype=np.float64
    )
    pe_attention_structured_values = np.array(
        [r.pe_attention_structured_first for r in results], dtype=np.float64
    )
    pe_basis_values = np.array([r.pe_basis_first for r in results], dtype=np.float64)
    pe_decoupled_values = np.array([r.pe_decoupled_first for r in results], dtype=np.float64)
    pe_auto_scale_values = np.array([r.pe_auto_scale_first for r in results], dtype=np.float64)
    pe_auto_shift_values = np.array([r.pe_auto_shift_first for r in results], dtype=np.float64)
    pe_mean_residual_values = np.array([r.pe_mean_residual_first for r in results], dtype=np.float64)
    pe_empty_prompt_values = np.array([r.pe_empty_prompt_first for r in results], dtype=np.float64)
    pe_cross_task_values = np.array([r.pe_cross_task_first for r in results], dtype=np.float64)
    pe_attn_head_ablation_values = np.array([r.pe_attn_head_ablation_first for r in results], dtype=np.float64)
    ae_values = np.array([r.ae_first for r in results], dtype=np.float64)
    icl_lift_values = np.array([r.icl_lift_first for r in results], dtype=np.float64)

    logit_pe_values = np.array([r.pe_logit_first for r in results], dtype=np.float64)
    logit_icl_lift_values = np.array(
        [r.icl_lift_logit_first for r in results], dtype=np.float64
    )

    pe_values_multi = np.array([r.pe_multi for r in results], dtype=np.float64)
    pe_random_values_multi = np.array(
        [r.pe_random_multi for r in results], dtype=np.float64
    )
    pe_corrupt_values_multi = np.array(
        [r.pe_corrupt_multi for r in results], dtype=np.float64
    )
    pe_null_values_multi = np.array([r.pe_null_multi for r in results], dtype=np.float64)
    pe_mean_pool_values_multi = np.array([r.pe_mean_pool_multi for r in results], dtype=np.float64)
    pe_shuffle_values_multi = np.array(
        [r.pe_shuffle_multi for r in results], dtype=np.float64
    )
    pe_gauss_values_multi = np.array([r.pe_gauss_multi for r in results], dtype=np.float64)
    pe_attention_values_multi = np.array(
        [r.pe_attention_multi for r in results], dtype=np.float64
    )
    pe_basis_values_multi = np.array(
        [r.pe_basis_multi for r in results], dtype=np.float64
    )
    pe_decoupled_values_multi = np.array([r.pe_decoupled_multi for r in results], dtype=np.float64)
    pe_auto_scale_values_multi = np.array([r.pe_auto_scale_multi for r in results], dtype=np.float64)
    pe_auto_shift_values_multi = np.array([r.pe_auto_shift_multi for r in results], dtype=np.float64)
    pe_mean_residual_values_multi = np.array([r.pe_mean_residual_multi for r in results], dtype=np.float64)
    pe_empty_prompt_values_multi = np.array([r.pe_empty_prompt_multi for r in results], dtype=np.float64)
    pe_cross_task_values_multi = np.array([r.pe_cross_task_multi for r in results], dtype=np.float64)
    pe_attn_head_ablation_values_multi = np.array([r.pe_attn_head_ablation_multi for r in results], dtype=np.float64)
    ae_values_multi = np.array([r.ae_multi for r in results], dtype=np.float64)
    icl_lift_values_multi = np.array(
        [r.icl_lift_multi for r in results], dtype=np.float64
    )

    # NLL/token diagnostics (stable in floor/ceiling regimes)
    nll_zs = np.array([r.nll_per_token_zs for r in results], dtype=np.float64)
    nll_icl = np.array([r.nll_per_token_icl for r in results], dtype=np.float64)
    nll_patched = np.array([r.nll_per_token_patched for r in results], dtype=np.float64)
    nll_rand = np.array(
        [r.nll_per_token_rand_patched for r in results], dtype=np.float64
    )
    nll_corrupt = np.array(
        [r.nll_per_token_corrupt_patched for r in results], dtype=np.float64
    )
    nll_null = np.array([r.nll_per_token_null_patched for r in results], dtype=np.float64)
    nll_mean_pool = np.array([r.nll_per_token_mean_pool_patched for r in results], dtype=np.float64)
    nll_shuffle = np.array(
        [r.nll_per_token_shuffle_patched for r in results], dtype=np.float64
    )
    nll_gauss = np.array(
        [r.nll_per_token_gauss_patched for r in results], dtype=np.float64
    )
    nll_attention = np.array(
        [r.nll_per_token_attention_patched for r in results], dtype=np.float64
    )
    nll_attention_structured = np.array(
        [r.nll_per_token_attention_structured for r in results], dtype=np.float64
    )
    nll_basis = np.array(
        [r.nll_per_token_basis_patched for r in results], dtype=np.float64
    )
    nll_decoupled = np.array([r.nll_per_token_decoupled_patched for r in results], dtype=np.float64)
    nll_auto_scale = np.array([r.nll_per_token_auto_scale_patched for r in results], dtype=np.float64)
    nll_auto_shift = np.array([r.nll_per_token_auto_shift_patched for r in results], dtype=np.float64)
    nll_mean_residual = np.array([r.nll_per_token_mean_residual_patched for r in results], dtype=np.float64)
    nll_empty_prompt = np.array([r.nll_per_token_empty_prompt_patched for r in results], dtype=np.float64)
    nll_cross_task = np.array([r.nll_per_token_cross_task_patched for r in results], dtype=np.float64)
    nll_attn_head_ablated = np.array([r.nll_per_token_attn_head_ablated for r in results], dtype=np.float64)
    nll_english_neutral_zs = np.array(
        [r.nll_per_token_english_neutral_zs for r in results], dtype=np.float64
    )
    nll_english_neutral_patched = np.array(
        [r.nll_per_token_english_neutral_patched for r in results], dtype=np.float64
    )
    nll_harm_english_neutral = np.array(
        [r.nll_harm_english_neutral_patch for r in results], dtype=np.float64
    )
    nll_abl = np.array([r.nll_per_token_ablated for r in results], dtype=np.float64)
    nll_pos1_zs = np.array([r.nll_pos1_zs for r in results], dtype=np.float64)
    nll_pos2_zs = np.array([r.nll_pos2_zs for r in results], dtype=np.float64)
    nll_pos3_zs = np.array([r.nll_pos3_zs for r in results], dtype=np.float64)
    nll_pos1_icl = np.array([r.nll_pos1_icl for r in results], dtype=np.float64)
    nll_pos2_icl = np.array([r.nll_pos2_icl for r in results], dtype=np.float64)
    nll_pos3_icl = np.array([r.nll_pos3_icl for r in results], dtype=np.float64)
    nll_pos1_patch = np.array([r.nll_pos1_patched for r in results], dtype=np.float64)
    nll_pos2_patch = np.array([r.nll_pos2_patched for r in results], dtype=np.float64)
    nll_pos3_patch = np.array([r.nll_pos3_patched for r in results], dtype=np.float64)
    span_found_zs = np.array([1.0 if r.query_span_found_zs else 0.0 for r in results], dtype=np.float64)
    span_found_icl = np.array([1.0 if r.query_span_found_icl else 0.0 for r in results], dtype=np.float64)
    span_found_selector = np.array(
        [1.0 if r.query_span_found_selector_reference else 0.0 for r in results], dtype=np.float64
    )
    local_window_zs = np.array([1.0 if r.local_window_exceeded_zs else 0.0 for r in results], dtype=np.float64)
    local_window_icl = np.array([1.0 if r.local_window_exceeded_icl else 0.0 for r in results], dtype=np.float64)
    local_window_selector = np.array(
        [1.0 if r.local_window_exceeded_selector_reference else 0.0 for r in results], dtype=np.float64
    )

    prob_comp_zs = np.array([r.prob_competitor_zs_first for r in results], dtype=np.float64)
    prob_comp_icl = np.array([r.prob_competitor_icl_first for r in results], dtype=np.float64)
    prob_comp_patch = np.array([r.prob_competitor_patched_first for r in results], dtype=np.float64)
    delta_comp_prob = np.array([r.delta_competitor_prob_patch for r in results], dtype=np.float64)
    delta_comp_logit = np.array([r.delta_competitor_logit_patch for r in results], dtype=np.float64)

    logit_zs_vals = np.array([r.logit_zs_first for r in results], dtype=np.float64)
    logit_icl_vals = np.array([r.logit_icl_first for r in results], dtype=np.float64)
    logit_patch_vals = np.array([r.logit_patched_first for r in results], dtype=np.float64)

    char_counts = np.array(
        [max(1, len(str(getattr(r, "word_hindi", "")))) for r in results], dtype=np.float64
    )
    logprob_zs_vals = np.array([r.logprob_zs for r in results], dtype=np.float64)
    logprob_icl_vals = np.array([r.logprob_icl for r in results], dtype=np.float64)
    logprob_patch_vals = np.array([r.logprob_patched for r in results], dtype=np.float64)
    nll_char_zs = np.where(np.isfinite(logprob_zs_vals), -logprob_zs_vals / np.maximum(1.0, char_counts), np.nan)
    nll_char_icl = np.where(np.isfinite(logprob_icl_vals), -logprob_icl_vals / np.maximum(1.0, char_counts), np.nan)
    nll_char_patch = np.where(np.isfinite(logprob_patch_vals), -logprob_patch_vals / np.maximum(1.0, char_counts), np.nan)

    active_icl_values = np.array(
        [r.active_features_icl for r in results], dtype=np.float64
    )
    active_zs_values = np.array(
        [r.active_features_zs for r in results], dtype=np.float64
    )
    coverage_values = np.array(
        [r.feature_coverage_ratio for r in results], dtype=np.float64
    )
    selected_frac_values = np.array(
        [r.selected_feature_fraction for r in results], dtype=np.float64
    )
    rec_cos_values = np.array(
        [r.reconstruction_cosine_icl for r in results], dtype=np.float64
    )
    rec_rel_values = np.array(
        [r.reconstruction_rel_error_icl for r in results], dtype=np.float64
    )
    rec_mse_values = np.array(
        [r.reconstruction_mse_icl for r in results], dtype=np.float64
    )
    feature_cos_values = np.array(
        [r.feature_cosine_zs_icl for r in results], dtype=np.float64
    )
    feature_jaccard_values = np.array(
        [r.feature_identity_jaccard_zs_icl for r in results], dtype=np.float64
    )
    patch_norm_ratio_values = np.array(
        [r.decoded_patch_norm_ratio for r in results], dtype=np.float64
    )
    dla_target_values = np.array(
        [getattr(r, "mean_selected_feature_dla_target", float("nan")) for r in results],
        dtype=np.float64,
    )
    dla_target_top_values = np.array(
        [getattr(r, "top_selected_feature_dla_target", float("nan")) for r in results],
        dtype=np.float64,
    )
    dla_comp_values = np.array(
        [
            getattr(r, "mean_selected_feature_dla_competitor", float("nan"))
            for r in results
        ],
        dtype=np.float64,
    )
    dla_delta_values = np.array(
        [getattr(r, "dla_target_minus_competitor", float("nan")) for r in results],
        dtype=np.float64,
    )

    script_zs_vals = [r.script_zs for r in results]
    script_icl_vals = [r.script_icl for r in results]
    script_patched_vals = [r.script_patched for r in results]
    
    decoupled_target_values = np.array([r.prob_decoupled_target_first for r in results], dtype=np.float64)
    attn_sink_excluded_values = np.array([getattr(r, "attn_sink_heads_excluded", 0) for r in results], dtype=np.float64)
    rope_gap_values = np.array([float(getattr(r, "rope_position_gap", float("nan"))) for r in results], dtype=np.float64)
    bos_next_zs_values = np.array(
        [float(getattr(r, "bos_attention_zs_next_layer", float("nan"))) for r in results],
        dtype=np.float64,
    )
    bos_next_patch_values = np.array(
        [float(getattr(r, "bos_attention_patched_next_layer", float("nan"))) for r in results],
        dtype=np.float64,
    )
    bos_next_delta_values = np.array(
        [
            float(getattr(r, "delta_bos_attention_next_layer_patch", float("nan")))
            for r in results
        ],
        dtype=np.float64,
    )

    max_selected_z_values = np.array(
        [getattr(r, "max_selected_feature_zscore", float("nan")) for r in results],
        dtype=np.float64,
    )
    selected_outlier_flags = np.array(
        [
            1.0
            if bool(getattr(r, "selected_outlier_gt5sigma", False))
            else 0.0
            for r in results
        ],
        dtype=np.float64,
    )

    input_token_counts = np.array(
        [float(getattr(r, "n_input_tokens_ood", float("nan"))) for r in results],
        dtype=np.float64,
    )
    input_char_counts = np.array(
        [max(1.0, float(len(str(getattr(r, "word_telugu", ""))))) for r in results],
        dtype=np.float64,
    )
    input_tokens_per_char = np.where(
        np.isfinite(input_token_counts),
        input_token_counts / np.maximum(1.0, input_char_counts),
        np.nan,
    )

    # Basic stats
    pe_mask = np.isfinite(pe_values)
    pe_random_mask = np.isfinite(pe_random_values)
    pe_corrupt_mask = np.isfinite(pe_corrupt_values)
    pe_null_mask = np.isfinite(pe_null_values)
    pe_mean_pool_mask = np.isfinite(pe_mean_pool_values)
    pe_shuffle_mask = np.isfinite(pe_shuffle_values)
    pe_gauss_mask = np.isfinite(pe_gauss_values)
    pe_attention_mask = np.isfinite(pe_attention_values)
    pe_attention_structured_mask = np.isfinite(pe_attention_structured_values)
    pe_basis_mask = np.isfinite(pe_basis_values)
    pe_decoupled_mask = np.isfinite(pe_decoupled_values)
    pe_auto_scale_mask = np.isfinite(pe_auto_scale_values)
    pe_auto_shift_mask = np.isfinite(pe_auto_shift_values)
    pe_mean_residual_mask = np.isfinite(pe_mean_residual_values)
    pe_empty_prompt_mask = np.isfinite(pe_empty_prompt_values)
    pe_cross_task_mask = np.isfinite(pe_cross_task_values)
    pe_attn_head_ablation_mask = np.isfinite(pe_attn_head_ablation_values)
    decoupled_target_mask = np.isfinite(decoupled_target_values)
    attn_sink_excluded_mask = np.isfinite(attn_sink_excluded_values)
    rope_gap_mask = np.isfinite(rope_gap_values)
    bos_next_zs_mask = np.isfinite(bos_next_zs_values)
    bos_next_patch_mask = np.isfinite(bos_next_patch_values)
    bos_next_delta_mask = np.isfinite(bos_next_delta_values)
    comp_zs_mask = np.isfinite(prob_comp_zs)
    comp_icl_mask = np.isfinite(prob_comp_icl)
    comp_patch_mask = np.isfinite(prob_comp_patch)
    delta_comp_prob_mask = np.isfinite(delta_comp_prob)
    delta_comp_logit_mask = np.isfinite(delta_comp_logit)
    dla_target_mask = np.isfinite(dla_target_values)
    dla_target_top_mask = np.isfinite(dla_target_top_values)
    dla_comp_mask = np.isfinite(dla_comp_values)
    dla_delta_mask = np.isfinite(dla_delta_values)
    nll_harm_english_mask = np.isfinite(nll_harm_english_neutral)
    max_selected_z_mask = np.isfinite(max_selected_z_values)
    selected_outlier_mask = np.isfinite(selected_outlier_flags)
    input_token_mask = np.isfinite(input_token_counts)
    input_tpc_mask = np.isfinite(input_tokens_per_char)
    ae_mask = np.isfinite(ae_values)
    icl_mask = np.isfinite(icl_lift_values)

    logit_pe_mask = np.isfinite(logit_pe_values)
    logit_icl_mask = np.isfinite(logit_icl_lift_values)

    pe_mask_multi = np.isfinite(pe_values_multi)
    pe_random_mask_multi = np.isfinite(pe_random_values_multi)
    pe_corrupt_mask_multi = np.isfinite(pe_corrupt_values_multi)
    pe_null_mask_multi = np.isfinite(pe_null_values_multi)
    pe_mean_pool_mask_multi = np.isfinite(pe_mean_pool_values_multi)
    pe_shuffle_mask_multi = np.isfinite(pe_shuffle_values_multi)
    pe_gauss_mask_multi = np.isfinite(pe_gauss_values_multi)
    pe_attention_mask_multi = np.isfinite(pe_attention_values_multi)
    pe_basis_mask_multi = np.isfinite(pe_basis_values_multi)
    pe_decoupled_mask_multi = np.isfinite(pe_decoupled_values_multi)
    pe_auto_scale_mask_multi = np.isfinite(pe_auto_scale_values_multi)
    pe_auto_shift_mask_multi = np.isfinite(pe_auto_shift_values_multi)
    pe_mean_residual_mask_multi = np.isfinite(pe_mean_residual_values_multi)
    pe_empty_prompt_mask_multi = np.isfinite(pe_empty_prompt_values_multi)
    pe_cross_task_mask_multi = np.isfinite(pe_cross_task_values_multi)
    pe_attn_head_ablation_mask_multi = np.isfinite(pe_attn_head_ablation_values_multi)
    ae_mask_multi = np.isfinite(ae_values_multi)
    icl_mask_multi = np.isfinite(icl_lift_values_multi)
    active_icl_mask = np.isfinite(active_icl_values)
    active_zs_mask = np.isfinite(active_zs_values)
    coverage_mask = np.isfinite(coverage_values)
    selected_frac_mask = np.isfinite(selected_frac_values)
    rec_cos_mask = np.isfinite(rec_cos_values)
    rec_rel_mask = np.isfinite(rec_rel_values)
    rec_mse_mask = np.isfinite(rec_mse_values)
    feature_cos_mask = np.isfinite(feature_cos_values)
    feature_jaccard_mask = np.isfinite(feature_jaccard_values)
    patch_norm_ratio_mask = np.isfinite(patch_norm_ratio_values)

    stats_dict = {
        "n_samples": len(results),
        "control_mode_requested": str(getattr(results[0], "control_mode_requested", "default")) if results else "default",
        "control_mode_resolved": str(getattr(results[0], "control_mode_resolved", "default")) if results else "default",
        "mean_pe": float(np.mean(pe_values[pe_mask]))
        if bool(np.any(pe_mask))
        else float("nan"),
        "std_pe": float(np.std(pe_values[pe_mask]))
        if bool(np.any(pe_mask))
        else float("nan"),
        "median_pe": float(np.median(pe_values[pe_mask]))
        if bool(np.any(pe_mask))
        else float("nan"),
        "positive_pe_rate": float(np.mean(pe_values[pe_mask] > 0))
        if bool(np.any(pe_mask))
        else float("nan"),
        "strong_pe_rate": float(np.mean(pe_values[pe_mask] > 0.1))
        if bool(np.any(pe_mask))
        else float("nan"),
        "mean_pe_random": float(np.mean(pe_random_values[pe_random_mask]))
        if bool(np.any(pe_random_mask))
        else float("nan"),
        "std_pe_random": float(np.std(pe_random_values[pe_random_mask]))
        if bool(np.any(pe_random_mask))
        else float("nan"),
        "mean_pe_corrupt": float(np.mean(pe_corrupt_values[pe_corrupt_mask]))
        if bool(np.any(pe_corrupt_mask))
        else float("nan"),
        "std_pe_corrupt": float(np.std(pe_corrupt_values[pe_corrupt_mask]))
        if bool(np.any(pe_corrupt_mask))
        else float("nan"),
        "mean_pe_null": float(np.mean(pe_null_values[pe_null_mask]))
        if bool(np.any(pe_null_mask))
        else float("nan"),
        "std_pe_null": float(np.std(pe_null_values[pe_null_mask]))
        if bool(np.any(pe_null_mask))
        else float("nan"),
        "mean_pe_mean_pool": float(np.mean(pe_mean_pool_values[pe_mean_pool_mask]))
        if bool(np.any(pe_mean_pool_mask))
        else float("nan"),
        "mean_pe_shuffle": float(np.mean(pe_shuffle_values[pe_shuffle_mask]))
        if bool(np.any(pe_shuffle_mask))
        else float("nan"),
        "std_pe_shuffle": float(np.std(pe_shuffle_values[pe_shuffle_mask]))
        if bool(np.any(pe_shuffle_mask))
        else float("nan"),
        "mean_pe_gauss": float(np.mean(pe_gauss_values[pe_gauss_mask]))
        if bool(np.any(pe_gauss_mask))
        else float("nan"),
        "std_pe_gauss": float(np.std(pe_gauss_values[pe_gauss_mask]))
        if bool(np.any(pe_gauss_mask))
        else float("nan"),
        "mean_pe_attention": float(np.mean(pe_attention_values[pe_attention_mask]))
        if bool(np.any(pe_attention_mask))
        else float("nan"),
        "std_pe_attention": float(np.std(pe_attention_values[pe_attention_mask]))
        if bool(np.any(pe_attention_mask))
        else float("nan"),
        "mean_pe_attention_structured": float(
            np.mean(pe_attention_structured_values[pe_attention_structured_mask])
        )
        if bool(np.any(pe_attention_structured_mask))
        else float("nan"),
        "std_pe_attention_structured": float(
            np.std(pe_attention_structured_values[pe_attention_structured_mask])
        )
        if bool(np.any(pe_attention_structured_mask))
        else float("nan"),
        "mean_pe_basis": float(np.mean(pe_basis_values[pe_basis_mask]))
        if bool(np.any(pe_basis_mask))
        else float("nan"),
        "std_pe_basis": float(np.std(pe_basis_values[pe_basis_mask]))
        if bool(np.any(pe_basis_mask))
        else float("nan"),
        "mean_pe_decoupled": float(np.mean(pe_decoupled_values[pe_decoupled_mask]))
        if bool(np.any(pe_decoupled_mask))
        else float("nan"),
        "std_pe_decoupled": float(np.std(pe_decoupled_values[pe_decoupled_mask]))
        if bool(np.any(pe_decoupled_mask))
        else float("nan"),
        "mean_pe_auto_scale": float(np.mean(pe_auto_scale_values[pe_auto_scale_mask]))
        if bool(np.any(pe_auto_scale_mask))
        else float("nan"),
        "mean_pe_auto_shift": float(np.mean(pe_auto_shift_values[pe_auto_shift_mask]))
        if bool(np.any(pe_auto_shift_mask))
        else float("nan"),
        "mean_pe_mean_residual": float(np.mean(pe_mean_residual_values[pe_mean_residual_mask]))
        if bool(np.any(pe_mean_residual_mask))
        else float("nan"),
        "mean_pe_empty_prompt": float(np.mean(pe_empty_prompt_values[pe_empty_prompt_mask]))
        if bool(np.any(pe_empty_prompt_mask))
        else float("nan"),
        "mean_pe_cross_task": float(np.mean(pe_cross_task_values[pe_cross_task_mask]))
        if bool(np.any(pe_cross_task_mask))
        else float("nan"),
        "mean_pe_attn_head_ablation": float(np.mean(pe_attn_head_ablation_values[pe_attn_head_ablation_mask]))
        if bool(np.any(pe_attn_head_ablation_mask))
        else float("nan"),
        "mean_prob_decoupled_target_first": float(np.mean(decoupled_target_values[decoupled_target_mask]))
        if bool(np.any(decoupled_target_mask))
        else float("nan"),
        "mean_attn_sink_heads_excluded": float(np.mean(attn_sink_excluded_values[attn_sink_excluded_mask]))
        if bool(np.any(attn_sink_excluded_mask))
        else float("nan"),
        "attn_sink_threshold": float(getattr(results[0], "attn_sink_threshold", 0.80)) if results else 0.80,
        "mean_max_selected_feature_zscore": float(np.mean(max_selected_z_values[max_selected_z_mask]))
        if bool(np.any(max_selected_z_mask))
        else float("nan"),
        "selected_outlier_gt5sigma_rate": float(np.mean(selected_outlier_flags[selected_outlier_mask]))
        if bool(np.any(selected_outlier_mask))
        else float("nan"),
        "mean_input_tokens_ood": float(np.mean(input_token_counts[input_token_mask]))
        if bool(np.any(input_token_mask))
        else float("nan"),
        "mean_input_tokens_per_char_ood": float(np.mean(input_tokens_per_char[input_tpc_mask]))
        if bool(np.any(input_tpc_mask))
        else float("nan"),
        "input_fragmentation_rate_ge_3_tokens": float(np.mean(input_token_counts[input_token_mask] >= 3.0))
        if bool(np.any(input_token_mask))
        else float("nan"),
        "high_fragmentation_warning": bool(
            bool(np.any(input_token_mask))
            and float(np.mean(input_token_counts[input_token_mask] >= 3.0)) > 0.50
        ),
        "mean_rope_position_gap": float(np.mean(rope_gap_values[rope_gap_mask]))
        if bool(np.any(rope_gap_mask))
        else float("nan"),
        "mean_prob_competitor_zs_first": float(np.mean(prob_comp_zs[comp_zs_mask]))
        if bool(np.any(comp_zs_mask))
        else float("nan"),
        "mean_prob_competitor_icl_first": float(np.mean(prob_comp_icl[comp_icl_mask]))
        if bool(np.any(comp_icl_mask))
        else float("nan"),
        "mean_prob_competitor_patched_first": float(np.mean(prob_comp_patch[comp_patch_mask]))
        if bool(np.any(comp_patch_mask))
        else float("nan"),
        "mean_delta_competitor_prob_patch": float(np.mean(delta_comp_prob[delta_comp_prob_mask]))
        if bool(np.any(delta_comp_prob_mask))
        else float("nan"),
        "mean_delta_competitor_logit_patch": float(np.mean(delta_comp_logit[delta_comp_logit_mask]))
        if bool(np.any(delta_comp_logit_mask))
        else float("nan"),
        "mean_logit_zs_first": float(np.mean(logit_zs_vals[np.isfinite(logit_zs_vals)]))
        if bool(np.any(np.isfinite(logit_zs_vals)))
        else float("nan"),
        "mean_logit_icl_first": float(np.mean(logit_icl_vals[np.isfinite(logit_icl_vals)]))
        if bool(np.any(np.isfinite(logit_icl_vals)))
        else float("nan"),
        "mean_logit_patched_first": float(np.mean(logit_patch_vals[np.isfinite(logit_patch_vals)]))
        if bool(np.any(np.isfinite(logit_patch_vals)))
        else float("nan"),
        "icl_logit_rate_gt25": float(np.mean(logit_icl_vals[np.isfinite(logit_icl_vals)] > 25.0))
        if bool(np.any(np.isfinite(logit_icl_vals)))
        else float("nan"),
        "patched_logit_rate_gt25": float(np.mean(logit_patch_vals[np.isfinite(logit_patch_vals)] > 25.0))
        if bool(np.any(np.isfinite(logit_patch_vals)))
        else float("nan"),
        "softcap_saturation_risk": bool(
            (
                bool(np.any(np.isfinite(logit_icl_vals)))
                and float(np.mean(logit_icl_vals[np.isfinite(logit_icl_vals)])) > 25.0
            )
            or (
                bool(np.any(np.isfinite(logit_patch_vals)))
                and float(np.mean(logit_patch_vals[np.isfinite(logit_patch_vals)])) > 25.0
            )
        ),
        "mean_selected_feature_dla_target": float(np.mean(dla_target_values[dla_target_mask]))
        if bool(np.any(dla_target_mask))
        else float("nan"),
        "mean_top_selected_feature_dla_target": float(
            np.mean(dla_target_top_values[dla_target_top_mask])
        )
        if bool(np.any(dla_target_top_mask))
        else float("nan"),
        "mean_selected_feature_dla_competitor": float(np.mean(dla_comp_values[dla_comp_mask]))
        if bool(np.any(dla_comp_mask))
        else float("nan"),
        "mean_dla_target_minus_competitor": float(np.mean(dla_delta_values[dla_delta_mask]))
        if bool(np.any(dla_delta_mask))
        else float("nan"),
        "mean_bos_attention_zs_next_layer": float(np.mean(bos_next_zs_values[bos_next_zs_mask]))
        if bool(np.any(bos_next_zs_mask))
        else float("nan"),
        "mean_bos_attention_patched_next_layer": float(
            np.mean(bos_next_patch_values[bos_next_patch_mask])
        )
        if bool(np.any(bos_next_patch_mask))
        else float("nan"),
        "mean_delta_bos_attention_next_layer_patch": float(
            np.mean(bos_next_delta_values[bos_next_delta_mask])
        )
        if bool(np.any(bos_next_delta_mask))
        else float("nan"),
        "context_expectation_warning_rate": float(
            np.mean(bos_next_delta_values[bos_next_delta_mask] > 0.10)
        )
        if bool(np.any(bos_next_delta_mask))
        else float("nan"),
        "mean_nll_harm_english_neutral_patch": float(
            np.mean(nll_harm_english_neutral[nll_harm_english_mask])
        )
        if bool(np.any(nll_harm_english_mask))
        else float("nan"),
        "feature_collision_risk_rate": float(
            np.mean(nll_harm_english_neutral[nll_harm_english_mask] > 0.05)
        )
        if bool(np.any(nll_harm_english_mask))
        else float("nan"),
        "query_span_success_rate_zs": float(np.mean(span_found_zs))
        if span_found_zs.size
        else float("nan"),
        "query_span_success_rate_icl": float(np.mean(span_found_icl))
        if span_found_icl.size
        else float("nan"),
        "query_span_success_rate_selector_reference": float(np.mean(span_found_selector))
        if span_found_selector.size
        else float("nan"),
        "local_window_exceeded_rate_zs": float(np.mean(local_window_zs))
        if local_window_zs.size
        else float("nan"),
        "local_window_exceeded_rate_icl": float(np.mean(local_window_icl))
        if local_window_icl.size
        else float("nan"),
        "local_window_exceeded_rate_selector_reference": float(np.mean(local_window_selector))
        if local_window_selector.size
        else float("nan"),
        "mean_ae": float(np.mean(ae_values[ae_mask]))
        if bool(np.any(ae_mask))
        else float("nan"),
        "std_ae": float(np.std(ae_values[ae_mask]))
        if bool(np.any(ae_mask))
        else float("nan"),
        "negative_ae_rate": float(np.mean(ae_values[ae_mask] < 0))
        if bool(np.any(ae_mask))
        else float("nan"),
        # Bootstrap CIs for reviewer-facing uncertainty reporting.
        "ci_pe_95": list(_bootstrap_mean_ci(pe_values[pe_mask], seed=0))
        if bool(np.any(pe_mask))
        else [float("nan"), float("nan")],
        "ci_pe_corrupt_95": list(_bootstrap_mean_ci(pe_corrupt_values[pe_corrupt_mask], seed=0))
        if bool(np.any(pe_corrupt_mask))
        else [float("nan"), float("nan")],
        "ci_ae_95": list(_bootstrap_mean_ci(ae_values[ae_mask], seed=0))
        if bool(np.any(ae_mask))
        else [float("nan"), float("nan")],
        "mean_icl_lift": float(np.mean(icl_lift_values[icl_mask]))
        if bool(np.any(icl_mask))
        else float("nan"),
        "mean_logit_pe": float(np.mean(logit_pe_values[logit_pe_mask]))
        if bool(np.any(logit_pe_mask))
        else float("nan"),
        "std_logit_pe": float(np.std(logit_pe_values[logit_pe_mask]))
        if bool(np.any(logit_pe_mask))
        else float("nan"),
        # Alias using PE naming to keep reports consistent with probability-space PE.
        "mean_pe_logit": float(np.mean(logit_pe_values[logit_pe_mask]))
        if bool(np.any(logit_pe_mask))
        else float("nan"),
        "std_pe_logit": float(np.std(logit_pe_values[logit_pe_mask]))
        if bool(np.any(logit_pe_mask))
        else float("nan"),
        "mean_logit_icl_lift": float(np.mean(logit_icl_lift_values[logit_icl_mask]))
        if bool(np.any(logit_icl_mask))
        else float("nan"),
        # Multi-token (teacher-forced joint probability) counterparts.
        "mean_pe_multi": float(np.mean(pe_values_multi[pe_mask_multi]))
        if bool(np.any(pe_mask_multi))
        else float("nan"),
        "std_pe_multi": float(np.std(pe_values_multi[pe_mask_multi]))
        if bool(np.any(pe_mask_multi))
        else float("nan"),
        "median_pe_multi": float(np.median(pe_values_multi[pe_mask_multi]))
        if bool(np.any(pe_mask_multi))
        else float("nan"),
        "positive_pe_rate_multi": float(np.mean(pe_values_multi[pe_mask_multi] > 0))
        if bool(np.any(pe_mask_multi))
        else float("nan"),
        "strong_pe_rate_multi": float(np.mean(pe_values_multi[pe_mask_multi] > 0.1))
        if bool(np.any(pe_mask_multi))
        else float("nan"),
        "mean_pe_random_multi": float(
            np.mean(pe_random_values_multi[pe_random_mask_multi])
        )
        if bool(np.any(pe_random_mask_multi))
        else float("nan"),
        "std_pe_random_multi": float(
            np.std(pe_random_values_multi[pe_random_mask_multi])
        )
        if bool(np.any(pe_random_mask_multi))
        else float("nan"),
        "mean_pe_corrupt_multi": float(
            np.mean(pe_corrupt_values_multi[pe_corrupt_mask_multi])
        )
        if bool(np.any(pe_corrupt_mask_multi))
        else float("nan"),
        "std_pe_corrupt_multi": float(
            np.std(pe_corrupt_values_multi[pe_corrupt_mask_multi])
        )
        if bool(np.any(pe_corrupt_mask_multi))
        else float("nan"),
        "mean_pe_null_multi": float(
            np.mean(pe_null_values_multi[pe_null_mask_multi])
        )
        if bool(np.any(pe_null_mask_multi))
        else float("nan"),
        "std_pe_null_multi": float(
            np.std(pe_null_values_multi[pe_null_mask_multi])
        )
        if bool(np.any(pe_null_mask_multi))
        else float("nan"),
        "mean_pe_mean_pool_multi": float(
            np.mean(pe_mean_pool_values_multi[pe_mean_pool_mask_multi])
        )
        if bool(np.any(pe_mean_pool_mask_multi))
        else float("nan"),
        "mean_pe_shuffle_multi": float(
            np.mean(pe_shuffle_values_multi[pe_shuffle_mask_multi])
        )
        if bool(np.any(pe_shuffle_mask_multi))
        else float("nan"),
        "std_pe_shuffle_multi": float(
            np.std(pe_shuffle_values_multi[pe_shuffle_mask_multi])
        )
        if bool(np.any(pe_shuffle_mask_multi))
        else float("nan"),
        "mean_pe_gauss_multi": float(np.mean(pe_gauss_values_multi[pe_gauss_mask_multi]))
        if bool(np.any(pe_gauss_mask_multi))
        else float("nan"),
        "std_pe_gauss_multi": float(np.std(pe_gauss_values_multi[pe_gauss_mask_multi]))
        if bool(np.any(pe_gauss_mask_multi))
        else float("nan"),
        "mean_pe_attention_multi": float(
            np.mean(pe_attention_values_multi[pe_attention_mask_multi])
        )
        if bool(np.any(pe_attention_mask_multi))
        else float("nan"),
        "std_pe_attention_multi": float(
            np.std(pe_attention_values_multi[pe_attention_mask_multi])
        )
        if bool(np.any(pe_attention_mask_multi))
        else float("nan"),
        "mean_pe_basis_multi": float(
            np.mean(pe_basis_values_multi[pe_basis_mask_multi])
        )
        if bool(np.any(pe_basis_mask_multi))
        else float("nan"),
        "std_pe_basis_multi": float(
            np.std(pe_basis_values_multi[pe_basis_mask_multi])
        )
        if bool(np.any(pe_basis_mask_multi))
        else float("nan"),
        "mean_pe_auto_scale_multi": float(
            np.mean(pe_auto_scale_values_multi[pe_auto_scale_mask_multi])
        )
        if bool(np.any(pe_auto_scale_mask_multi))
        else float("nan"),
        "mean_pe_auto_shift_multi": float(
            np.mean(pe_auto_shift_values_multi[pe_auto_shift_mask_multi])
        )
        if bool(np.any(pe_auto_shift_mask_multi))
        else float("nan"),
        "mean_pe_mean_residual_multi": float(
            np.mean(pe_mean_residual_values_multi[pe_mean_residual_mask_multi])
        )
        if bool(np.any(pe_mean_residual_mask_multi))
        else float("nan"),
        "mean_pe_empty_prompt_multi": float(
            np.mean(pe_empty_prompt_values_multi[pe_empty_prompt_mask_multi])
        )
        if bool(np.any(pe_empty_prompt_mask_multi))
        else float("nan"),
        "mean_pe_cross_task_multi": float(
            np.mean(pe_cross_task_values_multi[pe_cross_task_mask_multi])
        )
        if bool(np.any(pe_cross_task_mask_multi))
        else float("nan"),
        "mean_pe_attn_head_ablation_multi": float(
            np.mean(pe_attn_head_ablation_values_multi[pe_attn_head_ablation_mask_multi])
        )
        if bool(np.any(pe_attn_head_ablation_mask_multi))
        else float("nan"),
        "mean_ae_multi": float(np.mean(ae_values_multi[ae_mask_multi]))
        if bool(np.any(ae_mask_multi))
        else float("nan"),
        "std_ae_multi": float(np.std(ae_values_multi[ae_mask_multi]))
        if bool(np.any(ae_mask_multi))
        else float("nan"),
        "negative_ae_rate_multi": float(np.mean(ae_values_multi[ae_mask_multi] < 0))
        if bool(np.any(ae_mask_multi))
        else float("nan"),
        "mean_icl_lift_multi": float(np.mean(icl_lift_values_multi[icl_mask_multi]))
        if bool(np.any(icl_mask_multi))
        else float("nan"),
        "mean_active_features_icl": float(np.mean(active_icl_values[active_icl_mask]))
        if bool(np.any(active_icl_mask))
        else float("nan"),
        "mean_active_features_zs": float(np.mean(active_zs_values[active_zs_mask]))
        if bool(np.any(active_zs_mask))
        else float("nan"),
        "mean_feature_coverage_ratio": float(np.mean(coverage_values[coverage_mask]))
        if bool(np.any(coverage_mask))
        else float("nan"),
        "mean_selected_feature_fraction": float(
            np.mean(selected_frac_values[selected_frac_mask])
        )
        if bool(np.any(selected_frac_mask))
        else float("nan"),
        "mean_reconstruction_cosine_icl": float(np.mean(rec_cos_values[rec_cos_mask]))
        if bool(np.any(rec_cos_mask))
        else float("nan"),
        "mean_reconstruction_rel_error_icl": float(np.mean(rec_rel_values[rec_rel_mask]))
        if bool(np.any(rec_rel_mask))
        else float("nan"),
        "mean_reconstruction_mse_icl": float(np.mean(rec_mse_values[rec_mse_mask]))
        if bool(np.any(rec_mse_mask))
        else float("nan"),
        "mean_feature_cosine_zs_icl": float(np.mean(feature_cos_values[feature_cos_mask]))
        if bool(np.any(feature_cos_mask))
        else float("nan"),
        "mean_feature_identity_jaccard_zs_icl": float(np.mean(feature_jaccard_values[feature_jaccard_mask]))
        if bool(np.any(feature_jaccard_mask))
        else float("nan"),
        "mean_decoded_patch_norm_ratio": float(
            np.mean(patch_norm_ratio_values[patch_norm_ratio_mask])
        )
        if bool(np.any(patch_norm_ratio_mask))
        else float("nan"),
    }
    
    # Latent Romanization / Script Tracking
    if any(s for s in script_zs_vals if s):
        valid_script = [i for i, s in enumerate(script_zs_vals) if s]
        if valid_script:
            latin_zs = sum(1 for i in valid_script if script_zs_vals[i] == "Latin")
            latin_icl = sum(1 for i in valid_script if script_icl_vals[i] == "Latin")
            latin_patched = sum(1 for i in valid_script if script_patched_vals[i] == "Latin")
            stats_dict["script_tracking"] = {
                "n_evaluated": len(valid_script),
                "latin_fraction_zs": float(latin_zs / len(valid_script)),
                "latin_fraction_icl": float(latin_icl / len(valid_script)),
                "latin_fraction_patched": float(latin_patched / len(valid_script)),
            }

    # Bootstrap CIs (paired over samples) for key first-token metrics.
    lo, hi = _bootstrap_mean_ci(pe_values, seed=0)
    stats_dict["mean_pe_ci95"] = {"low": lo, "high": hi}

    if bool(np.any(np.isfinite(pe_values) & np.isfinite(pe_random_values))):
        delta = pe_values - pe_random_values
        lo, hi = _bootstrap_mean_ci(delta, seed=1)
        stats_dict["mean_pe_minus_random_ci95"] = {"low": lo, "high": hi}
    if bool(np.any(np.isfinite(pe_values) & np.isfinite(pe_corrupt_values))):
        delta = pe_values - pe_corrupt_values
        lo, hi = _bootstrap_mean_ci(delta, seed=2)
        stats_dict["mean_pe_minus_corrupt_ci95"] = {"low": lo, "high": hi}
    if bool(np.any(np.isfinite(pe_values) & np.isfinite(pe_shuffle_values))):
        delta = pe_values - pe_shuffle_values
        lo, hi = _bootstrap_mean_ci(delta, seed=3)
        stats_dict["mean_pe_minus_shuffle_ci95"] = {"low": lo, "high": hi}
    if bool(np.any(np.isfinite(pe_values) & np.isfinite(pe_gauss_values))):
        delta = pe_values - pe_gauss_values
        lo, hi = _bootstrap_mean_ci(delta, seed=4)
        stats_dict["mean_pe_minus_gauss_ci95"] = {"low": lo, "high": hi}
    if bool(np.any(np.isfinite(pe_values) & np.isfinite(pe_basis_values))):
        delta = pe_values - pe_basis_values
        lo, hi = _bootstrap_mean_ci(delta, seed=5)
        stats_dict["mean_pe_minus_basis_ci95"] = {"low": lo, "high": hi}

    # NLL/token aggregates (lower is better; report deltas as (ZS - patched) so >0 means improvement).
    nll_mask = np.isfinite(nll_zs)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_zs"] = float(np.mean(nll_zs[nll_mask]))
    nll_mask = np.isfinite(nll_icl)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_icl"] = float(np.mean(nll_icl[nll_mask]))
    nll_mask = np.isfinite(nll_patched)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_patched"] = float(np.mean(nll_patched[nll_mask]))
    for label, values in (
        ("zs", nll_pos1_zs),
        ("icl", nll_pos1_icl),
        ("patched", nll_pos1_patch),
    ):
        m = np.isfinite(values)
        if bool(np.any(m)):
            stats_dict[f"mean_nll_target_pos1_{label}"] = float(np.mean(values[m]))
    for label, values in (
        ("zs", nll_pos2_zs),
        ("icl", nll_pos2_icl),
        ("patched", nll_pos2_patch),
    ):
        m = np.isfinite(values)
        if bool(np.any(m)):
            stats_dict[f"mean_nll_target_pos2_{label}"] = float(np.mean(values[m]))
    for label, values in (
        ("zs", nll_pos3_zs),
        ("icl", nll_pos3_icl),
        ("patched", nll_pos3_patch),
    ):
        m = np.isfinite(values)
        if bool(np.any(m)):
            stats_dict[f"mean_nll_target_pos3_{label}"] = float(np.mean(values[m]))
    nll_mask = np.isfinite(nll_null)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_null_patched"] = float(np.mean(nll_null[nll_mask]))
    nll_mask = np.isfinite(nll_attention)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_attention_patched"] = float(
            np.mean(nll_attention[nll_mask])
        )
    nll_mask = np.isfinite(nll_attention_structured)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_attention_structured"] = float(
            np.mean(nll_attention_structured[nll_mask])
        )
    nll_mask = np.isfinite(nll_basis)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_basis_patched"] = float(
            np.mean(nll_basis[nll_mask])
        )
    nll_mask = np.isfinite(nll_auto_scale)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_auto_scale_patched"] = float(
            np.mean(nll_auto_scale[nll_mask])
        )
    nll_mask = np.isfinite(nll_auto_shift)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_auto_shift_patched"] = float(
            np.mean(nll_auto_shift[nll_mask])
        )
    nll_mask = np.isfinite(nll_mean_residual)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_mean_residual_patched"] = float(
            np.mean(nll_mean_residual[nll_mask])
        )
    nll_mask = np.isfinite(nll_empty_prompt)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_empty_prompt_patched"] = float(
            np.mean(nll_empty_prompt[nll_mask])
        )
    nll_mask = np.isfinite(nll_cross_task)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_cross_task_patched"] = float(
            np.mean(nll_cross_task[nll_mask])
        )
    nll_mask = np.isfinite(nll_attn_head_ablated)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_attn_head_ablated"] = float(
            np.mean(nll_attn_head_ablated[nll_mask])
        )
    nll_mask = np.isfinite(nll_english_neutral_zs)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_english_neutral_zs"] = float(
            np.mean(nll_english_neutral_zs[nll_mask])
        )
    nll_mask = np.isfinite(nll_english_neutral_patched)
    if bool(np.any(nll_mask)):
        stats_dict["mean_nll_per_token_english_neutral_patched"] = float(
            np.mean(nll_english_neutral_patched[nll_mask])
        )

    # Character-normalized NLL sanity metrics (tokenization confound check)
    m = np.isfinite(nll_char_zs)
    if bool(np.any(m)):
        stats_dict["mean_nll_per_char_zs"] = float(np.mean(nll_char_zs[m]))
    m = np.isfinite(nll_char_icl)
    if bool(np.any(m)):
        stats_dict["mean_nll_per_char_icl"] = float(np.mean(nll_char_icl[m]))
    m = np.isfinite(nll_char_patch)
    if bool(np.any(m)):
        stats_dict["mean_nll_per_char_patched"] = float(np.mean(nll_char_patch[m]))

    m = np.isfinite(nll_char_zs) & np.isfinite(nll_char_patch)
    if bool(np.any(m)):
        stats_dict["mean_nll_per_char_improvement_patch"] = float(
            np.mean(nll_char_zs[m] - nll_char_patch[m])
        )

    # Paired NLL improvements
    m = np.isfinite(nll_zs) & np.isfinite(nll_patched)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_patched[m]
        stats_dict["mean_nll_improvement_patch"] = float(np.mean(delta))
        lo, hi = _bootstrap_mean_ci(delta, seed=10)
        stats_dict["mean_nll_improvement_patch_ci95"] = {"low": lo, "high": hi}

    m = np.isfinite(nll_zs) & np.isfinite(nll_rand)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_rand[m]
        stats_dict["mean_nll_improvement_random_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_corrupt)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_corrupt[m]
        stats_dict["mean_nll_improvement_corrupt_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_null)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_null[m]
        stats_dict["mean_nll_improvement_null_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_mean_pool)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_mean_pool[m]
        stats_dict["mean_nll_improvement_mean_pool_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_shuffle)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_shuffle[m]
        stats_dict["mean_nll_improvement_shuffle_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_gauss)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_gauss[m]
        stats_dict["mean_nll_improvement_gauss_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_attention)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_attention[m]
        stats_dict["mean_nll_improvement_attention_patch"] = float(np.mean(delta))

    # Early target-position diagnostics to separate script entry from continuation.
    for name, arr in (
        ("nll_pos1_zs", nll_pos1_zs),
        ("nll_pos2_zs", nll_pos2_zs),
        ("nll_pos3_zs", nll_pos3_zs),
        ("nll_pos1_icl", nll_pos1_icl),
        ("nll_pos2_icl", nll_pos2_icl),
        ("nll_pos3_icl", nll_pos3_icl),
        ("nll_pos1_patched", nll_pos1_patch),
        ("nll_pos2_patched", nll_pos2_patch),
        ("nll_pos3_patched", nll_pos3_patch),
    ):
        mask = np.isfinite(arr)
        if bool(np.any(mask)):
            stats_dict[f"mean_{name}"] = float(np.mean(arr[mask]))

    m = np.isfinite(nll_zs) & np.isfinite(nll_basis)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_basis[m]
        stats_dict["mean_nll_improvement_basis_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_decoupled)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_decoupled[m]
        stats_dict["mean_nll_improvement_decoupled_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_auto_scale)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_auto_scale[m]
        stats_dict["mean_nll_improvement_auto_scale_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_auto_shift)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_auto_shift[m]
        stats_dict["mean_nll_improvement_auto_shift_patch"] = float(np.mean(delta))

    m_mult = np.isfinite(nll_zs) & np.isfinite(nll_auto_scale) & np.isfinite(nll_patched)
    m_add = np.isfinite(nll_zs) & np.isfinite(nll_auto_shift) & np.isfinite(nll_patched)
    if bool(np.any(m_mult)):
        num_mult = float(np.mean(nll_zs[m_mult] - nll_auto_scale[m_mult]))
        den_mult = float(np.mean(nll_zs[m_mult] - nll_patched[m_mult]))
        raw_mult = float(num_mult / max(1e-8, den_mult))
        stats_dict["auto_scale_ratio_mult_raw"] = raw_mult
        stats_dict["auto_scale_ratio_mult"] = float(min(1.0, raw_mult))
        stats_dict["auto_scale_mult_overshoot"] = bool(raw_mult > 1.0)

    if bool(np.any(m_add)):
        num_add = float(np.mean(nll_zs[m_add] - nll_auto_shift[m_add]))
        den_add = float(np.mean(nll_zs[m_add] - nll_patched[m_add]))
        raw_add = float(num_add / max(1e-8, den_add))
        stats_dict["auto_scale_ratio_add_raw"] = raw_add
        stats_dict["auto_scale_ratio_add"] = float(min(1.0, raw_add))
        stats_dict["auto_scale_add_overshoot"] = bool(raw_add > 1.0)

    if bool(np.any(m_mult)) or bool(np.any(m_add)):
        raw_mult = float(stats_dict.get("auto_scale_ratio_mult_raw", float("nan")))
        raw_add = float(stats_dict.get("auto_scale_ratio_add_raw", float("nan")))
        candidates = [v for v in [raw_mult, raw_add] if np.isfinite(v)]
        if candidates:
            raw_combo = float(max(candidates))
            stats_dict["auto_scale_ratio_raw"] = raw_combo
            stats_dict["auto_scale_ratio"] = float(min(1.0, raw_combo))
            stats_dict["auto_scale_overshoot"] = bool(raw_combo > 1.0)
            stats_dict["auto_scale_intervention_artifact"] = bool(raw_combo >= 1.2)

    mean_pe_main = float(stats_dict.get("mean_pe", float("nan")))
    mean_pe_auto = float(stats_dict.get("mean_pe_auto_scale", float("nan")))
    if np.isfinite(mean_pe_main) and abs(mean_pe_main) > 1e-8 and np.isfinite(mean_pe_auto):
        raw_pe_ratio = float(abs(mean_pe_auto) / max(1e-8, abs(mean_pe_main)))
        stats_dict["auto_scale_ratio_pe_raw"] = raw_pe_ratio
        stats_dict["auto_scale_ratio_pe"] = float(min(1.0, raw_pe_ratio))

    softcap_risk = bool(stats_dict.get("softcap_saturation_risk", False))
    ratio_nll = float(stats_dict.get("auto_scale_ratio", float("nan")))
    ratio_pe = float(stats_dict.get("auto_scale_ratio_pe", float("nan")))
    if softcap_risk and np.isfinite(ratio_pe):
        stats_dict["auto_scale_ratio_adjudicated"] = ratio_pe
        stats_dict["auto_scale_ratio_metric"] = "pe"
    elif np.isfinite(ratio_nll):
        stats_dict["auto_scale_ratio_adjudicated"] = ratio_nll
        stats_dict["auto_scale_ratio_metric"] = "nll"
    elif np.isfinite(ratio_pe):
        stats_dict["auto_scale_ratio_adjudicated"] = ratio_pe
        stats_dict["auto_scale_ratio_metric"] = "pe_fallback"

    m = np.isfinite(nll_zs) & np.isfinite(nll_mean_residual)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_mean_residual[m]
        stats_dict["mean_nll_improvement_mean_residual_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_empty_prompt)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_empty_prompt[m]
        stats_dict["mean_nll_improvement_empty_prompt_patch"] = float(np.mean(delta))

    m = np.isfinite(nll_zs) & np.isfinite(nll_cross_task)
    if bool(np.any(m)):
        delta = nll_zs[m] - nll_cross_task[m]
        stats_dict["mean_nll_improvement_cross_task_patch"] = float(np.mean(delta))

    main_improvement = float(stats_dict.get("mean_nll_improvement_patch", float("nan")))
    empty_improvement = float(
        stats_dict.get("mean_nll_improvement_empty_prompt_patch", float("nan"))
    )
    if np.isfinite(main_improvement) and np.isfinite(empty_improvement) and abs(main_improvement) > 1e-8:
        stats_dict["bracket_rescue_ratio"] = float(
            abs(empty_improvement) / max(1e-8, abs(main_improvement))
        )

    m = np.isfinite(nll_icl) & np.isfinite(nll_attn_head_ablated)
    if bool(np.any(m)):
        delta = nll_attn_head_ablated[m] - nll_icl[m]
        stats_dict["mean_nll_harm_attn_head_ablation"] = float(np.mean(delta))

    # Sweet spot analysis
    # ==================
    # PRE-REGISTRATION NOTE: These thresholds were derived from theoretical reasoning
    # and initial exploratory analysis. For confirmatory claims, thresholds should be
    # pre-registered BEFORE data collection. Current thresholds are:
    # - Below 0.02: Floor effect - model has essentially no signal, patching can't help
    # - Above 0.35: Ceiling effect - model already handles the input well, less room
    #
    # STATUS: EXPLORATORY - these thresholds were determined post-hoc from initial
    # experiments. For publication, either (a) pre-register on a new dataset, or
    # (b) label sweet spot analysis as "exploratory" and report sensitivity to
    # alternative thresholds (computed below).
    #
    # Alternative thresholds are computed to assess robustness of findings.
    SWEET_SPOT_LOW = 0.02  # Configurable: minimum baseline probability
    SWEET_SPOT_HIGH = 0.35  # Configurable: maximum baseline probability

    prob_zs_values = np.array([r.prob_zs_first for r in results], dtype=np.float64)
    ss_mask = np.isfinite(prob_zs_values) & np.isfinite(pe_values)
    prob_ok = prob_zs_values[ss_mask]
    pe_ok = pe_values[ss_mask]
    sweet = (prob_ok >= SWEET_SPOT_LOW) & (prob_ok <= SWEET_SPOT_HIGH)
    pe_sweet = pe_ok[sweet]
    pe_non_sweet = pe_ok[~sweet]

    stats_dict["sweet_spot"] = {
        "threshold_low": SWEET_SPOT_LOW,
        "threshold_high": SWEET_SPOT_HIGH,
        "n_in": int(pe_sweet.shape[0]),
        "n_out": int(pe_non_sweet.shape[0]),
        "mean_pe_in": float(np.mean(pe_sweet)) if pe_sweet.shape[0] else float("nan"),
        "mean_pe_out": float(np.mean(pe_non_sweet))
        if pe_non_sweet.shape[0]
        else float("nan"),
    }

    # Cross-validation: also report for alternative thresholds
    alt_thresholds = [(0.01, 0.25), (0.01, 0.50), (0.05, 0.30)]
    alt_results = []
    for low, high in alt_thresholds:
        alt_sweet = (prob_ok >= low) & (prob_ok <= high)
        alt_pe_sweet = pe_ok[alt_sweet]
        alt_pe_non = pe_ok[~alt_sweet]
        alt_results.append(
            {
                "threshold_low": low,
                "threshold_high": high,
                "n_in": int(alt_pe_sweet.shape[0]),
                "mean_pe_in": float(np.mean(alt_pe_sweet))
                if alt_pe_sweet.shape[0]
                else float("nan"),
            }
        )
    stats_dict["sweet_spot"]["alternative_thresholds"] = alt_results

    # T-test if enough samples
    if pe_sweet.shape[0] >= 2 and pe_non_sweet.shape[0] >= 2:
        t_stat, t_pval = stats.ttest_ind(pe_sweet, pe_non_sweet)
        stats_dict["sweet_spot"]["t_stat"] = float(t_stat)
        stats_dict["sweet_spot"]["p_value"] = float(t_pval)
        # Ensure JSON-serializable (avoid numpy.bool_)
        stats_dict["sweet_spot"]["significant"] = bool(t_pval < 0.05)

    # Correlation
    if pe_ok.shape[0] >= 3:
        r, p = stats.pearsonr(prob_ok, pe_ok)
        stats_dict["correlation"] = {
            "prob_zs_vs_pe": float(r),
            "p_value": float(p),
            # Ensure JSON-serializable (avoid numpy.bool_)
            "significant": bool(p < 0.05),
        }

    # Random-control sanity check: rescue should beat random on average
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_random_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_random_values[delta_mask]
        stats_dict["random_control"] = {
            "mean_pe_minus_random": float(np.mean(delta)),
            "std_pe_minus_random": float(np.std(delta)),
        }

    # Shuffle-index control sanity check: rescue should beat shuffle on average
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_shuffle_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_shuffle_values[delta_mask]
        stats_dict["shuffle_control"] = {
            "mean_pe_minus_shuffle": float(np.mean(delta)),
            "std_pe_minus_shuffle": float(np.std(delta)),
        }

    # Gaussian-noise control sanity check: rescue should beat gaussian on average
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_gauss_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_gauss_values[delta_mask]
        stats_dict["gauss_control"] = {
            "mean_pe_minus_gauss": float(np.mean(delta)),
            "std_pe_minus_gauss": float(np.std(delta)),
        }

    # Attention-only control sanity check: rescue should beat attention-only intervention.
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_attention_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_attention_values[delta_mask]
        stats_dict["attention_control"] = {
            "mean_pe_minus_attention": float(np.mean(delta)),
            "std_pe_minus_attention": float(np.std(delta)),
        }

    # Random-basis control sanity check: rescue should beat SRHT basis control.
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_basis_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_basis_values[delta_mask]
        stats_dict["random_basis_control"] = {
            "method": "srht_topk",
            "mean_pe_minus_basis": float(np.mean(delta)),
            "std_pe_minus_basis": float(np.std(delta)),
        }

    # Multi-token random-control sanity check
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_random_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_random_values_multi[delta_mask]
        stats_dict["random_control_multi"] = {
            "mean_pe_minus_random_multi": float(np.mean(delta)),
            "std_pe_minus_random_multi": float(np.std(delta)),
        }

    # Multi-token shuffle-control sanity check
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_shuffle_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_shuffle_values_multi[delta_mask]
        stats_dict["shuffle_control_multi"] = {
            "mean_pe_minus_shuffle_multi": float(np.mean(delta)),
            "std_pe_minus_shuffle_multi": float(np.std(delta)),
        }

    # Multi-token gaussian-control sanity check
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_gauss_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_gauss_values_multi[delta_mask]
        stats_dict["gauss_control_multi"] = {
            "mean_pe_minus_gauss_multi": float(np.mean(delta)),
            "std_pe_minus_gauss_multi": float(np.std(delta)),
        }

    # Multi-token attention-control sanity check
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_attention_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_attention_values_multi[delta_mask]
        stats_dict["attention_control_multi"] = {
            "mean_pe_minus_attention_multi": float(np.mean(delta)),
            "std_pe_minus_attention_multi": float(np.std(delta)),
        }

    # Multi-token random-basis control sanity check
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_basis_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_basis_values_multi[delta_mask]
        stats_dict["random_basis_control_multi"] = {
            "method": "srht_topk",
            "mean_pe_minus_basis_multi": float(np.mean(delta)),
            "std_pe_minus_basis_multi": float(np.std(delta)),
        }

    # Task-matched corrupted-ICL control: patch should beat "wrong mapping" on average.
    delta_mask = np.isfinite(pe_values) & np.isfinite(pe_corrupt_values)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values[delta_mask] - pe_corrupt_values[delta_mask]
        stats_dict["task_matched_control"] = {
            "mean_pe_minus_corrupt": float(np.mean(delta)),
            "std_pe_minus_corrupt": float(np.std(delta)),
        }

    # Multi-token task-matched corrupted-ICL control (if available; may be NaN in older runs).
    delta_mask = np.isfinite(pe_values_multi) & np.isfinite(pe_corrupt_values_multi)
    if bool(np.sum(delta_mask) >= 2):
        delta = pe_values_multi[delta_mask] - pe_corrupt_values_multi[delta_mask]
        stats_dict["task_matched_control_multi"] = {
            "mean_pe_minus_corrupt_multi": float(np.mean(delta)),
            "std_pe_minus_corrupt_multi": float(np.std(delta)),
        }

    # =========================================================================
    # EFFECT SIZE AND STATISTICAL POWER (Added for reviewer-robustness)
    # =========================================================================

    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size between two groups."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return float("nan")
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std < 1e-9:
            return float("nan")
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    def _cohens_d_paired(diff: np.ndarray) -> float:
        """Compute Cohen's d for paired samples (diff / std_diff)."""
        if len(diff) < 2:
            return float("nan")
        std_diff = np.std(diff, ddof=1)
        if std_diff < 1e-9:
            return float("nan")
        return float(np.mean(diff) / std_diff)

    # Helper: safe Wilcoxon signed-rank test (handles edge cases)
    def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
        """Wilcoxon signed-rank test p-value (two-sided). Returns NaN on error."""
        try:
            diff = x - y
            # Remove zero differences (Wilcoxon requirement)
            nonzero = diff[diff != 0]
            if len(nonzero) < 10:
                return float("nan")
            _, p = stats.wilcoxon(nonzero, alternative="two-sided")
            return float(p)
        except Exception:
            return float("nan")

    def _safe_wilcoxon_greater(x: np.ndarray, y: np.ndarray) -> float:
        """One-sided Wilcoxon: is x > y? Returns NaN on error."""
        try:
            diff = x - y
            nonzero = diff[diff != 0]
            if len(nonzero) < 10:
                return float("nan")
            _, p = stats.wilcoxon(nonzero, alternative="greater")
            return float(p)
        except Exception:
            return float("nan")

    # Effect size: PE vs Random (paired)
    pe_rand_mask = np.isfinite(pe_values) & np.isfinite(pe_random_values)
    if bool(np.sum(pe_rand_mask) >= 2):
        diff = pe_values[pe_rand_mask] - pe_random_values[pe_rand_mask]
        d = _cohens_d_paired(diff)
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(
            pe_values[pe_rand_mask], pe_random_values[pe_rand_mask]
        )
        p_perm = _paired_perm_pvalue(
            pe_values[pe_rand_mask], pe_random_values[pe_rand_mask], seed=101
        )
        p_wilcox = _safe_wilcoxon_greater(pe_values[pe_rand_mask], pe_random_values[pe_rand_mask])
        stats_dict["effect_size_pe_vs_random"] = {
            "cohens_d": d,
            "interpretation": "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible",
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
            "paired_permutation_p": float(p_perm),
            "wilcoxon_p": float(p_wilcox),
            "significant": bool(p_val < 0.05),
        }

    # Effect size: PE vs Corrupt (paired) — PRIMARY CONFIRMATORY COMPARISON
    pe_corr_mask = np.isfinite(pe_values) & np.isfinite(pe_corrupt_values)
    if bool(np.sum(pe_corr_mask) >= 2):
        diff = pe_values[pe_corr_mask] - pe_corrupt_values[pe_corr_mask]
        d = _cohens_d_paired(diff)
        t_stat, p_val = stats.ttest_rel(
            pe_values[pe_corr_mask], pe_corrupt_values[pe_corr_mask]
        )
        p_perm = _paired_perm_pvalue(
            pe_values[pe_corr_mask], pe_corrupt_values[pe_corr_mask], seed=202
        )
        p_wilcox = _safe_wilcoxon_greater(pe_values[pe_corr_mask], pe_corrupt_values[pe_corr_mask])
        stats_dict["effect_size_pe_vs_corrupt"] = {
            "cohens_d": d,
            "interpretation": "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible",
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
            "paired_permutation_p": float(p_perm),
            "wilcoxon_p": float(p_wilcox),
            "significant": bool(p_val < 0.05),
        }

    # Effect size: NLL Improvement vs Corrupt (paired) — PRIMARY CONFIRMATORY
    # nll_improvement = nll_zs - nll_patched
    # We want to test if nll_patched < nll_corrupt
    nll_corr_mask_eff = np.isfinite(nll_patched) & np.isfinite(nll_corrupt)
    if bool(np.sum(nll_corr_mask_eff) >= 2):
        # We compare nll_corrupt (baseline) to nll_patched. Positive diff means patched is lower NLL (better).
        diff = nll_corrupt[nll_corr_mask_eff] - nll_patched[nll_corr_mask_eff]
        d = _cohens_d_paired(diff)
        t_stat, p_val = stats.ttest_rel(
            nll_corrupt[nll_corr_mask_eff], nll_patched[nll_corr_mask_eff]
        )
        p_perm = _paired_perm_pvalue(
            nll_corrupt[nll_corr_mask_eff], nll_patched[nll_corr_mask_eff], seed=202
        )
        # Wilcoxon: is nll_corrupt > nll_patched? (i.e., patched is better)
        p_wilcox = _safe_wilcoxon_greater(nll_corrupt[nll_corr_mask_eff], nll_patched[nll_corr_mask_eff])
        stats_dict["effect_size_nll_improvement_vs_corrupt"] = {
            "cohens_d": d,
            "interpretation": "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible",
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
            "paired_permutation_p": float(p_perm),
            "wilcoxon_p": float(p_wilcox),
            "significant": bool(p_val < 0.05),
        }

    # Effect size: PE vs Shuffle (paired)
    pe_shuf_mask = np.isfinite(pe_values) & np.isfinite(pe_shuffle_values)
    if bool(np.sum(pe_shuf_mask) >= 2):
        diff = pe_values[pe_shuf_mask] - pe_shuffle_values[pe_shuf_mask]
        d = _cohens_d_paired(diff)
        t_stat, p_val = stats.ttest_rel(
            pe_values[pe_shuf_mask], pe_shuffle_values[pe_shuf_mask]
        )
        p_perm = _paired_perm_pvalue(
            pe_values[pe_shuf_mask], pe_shuffle_values[pe_shuf_mask], seed=303
        )
        p_wilcox = _safe_wilcoxon_greater(pe_values[pe_shuf_mask], pe_shuffle_values[pe_shuf_mask])
        stats_dict["effect_size_pe_vs_shuffle"] = {
            "cohens_d": d,
            "interpretation": "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible",
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
            "paired_permutation_p": float(p_perm),
            "wilcoxon_p": float(p_wilcox),
            "significant": bool(p_val < 0.05),
        }

    # Effect size: PE vs Random-basis (paired)
    pe_basis_mask_eff = np.isfinite(pe_values) & np.isfinite(pe_basis_values)
    if bool(np.sum(pe_basis_mask_eff) >= 2):
        diff = pe_values[pe_basis_mask_eff] - pe_basis_values[pe_basis_mask_eff]
        d = _cohens_d_paired(diff)
        t_stat, p_val = stats.ttest_rel(
            pe_values[pe_basis_mask_eff], pe_basis_values[pe_basis_mask_eff]
        )
        p_perm = _paired_perm_pvalue(
            pe_values[pe_basis_mask_eff], pe_basis_values[pe_basis_mask_eff], seed=404
        )
        p_wilcox = _safe_wilcoxon_greater(pe_values[pe_basis_mask_eff], pe_basis_values[pe_basis_mask_eff])
        stats_dict["effect_size_pe_vs_basis"] = {
            "method": "srht_topk",
            "cohens_d": d,
            "interpretation": "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible",
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
            "paired_permutation_p": float(p_perm),
            "wilcoxon_p": float(p_wilcox),
            "significant": bool(p_val < 0.05),
        }

    # Effect size: Sweet spot in vs out (independent)
    if pe_sweet.shape[0] >= 2 and pe_non_sweet.shape[0] >= 2:
        d = _cohens_d(pe_sweet, pe_non_sweet)
        stats_dict["sweet_spot"]["cohens_d"] = d
        stats_dict["sweet_spot"]["effect_interpretation"] = (
            "large"
            if abs(d) >= 0.8
            else "medium"
            if abs(d) >= 0.5
            else "small"
            if abs(d) >= 0.2
            else "negligible"
        )

    # One-sample t-test: Is mean PE significantly greater than 0?
    pe_finite = pe_values[np.isfinite(pe_values)]
    if len(pe_finite) >= 2:
        t_stat, p_val = stats.ttest_1samp(pe_finite, 0)
        # One-tailed p-value (we care if PE > 0)
        p_val_one_tailed = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        stats_dict["one_sample_test_pe_gt_0"] = {
            "t_stat": float(t_stat),
            "p_value_two_tailed": float(p_val),
            "p_value_one_tailed": float(p_val_one_tailed),
            "significant": bool(p_val_one_tailed < 0.05),
        }

    # One-sample t-test: Is mean AE significantly less than 0? (necessity)
    ae_finite = ae_values[np.isfinite(ae_values)]
    if len(ae_finite) >= 2:
        t_stat, p_val = stats.ttest_1samp(ae_finite, 0)
        # One-tailed p-value (we care if AE < 0)
        p_val_one_tailed = p_val / 2 if t_stat < 0 else 1 - p_val / 2
        stats_dict["one_sample_test_ae_lt_0"] = {
            "t_stat": float(t_stat),
            "p_value_two_tailed": float(p_val),
            "p_value_one_tailed": float(p_val_one_tailed),
            "significant": bool(p_val_one_tailed < 0.05),
        }

    # =========================================================================
    # GENERATION EVALUATION (CER, exact match, script correctness)
    # =========================================================================
    # Only computed when eval_generation=True populated gen_patched fields.
    gen_results_with_data = [
        r for r in results if r.gen_patched and r.word_hindi
    ]
    if gen_results_with_data:
        def _char_error_rate(pred: str, ref: str) -> float:
            """Compute Character Error Rate via Levenshtein distance."""
            if not ref:
                return 0.0 if not pred else 1.0
            # Standard dynamic-programming edit distance.
            n, m = len(ref), len(pred)
            dp = list(range(n + 1))
            for j in range(1, m + 1):
                prev, dp[0] = dp[0], j
                for i in range(1, n + 1):
                    temp = dp[i]
                    if ref[i - 1] == pred[j - 1]:
                        dp[i] = prev
                    else:
                        dp[i] = 1 + min(dp[i], dp[i - 1], prev)
                    prev = temp
            return dp[n] / max(1, n)

        cer_zs_list = []
        cer_icl_list = []
        cer_patched_list = []
        exact_zs = 0
        exact_icl = 0
        exact_patched = 0

        for r in gen_results_with_data:
            ref = r.word_hindi.strip()
            if r.gen_zs:
                gen = r.gen_zs.strip().split()[0] if r.gen_zs.strip() else ""
                cer_zs_list.append(_char_error_rate(gen, ref))
                if gen == ref:
                    exact_zs += 1
            if r.gen_icl:
                gen = r.gen_icl.strip().split()[0] if r.gen_icl.strip() else ""
                cer_icl_list.append(_char_error_rate(gen, ref))
                if gen == ref:
                    exact_icl += 1
            if r.gen_patched:
                gen = r.gen_patched.strip().split()[0] if r.gen_patched.strip() else ""
                cer_patched_list.append(_char_error_rate(gen, ref))
                if gen == ref:
                    exact_patched += 1

        n_gen = len(gen_results_with_data)
        stats_dict["generation_eval"] = {
            "n_evaluated": n_gen,
            "mean_cer_zs": float(np.mean(cer_zs_list)) if cer_zs_list else float("nan"),
            "mean_cer_icl": float(np.mean(cer_icl_list)) if cer_icl_list else float("nan"),
            "mean_cer_patched": float(np.mean(cer_patched_list)) if cer_patched_list else float("nan"),
            "exact_match_rate_zs": exact_zs / max(1, n_gen),
            "exact_match_rate_icl": exact_icl / max(1, n_gen),
            "exact_match_rate_patched": exact_patched / max(1, n_gen),
        }

    return stats_dict
