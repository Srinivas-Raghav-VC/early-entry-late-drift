from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rescue_research.analysis.stats import bootstrap_ci_mean


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def prompt_fingerprint(*, raw_prompt: str, rendered_prompt: str) -> dict[str, Any]:
    raw = str(raw_prompt)
    rendered = str(rendered_prompt)
    return {
        "raw_length_chars": len(raw),
        "rendered_length_chars": len(rendered),
        "raw_sha256": sha256_text(raw),
        "rendered_sha256": sha256_text(rendered),
    }


def prompt_template_fingerprint(tokenizer: Any) -> dict[str, Any]:
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "chat_template_length_chars": len(chat_template),
        "chat_template_sha256": sha256_text(chat_template),
    }


def runtime_identity(*, model_key: str, hf_id: str, tokenizer: Any, model: Any) -> dict[str, Any]:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "")
    tokenizer_revision = ""
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, Mapping):
        tokenizer_revision = str(tokenizer.init_kwargs.get("revision", "") or "")
    tokenizer_class = str(getattr(tokenizer, "__class__", type(tokenizer)).__name__)
    model_name = ""
    model_revision = ""
    config = getattr(model, "config", None)
    if config is not None:
        model_name = str(getattr(config, "_name_or_path", "") or "")
        model_revision = str(getattr(config, "_commit_hash", "") or "")
    model_class = str(getattr(model, "__class__", type(model)).__name__)
    transformers_version = ""
    torch_version = ""
    try:
        import transformers  # type: ignore

        transformers_version = str(getattr(transformers, "__version__", "") or "")
    except Exception:
        transformers_version = ""
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "") or "")
    except Exception:
        torch_version = ""
    return {
        "model_key": str(model_key),
        "hf_id": str(hf_id),
        "tokenizer_name_or_path": tokenizer_name,
        "tokenizer_class": tokenizer_class,
        "tokenizer_revision": tokenizer_revision,
        "model_name_or_path": model_name,
        "model_class": model_class,
        "model_revision": model_revision,
        "transformers_version": transformers_version,
        "torch_version": torch_version,
    }


def site_alignment_verdict(*, exact_match: bool, family_match: bool) -> str:
    if bool(exact_match):
        return "exact_artifact_match"
    if bool(family_match):
        return "artifact_family_match_but_operational_offset"
    return "mismatch_do_not_claim_artifact_alignment"


def premise_gap_summary(
    explicit_zs: Sequence[float],
    icl64: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(explicit_zs) != len(icl64):
        raise ValueError("Premise-gate samples must be paired and equal-length.")
    paired = [
        float(b) - float(a)
        for a, b in zip(explicit_zs, icl64)
        if np.isfinite(float(a)) and np.isfinite(float(b))
    ]
    if not paired:
        return {
            "gap_mean": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "ci_excludes_zero": False,
            "n_pairs": 0,
        }
    lo, hi = bootstrap_ci_mean(paired, n_bootstrap=int(n_bootstrap), seed=int(seed))
    return {
        "gap_mean": float(np.mean(paired)),
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        "n_pairs": int(len(paired)),
    }


def local_stability_window(
    *,
    layer: int,
    topk: int,
    valid_layers: Iterable[int],
    topk_ladder: Sequence[int] = (4, 8, 16, 32),
) -> dict[str, Any]:
    layers = sorted({int(x) for x in valid_layers})
    ladder = [int(x) for x in topk_ladder]
    return {
        "reporting_only": True,
        "layers": [x for x in (int(layer) - 1, int(layer), int(layer) + 1) if x in layers],
        "topk_values": [x for x in ladder if abs(int(x) - int(topk)) <= int(topk)],
    }
