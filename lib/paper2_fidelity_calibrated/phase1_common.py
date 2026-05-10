from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    _bottomk_indices_abs,
    _build_patch_vector,
    _extract_mlp_io_at_position_from_input_ids,
    _find_last_subsequence,
    _topk_indices,
    apply_chat_template,
    build_corrupted_icl_prompt,
    build_task_prompt,
    load_transcoder,
    split_data_three_way,
)
from rescue_research.data_pipeline.ingest import (  # noqa: E402
    get_pair_prompt_metadata,
    load_pair_records_bundle,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _prompt_naming(prompt_meta: Dict[str, str]) -> Tuple[str, str, str]:
    source_language = str(prompt_meta.get("source_language", "")).strip() or "Hindi"
    input_script_name = str(prompt_meta.get("source_script", "")).strip() or "Latin"
    output_script_name = str(prompt_meta.get("target_script", "")).strip() or "Devanagari"
    return source_language, input_script_name, output_script_name


def _load_words(
    pair_id: str,
    *,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    bundle = load_pair_records_bundle(str(pair_id), include_builtin=not bool(external_only))
    total = int(len(bundle.rows))
    source_names = [str(s.name) for s in bundle.sources]
    external_sources = [name for name in source_names if name and name != "config_multiscript"]

    if bool(require_external_sources) and not external_sources:
        raise RuntimeError(
            f"Pair {pair_id!r} has no external sources (only builtin). "
            "Provide external data under lib/data/transliteration/ or disable --require-external-sources."
        )
    if int(min_pool_size) > 0 and total < int(min_pool_size):
        raise RuntimeError(f"Pair {pair_id!r} pool too small: total={total} < {int(min_pool_size)}")

    words = [
        {
            "english": str(row["english"]),
            "hindi": str(row["target"]),
            "ood": str(row["source"]),
        }
        for row in bundle.rows
    ]
    meta = {
        "pair_id": str(pair_id),
        "total_rows": total,
        "sources": [
            {
                "name": str(source.name),
                "url": str(source.url),
                "license": str(source.license),
                "checksum": str(source.checksum),
                "version_date": str(source.version_date),
            }
            for source in bundle.sources
        ],
        "source_counts": dict(bundle.source_counts),
    }
    return words, meta


def resolve_stagea_path(pair_id: str, model_key: str, stagea_path: str = "") -> Path:
    if str(stagea_path).strip():
        return Path(stagea_path).resolve()
    return (
        PROJECT_ROOT
        / "paper2_fidelity_calibrated"
        / "results"
        / pair_id
        / model_key
        / f"paper2_fidelity_calibrated_{model_key}.json"
    )


def load_stagea_best(stagea_path: Path, *, seed: int) -> Dict[str, Any]:
    payload = json.loads(stagea_path.read_text(encoding="utf-8"))
    seed_key = str(int(seed))
    if seed_key not in payload.get("seeds", {}):
        raise KeyError(f"Seed {seed_key} missing from Stage A artifact: {stagea_path}")
    best = dict(payload["seeds"][seed_key].get("best") or {})
    if not best or best.get("layer") is None or best.get("topk") is None or not best.get("variant"):
        raise RuntimeError(f"No valid best config for seed {seed_key} in {stagea_path}")
    cfg = dict(payload.get("config") or {})
    return {
        "pair": str(payload.get("pair", "")),
        "model_key": str(payload.get("model_key", "")),
        "seed": int(seed),
        "variant": str(best["variant"]),
        "layer": int(best["layer"]),
        "topk": int(best["topk"]),
        "score": float(best.get("score", float("nan"))),
        "patch_style": str(cfg.get("patch_style", "sparse")),
        "feature_selection": str(cfg.get("feature_selection", "topk_abs_delta")),
        "selector_reference": str(cfg.get("selector_reference", "zs")),
        "prompt_variant": str(cfg.get("prompt_variant", "canonical")),
        "norm_matching": bool(cfg.get("norm_matching", True)),
        "require_query_span_match": bool(cfg.get("require_query_span_match", False)),
        "stagea_path": str(stagea_path),
    }


def resolve_alt_pair(pair_id: str, alt_pair: str = "") -> str:
    if str(alt_pair).strip():
        return str(alt_pair).strip()
    if pair_id == "aksharantar_hin_latin":
        return "aksharantar_tel_latin"
    if pair_id == "aksharantar_tel_latin":
        return "aksharantar_hin_latin"
    raise ValueError(
        f"No default alt pair for {pair_id!r}; pass --alt-pair explicitly."
    )


def load_pair_split(
    pair_id: str,
    *,
    seed: int,
    n_icl: int,
    n_select: int,
    n_eval: int,
    external_only: bool,
    require_external_sources: bool,
    min_pool_size: int,
) -> Dict[str, Any]:
    words, provenance = _load_words(
        pair_id,
        external_only=bool(external_only),
        require_external_sources=bool(require_external_sources),
        min_pool_size=int(min_pool_size),
    )
    prompt_meta = dict(get_pair_prompt_metadata(pair_id))
    source_language, input_script_name, output_script_name = _prompt_naming(prompt_meta)
    icl_examples, select_rows, eval_rows = split_data_three_way(
        words,
        n_icl=int(n_icl),
        n_select=int(n_select),
        n_eval=int(n_eval),
        seed=int(seed),
    )
    return {
        "pair": str(pair_id),
        "words": words,
        "provenance": provenance,
        "prompt_meta": prompt_meta,
        "source_language": source_language,
        "input_script_name": input_script_name,
        "output_script_name": output_script_name,
        "icl_examples": icl_examples,
        "select_rows": select_rows,
        "eval_rows": eval_rows,
    }


def get_final_norm(model: Any):
    chains = [
        ("model", "norm"),
        ("language_model", "norm"),
        ("model", "model", "norm"),
        ("language_model", "model", "norm"),
        ("model", "decoder", "norm"),
    ]
    for chain in chains:
        cur = model
        ok = True
        for name in chain:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    raise AttributeError(f"Could not locate final norm for model type {type(model).__name__}")


def parse_selected_feature_indices(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _prepare_prompts_and_positions(
    *,
    tokenizer: Any,
    word: Dict[str, str],
    icl_examples: List[Dict[str, str]],
    prompt_variant: str,
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
    selector_reference: str,
    patch_position_mode: str,
    seed: int,
) -> Dict[str, Any]:
    telugu = str(word["ood"])
    hindi = str(word["hindi"])

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
    corrupt_prompt = build_corrupted_icl_prompt(
        telugu,
        icl_examples,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        seed=int(seed),
    )

    zs_rendered = apply_chat_template(tokenizer, zs_prompt)
    icl_rendered = apply_chat_template(tokenizer, icl_prompt)
    corrupt_rendered = apply_chat_template(tokenizer, corrupt_prompt)

    zs_input_ids = tokenizer(zs_rendered, return_tensors="pt").to(device).input_ids
    icl_input_ids = tokenizer(icl_rendered, return_tensors="pt").to(device).input_ids
    corrupt_input_ids = tokenizer(corrupt_rendered, return_tensors="pt").to(device).input_ids

    target_ids = tokenizer.encode(hindi, add_special_tokens=False)
    target_tensor = (
        torch.tensor(target_ids, device=zs_input_ids.device, dtype=zs_input_ids.dtype).unsqueeze(0)
        if target_ids
        else None
    )
    zs_tf_input_ids = torch.cat([zs_input_ids, target_tensor], dim=1) if target_tensor is not None else zs_input_ids
    icl_tf_input_ids = torch.cat([icl_input_ids, target_tensor], dim=1) if target_tensor is not None else icl_input_ids
    corrupt_tf_input_ids = torch.cat([corrupt_input_ids, target_tensor], dim=1) if target_tensor is not None else corrupt_input_ids

    if patch_position_mode == "target_pos1":
        if target_tensor is None:
            raise ValueError("target_pos1 patching requires non-empty target tokenization")
        zs_patch_position = int(zs_input_ids.shape[1])
        icl_feature_position = int(icl_input_ids.shape[1])
        selector_ref_position = (
            int(corrupt_input_ids.shape[1]) if selector_reference == "corrupt_icl" else int(zs_input_ids.shape[1])
        )
        feature_icl_input_ids = icl_tf_input_ids
        feature_selector_input_ids = corrupt_tf_input_ids if selector_reference == "corrupt_icl" else zs_tf_input_ids
    else:
        ood_ids = tokenizer.encode(telugu, add_special_tokens=False)
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
        if span_zs is None or span_icl is None:
            raise ValueError(
                f"Fail-closed query span localization failed: zs={span_zs is not None} icl={span_icl is not None}"
            )
        if selector_reference == "corrupt_icl" and span_corrupt is None:
            raise ValueError("Fail-closed selector-reference span localization failed for corrupt_icl")
        zs_patch_position = int(span_zs[1] - 1)
        icl_feature_position = int(span_icl[1] - 1)
        selector_ref_position = int(span_corrupt[1] - 1) if selector_reference == "corrupt_icl" else int(span_zs[1] - 1)
        feature_icl_input_ids = icl_input_ids
        feature_selector_input_ids = corrupt_input_ids if selector_reference == "corrupt_icl" else zs_input_ids

    return {
        "word": dict(word),
        "zs_prompt": zs_prompt,
        "icl_prompt": icl_prompt,
        "corrupt_prompt": corrupt_prompt,
        "zs_rendered": zs_rendered,
        "icl_rendered": icl_rendered,
        "corrupt_rendered": corrupt_rendered,
        "zs_input_ids": zs_input_ids,
        "icl_input_ids": icl_input_ids,
        "corrupt_input_ids": corrupt_input_ids,
        "zs_tf_input_ids": zs_tf_input_ids,
        "icl_tf_input_ids": icl_tf_input_ids,
        "corrupt_tf_input_ids": corrupt_tf_input_ids,
        "target_ids": target_ids,
        "target_id": int(target_ids[0]) if target_ids else -1,
        "zs_patch_position": int(zs_patch_position),
        "icl_feature_position": int(icl_feature_position),
        "selector_ref_position": int(selector_ref_position),
        "feature_icl_input_ids": feature_icl_input_ids,
        "feature_selector_input_ids": feature_selector_input_ids,
    }


def build_patch_packet(
    *,
    model: Any,
    tokenizer: Any,
    transcoder: Any,
    word: Dict[str, str],
    icl_examples: List[Dict[str, str]],
    stagea_best: Dict[str, Any],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
) -> Dict[str, Any]:
    selector_reference = str(stagea_best.get("selector_reference", "zs")).strip().lower()
    if selector_reference not in {"zs", "corrupt_icl"}:
        raise ValueError(f"Unsupported selector_reference={selector_reference!r}")

    feature_selection = str(stagea_best.get("feature_selection", "topk_abs_delta")).strip().lower()
    patch_position_mode = str(stagea_best.get("patch_position_mode", "source_last_subtoken")).strip().lower()
    if patch_position_mode == "target_pos1_teacher_forced":
        patch_position_mode = "target_pos1"
    if patch_position_mode not in {"source_last_subtoken", "target_pos1"}:
        raise ValueError(f"Unsupported patch_position_mode={patch_position_mode!r}")

    prompt_variant = str(stagea_best.get("prompt_variant", "canonical"))
    topk = int(stagea_best["topk"])
    layer = int(stagea_best["layer"])

    prompt_packet = _prepare_prompts_and_positions(
        tokenizer=tokenizer,
        word=word,
        icl_examples=icl_examples,
        prompt_variant=prompt_variant,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        device=device,
        selector_reference=selector_reference,
        patch_position_mode=patch_position_mode,
        seed=int(stagea_best.get("seed", 0)),
    )

    mlp_in_icl, mlp_out_icl = _extract_mlp_io_at_position_from_input_ids(
        model=model,
        input_ids=prompt_packet["feature_icl_input_ids"],
        layer=layer,
        position=int(prompt_packet["icl_feature_position"]),
    )
    mlp_in_sel, _ = _extract_mlp_io_at_position_from_input_ids(
        model=model,
        input_ids=prompt_packet["feature_selector_input_ids"],
        layer=layer,
        position=int(prompt_packet["selector_ref_position"]),
    )

    icl_feats = transcoder.encode(mlp_in_icl.unsqueeze(0)).squeeze(0)
    selector_ref_feats = transcoder.encode(mlp_in_sel.unsqueeze(0)).squeeze(0)

    if feature_selection == "topk_abs_icl":
        idx = _topk_indices(icl_feats, topk)
    elif feature_selection == "bottomk_abs_icl":
        idx = _bottomk_indices_abs(icl_feats, topk)
    elif feature_selection == "topk_abs_delta":
        idx = _topk_indices(icl_feats - selector_ref_feats, topk)
    elif feature_selection == "bottomk_abs_delta":
        idx = _bottomk_indices_abs(icl_feats - selector_ref_feats, topk)
    else:
        raise ValueError(f"Unsupported feature_selection={feature_selection!r}")

    patch_style = str(stagea_best.get("patch_style", "sparse")).strip().lower()
    if patch_style == "sparse":
        patch_feats = _build_patch_vector(icl_feats, idx)
    elif patch_style == "hybrid":
        patch_feats = selector_ref_feats.clone()
        if idx.numel() > 0:
            patch_feats[idx] = icl_feats[idx]
    else:
        raise ValueError(f"Unsupported patch_style={patch_style!r}")

    return {
        **prompt_packet,
        "selected_idx": [int(i) for i in idx.detach().to("cpu", dtype=torch.long).tolist()],
        "selected_idx_tensor": idx.detach(),
        "icl_feats": icl_feats.detach(),
        "selector_ref_feats": selector_ref_feats.detach(),
        "patch_feats": patch_feats.detach(),
        "target_output_norm": float(torch.norm(mlp_out_icl.detach().float()).item()) if bool(stagea_best.get("norm_matching", True)) else None,
        "layer": layer,
        "patch_style": patch_style,
        "feature_selection": feature_selection,
        "patch_position_mode": patch_position_mode,
    }


def extract_feature_delta_vector(
    *,
    model: Any,
    tokenizer: Any,
    transcoder: Any,
    layer: int,
    word: Dict[str, str],
    icl_examples: List[Dict[str, str]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    prompt_variant: str,
    selector_reference: str,
    patch_position_mode: str,
    device: str,
    seed: int,
) -> torch.Tensor:
    packet = _prepare_prompts_and_positions(
        tokenizer=tokenizer,
        word=word,
        icl_examples=icl_examples,
        prompt_variant=prompt_variant,
        input_script_name=input_script_name,
        source_language=source_language,
        output_script_name=output_script_name,
        device=device,
        selector_reference=selector_reference,
        patch_position_mode=patch_position_mode,
        seed=int(seed),
    )
    mlp_in_icl, _ = _extract_mlp_io_at_position_from_input_ids(
        model=model,
        input_ids=packet["feature_icl_input_ids"],
        layer=int(layer),
        position=int(packet["icl_feature_position"]),
    )
    mlp_in_ref, _ = _extract_mlp_io_at_position_from_input_ids(
        model=model,
        input_ids=packet["feature_selector_input_ids"],
        layer=int(layer),
        position=int(packet["selector_ref_position"]),
    )
    icl_feats = transcoder.encode(mlp_in_icl.unsqueeze(0)).squeeze(0)
    ref_feats = transcoder.encode(mlp_in_ref.unsqueeze(0)).squeeze(0)
    return (icl_feats - ref_feats).detach()


def load_transcoder_for_stagea(model: Any, stagea_best: Dict[str, Any], device: str):
    scope_repo = str(stagea_best.get("scope_repo", "")).strip()
    if not scope_repo:
        raise ValueError("stagea_best is missing scope_repo")
    return load_transcoder(
        model,
        scope_repo,
        int(stagea_best["layer"]),
        device,
        variant=str(stagea_best["variant"]),
    )
