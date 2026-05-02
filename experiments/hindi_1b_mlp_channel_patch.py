#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import (  # noqa: E402
    _find_last_subsequence,
    apply_chat_template,
    build_task_prompt,
    load_model,
    set_all_seeds,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402

CONDITIONS = ("zs", "icl_helpful")
PATCH_DIRECTIONS = (("icl_helpful", "zs"), ("zs", "icl_helpful"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_csv_int(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw or "").split(",") if x.strip()]


def _parse_selector_policies(raw: str) -> List[str]:
    vals = [str(x).strip().lower() for x in str(raw or "").split(",") if str(x).strip()]
    allowed = {"abs", "pos", "neg"}
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"Unexpected selector policies: {bad}; allowed={sorted(allowed)}")
    vals = vals or ["abs"]
    out: List[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def _detect_script(text: str) -> str:
    scripts: Counter[str] = Counter()
    for ch in str(text or ""):
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            name = ""
        if "DEVANAGARI" in name:
            scripts["devanagari"] += 1
        elif "TELUGU" in name:
            scripts["telugu"] += 1
        elif "BENGALI" in name or "BANGLA" in name:
            scripts["bengali"] += 1
        elif "TAMIL" in name:
            scripts["tamil"] += 1
        elif "LATIN" in name or ch.isascii():
            scripts["latin"] += 1
        else:
            scripts["other"] += 1
    if not scripts:
        return "unknown"
    return scripts.most_common(1)[0][0]


def _token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=True)).replace("\n", " ").strip()


def _build_latin_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    flags: List[bool] = []
    for token_id in range(int(vocab_size)):
        text = _token_text(tokenizer, token_id)
        flags.append(_detect_script(text) == "latin")
    if not any(flags):
        raise RuntimeError("Latin token mask is empty; token classification failed.")
    return torch.tensor(flags, dtype=torch.bool)


def _first_step_stats(
    *,
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_id: int,
    latin_mask: torch.Tensor,
    hooks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    active = list(hooks or [])
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, int(input_ids.shape[1] - 1), :].float()
        probs = torch.softmax(logits, dim=-1)

        top1_id = int(torch.argmax(logits).item())
        top1_text = _token_text(tokenizer, top1_id)

        logits_no_target = logits.clone()
        logits_no_target[int(target_id)] = -float("inf")
        competitor_id = int(torch.argmax(logits_no_target).item())
        competitor_text = _token_text(tokenizer, competitor_id)

        lat_mask = latin_mask.to(device=logits.device)
        logits_latin = logits.clone()
        logits_latin[~lat_mask] = -float("inf")
        latin_id = int(torch.argmax(logits_latin).item())
        latin_text = _token_text(tokenizer, latin_id)

        return {
            "target_id": int(target_id),
            "target_token_text": _token_text(tokenizer, int(target_id)),
            "target_prob": float(probs[int(target_id)].item()),
            "target_logit": float(logits[int(target_id)].item()),
            "top1_id": int(top1_id),
            "top1_token_text": top1_text,
            "top1_script": _detect_script(top1_text),
            "top1_prob": float(probs[top1_id].item()),
            "top1_is_target": bool(top1_id == int(target_id)),
            "competitor_id": int(competitor_id),
            "competitor_token_text": competitor_text,
            "competitor_script": _detect_script(competitor_text),
            "competitor_logit": float(logits[competitor_id].item()),
            "target_minus_competitor_logit": float(logits[int(target_id)].item() - logits[competitor_id].item()),
            "latin_competitor_id": int(latin_id),
            "latin_competitor_token_text": latin_text,
            "latin_competitor_script": _detect_script(latin_text),
            "latin_competitor_logit": float(logits[latin_id].item()),
            "target_minus_latin_logit": float(logits[int(target_id)].item() - logits[latin_id].item()),
        }
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _get_mlp(model: Any, layer: int):
    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate model.model.layers")
    return layers[int(layer)].mlp


def _extract_mlp_channel_vector(
    model: Any,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    mlp = _get_mlp(model, int(layer))
    captured: Dict[str, torch.Tensor] = {}

    def pre_hook(module, inputs_tuple):
        x = inputs_tuple[0] if inputs_tuple else None
        if torch.is_tensor(x):
            captured["channel_in"] = x.detach()

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()
    if "channel_in" not in captured:
        raise RuntimeError("Failed to capture MLP channel vector at down_proj input.")
    seq_len = int(captured["channel_in"].shape[1])
    pos = int(max(0, min(int(position), seq_len - 1)))
    return captured["channel_in"][0, pos, :]


def _register_partial_mlp_channel_replace_hook(
    model: Any,
    layer: int,
    donor_channel_vector: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    patch_position: int,
):
    mlp = _get_mlp(model, int(layer))
    donor_channel_vector = donor_channel_vector.detach()
    selected_idx = selected_idx.detach().to(dtype=torch.long)

    def pre_hook(module, inputs_tuple):
        if not inputs_tuple:
            return None
        x = inputs_tuple[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None
        seq_len = int(x.shape[1])
        pos = int(max(0, min(int(patch_position), seq_len - 1)))
        idx = selected_idx.to(device=x.device)
        if idx.numel() == 0:
            return None
        idx = idx[(idx >= 0) & (idx < int(x.shape[2]))]
        if idx.numel() == 0:
            return None
        donor = donor_channel_vector.to(device=x.device, dtype=x.dtype).view(-1)
        x_new = x.clone()
        x_new[:, pos, idx] = donor[idx].view(1, -1)
        if len(inputs_tuple) == 1:
            return (x_new,)
        return (x_new,) + tuple(inputs_tuple[1:])

    return mlp.down_proj.register_forward_pre_hook(pre_hook)


def _prepare_condition_inputs(
    *,
    tokenizer: Any,
    word: Mapping[str, Any],
    icl_examples: List[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    query_text = str(word["ood"])
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    if not query_ids:
        raise ValueError(f"Failed to tokenize query text: {query_text!r}")
    for condition in CONDITIONS:
        prompt = build_task_prompt(
            query_text,
            None if condition == "zs" else icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant="canonical",
        )
        rendered = apply_chat_template(tokenizer, prompt)
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
        span = _find_last_subsequence(
            input_ids[0].detach().cpu().tolist(),
            [int(x) for x in query_ids],
        )
        if span is None:
            raise RuntimeError(f"Failed to locate query span for condition={condition} item={query_text}")
        out[condition] = {
            "rendered": rendered,
            "input_ids": input_ids,
            "query_position": int(span[1] - 1),
            "last_position": int(input_ids.shape[1] - 1),
        }
    return out


def _deterministic_random_indices(
    *,
    size: int,
    k: int,
    seed: int,
    label: str,
) -> torch.Tensor:
    msg = f"{seed}::{label}::{size}::{k}".encode("utf-8")
    seed32 = int.from_bytes(hashlib.sha256(msg).digest()[:4], "little", signed=False)
    rng = np.random.default_rng(seed32)
    perm = rng.permutation(int(size))
    return torch.tensor(perm[: int(k)], dtype=torch.long)


def _default_k_grid(channel_size: int) -> List[int]:
    vals = [1, 4, 16, 64, 256, 512, 1024, 2048, int(channel_size)]
    out: List[int] = []
    for v in vals:
        vv = int(min(max(1, v), int(channel_size)))
        if vv not in out:
            out.append(vv)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MLP-channel-basis patching for Hindi 1B L25 last-token site.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--selector-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--ks", type=str, default="")
    ap.add_argument("--selector-policies", type=str, default="abs,pos,neg")
    ap.add_argument("--random-ks", type=str, default="4,16,64,256")
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--no-op-k", type=int, default=64)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_all_seeds(int(args.seed))

    bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=True,
        require_external_sources=True,
        min_pool_size=500,
    )
    select_rows = list(bundle["select_rows"][: max(1, int(args.selector_max_items))])
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)

    mlp = _get_mlp(model, int(args.layer))
    channel_size = int(getattr(mlp.down_proj, "in_features", 0))
    if channel_size <= 0:
        raise RuntimeError("Could not determine MLP channel size from down_proj.in_features.")

    k_grid = _parse_csv_int(str(args.ks)) or _default_k_grid(channel_size)
    k_grid = [int(min(max(1, k), channel_size)) for k in k_grid]
    k_grid = list(dict.fromkeys(k_grid))
    selector_policies = _parse_selector_policies(str(args.selector_policies))
    random_k_grid = [int(min(max(1, k), channel_size)) for k in (_parse_csv_int(str(args.random_ks)) or [])]
    random_k_grid = [k for k in dict.fromkeys(random_k_grid) if k in set(k_grid) and k < channel_size]
    no_op_k = int(min(max(1, int(args.no_op_k)), channel_size))

    out_root = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_mlp_channel_patch_v1" / str(args.model) / str(args.pair) / f"nicl{int(args.n_icl)}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running Hindi L{args.layer} MLP channel patch: model={args.model} pair={args.pair} channel_size={channel_size} select_items={len(select_rows)} eval_items={len(eval_rows)} selector_policies={selector_policies} k_grid={k_grid} random_k_grid={random_k_grid}",
        flush=True,
    )

    # Selection split ranking in MLP-channel basis.
    abs_delta_sum = torch.zeros(channel_size, dtype=torch.float32)
    signed_delta_sum = torch.zeros(channel_size, dtype=torch.float32)
    n_selector_items = 0
    for idx, word in enumerate(select_rows, start=1):
        if idx == 1 or idx == len(select_rows) or idx % 25 == 0:
            print(f"[selector {idx}/{len(select_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
        cond_inputs = _prepare_condition_inputs(
            tokenizer=tokenizer,
            word=word,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
        )
        zs_vec = _extract_mlp_channel_vector(model, cond_inputs["zs"]["input_ids"], int(args.layer), int(cond_inputs["zs"]["last_position"])).detach().float().cpu()
        helpful_vec = _extract_mlp_channel_vector(model, cond_inputs["icl_helpful"]["input_ids"], int(args.layer), int(cond_inputs["icl_helpful"]["last_position"])).detach().float().cpu()
        delta = zs_vec - helpful_vec
        abs_delta_sum += torch.abs(delta)
        signed_delta_sum += delta
        n_selector_items += 1
    if n_selector_items <= 0:
        raise RuntimeError("No selector items were processed.")

    mean_abs_delta = abs_delta_sum / float(n_selector_items)
    mean_signed_delta = signed_delta_sum / float(n_selector_items)
    selector_rank_tensors: Dict[str, torch.Tensor] = {
        "abs": torch.argsort(mean_abs_delta, descending=True).to(dtype=torch.long),
        "pos": torch.nonzero(mean_signed_delta > 0, as_tuple=False).flatten(),
        "neg": torch.nonzero(mean_signed_delta < 0, as_tuple=False).flatten(),
    }
    if selector_rank_tensors["pos"].numel() > 0:
        pos_scores = mean_signed_delta[selector_rank_tensors["pos"]]
        selector_rank_tensors["pos"] = selector_rank_tensors["pos"][torch.argsort(pos_scores, descending=True)]
    if selector_rank_tensors["neg"].numel() > 0:
        neg_scores = torch.abs(mean_signed_delta[selector_rank_tensors["neg"]])
        selector_rank_tensors["neg"] = selector_rank_tensors["neg"][torch.argsort(neg_scores, descending=True)]

    selector_rank_rows_by_policy: Dict[str, List[Dict[str, Any]]] = {}
    for policy_name, rank_idx in selector_rank_tensors.items():
        rows: List[Dict[str, Any]] = []
        for rank, dim in enumerate(rank_idx[:128].tolist(), start=1):
            rows.append(
                {
                    "rank": int(rank),
                    "channel": int(dim),
                    "mean_abs_delta": float(mean_abs_delta[int(dim)].item()),
                    "mean_signed_delta": float(mean_signed_delta[int(dim)].item()),
                }
            )
        selector_rank_rows_by_policy[policy_name] = rows

    # Eval split causal test.
    item_rows: List[Dict[str, Any]] = []
    for item_idx, word in enumerate(eval_rows, start=1):
        if item_idx == 1 or item_idx == len(eval_rows) or item_idx % 5 == 0:
            print(f"[eval {item_idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
        target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
        if not target_ids:
            continue
        target_id = int(target_ids[0])

        cond_inputs = _prepare_condition_inputs(
            tokenizer=tokenizer,
            word=word,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
        )

        base_by_condition: Dict[str, Dict[str, Any]] = {}
        donor_channels_by_condition: Dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            input_ids = cond_inputs[condition]["input_ids"]
            base_by_condition[condition] = _first_step_stats(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
            )
            donor_channels_by_condition[condition] = _extract_mlp_channel_vector(
                model,
                input_ids,
                int(args.layer),
                int(cond_inputs[condition]["last_position"]),
            ).detach()

        interventions: List[Dict[str, Any]] = []
        for recipient_condition, donor_condition in PATCH_DIRECTIONS:
            recipient_input_ids = cond_inputs[recipient_condition]["input_ids"]
            patch_position = int(cond_inputs[recipient_condition]["last_position"])
            donor_channels = donor_channels_by_condition[donor_condition]
            base_stats = base_by_condition[recipient_condition]

            for selector_policy in selector_policies:
                selector_rank = selector_rank_tensors[str(selector_policy)]
                for k in k_grid:
                    idx_selected = selector_rank[: int(min(int(k), int(selector_rank.numel())))]
                    if idx_selected.numel() <= 0:
                        continue
                    hook = _register_partial_mlp_channel_replace_hook(
                        model,
                        int(args.layer),
                        donor_channels,
                        idx_selected,
                        patch_position=patch_position,
                    )
                    patched = _first_step_stats(
                        model=model,
                        input_ids=recipient_input_ids,
                        tokenizer=tokenizer,
                        target_id=target_id,
                        latin_mask=latin_mask,
                        hooks=[hook],
                    )
                    interventions.append(
                        {
                            "recipient_condition": str(recipient_condition),
                            "donor_condition": str(donor_condition),
                            "selector_kind": "selected_topk",
                            "selector_policy": str(selector_policy),
                            "k": int(k),
                            "actual_k": int(idx_selected.numel()),
                            "random_repeat": None,
                            "selected_channels": [int(x) for x in idx_selected.detach().cpu().tolist()],
                            "base": base_stats,
                            "patched": patched,
                            "delta": {
                                "target_prob": float(patched["target_prob"] - base_stats["target_prob"]),
                                "target_minus_competitor_logit": float(patched["target_minus_competitor_logit"] - base_stats["target_minus_competitor_logit"]),
                                "target_minus_latin_logit": float(patched["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]),
                                "top1_is_target": float(float(patched["top1_is_target"]) - float(base_stats["top1_is_target"])),
                            },
                        }
                    )

            for k in random_k_grid:
                for random_repeat in range(int(args.n_random)):
                    idx_random = _deterministic_random_indices(
                        size=channel_size,
                        k=int(k),
                        seed=int(args.seed) + int(random_repeat),
                        label=f"{recipient_condition}<-{donor_condition}::k={k}::repeat={random_repeat}",
                    )
                    hook = _register_partial_mlp_channel_replace_hook(
                        model,
                        int(args.layer),
                        donor_channels,
                        idx_random,
                        patch_position=patch_position,
                    )
                    patched = _first_step_stats(
                        model=model,
                        input_ids=recipient_input_ids,
                        tokenizer=tokenizer,
                        target_id=target_id,
                        latin_mask=latin_mask,
                        hooks=[hook],
                    )
                    interventions.append(
                        {
                            "recipient_condition": str(recipient_condition),
                            "donor_condition": str(donor_condition),
                            "selector_kind": "random_topk",
                            "selector_policy": "random",
                            "k": int(k),
                            "actual_k": int(idx_random.numel()),
                            "random_repeat": int(random_repeat),
                            "selected_channels": [int(x) for x in idx_random.detach().cpu().tolist()],
                            "base": base_stats,
                            "patched": patched,
                            "delta": {
                                "target_prob": float(patched["target_prob"] - base_stats["target_prob"]),
                                "target_minus_competitor_logit": float(patched["target_minus_competitor_logit"] - base_stats["target_minus_competitor_logit"]),
                                "target_minus_latin_logit": float(patched["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]),
                                "top1_is_target": float(float(patched["top1_is_target"]) - float(base_stats["top1_is_target"])),
                            },
                        }
                    )

            idx_noop = selector_rank_tensors["abs"][: int(min(int(no_op_k), int(selector_rank_tensors["abs"].numel())))]
            self_channels = donor_channels_by_condition[recipient_condition]
            hook = _register_partial_mlp_channel_replace_hook(
                model,
                int(args.layer),
                self_channels,
                idx_noop,
                patch_position=patch_position,
            )
            patched = _first_step_stats(
                model=model,
                input_ids=recipient_input_ids,
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
                hooks=[hook],
            )
            interventions.append(
                {
                    "recipient_condition": str(recipient_condition),
                    "donor_condition": str(recipient_condition),
                    "selector_kind": "self_noop",
                    "selector_policy": "abs",
                    "k": int(no_op_k),
                    "actual_k": int(idx_noop.numel()),
                    "random_repeat": None,
                    "selected_channels": [int(x) for x in idx_noop.detach().cpu().tolist()],
                    "base": base_stats,
                    "patched": patched,
                    "delta": {
                        "target_prob": float(patched["target_prob"] - base_stats["target_prob"]),
                        "target_minus_competitor_logit": float(patched["target_minus_competitor_logit"] - base_stats["target_minus_competitor_logit"]),
                        "target_minus_latin_logit": float(patched["target_minus_latin_logit"] - base_stats["target_minus_latin_logit"]),
                        "top1_is_target": float(float(patched["top1_is_target"]) - float(base_stats["top1_is_target"])),
                    },
                }
            )

        item_rows.append(
            {
                "item_index": int(item_idx - 1),
                "word_ood": str(word["ood"]),
                "word_hindi": str(word["hindi"]),
                "target_id": int(target_id),
                "patch_position_mode": "last_token",
                "last_position_by_condition": {k: int(v["last_position"]) for k, v in cond_inputs.items()},
                "base_by_condition": base_by_condition,
                "interventions": interventions,
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in PATCH_DIRECTIONS:
        for selector_kind in ("selected_topk", "random_topk", "self_noop"):
            selector_policy_values = ["abs"] if selector_kind == "self_noop" else (["random"] if selector_kind == "random_topk" else selector_policies)
            candidate_ks = [no_op_k] if selector_kind == "self_noop" else (random_k_grid if selector_kind == "random_topk" else k_grid)
            for selector_policy in selector_policy_values:
                for k in candidate_ks:
                    repeat_ids = [None]
                    if selector_kind == "random_topk":
                        repeat_ids = list(range(int(args.n_random)))
                    for repeat_id in repeat_ids:
                        rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                        for item in item_rows:
                            for intr in item["interventions"]:
                                if (
                                    intr["recipient_condition"] == recipient_condition
                                    and intr["donor_condition"] == (recipient_condition if selector_kind == "self_noop" else donor_condition)
                                    and intr["selector_kind"] == selector_kind
                                    and intr.get("selector_policy", "abs") == selector_policy
                                    and int(intr["k"]) == int(k)
                                    and intr.get("random_repeat") == repeat_id
                                ):
                                    rows.append((item, intr))
                        if not rows:
                            continue
                        base_success = [1.0 if bool(intr["base"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                        patched_success = [1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in rows]
                        failed_base_rows = [(item, intr) for item, intr in rows if not bool(intr["base"]["top1_is_target"])]
                        succeeded_base_rows = [(item, intr) for item, intr in rows if bool(intr["base"]["top1_is_target"])]
                        patched_script_counts = Counter(str(intr["patched"]["top1_script"]) for _item, intr in rows)
                        summary_rows.append(
                            {
                                "recipient_condition": str(recipient_condition),
                                "donor_condition": str(recipient_condition if selector_kind == "self_noop" else donor_condition),
                                "selector_kind": str(selector_kind),
                                "selector_policy": str(selector_policy),
                                "k": int(k),
                                "random_repeat": repeat_id,
                                "actual_k": int(np.nanmean([intr.get("actual_k", intr["k"]) for _item, intr in rows])),
                                "n_items": int(len(rows)),
                                "delta_mean_target_prob": float(np.nanmean([intr["delta"]["target_prob"] for _item, intr in rows])),
                                "delta_mean_gap_overall": float(np.nanmean([intr["delta"]["target_minus_competitor_logit"] for _item, intr in rows])),
                                "delta_mean_gap_latin": float(np.nanmean([intr["delta"]["target_minus_latin_logit"] for _item, intr in rows])),
                                "patched_top1_target_rate": float(np.nanmean(patched_success)),
                                "delta_top1_target_rate": float(np.nanmean([p - b for p, b in zip(patched_success, base_success)])),
                                "rescue_rate_on_base_failures": float(
                                    np.nanmean([1.0 if bool(intr["patched"]["top1_is_target"]) else 0.0 for _item, intr in failed_base_rows])
                                ) if failed_base_rows else None,
                                "harm_rate_on_base_successes": float(
                                    np.nanmean([0.0 if bool(intr["patched"]["top1_is_target"]) else 1.0 for _item, intr in succeeded_base_rows])
                                ) if succeeded_base_rows else None,
                                "patched_top1_script_counts": dict(patched_script_counts),
                            }
                        )

    comparison_rows: List[Dict[str, Any]] = []
    for recipient_condition, donor_condition in PATCH_DIRECTIONS:
        for selector_policy in selector_policies:
            for k in k_grid:
                selected = [
                    row for row in summary_rows
                    if row["recipient_condition"] == recipient_condition
                    and row["donor_condition"] == donor_condition
                    and row["selector_kind"] == "selected_topk"
                    and row.get("selector_policy", "abs") == selector_policy
                    and int(row["k"]) == int(k)
                ]
                if not selected:
                    continue
                selected_row = selected[0]
                random_rows = [
                    row for row in summary_rows
                    if row["recipient_condition"] == recipient_condition
                    and row["donor_condition"] == donor_condition
                    and row["selector_kind"] == "random_topk"
                    and int(row["k"]) == int(k)
                ]
                dense_rows = [
                    row for row in summary_rows
                    if row["recipient_condition"] == recipient_condition
                    and row["donor_condition"] == donor_condition
                    and row["selector_kind"] == "selected_topk"
                    and row.get("selector_policy", "abs") == selector_policy
                    and int(row["actual_k"]) == int(channel_size)
                ]
                dense_gap = float(dense_rows[0]["delta_mean_gap_latin"]) if dense_rows else float("nan")
                selected_gap = float(selected_row["delta_mean_gap_latin"])
                comparison_rows.append(
                    {
                        "recipient_condition": str(recipient_condition),
                        "donor_condition": str(donor_condition),
                        "selector_policy": str(selector_policy),
                        "k": int(k),
                        "actual_k": int(selected_row.get("actual_k", k)),
                        "selected_delta_mean_gap_latin": selected_gap,
                        "selected_delta_top1_target_rate": float(selected_row["delta_top1_target_rate"]),
                        "selected_rescue_rate_on_base_failures": selected_row["rescue_rate_on_base_failures"],
                        "random_mean_delta_gap_latin": float(np.nanmean([row["delta_mean_gap_latin"] for row in random_rows])) if random_rows else None,
                        "random_std_delta_gap_latin": float(np.nanstd([row["delta_mean_gap_latin"] for row in random_rows])) if random_rows else None,
                        "selected_minus_random_mean_gap_latin": float(selected_gap - np.nanmean([row["delta_mean_gap_latin"] for row in random_rows])) if random_rows else None,
                        "dense_delta_mean_gap_latin": dense_gap,
                        "fraction_of_dense_gap_latin": float(selected_gap / dense_gap) if np.isfinite(dense_gap) and abs(dense_gap) > 1e-8 else None,
                    }
                )

    payload = {
        "experiment": "hindi_1b_mlp_channel_patch",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "selector_max_items": int(args.selector_max_items),
        "max_items": int(args.max_items),
        "layer": int(args.layer),
        "component": "mlp_channel_basis",
        "patch_position_mode": "last_token",
        "hidden_size": int(hidden_size),
        "channel_size": int(channel_size),
        "k_grid": [int(x) for x in k_grid],
        "random_k_grid": [int(x) for x in random_k_grid],
        "n_random": int(args.n_random),
        "selector": {
            "selection_split_only": True,
            "score": "mean_abs_delta(zs_mlp_channels_last_token - icl_helpful_mlp_channels_last_token)",
            "requested_policies": list(selector_policies),
            "top_ranked_channels_by_policy": selector_rank_rows_by_policy,
        },
        "oracle": {
            "description": "Held-out eval test of whether the Hindi L25 bottleneck becomes more concentrated in the actual MLP channel basis than in the raw residual-output basis.",
            "primary_hypothesis": "If the channel basis is more mechanistic than the raw output basis, smaller selected channel sets should recover a larger fraction of the dense effect.",
            "secondary_hypotheses": [
                "Sign-split channel subsets may separate rescuing versus harming directions more cleanly than raw output coordinates.",
                "Self-patching at selected channels should be a no-op sanity check.",
            ],
        },
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "item_rows": item_rows,
    }

    out_path = out_root / "hindi_1b_mlp_channel_patch.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)

    print("\n=== Helpful<-zs selected-topk curve (MLP channel basis) ===", flush=True)
    for selector_policy in selector_policies:
        print(f"policy={selector_policy}", flush=True)
        helpful_rows = [
            row for row in summary_rows
            if row["recipient_condition"] == "icl_helpful"
            and row["donor_condition"] == "zs"
            and row["selector_kind"] == "selected_topk"
            and row.get("selector_policy", "abs") == selector_policy
        ]
        helpful_rows = sorted(helpful_rows, key=lambda row: int(row["k"]))
        for row in helpful_rows:
            print(
                f"  k={row['k']:>4d} actual_k={int(row['actual_k']):>4d} dgapLatin={row['delta_mean_gap_latin']:.3f} dtop1={row['delta_top1_target_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
