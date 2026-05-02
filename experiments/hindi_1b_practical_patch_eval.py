#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import load_model, set_all_seeds  # noqa: E402
from experiments.hindi_1b_mlp_channel_subset_panel import (  # noqa: E402
    _build_latin_mask,
    _extract_mlp_channel_vector,
    _first_step_stats,
    _get_mlp,
    _prepare_condition_inputs,
)
from paper2_fidelity_calibrated.eval_utils import (  # noqa: E402
    akshara_cer,
    continuation_akshara_cer,
    exact_match,
    first_entry_correct,
    script_compliance,
)
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402
from research.modules.eval.output_extraction import (  # noqa: E402
    analyze_generation_text,
    extract_transliteration_candidate,
    resolve_generation_stop_ids,
    resolve_pad_token_id,
)


INTERVENTIONS = (
    "baseline_no_patch",
    "chosen_mean_shift",
    "chosen_sign_flip",
    "chosen_zero_ablate",
)


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


def _parse_channels(text: str) -> List[int]:
    out = [int(p.strip()) for p in str(text).split(",") if p.strip()]
    if not out:
        raise ValueError("No channels provided.")
    return out


def _parse_alpha_grid(text: str) -> List[float]:
    vals = [float(p.strip()) for p in str(text).split(",") if p.strip()]
    if not vals:
        raise ValueError("No alpha values provided.")
    out: List[float] = []
    for v in vals:
        if v not in out:
            out.append(float(v))
    return out


def _load_external_patch(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected_mean_delta")
    if not isinstance(selected, Mapping):
        raise ValueError(f"External patch payload missing selected_mean_delta: {path}")
    mean_delta = [float(x) for x in selected.get("mean_delta", [])]
    channels = [int(x) for x in selected.get("channels", payload.get("channels", []))]
    if not mean_delta:
        raise ValueError(f"External patch payload missing mean_delta values: {path}")
    if not channels:
        raise ValueError(f"External patch payload missing channels: {path}")
    if len(mean_delta) != len(channels):
        raise ValueError(
            f"External patch mean_delta/channels length mismatch: {len(mean_delta)} vs {len(channels)} in {path}"
        )
    return {
        "path": str(Path(path)),
        "channels": channels,
        "mean_delta": mean_delta,
        "selected_alpha": float(payload.get("selected_alpha", 1.0)),
        "selected_mean_delta": selected,
    }


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, n_boot: int = 5000) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if arr.size == 1:
        val = float(arr[0])
        return {"mean": val, "ci_low": val, "ci_high": val, "n": 1}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(arr.size)}


def _deterministic_random_pairs(
    *,
    channel_size: int,
    chosen_channels: Sequence[int],
    n_random: int,
    seed: int,
) -> List[List[int]]:
    forbidden = set(int(c) for c in chosen_channels)
    pool = [i for i in range(int(channel_size)) if i not in forbidden]
    rng_seed = int.from_bytes(
        hashlib.sha256(f"random_pairs::{seed}::{channel_size}::{sorted(forbidden)}::{n_random}".encode("utf-8")).digest()[:4],
        "little",
        signed=False,
    )
    rng = np.random.default_rng(rng_seed)
    out: List[List[int]] = []
    seen = set()
    while len(out) < int(n_random):
        pair = tuple(sorted(int(x) for x in rng.choice(pool, size=len(chosen_channels), replace=False).tolist()))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(list(pair))
    return out


def _register_channel_add_hook(
    model: Any,
    *,
    layer: int,
    patch_position: int,
    channel_indices: Sequence[int],
    delta_values: Sequence[float],
):
    mlp = _get_mlp(model, int(layer))
    idx = torch.tensor([int(x) for x in channel_indices], dtype=torch.long)
    delta = torch.tensor([float(x) for x in delta_values], dtype=torch.float32)

    def pre_hook(module, inputs_tuple):
        if not inputs_tuple:
            return None
        x = inputs_tuple[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None
        seq_len = int(x.shape[1])
        pos = int(max(0, min(int(patch_position), seq_len - 1)))
        use_idx = idx.to(device=x.device)
        use_delta = delta.to(device=x.device, dtype=x.dtype)
        x_new = x.clone()
        x_new[:, pos, use_idx] = x_new[:, pos, use_idx] + use_delta.view(1, -1)
        if len(inputs_tuple) == 1:
            return (x_new,)
        return (x_new,) + tuple(inputs_tuple[1:])

    return mlp.down_proj.register_forward_pre_hook(pre_hook)


def _register_channel_zero_hook(
    model: Any,
    *,
    layer: int,
    patch_position: int,
    channel_indices: Sequence[int],
):
    mlp = _get_mlp(model, int(layer))
    idx = torch.tensor([int(x) for x in channel_indices], dtype=torch.long)

    def pre_hook(module, inputs_tuple):
        if not inputs_tuple:
            return None
        x = inputs_tuple[0]
        if not torch.is_tensor(x) or x.ndim != 3:
            return None
        seq_len = int(x.shape[1])
        pos = int(max(0, min(int(patch_position), seq_len - 1)))
        use_idx = idx.to(device=x.device)
        x_new = x.clone()
        x_new[:, pos, use_idx] = 0.0
        if len(inputs_tuple) == 1:
            return (x_new,)
        return (x_new,) + tuple(inputs_tuple[1:])

    return mlp.down_proj.register_forward_pre_hook(pre_hook)


def _generate_raw_text(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_ids: int | list[int],
    pad_id: int,
    hooks: Sequence[Any] | None = None,
) -> str:
    active = list(hooks or [])
    try:
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=False,
                eos_token_id=stop_ids,
                pad_token_id=int(pad_id),
            )
        new_tokens = out[0, int(input_ids.shape[1]) :]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        for hook in reversed(active):
            try:
                hook.remove()
            except Exception:
                pass


def _evaluate_generation(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    target_text: str,
    output_script_name: str,
    max_new_tokens: int,
    stop_ids: int | list[int],
    pad_id: int,
    hooks: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    raw_text = _generate_raw_text(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        pad_id=pad_id,
        hooks=hooks,
    )
    pred = extract_transliteration_candidate(raw_text, script_name=str(output_script_name), min_script_ratio=0.80)
    audit = analyze_generation_text(raw_text, pred)
    gold = str(target_text)
    return {
        "raw_text": raw_text,
        "prediction": pred,
        "exact_match": float(exact_match(pred, gold)),
        "akshara_cer": float(akshara_cer(pred, gold)),
        "first_entry_correct": float(first_entry_correct(pred, gold)),
        "continuation_akshara_cer": float(continuation_akshara_cer(pred, gold)),
        "script_compliance": float(script_compliance(pred, str(output_script_name))),
        "raw_strict_word_only": float(int(audit.get("strict_word_only", False))),
        "has_leading_text": float(int(audit.get("has_leading_text", False))),
        "has_trailing_text": float(int(audit.get("has_trailing_text", False))),
    }


def _mean_delta_vector(
    *,
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    icl_examples: List[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
    layer: int,
    channels: Sequence[int],
    prompt_variant: str,
) -> Dict[str, Any]:
    diffs: List[np.ndarray] = []
    for word in rows:
        cond_inputs = _prepare_condition_inputs(
            tokenizer=tokenizer,
            word=word,
            icl_examples=icl_examples,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            device=device,
            prompt_variant=str(prompt_variant),
        )
        helpful_vec = _extract_mlp_channel_vector(model, cond_inputs["icl_helpful"]["input_ids"], int(layer), int(cond_inputs["icl_helpful"]["last_position"]))
        zs_vec = _extract_mlp_channel_vector(model, cond_inputs["zs"]["input_ids"], int(layer), int(cond_inputs["zs"]["last_position"]))
        diff = (zs_vec.detach().float().cpu().numpy() - helpful_vec.detach().float().cpu().numpy())[[int(c) for c in channels]]
        diffs.append(diff.astype(np.float64))
    arr = np.stack(diffs, axis=0) if diffs else np.zeros((0, len(channels)), dtype=np.float64)
    return {
        "channels": [int(c) for c in channels],
        "n_items": int(arr.shape[0]),
        "mean_delta": [float(x) for x in arr.mean(axis=0)] if arr.size else [0.0 for _ in channels],
        "mean_delta_norm": float(np.linalg.norm(arr.mean(axis=0))) if arr.size else 0.0,
        "per_channel_mean_abs": [float(x) for x in np.mean(np.abs(arr), axis=0)] if arr.size else [0.0 for _ in channels],
    }


def _selection_alpha_sweep(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    icl_examples: List[Dict[str, Any]],
    input_script_name: str,
    source_language: str,
    output_script_name: str,
    device: str,
    layer: int,
    channels: Sequence[int],
    mean_delta: Sequence[float],
    alpha_grid: Sequence[float],
    latin_mask: torch.Tensor,
    prompt_variant: str,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for alpha in alpha_grid:
        delta = [float(alpha) * float(v) for v in mean_delta]
        gains_gap: List[float] = []
        gains_target: List[float] = []
        gains_latin_rate: List[float] = []
        top1_target: List[float] = []
        top1_latin: List[float] = []
        for word in rows:
            target_ids = tokenizer.encode(str(word["hindi"]), add_special_tokens=False)
            if not target_ids:
                continue
            target_id = int(target_ids[0])
            cond_inputs = _prepare_condition_inputs(
                tokenizer=tokenizer,
                word=word,
                icl_examples=icl_examples,
                input_script_name=input_script_name,
                source_language=source_language,
                output_script_name=output_script_name,
                device=device,
                prompt_variant=str(prompt_variant),
            )
            helpful = cond_inputs["icl_helpful"]
            base = _first_step_stats(
                model=model,
                input_ids=helpful["input_ids"],
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
            )
            hook = _register_channel_add_hook(
                model,
                layer=int(layer),
                patch_position=int(helpful["last_position"]),
                channel_indices=channels,
                delta_values=delta,
            )
            patched = _first_step_stats(
                model=model,
                input_ids=helpful["input_ids"],
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
                hooks=[hook],
            )
            gains_gap.append(float(patched["target_minus_latin_logit"] - base["target_minus_latin_logit"]))
            gains_target.append(float(patched["target_prob"] - base["target_prob"]))
            gains_latin_rate.append(float((patched["top1_script"] == "latin") - (base["top1_script"] == "latin")))
            top1_target.append(float(patched["top1_is_target"]))
            top1_latin.append(float(patched["top1_script"] == "latin"))
        rows_out.append(
            {
                "alpha": float(alpha),
                "n_items": int(len(gains_gap)),
                "delta_mean_target_minus_latin_logit": float(np.mean(gains_gap)) if gains_gap else float("nan"),
                "delta_mean_target_prob": float(np.mean(gains_target)) if gains_target else float("nan"),
                "delta_latin_top1_rate": float(np.mean(gains_latin_rate)) if gains_latin_rate else float("nan"),
                "patched_top1_target_rate": float(np.mean(top1_target)) if top1_target else float("nan"),
                "patched_top1_latin_rate": float(np.mean(top1_latin)) if top1_latin else float("nan"),
            }
        )
    return rows_out


def _summarize_intervention(
    *,
    name: str,
    items: Sequence[Mapping[str, Any]],
    key_prefix: str,
    base_prefix: str = "baseline_no_patch",
    seed: int,
) -> Dict[str, Any]:
    exact = [float(item[key_prefix]["generation"]["exact_match"]) for item in items]
    cer = [float(item[key_prefix]["generation"]["akshara_cer"]) for item in items]
    first = [float(item[key_prefix]["generation"]["first_entry_correct"]) for item in items]
    script = [float(item[key_prefix]["generation"]["script_compliance"]) for item in items]
    gap = [float(item[key_prefix]["first_step"]["target_minus_latin_logit"]) for item in items]
    target_top1 = [float(item[key_prefix]["first_step"]["top1_is_target"]) for item in items]
    latin_top1 = [float(item[key_prefix]["first_step"]["top1_script"] == "latin") for item in items]

    delta_exact = [float(item[key_prefix]["generation"]["exact_match"] - item[base_prefix]["generation"]["exact_match"]) for item in items]
    delta_cer_improvement = [float(item[base_prefix]["generation"]["akshara_cer"] - item[key_prefix]["generation"]["akshara_cer"]) for item in items]
    delta_first = [float(item[key_prefix]["generation"]["first_entry_correct"] - item[base_prefix]["generation"]["first_entry_correct"]) for item in items]
    delta_gap = [float(item[key_prefix]["first_step"]["target_minus_latin_logit"] - item[base_prefix]["first_step"]["target_minus_latin_logit"]) for item in items]
    delta_latin = [float((item[key_prefix]["first_step"]["top1_script"] == "latin") - (item[base_prefix]["first_step"]["top1_script"] == "latin")) for item in items]

    base_fail = [item for item in items if float(item[base_prefix]["generation"]["exact_match"]) < 0.5]
    base_success = [item for item in items if float(item[base_prefix]["generation"]["exact_match"]) > 0.5]
    rescue_rate = float(np.mean([float(item[key_prefix]["generation"]["exact_match"]) for item in base_fail])) if base_fail else float("nan")
    harm_rate = float(np.mean([1.0 - float(item[key_prefix]["generation"]["exact_match"]) for item in base_success])) if base_success else float("nan")

    return {
        "intervention": str(name),
        "n_items": int(len(items)),
        "exact_match": _bootstrap_mean_ci(exact, seed=seed + 11),
        "akshara_cer": _bootstrap_mean_ci(cer, seed=seed + 17),
        "first_entry_correct": _bootstrap_mean_ci(first, seed=seed + 23),
        "script_compliance": _bootstrap_mean_ci(script, seed=seed + 29),
        "first_token_gap_latin": _bootstrap_mean_ci(gap, seed=seed + 31),
        "first_token_top1_target_rate": _bootstrap_mean_ci(target_top1, seed=seed + 37),
        "first_token_top1_latin_rate": _bootstrap_mean_ci(latin_top1, seed=seed + 41),
        "delta_exact_match": _bootstrap_mean_ci(delta_exact, seed=seed + 43),
        "delta_cer_improvement": _bootstrap_mean_ci(delta_cer_improvement, seed=seed + 47),
        "delta_first_entry_correct": _bootstrap_mean_ci(delta_first, seed=seed + 53),
        "delta_first_token_gap_latin": _bootstrap_mean_ci(delta_gap, seed=seed + 59),
        "delta_first_token_top1_latin_rate": _bootstrap_mean_ci(delta_latin, seed=seed + 61),
        "rescue_rate_on_base_failures": rescue_rate,
        "harm_rate_on_base_successes": harm_rate,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a practical Hindi 1B inference-time patch derived from the bounded channel mechanism.")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--select-max-items", type=int, default=100)
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--channels", type=str, default="5486,2299")
    ap.add_argument("--alpha-grid", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--prompt-variant", type=str, default="canonical")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--external-patch-json", type=str, default="")
    ap.add_argument("--external-patch-use-selected-alpha", action="store_true")
    ap.add_argument("--override-alpha", type=float, default=float("nan"))
    ap.add_argument("--external-only", action="store_true")
    ap.add_argument("--require-external-sources", action="store_true")
    ap.add_argument("--min-pool-size", type=int, default=500)
    ap.add_argument("--out-root", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    channels = _parse_channels(args.channels)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    set_all_seeds(int(args.seed))

    bundle = load_pair_split(
        str(args.pair),
        seed=int(args.seed),
        n_icl=int(args.n_icl),
        n_select=int(args.n_select),
        n_eval=int(args.n_eval),
        external_only=bool(args.external_only),
        require_external_sources=bool(args.require_external_sources),
        min_pool_size=int(args.min_pool_size),
    )
    select_rows = list(bundle["select_rows"][: max(1, int(args.select_max_items))])
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_items))])

    model, tokenizer = load_model(str(args.model), device=str(args.device))
    device = str(next(model.parameters()).device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 0)))
    latin_mask = _build_latin_mask(tokenizer, vocab_size)
    stop_ids = resolve_generation_stop_ids(tokenizer)
    pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=stop_ids)
    channel_size = int(_get_mlp(model, int(args.layer)).down_proj.in_features)
    random_pairs = _deterministic_random_pairs(
        channel_size=channel_size,
        chosen_channels=channels,
        n_random=int(args.n_random),
        seed=int(args.seed),
    )

    external_patch: Dict[str, Any] | None = None
    if str(args.external_patch_json).strip():
        external_patch = _load_external_patch(str(args.external_patch_json).strip())
        channels = [int(x) for x in external_patch["channels"]]

    print(
        f"Running Hindi practical patch eval: model={args.model} pair={args.pair} select_items={len(select_rows)} eval_items={len(eval_rows)} channels={channels}",
        flush=True,
    )
    if external_patch is not None:
        print(
            f"Using external patch source: {external_patch['path']} | selected_alpha={external_patch['selected_alpha']:.3f}",
            flush=True,
        )

    if external_patch is not None:
        chosen_mean = {
            "channels": [int(x) for x in external_patch["channels"]],
            "n_items": int(external_patch["selected_mean_delta"].get("n_items", 0)),
            "mean_delta": [float(x) for x in external_patch["mean_delta"]],
            "mean_delta_norm": float(external_patch["selected_mean_delta"].get("mean_delta_norm", np.linalg.norm(external_patch["mean_delta"]))),
            "per_channel_mean_abs": [float(x) for x in external_patch["selected_mean_delta"].get("per_channel_mean_abs", [abs(float(x)) for x in external_patch["mean_delta"]])],
        }
    else:
        chosen_mean = _mean_delta_vector(
            model=model,
            rows=select_rows,
            tokenizer=tokenizer,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
            layer=int(args.layer),
            channels=channels,
            prompt_variant=str(args.prompt_variant),
        )

    use_override_alpha = not math.isnan(float(args.override_alpha))
    if external_patch is not None and (bool(args.external_patch_use_selected_alpha) or use_override_alpha):
        chosen_alpha = float(args.override_alpha) if use_override_alpha else float(external_patch["selected_alpha"])
        selection_rows = []
    else:
        selection_rows = _selection_alpha_sweep(
            model=model,
            tokenizer=tokenizer,
            rows=select_rows,
            icl_examples=bundle["icl_examples"],
            input_script_name=bundle["input_script_name"],
            source_language=bundle["source_language"],
            output_script_name=bundle["output_script_name"],
            device=device,
            layer=int(args.layer),
            channels=channels,
            mean_delta=chosen_mean["mean_delta"],
            alpha_grid=alpha_grid,
            latin_mask=latin_mask,
            prompt_variant=str(args.prompt_variant),
        )
        best_row = max(selection_rows, key=lambda row: (float(row["delta_mean_target_minus_latin_logit"]), float(row["delta_mean_target_prob"])))
        chosen_alpha = float(best_row["alpha"])
    chosen_delta = [chosen_alpha * float(v) for v in chosen_mean["mean_delta"]]

    random_pair_specs: List[Dict[str, Any]] = []
    for i, pair in enumerate(random_pairs):
        random_pair_specs.append(
            {
                "name": f"random_pair_{i+1}",
                "channels": [int(x) for x in pair],
                "delta": [float(x) for x in chosen_delta],
            }
        )

    item_rows: List[Dict[str, Any]] = []
    for idx, word in enumerate(eval_rows, start=1):
        if idx == 1 or idx == len(eval_rows) or idx % 5 == 0:
            print(f"[eval {idx}/{len(eval_rows)}] {word['ood']} -> {word['hindi']}", flush=True)
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
            prompt_variant=str(args.prompt_variant),
        )
        helpful = cond_inputs["icl_helpful"]
        base_first = _first_step_stats(
            model=model,
            input_ids=helpful["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
        )
        base_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=helpful["input_ids"],
            target_text=str(word["hindi"]),
            output_script_name=bundle["output_script_name"],
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
        )

        item_payload: Dict[str, Any] = {
            "item_index": int(idx - 1),
            "word_ood": str(word["ood"]),
            "word_hindi": str(word["hindi"]),
            "baseline_no_patch": {
                "first_step": base_first,
                "generation": base_gen,
            },
        }

        hook = _register_channel_add_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
            delta_values=chosen_delta,
        )
        patched_first = _first_step_stats(
            model=model,
            input_ids=helpful["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
            hooks=[hook],
        )
        hook = _register_channel_add_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
            delta_values=chosen_delta,
        )
        patched_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=helpful["input_ids"],
            target_text=str(word["hindi"]),
            output_script_name=bundle["output_script_name"],
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
            hooks=[hook],
        )
        item_payload["chosen_mean_shift"] = {"first_step": patched_first, "generation": patched_gen}

        sign_hook = _register_channel_add_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
            delta_values=[-float(x) for x in chosen_delta],
        )
        sign_first = _first_step_stats(
            model=model,
            input_ids=helpful["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
            hooks=[sign_hook],
        )
        sign_hook = _register_channel_add_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
            delta_values=[-float(x) for x in chosen_delta],
        )
        sign_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=helpful["input_ids"],
            target_text=str(word["hindi"]),
            output_script_name=bundle["output_script_name"],
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
            hooks=[sign_hook],
        )
        item_payload["chosen_sign_flip"] = {"first_step": sign_first, "generation": sign_gen}

        zero_hook = _register_channel_zero_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
        )
        zero_first = _first_step_stats(
            model=model,
            input_ids=helpful["input_ids"],
            tokenizer=tokenizer,
            target_id=target_id,
            latin_mask=latin_mask,
            hooks=[zero_hook],
        )
        zero_hook = _register_channel_zero_hook(
            model,
            layer=int(args.layer),
            patch_position=int(helpful["last_position"]),
            channel_indices=channels,
        )
        zero_gen = _evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=helpful["input_ids"],
            target_text=str(word["hindi"]),
            output_script_name=bundle["output_script_name"],
            max_new_tokens=int(args.max_new_tokens),
            stop_ids=stop_ids,
            pad_id=pad_id,
            hooks=[zero_hook],
        )
        item_payload["chosen_zero_ablate"] = {"first_step": zero_first, "generation": zero_gen}

        random_payloads: List[Dict[str, Any]] = []
        for spec in random_pair_specs:
            rand_hook = _register_channel_add_hook(
                model,
                layer=int(args.layer),
                patch_position=int(helpful["last_position"]),
                channel_indices=spec["channels"],
                delta_values=spec["delta"],
            )
            rand_first = _first_step_stats(
                model=model,
                input_ids=helpful["input_ids"],
                tokenizer=tokenizer,
                target_id=target_id,
                latin_mask=latin_mask,
                hooks=[rand_hook],
            )
            rand_hook = _register_channel_add_hook(
                model,
                layer=int(args.layer),
                patch_position=int(helpful["last_position"]),
                channel_indices=spec["channels"],
                delta_values=spec["delta"],
            )
            rand_gen = _evaluate_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=helpful["input_ids"],
                target_text=str(word["hindi"]),
                output_script_name=bundle["output_script_name"],
                max_new_tokens=int(args.max_new_tokens),
                stop_ids=stop_ids,
                pad_id=pad_id,
                hooks=[rand_hook],
            )
            random_payloads.append(
                {
                    "name": str(spec["name"]),
                    "channels": [int(x) for x in spec["channels"]],
                    "first_step": rand_first,
                    "generation": rand_gen,
                }
            )
        item_payload["random_controls"] = random_payloads
        item_rows.append(item_payload)

    summary_rows = [
        _summarize_intervention(name="baseline_no_patch", items=item_rows, key_prefix="baseline_no_patch", seed=int(args.seed)),
        _summarize_intervention(name="chosen_mean_shift", items=item_rows, key_prefix="chosen_mean_shift", seed=int(args.seed) + 100),
        _summarize_intervention(name="chosen_sign_flip", items=item_rows, key_prefix="chosen_sign_flip", seed=int(args.seed) + 200),
        _summarize_intervention(name="chosen_zero_ablate", items=item_rows, key_prefix="chosen_zero_ablate", seed=int(args.seed) + 300),
    ]

    random_summary_rows: List[Dict[str, Any]] = []
    for ridx, spec in enumerate(random_pair_specs):
        proxy_items = []
        for item in item_rows:
            match = next(r for r in item["random_controls"] if r["name"] == spec["name"])
            proxy_items.append(
                {
                    "baseline_no_patch": item["baseline_no_patch"],
                    spec["name"]: {"first_step": match["first_step"], "generation": match["generation"]},
                }
            )
        random_summary_rows.append(
            _summarize_intervention(
                name=str(spec["name"]),
                items=proxy_items,
                key_prefix=str(spec["name"]),
                seed=int(args.seed) + 500 + 10 * ridx,
            )
        )

    random_agg = {
        "intervention": "random_mean_shift_avg",
        "n_random": int(len(random_summary_rows)),
        "exact_match_mean": float(np.mean([r["exact_match"]["mean"] for r in random_summary_rows])) if random_summary_rows else float("nan"),
        "akshara_cer_mean": float(np.mean([r["akshara_cer"]["mean"] for r in random_summary_rows])) if random_summary_rows else float("nan"),
        "delta_exact_match_mean": float(np.mean([r["delta_exact_match"]["mean"] for r in random_summary_rows])) if random_summary_rows else float("nan"),
        "delta_cer_improvement_mean": float(np.mean([r["delta_cer_improvement"]["mean"] for r in random_summary_rows])) if random_summary_rows else float("nan"),
        "delta_first_token_gap_latin_mean": float(np.mean([r["delta_first_token_gap_latin"]["mean"] for r in random_summary_rows])) if random_summary_rows else float("nan"),
    }

    comparison_rows = []
    summary_map = {row["intervention"]: row for row in summary_rows}
    chosen = summary_map["chosen_mean_shift"]
    sign = summary_map["chosen_sign_flip"]
    zero = summary_map["chosen_zero_ablate"]
    base = summary_map["baseline_no_patch"]
    comparison_rows.append(
        {
            "comparison": "chosen_minus_baseline",
            "delta_exact_match": chosen["delta_exact_match"],
            "delta_cer_improvement": chosen["delta_cer_improvement"],
            "delta_first_entry_correct": chosen["delta_first_entry_correct"],
            "delta_first_token_gap_latin": chosen["delta_first_token_gap_latin"],
        }
    )
    comparison_rows.append(
        {
            "comparison": "chosen_minus_signflip_exact",
            "value": float(chosen["exact_match"]["mean"] - sign["exact_match"]["mean"]),
        }
    )
    comparison_rows.append(
        {
            "comparison": "chosen_minus_random_avg_exact",
            "value": float(chosen["exact_match"]["mean"] - random_agg["exact_match_mean"]),
        }
    )
    comparison_rows.append(
        {
            "comparison": "chosen_minus_zero_ablate_exact",
            "value": float(chosen["exact_match"]["mean"] - zero["exact_match"]["mean"]),
        }
    )

    payload = {
        "experiment": "hindi_1b_practical_patch_eval",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "pair": str(args.pair),
        "seed": int(args.seed),
        "n_icl": int(args.n_icl),
        "n_select": int(args.n_select),
        "n_eval": int(args.n_eval),
        "select_max_items": int(args.select_max_items),
        "max_items": int(args.max_items),
        "layer": int(args.layer),
        "channels": [int(c) for c in channels],
        "prompt_variant": str(args.prompt_variant),
        "alpha_grid": [float(a) for a in alpha_grid],
        "selection_mode": (
            "external_selected_alpha"
            if external_patch is not None and (bool(args.external_patch_use_selected_alpha) or use_override_alpha)
            else "selection_sweep"
        ),
        "external_patch_source": external_patch["path"] if external_patch is not None else "",
        "selected_alpha": chosen_alpha,
        "selected_mean_delta": chosen_mean,
        "selection_rows": selection_rows,
        "random_pair_specs": random_pair_specs,
        "summary_rows": summary_rows,
        "random_summary_rows": random_summary_rows,
        "random_aggregate": random_agg,
        "comparison_rows": comparison_rows,
        "item_rows": item_rows,
    }

    out_root = (
        Path(args.out_root).resolve()
        if str(args.out_root).strip()
        else PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_practical_patch_eval_v1" / str(args.model) / str(args.pair) / f"seed{int(args.seed)}" / f"nicl{int(args.n_icl)}"
    )
    out_path = out_root / "hindi_1b_practical_patch_eval.json"
    _write_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    print("=== Eval summary ===", flush=True)
    for row in summary_rows:
        print(
            f"  {row['intervention']:<20} EM={row['exact_match']['mean']:.3f} CER={row['akshara_cer']['mean']:.3f} "
            f"dEM={row['delta_exact_match']['mean']:.3f} dCERimp={row['delta_cer_improvement']['mean']:.3f} dGap={row['delta_first_token_gap_latin']['mean']:.3f}",
            flush=True,
        )
    print(
        f"  {'random_mean_shift_avg':<20} EM={random_agg['exact_match_mean']:.3f} dEM={random_agg['delta_exact_match_mean']:.3f} dCERimp={random_agg['delta_cer_improvement_mean']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
