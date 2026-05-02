#!/usr/bin/env python3
"""
Mechanistic localizer: 1B Hindi early-routing layerwise trace.

For each eval word under zs / helpful / corrupt conditions,
does a teacher-forced forward pass and records at every layer:
  - P(correct_target_token)
  - logit(correct_target_token)
  - rank of correct_target_token
  - identity / logit / script of the top-1 competitor
  - top-5 tokens

This lets us see WHERE in the network the correct first token loses
the competition — the central mechanistic question for 1B Hindi.

Usage (on VM):
  python3 experiments/mech_localizer_hindi_routing.py \
      --model 1b --pair aksharantar_hin_latin \
      --n-icl 64 --seed 42 --max-words 30 \
      --out research/results/autoresearch/mech_localizer_v1/1b_hindi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


CONFIRMATORY_SYSTEM_PROMPT = (
    "You are a transliteration assistant. Convert only the input token from the "
    "input script to the output script. Return only the transliterated token "
    "without explanation."
)

# ---------------------------------------------------------------------------
# Bootstrap imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = PROJECT_ROOT / "Draft_Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAFT_ROOT))

from core import apply_chat_template as core_apply_chat_template  # noqa: E402
from core import build_corrupted_icl_prompt, build_task_prompt  # noqa: E402
from paper2_fidelity_calibrated.phase1_common import load_pair_split  # noqa: E402


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return str(v)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _detect_script(text: str) -> str:
    """Detect dominant Unicode script of a short string."""
    scripts: dict[str, int] = {}
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            name = ""
        if "DEVANAGARI" in name:
            scripts["devanagari"] = scripts.get("devanagari", 0) + 1
        elif "TELUGU" in name:
            scripts["telugu"] = scripts.get("telugu", 0) + 1
        elif "BENGALI" in name or "BANGLA" in name:
            scripts["bengali"] = scripts.get("bengali", 0) + 1
        elif "TAMIL" in name:
            scripts["tamil"] = scripts.get("tamil", 0) + 1
        elif "LATIN" in name or ch.isascii():
            scripts["latin"] = scripts.get("latin", 0) + 1
        else:
            scripts["other"] = scripts.get("other", 0) + 1
    if not scripts:
        return "unknown"
    return max(scripts, key=scripts.get)


def _set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading  (lightweight, no legacy core.py dependency)
# ---------------------------------------------------------------------------
def _load_model(model_tag: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_map = {
        "270m": "google/gemma-3-270m-it",
        "1b": "google/gemma-3-1b-it",
        "4b": "google/gemma-3-4b-it",
    }
    name = model_map.get(model_tag, model_tag)
    print(f"Loading {name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _get_final_norm(model) -> torch.nn.Module:
    """Return the final RMSNorm / LayerNorm before unembedding."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise RuntimeError("Cannot locate final norm layer")


def _get_unembed(model) -> torch.Tensor:
    """Return the unembedding weight matrix [vocab, hidden]."""
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head weight")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _apply_chat_template(tokenizer, user_text: str) -> str:
    """Wrap user text in the same canonical system+user chat template used by the working behavioral scripts."""
    messages = [
        {"role": "system", "content": CONFIRMATORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        rich = [
            {"role": str(m["role"]), "content": [{"type": "text", "text": str(m["content"])}]}
            for m in messages
        ]
        try:
            return tokenizer.apply_chat_template(
                rich, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return f"{CONFIRMATORY_SYSTEM_PROMPT}\n\n{user_text}"


# Prompt format must match the canonical format used by the working behavioral scripts.
# See: Draft_Results/rescue_research/prompts/templates.py :: confirmatory_user_prompt
SCRIPT_TO_LANGUAGE = {
    "Devanagari": "Hindi",
    "Telugu": "Telugu",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
}


def _build_user_prompt(
    word_source: str,
    output_script_name: str,
    icl_examples: Optional[List[dict]] = None,
) -> str:
    """
    Build the user-text prompt in the canonical format that the model
    was behaviorally validated on.  NOT YET chat-wrapped.
    """
    lang = SCRIPT_TO_LANGUAGE.get(output_script_name, "Hindi")
    lines = [
        f"Task: Transliterate {lang} written in Latin into {output_script_name}.",
        "Output only the transliterated token.",
    ]
    if icl_examples:
        lines.append("Examples:")
        for ex in icl_examples:
            lines.append(f"{ex['source']} -> {ex['target']}")
    lines.append("Now transliterate:")
    lines.append(f"{word_source} ->")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core: layerwise logit-lens trace
# ---------------------------------------------------------------------------
def _layerwise_trace(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_id: int,
    tokenizer: Any,
    final_norm: torch.nn.Module,
    unembed: torch.Tensor,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run a teacher-forced forward pass and at each layer project the residual stream
    to vocabulary space.

    Important Gemma-specific detail:
    - intermediate hidden_states need the final norm before unembedding
    - the final hidden_states entry already corresponds to the model's final output
      state closely enough that we should use `out.logits` directly rather than
      applying the final norm a second time.
    """
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple of (batch, seq, hidden)
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states")

    rows = []
    last_idx = len(hidden_states) - 1
    for layer_idx, h in enumerate(hidden_states):
        if layer_idx == last_idx:
            logits = out.logits[0, -1, :].float()
        else:
            vec = h[0, -1, :].detach().to(dtype=unembed.dtype)
            normed = final_norm(vec.unsqueeze(0))
            logits = torch.nn.functional.linear(normed, unembed).float()[0]
        probs = torch.softmax(logits, dim=-1)

        target_prob = float(probs[target_id].item())
        target_logit = float(logits[target_id].item())
        target_rank = int(torch.sum(logits > logits[target_id]).item()) + 1

        # Top-k
        topk = torch.topk(logits, k=min(top_k, int(logits.numel())))
        topk_ids = topk.indices.tolist()
        topk_logits = [float(x) for x in topk.values.tolist()]
        topk_probs = [float(probs[i].item()) for i in topk_ids]
        topk_tokens = [tokenizer.decode([i]).strip() for i in topk_ids]
        topk_scripts = [_detect_script(t) for t in topk_tokens]

        # Best competitor (not the target)
        logits_masked = logits.clone()
        logits_masked[target_id] = -float("inf")
        comp_id = int(torch.argmax(logits_masked).item())
        comp_token = tokenizer.decode([comp_id]).strip()
        comp_script = _detect_script(comp_token)
        comp_logit = float(logits[comp_id].item())
        comp_prob = float(probs[comp_id].item())

        rows.append({
            "layer": layer_idx,
            "target_prob": target_prob,
            "target_logit": target_logit,
            "target_rank": target_rank,
            "target_minus_competitor_logit": target_logit - comp_logit,
            "competitor_id": comp_id,
            "competitor_token": comp_token,
            "competitor_script": comp_script,
            "competitor_logit": comp_logit,
            "competitor_prob": comp_prob,
            "top1_is_target": topk_ids[0] == target_id if topk_ids else False,
            "top_k_tokens": topk_tokens,
            "top_k_logits": topk_logits,
            "top_k_probs": topk_probs,
            "top_k_scripts": topk_scripts,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="1B Hindi early-routing layerwise localizer")
    ap.add_argument("--model", type=str, default="1b")
    ap.add_argument("--pair", type=str, default="aksharantar_hin_latin")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-icl", type=int, default=64)
    ap.add_argument("--n-select", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    _set_all_seeds(args.seed)

    # ---- Data: exact audited split path ----
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
    helpful_examples = list(bundle["icl_examples"])
    eval_rows = list(bundle["eval_rows"][: max(1, int(args.max_words))])
    input_script_name = str(bundle["input_script_name"])
    source_language = str(bundle["source_language"])
    output_script_name = str(bundle["output_script_name"])
    print(
        f"Loaded audited split: icl={len(helpful_examples)} select={int(args.n_select)} eval_available={len(bundle['eval_rows'])} using={len(eval_rows)}"
    )

    # ---- Model ----
    model, tokenizer = _load_model(args.model, args.device)
    device = str(next(model.parameters()).device)
    final_norm = _get_final_norm(model).to(device)
    unembed = _get_unembed(model).to(device)

    n_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else "?"
    print(f"Model loaded: {args.model}, layers={n_layers}, device={device}", flush=True)

    # ---- Output root ----
    out_root = Path(args.out).resolve() if args.out else (
        PROJECT_ROOT / "research" / "results" / "autoresearch" /
        "mech_localizer_v1" / args.model / args.pair / f"nicl{args.n_icl}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Run ----
    condition_names = ["zs", "icl_helpful", "icl_corrupt"]
    all_items: List[Dict[str, Any]] = []

    for item_idx, row in enumerate(eval_rows):
        word_source = row["ood"]
        word_target = row["hindi"]
        print(f"[{item_idx+1}/{len(eval_rows)}] {word_source} -> {word_target}", flush=True)

        prompt_by_condition = {
            "zs": build_task_prompt(
                word_source,
                None,
                input_script_name=input_script_name,
                source_language=source_language,
                output_script_name=output_script_name,
                prompt_variant="canonical",
            ),
            "icl_helpful": build_task_prompt(
                word_source,
                helpful_examples,
                input_script_name=input_script_name,
                source_language=source_language,
                output_script_name=output_script_name,
                prompt_variant="canonical",
            ),
            "icl_corrupt": build_corrupted_icl_prompt(
                word_source,
                helpful_examples,
                input_script_name=input_script_name,
                source_language=source_language,
                output_script_name=output_script_name,
                seed=int(args.seed),
            ),
        }

        # Get the correct first target token
        target_tokens = tokenizer.encode(word_target, add_special_tokens=False)
        if not target_tokens:
            print(f"  SKIP: empty target tokenization", flush=True)
            continue
        first_target_id = target_tokens[0]
        first_target_text = tokenizer.decode([first_target_id]).strip()

        item_result = {
            "item_index": item_idx,
            "word_source": word_source,
            "word_target": word_target,
            "first_target_token_id": first_target_id,
            "first_target_token_text": first_target_text,
            "first_target_token_script": _detect_script(first_target_text),
            "conditions": {},
        }

        for cond_name, prompt_text in prompt_by_condition.items():
            rendered = core_apply_chat_template(tokenizer, str(prompt_text))
            input_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)

            layer_rows = _layerwise_trace(
                model=model,
                input_ids=input_ids,
                target_id=first_target_id,
                tokenizer=tokenizer,
                final_norm=final_norm,
                unembed=unembed,
                top_k=5,
            )

            item_result["conditions"][cond_name] = {
                "prompt_tokens": int(input_ids.shape[1]),
                "layers": layer_rows,
            }

        all_items.append(item_result)

    # ---- Aggregate summary ----
    summary_rows = []
    if all_items:
        cond_names = list(all_items[0]["conditions"].keys())
        # Get layer count from first item
        n_layers_actual = len(all_items[0]["conditions"][cond_names[0]]["layers"])
        for cond in cond_names:
            for layer_idx in range(n_layers_actual):
                probs = []
                ranks = []
                gaps = []
                top1_is_target_count = 0
                comp_scripts: dict[str, int] = {}
                for item in all_items:
                    lr = item["conditions"][cond]["layers"][layer_idx]
                    probs.append(lr["target_prob"])
                    ranks.append(lr["target_rank"])
                    gaps.append(lr["target_minus_competitor_logit"])
                    if lr["top1_is_target"]:
                        top1_is_target_count += 1
                    cs = lr["competitor_script"]
                    comp_scripts[cs] = comp_scripts.get(cs, 0) + 1

                n = len(probs)
                summary_rows.append({
                    "condition": cond,
                    "layer": layer_idx,
                    "n_items": n,
                    "mean_target_prob": float(np.mean(probs)),
                    "mean_target_rank": float(np.mean(ranks)),
                    "median_target_rank": float(np.median(ranks)),
                    "mean_target_minus_competitor_logit": float(np.mean(gaps)),
                    "top1_target_rate": top1_is_target_count / max(n, 1),
                    "competitor_script_counts": comp_scripts,
                })

    payload = {
        "experiment": "mech_localizer_layerwise_routing",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "pair": args.pair,
        "seed": args.seed,
        "n_icl": args.n_icl,
        "max_words": args.max_words,
        "n_items_actual": len(all_items),
        "conditions": list(condition_names),
        "oracle": {
            "description": "At each layer, does the correct first target token still lead the logit competition?",
            "success_criterion": "Identify a layer band where target_prob drops and competitor_script shifts toward Latin under helpful ICL but not under zs.",
            "failure_criterion": "If target_prob is uniformly low or high across all layers and conditions, localization has failed.",
        },
        "summary": summary_rows,
        "items": all_items,
    }

    _write_json(out_root / "layerwise_routing_trace.json", payload)
    print(f"\nSaved: {out_root / 'layerwise_routing_trace.json'}", flush=True)

    # Quick console summary
    print("\n=== Quick summary (mean target prob by layer, last 10 layers) ===")
    for cond in condition_names:
        cond_rows = [r for r in summary_rows if r["condition"] == cond]
        if cond_rows:
            last10 = cond_rows[-10:]
            vals = ", ".join(f"L{r['layer']}={r['mean_target_prob']:.3f}" for r in last10)
            print(f"  {cond}: {vals}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
