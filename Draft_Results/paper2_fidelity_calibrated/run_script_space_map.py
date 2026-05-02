#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import get_model_config
from core import _get_unembedding_weight, apply_chat_template, build_task_prompt, load_model, split_data_three_way
from paper2_fidelity_calibrated.protocol_utils import prompt_fingerprint, prompt_template_fingerprint, runtime_identity
from paper2_fidelity_calibrated.run import _load_words, _prompt_naming
from rescue_research.data_pipeline.ingest import get_pair_prompt_metadata

SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "telugu": [(0x0C00, 0x0C7F)],
}

EXPECTED_ARCH: Dict[str, Dict[str, Any]] = {
    "1b": {
        "num_hidden_layers": 26,
        "hidden_size": 1152,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "sliding_window": 512,
        "global_layers": [5, 11, 17, 23],
        "max_position_embeddings": 32768,
    },
    "4b": {
        "num_hidden_layers": 34,
        "hidden_size": 2560,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "sliding_window": 1024,
        "global_layers": [5, 11, 17, 23, 29],
        "max_position_embeddings": 131072,
    },
}

DEFAULT_TASKS = [
    {"model": "4b", "pair": "aksharantar_hin_latin"},
    {"model": "1b", "pair": "aksharantar_hin_latin"},
    {"model": "4b", "pair": "aksharantar_tel_latin"},
    {"model": "1b", "pair": "aksharantar_tel_latin"},
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(base: Path, rows: List[Dict[str, Any]]) -> None:
    _write_json(base.with_suffix('.json'), rows)
    keys = sorted({str(k) for row in rows for k in row.keys()}) if rows else []
    with base.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        if not keys:
            f.write('')
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in keys})


def _mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(statistics.fmean(xs)) if xs else float('nan')


def _parse_tasks(raw: str) -> List[Dict[str, str]]:
    if not str(raw or '').strip():
        return list(DEFAULT_TASKS)
    out: List[Dict[str, str]] = []
    for chunk in str(raw).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        model, pair = chunk.split(':', 1)
        out.append({"model": model.strip(), "pair": pair.strip()})
    return out


def _text_config(model: Any) -> Any:
    return getattr(model.config, 'text_config', model.config)


def _derive_global_layers(text_cfg: Any) -> Tuple[List[int], List[int], str]:
    n_layers = int(getattr(text_cfg, 'num_hidden_layers', 0) or 0)
    layer_types = getattr(text_cfg, 'layer_types', None)
    if isinstance(layer_types, (list, tuple)) and len(layer_types) == n_layers:
        global_layers = [i for i, typ in enumerate(layer_types) if str(typ) == 'full_attention']
        local_layers = [i for i in range(n_layers) if i not in global_layers]
        return global_layers, local_layers, 'layer_types'
    pattern = int(getattr(text_cfg, 'sliding_window_pattern', 0) or 0)
    if pattern > 0:
        global_layers = [i for i in range(n_layers) if (i % pattern) == (pattern - 1)]
        local_layers = [i for i in range(n_layers) if i not in global_layers]
        return global_layers, local_layers, 'sliding_window_pattern'
    return [], list(range(n_layers)), 'none'


def _get_final_norm(model: Any) -> Tuple[Any, str]:
    candidates = [
        ('model.norm', lambda m: getattr(getattr(m, 'model', None), 'norm', None)),
        ('language_model.norm', lambda m: getattr(getattr(m, 'language_model', None), 'norm', None)),
        ('model.language_model.norm', lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'norm', None)),
        ('model.model.norm', lambda m: getattr(getattr(getattr(m, 'model', None), 'model', None), 'norm', None)),
    ]
    for path, getter in candidates:
        try:
            obj = getter(model)
        except Exception:
            obj = None
        if obj is not None:
            return obj, path
    raise RuntimeError(f'Could not locate final norm for {type(model).__name__}')


def _norm_device(mod: Any) -> torch.device:
    for p in mod.parameters(recurse=True):
        return p.device
    for b in mod.buffers(recurse=True):
        return b.device
    return torch.device('cpu')


def _architecture_packet(model_key: str, model: Any, tokenizer: Any, final_norm_path: str, unemb: torch.Tensor) -> Dict[str, Any]:
    text_cfg = _text_config(model)
    global_layers, local_layers, source = _derive_global_layers(text_cfg)
    actual = {
        'model_key': str(model_key),
        'hf_id': str(get_model_config(str(model_key)).hf_id),
        'tokenizer_name_or_path': str(getattr(tokenizer, 'name_or_path', '') or ''),
        'tokenizer_vocab_size': int(len(tokenizer)),
        'unembedding_vocab_size': int(unemb.shape[0]),
        'num_hidden_layers': int(getattr(text_cfg, 'num_hidden_layers', 0) or 0),
        'hidden_size': int(getattr(text_cfg, 'hidden_size', 0) or 0),
        'num_attention_heads': int(getattr(text_cfg, 'num_attention_heads', 0) or 0),
        'num_key_value_heads': int(getattr(text_cfg, 'num_key_value_heads', 0) or 0),
        'head_dim': int(getattr(text_cfg, 'head_dim', 0) or 0),
        'sliding_window': int(getattr(text_cfg, 'sliding_window', 0) or 0),
        'sliding_window_pattern': getattr(text_cfg, 'sliding_window_pattern', None),
        'layer_types': list(getattr(text_cfg, 'layer_types', []) or []),
        'max_position_embeddings': int(getattr(text_cfg, 'max_position_embeddings', 0) or 0),
        'global_layers': global_layers,
        'local_layers': local_layers,
        'global_layer_source': source,
        'final_norm_path': final_norm_path,
        'runtime_identity': runtime_identity(model_key=str(model_key), hf_id=str(get_model_config(str(model_key)).hf_id), tokenizer=tokenizer, model=model),
    }
    expected = EXPECTED_ARCH.get(str(model_key), {})
    checks = {}
    for key, expected_value in expected.items():
        checks[key] = {
            'expected': expected_value,
            'actual': actual.get(key),
            'match': actual.get(key) == expected_value,
        }
    actual['verification'] = checks
    mismatches = [k for k, v in checks.items() if not bool(v['match'])]
    actual['verification_status'] = 'ok' if not mismatches else 'mismatch'
    actual['verification_mismatches'] = mismatches
    if mismatches:
        raise RuntimeError(f'Architecture mismatch for {model_key}: {mismatches}')
    return actual


def _language_name(pair_id: str) -> str:
    meta = get_pair_prompt_metadata(pair_id)
    label = str(meta.get('source_language', '')).strip().lower()
    if label:
        return label
    parts = str(pair_id).split('_')
    return parts[1] if len(parts) > 1 else str(pair_id)


def _script_name(pair_id: str) -> str:
    meta = get_pair_prompt_metadata(pair_id)
    target = str(meta.get('target_script', '')).strip().lower()
    if 'devanagari' in target:
        return 'devanagari'
    if 'telugu' in target:
        return 'telugu'
    raise RuntimeError(f'Unsupported target script for pair {pair_id}: {target}')


def _char_in_script(ch: str, script_name: str) -> bool:
    cp = ord(ch)
    for lo, hi in SCRIPT_RANGES[script_name]:
        if lo <= cp <= hi:
            return True
    return False


def _script_token_cache_path(out_root: Path, tokenizer_name: str, script_name: str) -> Path:
    import hashlib
    key = hashlib.sha256(f'{tokenizer_name}::{script_name}'.encode('utf-8')).hexdigest()[:16]
    return out_root / 'token_script_id_cache' / f'{key}_{script_name}.json'


def _script_token_ids(tokenizer: Any, model_vocab_size: int, script_name: str, out_root: Path) -> List[int]:
    cache_path = _script_token_cache_path(out_root, str(getattr(tokenizer, 'name_or_path', '') or 'tokenizer'), script_name)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            ids = [int(x) for x in data.get('token_ids', [])]
            if ids:
                return ids
        except Exception:
            pass
    max_ids = min(int(model_vocab_size), int(len(tokenizer)))
    ids: List[int] = []
    for token_id in range(max_ids):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if any(_char_in_script(ch, script_name) for ch in token_text):
            ids.append(token_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({'script_name': script_name, 'token_ids': ids}, indent=2), encoding='utf-8')
    return ids


def _build_prompts(word: Mapping[str, str], *, icl_examples: Sequence[Mapping[str, str]], pair_id: str) -> Dict[str, str]:
    meta = get_pair_prompt_metadata(pair_id)
    source_language, input_script_name, output_script_name = _prompt_naming(dict(meta))
    query = str(word['ood'])
    return {
        'explicit_zs': build_task_prompt(
            query,
            None,
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant='canonical',
        ),
        'icl64': build_task_prompt(
            query,
            list(icl_examples),
            input_script_name=input_script_name,
            source_language=source_language,
            output_script_name=output_script_name,
            prompt_variant='canonical',
        ),
    }


def _topk_counts(topk_ids: torch.Tensor, script_mask: torch.Tensor) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for k in (1, 5, 10, 50):
        kk = min(k, int(topk_ids.shape[1]))
        counts = script_mask[topk_ids[:, :kk]].sum(dim=1).detach().cpu().to(torch.int64).tolist()
        out[f'target_script_count_top{k}'] = counts
        out[f'target_script_any_top{k}'] = [int(x > 0) for x in counts]
    return out


def _run_condition(
    *,
    model: Any,
    tokenizer: Any,
    final_norm: Any,
    final_norm_path: str,
    unemb: torch.Tensor,
    script_ids: List[int],
    model_key: str,
    pair_id: str,
    condition_name: str,
    eval_rows: Sequence[Mapping[str, str]],
    icl_examples: Sequence[Mapping[str, str]],
    out_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    final_norm_device = _norm_device(final_norm)
    unemb_device = unemb.device
    if final_norm_device != unemb_device:
        raise RuntimeError(f'Final norm and unembedding are on different devices: {final_norm_device} vs {unemb_device}')
    script_ids_tensor = torch.tensor(script_ids, device=unemb_device, dtype=torch.long)
    script_mask = torch.zeros((int(unemb.shape[0]),), device=unemb_device, dtype=torch.bool)
    script_mask[script_ids_tensor] = True

    row_records: List[Dict[str, Any]] = []
    summary_group: Dict[int, List[Dict[str, Any]]] = { }
    pair_meta = get_pair_prompt_metadata(pair_id)
    target_script = _script_name(pair_id)
    n_layers = int(_text_config(model).num_hidden_layers)
    global_layers, _, _ = _derive_global_layers(_text_config(model))

    for idx, word in enumerate(eval_rows):
        prompts = _build_prompts(word, icl_examples=icl_examples, pair_id=pair_id)
        raw_prompt = prompts[condition_name]
        rendered = apply_chat_template(tokenizer, raw_prompt)
        prompt_inputs = tokenizer(rendered, return_tensors='pt')
        prompt_ids = prompt_inputs.input_ids.to(unemb_device)
        prompt_len = int(prompt_ids.shape[1])
        target_ids = tokenizer.encode(str(word['hindi']), add_special_tokens=False)
        if not target_ids:
            continue
        gold_first_id = int(target_ids[0])
        full_ids = torch.cat(
            [prompt_ids, torch.tensor(target_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)],
            dim=1,
        )
        attention_mask = torch.ones_like(full_ids)
        generation_start_index = int(prompt_len - 1)
        log(f'{model_key} {pair_id} {condition_name}: item={idx}')
        with torch.inference_mode():
            outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states[1:])
        if len(hidden_states) != n_layers:
            raise RuntimeError(f'Unexpected hidden-state count for {model_key}: {len(hidden_states)} vs {n_layers}')
        layer_vectors = []
        for hs in hidden_states:
            vec = hs[0, generation_start_index, :].detach().to(unemb_device)
            layer_vectors.append(vec)
        layer_stack = torch.stack(layer_vectors, dim=0)
        normed = final_norm(layer_stack)
        logits = F.linear(normed.to(dtype=unemb.dtype), unemb).float()
        log_denom = torch.logsumexp(logits, dim=1)
        script_logits = logits.index_select(1, script_ids_tensor)
        script_logsumexp = torch.logsumexp(script_logits, dim=1)
        script_mass = torch.exp(script_logsumexp - log_denom)
        gold_logits = logits[:, gold_first_id]
        gold_rank = (logits > gold_logits.unsqueeze(1)).sum(dim=1).to(torch.int64) + 1
        best_script_vals, best_script_idx_local = script_logits.max(dim=1)
        best_script_token_ids = script_ids_tensor[best_script_idx_local]
        best_script_rank = (logits > best_script_vals.unsqueeze(1)).sum(dim=1).to(torch.int64) + 1
        best_script_prob = torch.exp(best_script_vals - log_denom)
        top50_ids = torch.topk(logits, k=min(50, logits.shape[1]), dim=1).indices
        topk_stats = _topk_counts(top50_ids, script_mask)
        top1_ids = top50_ids[:, 0].detach().cpu().tolist()

        for layer_idx in range(n_layers):
            row = {
                'model': model_key,
                'pair': pair_id,
                'language': _language_name(pair_id),
                'condition': condition_name,
                'item_index': int(idx),
                'word_source_romanized': str(word['ood']),
                'word_target': str(word['hindi']),
                'gold_first_token_id': int(gold_first_id),
                'layer': int(layer_idx),
                'is_global': bool(layer_idx in global_layers),
                'target_position_index': int(generation_start_index),
                'prompt_length_tokens': int(prompt_len),
                'target_length_tokens': int(len(target_ids)),
                'target_script': target_script,
                'target_script_probability_mass': float(script_mass[layer_idx].item()),
                'gold_token_rank': int(gold_rank[layer_idx].item()),
                'best_target_script_token_id': int(best_script_token_ids[layer_idx].item()),
                'best_target_script_token_rank': int(best_script_rank[layer_idx].item()),
                'best_target_script_token_prob': float(best_script_prob[layer_idx].item()),
                'top1_token_id': int(top1_ids[layer_idx]),
                'top1_in_target_script': int(topk_stats['target_script_any_top1'][layer_idx]),
                'target_script_count_top1': int(topk_stats['target_script_count_top1'][layer_idx]),
                'target_script_count_top5': int(topk_stats['target_script_count_top5'][layer_idx]),
                'target_script_count_top10': int(topk_stats['target_script_count_top10'][layer_idx]),
                'target_script_count_top50': int(topk_stats['target_script_count_top50'][layer_idx]),
                'target_script_any_top5': int(topk_stats['target_script_any_top5'][layer_idx]),
                'target_script_any_top10': int(topk_stats['target_script_any_top10'][layer_idx]),
                'target_script_any_top50': int(topk_stats['target_script_any_top50'][layer_idx]),
                'prompt_fingerprint': prompt_fingerprint(raw_prompt=raw_prompt, rendered_prompt=rendered),
                'prompt_template_fingerprint': prompt_template_fingerprint(tokenizer),
                'runtime_identity': runtime_identity(model_key=model_key, hf_id=str(get_model_config(model_key).hf_id), tokenizer=tokenizer, model=model),
                'final_norm_path': final_norm_path,
            }
            row_records.append(row)
            summary_group.setdefault(layer_idx, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for layer_idx in range(n_layers):
        rows = summary_group.get(layer_idx, [])
        summary_rows.append({
            'model': model_key,
            'pair': pair_id,
            'language': _language_name(pair_id),
            'condition': condition_name,
            'layer': int(layer_idx),
            'is_global': bool(layer_idx in global_layers),
            'n_items': int(len(rows)),
            'mean_target_script_probability_mass': _mean(r['target_script_probability_mass'] for r in rows),
            'mean_gold_token_rank': _mean(r['gold_token_rank'] for r in rows),
            'mean_best_target_script_token_rank': _mean(r['best_target_script_token_rank'] for r in rows),
            'mean_best_target_script_token_prob': _mean(r['best_target_script_token_prob'] for r in rows),
            'fraction_top1_in_target_script': _mean(r['top1_in_target_script'] for r in rows),
            'fraction_any_target_script_top5': _mean(r['target_script_any_top5'] for r in rows),
            'fraction_any_target_script_top10': _mean(r['target_script_any_top10'] for r in rows),
            'fraction_any_target_script_top50': _mean(r['target_script_any_top50'] for r in rows),
            'mean_target_script_count_top5': _mean(r['target_script_count_top5'] for r in rows),
            'mean_target_script_count_top10': _mean(r['target_script_count_top10'] for r in rows),
            'mean_target_script_count_top50': _mean(r['target_script_count_top50'] for r in rows),
        })
    meta = {
        'model': model_key,
        'pair': pair_id,
        'language': _language_name(pair_id),
        'condition': condition_name,
        'n_items': int(len(eval_rows)),
        'global_layers': global_layers,
        'target_script': target_script,
        'script_token_count': int(len(script_ids)),
    }
    return row_records, summary_rows, meta


def _plot_script_mass(summary_rows: Sequence[Dict[str, Any]], *, model_key: str, language: str, global_layers: Sequence[int], out_path: Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row['condition']), []).append(row)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {'explicit_zs': '#1f77b4', 'icl64': '#d62728'}
    for condition, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r['layer']))
        ax.plot(
            [int(r['layer']) for r in rows],
            [float(r['mean_target_script_probability_mass']) for r in rows],
            label=condition,
            color=colors.get(condition),
            linewidth=2,
        )
    for g in global_layers:
        ax.axvline(int(g), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'Script-space map: {model_key} {language}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean target-script probability mass')
    ax.legend()
    ax.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run layer-wise script-space map at the first target position.')
    ap.add_argument('--tasks', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-icl', type=int, default=64)
    ap.add_argument('--n-select', type=int, default=300)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--max-words', type=int, default=50)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--external-only', action='store_true')
    ap.add_argument('--require-external-sources', action='store_true')
    ap.add_argument('--min-pool-size', type=int, default=500)
    ap.add_argument('--out-root', type=str, default='artifacts/phase2')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = (PROJECT_ROOT / str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    phase0_root = (PROJECT_ROOT / 'artifacts/phase0').resolve()
    phase0_root.mkdir(parents=True, exist_ok=True)
    tasks = _parse_tasks(args.tasks)

    for task in tasks:
        model_key = str(task['model'])
        pair_id = str(task['pair'])
        language = _language_name(pair_id)
        script_name = _script_name(pair_id)
        log(f'loading model={model_key} pair={pair_id}')
        model, tokenizer = load_model(model_key, device=str(args.device))
        unemb = _get_unembedding_weight(model)
        if unemb is None:
            raise RuntimeError(f'No unembedding weight found for {model_key}')
        final_norm, final_norm_path = _get_final_norm(model)
        arch = _architecture_packet(model_key, model, tokenizer, final_norm_path, unemb)
        arch_path = phase0_root / f'architecture_packet_{model_key}.json'
        _write_json(arch_path, arch)

        words, provenance = _load_words(
            pair_id,
            external_only=bool(args.external_only),
            require_external_sources=bool(args.require_external_sources),
            min_pool_size=int(args.min_pool_size),
        )
        icl, _, ev = split_data_three_way(
            words=words,
            n_icl=int(args.n_icl),
            n_select=int(args.n_select),
            n_eval=int(args.n_eval),
            seed=int(args.seed),
        )
        eval_rows = list(ev[: int(args.max_words)])
        script_ids = _script_token_ids(tokenizer, int(unemb.shape[0]), script_name, out_root)
        if not script_ids:
            raise RuntimeError(f'No token ids found for script={script_name}')

        all_summary_rows: List[Dict[str, Any]] = []
        task_meta = {
            'model': model_key,
            'pair': pair_id,
            'language': language,
            'script_name': script_name,
            'seed': int(args.seed),
            'n_icl': int(args.n_icl),
            'n_eval_total': int(args.n_eval),
            'n_eval_used': int(len(eval_rows)),
            'provenance': provenance,
            'architecture_packet_path': str(arch_path.relative_to(PROJECT_ROOT)),
        }
        for condition_name in ('explicit_zs', 'icl64'):
            log(f'running script-space map model={model_key} pair={pair_id} condition={condition_name}')
            item_rows, summary_rows, cond_meta = _run_condition(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                final_norm_path=final_norm_path,
                unemb=unemb,
                script_ids=script_ids,
                model_key=model_key,
                pair_id=pair_id,
                condition_name=condition_name,
                eval_rows=eval_rows,
                icl_examples=icl,
                out_root=out_root,
            )
            payload = {
                'meta': {**task_meta, **cond_meta},
                'rows': item_rows,
            }
            _write_json(out_root / f'script_space_map_{model_key}_{language}_{condition_name}.json', payload)
            all_summary_rows.extend(summary_rows)
        _write_rows(out_root / f'script_space_summary_{model_key}_{language}', all_summary_rows)
        _plot_script_mass(
            all_summary_rows,
            model_key=model_key,
            language=language,
            global_layers=arch['global_layers'],
            out_path=out_root / f'script_mass_by_layer_{model_key}_{language}.png',
        )
        log(f'finished model={model_key} pair={pair_id}')
        del model
        del tokenizer
        del unemb
        del final_norm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
