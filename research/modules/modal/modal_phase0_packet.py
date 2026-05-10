from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import modal


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "research").exists():
            return candidate
    return Path(os.environ.get("PHASE0_REPO_ROOT", "/repo"))


PROJECT_ROOT = _resolve_project_root()
N_VALUES = [0, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

FROZEN_SNAPSHOTS = {
    "Hindi": PROJECT_ROOT
    / "lib/paper2_fidelity_calibrated/split_snapshots/aksharantar_hin_latin_split_seed42_nicl16_nselect300_neval50.json",
    "Telugu": PROJECT_ROOT
    / "lib/paper2_fidelity_calibrated/split_snapshots/aksharantar_tel_latin_split_seed42_nicl16_nselect300_neval50.json",
}

LANGUAGE_CODE_TO_NAME = {
    "hin": "Hindi",
    "tel": "Telugu",
    "ben": "Bengali",
    "tam": "Tamil",
    "mar": "Marathi",
}


def _parse_language_codes(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw).split(","):
        code = part.strip().lower()
        if not code:
            continue
        if code not in out:
            out.append(code)
    return out


def _parse_n_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        values = list(N_VALUES)
    values = sorted(set(values))
    if 0 not in values:
        values = [0, *values]
    return values


def _parse_bool(raw: str, default: bool = False) -> bool:
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "y", "on"}


def _run_snapshots_for_seed(
    *,
    split_seed: int,
    n_candidate: int = 300,
    n_eval: int = 50,
    language_codes: list[str] | None = None,
    snapshots_dir: str = "research/results/phase0/snapshots",
) -> dict[str, Path]:
    base = PROJECT_ROOT / str(snapshots_dir)
    codes = list(language_codes or ["hin", "tel"])
    out: dict[str, Path] = {}
    for code in codes:
        label = LANGUAGE_CODE_TO_NAME.get(code, code.upper())
        out[label] = (
            base
            / f"aksharantar_{code}_latin_unique_seed{split_seed}_ncand{n_candidate}_neval{n_eval}.json"
        )
    return out


def _render_model_prompts(tokenizer: Any, prompts: list[str]) -> tuple[list[str], bool]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return list(prompts), False

    rendered: list[str] = []
    used_chat_template = False
    for prompt in prompts:
        final_prompt = prompt
        try:
            candidate = apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Follow the user's instruction exactly. Reply only with the final answer and nothing else.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(candidate, str) and candidate.strip():
                final_prompt = candidate
                used_chat_template = True
        except Exception:
            final_prompt = prompt
        rendered.append(final_prompt)
    return rendered, used_chat_template

app = modal.App("gemma-phase0a-packet")
artifacts_volume = modal.Volume.from_name("phase0a-verification-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "datasets==3.6.0",
        "accelerate==1.2.1",
        "sentencepiece==0.2.0",
        "numpy==1.26.4",
        "sae-lens==6.39.0",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "PHASE0_REPO_ROOT": "/repo",
        }
    )
    .add_local_dir(PROJECT_ROOT / "research", remote_path="/repo/research")
    .add_local_dir(PROJECT_ROOT / "lib", remote_path="/repo/lib")
)


@app.function(image=image, gpu="A100-40GB", timeout=14400)
def run_phase0a(
    *,
    hf_token: str = "",
    google_api_key: str = "",
    max_new_tokens: int = 32,
    max_eval_rows: int = 0,
    split_seed: int = 42,
    n_candidate: int = 300,
    n_eval: int = 50,
    n_values_csv: str = "0,4,8,16,32,48,64,96,128,192,256",
    batch_size: int = 16,
    judge_enabled: bool = False,
    judge_probe_per_condition: int = 0,
    judge_model: str = "gemini-2.0-flash-lite",
    snapshots_dir: str = "research/results/phase0/snapshots",
    language_codes_csv: str = "hin,tel",
    prompt_templates_csv: str = "canonical,output_only,task_tagged",
    icl_variants_csv: str = "helpful,random,shuffled_targets,corrupted_targets",
) -> dict[str, Any]:
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_root = Path(os.environ.get("PHASE0_REPO_ROOT", "/repo"))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from research.modules.behavior.icl_variants import materialize_icl_variant, parse_variant_csv
    from research.modules.data.row_schema import get_target_text
    from research.modules.data.token_count_probe import probe_prompt_visibility
    from research.modules.eval.judge_wrapper import judge_transliteration, run_judge_sanity_packet
    from research.modules.eval.metrics import (
        akshara_cer,
        empty_or_refusal,
        exact_match,
        normalize_text,
        script_valid,
        standalone_answer,
        summarize_rows,
    )
    from research.modules.eval.output_extraction import (
        analyze_generation_text,
        extract_transliteration_candidate,
        resolve_generation_stop_ids,
        resolve_pad_token_id,
    )
    from research.modules.eval.verification_packet import (
        run_deterministic_metric_sanity,
        validate_snapshot,
    )
    from research.modules.prompts.prompt_templates import (
        build_prompt,
        parse_prompt_template_csv,
    )

    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _target(row: dict[str, str]) -> str:
        return get_target_text(row)

    def _stable_seed(*parts: Any) -> int:
        acc = 17
        for part in parts:
            for ch in str(part):
                acc = (acc * 131 + ord(ch)) % 2_147_483_647
        return int(acc)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    language_codes = _parse_language_codes(language_codes_csv)
    if not language_codes:
        language_codes = ["hin", "tel"]

    n_values = _parse_n_values(n_values_csv)

    prompt_templates = parse_prompt_template_csv(prompt_templates_csv)
    if "canonical" not in prompt_templates:
        prompt_templates = ["canonical", *prompt_templates]

    icl_variants = parse_variant_csv(icl_variants_csv)
    if "helpful" not in icl_variants:
        icl_variants = ["helpful", *icl_variants]

    judge_enabled_flag = bool(judge_enabled)
    if not judge_enabled_flag:
        judge_probe_per_condition = 0

    snapshot_reports = {
        language: validate_snapshot(path)
        for language, path in FROZEN_SNAPSHOTS.items()
    }
    if not all(report["ok"] for report in snapshot_reports.values()):
        return {
            "status": "stop",
            "stop_reason": "snapshot_validation_failed",
            "created_at_utc": _now(),
            "run_config": {
                "split_seed": int(split_seed),
                "n_candidate": int(n_candidate),
                "n_eval": int(n_eval),
                "max_new_tokens": int(max_new_tokens),
                "n_values": list(n_values),
                "batch_size": int(batch_size),
                "judge_probe_per_condition": int(judge_probe_per_condition),
                "judge_enabled": bool(judge_enabled_flag),
                "judge_model": str(judge_model),
                "prompt_templates": list(prompt_templates),
                "icl_variants": list(icl_variants),
                "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
            },
            "snapshot_reports": snapshot_reports,
        }

    run_snapshots = _run_snapshots_for_seed(
        split_seed=int(split_seed),
        n_candidate=int(n_candidate),
        n_eval=int(n_eval),
        language_codes=language_codes,
        snapshots_dir=str(snapshots_dir),
    )

    run_snapshot_reports = {
        language: validate_snapshot(path)
        for language, path in run_snapshots.items()
    }
    if not all(report["ok"] for report in run_snapshot_reports.values()):
        return {
            "status": "stop",
            "stop_reason": "run_snapshot_validation_failed",
            "created_at_utc": _now(),
            "run_config": {
                "split_seed": int(split_seed),
                "n_candidate": int(n_candidate),
                "n_eval": int(n_eval),
                "max_new_tokens": int(max_new_tokens),
                "n_values": list(n_values),
                "batch_size": int(batch_size),
                "judge_probe_per_condition": int(judge_probe_per_condition),
                "judge_enabled": bool(judge_enabled_flag),
                "judge_model": str(judge_model),
                "prompt_templates": list(prompt_templates),
                "icl_variants": list(icl_variants),
                "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
            },
            "snapshot_reports": snapshot_reports,
            "run_snapshot_reports": run_snapshot_reports,
        }

    deterministic_sanity = run_deterministic_metric_sanity()
    if not deterministic_sanity["ok"]:
        return {
            "status": "stop",
            "stop_reason": "deterministic_eval_sanity_failed",
            "created_at_utc": _now(),
            "run_config": {
                "split_seed": int(split_seed),
                "n_candidate": int(n_candidate),
                "n_eval": int(n_eval),
                "max_new_tokens": int(max_new_tokens),
                "n_values": list(n_values),
                "batch_size": int(batch_size),
                "judge_probe_per_condition": int(judge_probe_per_condition),
                "judge_enabled": bool(judge_enabled_flag),
                "judge_model": str(judge_model),
                "prompt_templates": list(prompt_templates),
                "icl_variants": list(icl_variants),
                "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
            },
            "snapshot_reports": snapshot_reports,
            "run_snapshot_reports": run_snapshot_reports,
            "deterministic_sanity": deterministic_sanity,
        }

    print("[phase0a] loading tokenizer/model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer.padding_side = "left"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    device = str(next(model.parameters()).device)
    text_cfg = getattr(model.config, "text_config", model.config)
    local_window = int(getattr(text_cfg, "sliding_window", 0) or 1024)

    generation_stop_ids = resolve_generation_stop_ids(tokenizer)
    generation_pad_id = resolve_pad_token_id(tokenizer, fallback_stop_ids=generation_stop_ids)

    print(
        f"[phase0a][gpu] cuda_available={torch.cuda.is_available()} model_device={device}",
        flush=True,
    )
    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        print(
            "[phase0a][gpu] "
            f"device_index={current_idx} "
            f"name={torch.cuda.get_device_name(current_idx)} "
            f"capability={torch.cuda.get_device_capability(current_idx)} "
            f"allocated_mb={torch.cuda.memory_allocated(current_idx)/1024/1024:.1f}",
            flush=True,
        )
    print(
        f"[phase0a][gpu] hf_device_map={getattr(model, 'hf_device_map', {})}",
        flush=True,
    )

    prompt_separator = "->"

    all_summary_rows: list[dict[str, Any]] = []
    token_probe_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    language_payloads: dict[str, dict[str, Any]] = {}
    for language, path in run_snapshots.items():
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
        script_name = str(snapshot["output_script_name"])
        candidate_pool = list(snapshot.get("icl_bank", snapshot.get("candidate_pool", [])))
        eval_rows = list(snapshot.get("eval_rows", []))
        icl_presets = snapshot.get("icl_presets", {})

        required_n = max(n_values)
        if len(candidate_pool) < required_n:
            blockers.append(
                f"{language}: icl_bank has {len(candidate_pool)} rows (< {required_n})."
            )
            continue

        if not isinstance(icl_presets, dict):
            blockers.append(f"{language}: icl_presets missing or invalid.")
            continue

        language_payloads[language] = {
            "script": script_name,
            "eval_rows": eval_rows,
            "candidate_pool": candidate_pool,
            "icl_presets": icl_presets,
            "icl_unique_count": len(candidate_pool),
            "icl_repeats_added": 0,
        }

    if blockers:
        return {
            "status": "stop",
            "stop_reason": "insufficient_icl_pool",
            "created_at_utc": _now(),
            "run_config": {
                "split_seed": int(split_seed),
                "n_candidate": int(n_candidate),
                "n_eval": int(n_eval),
                "max_new_tokens": int(max_new_tokens),
                "n_values": list(n_values),
                "batch_size": int(batch_size),
                "judge_probe_per_condition": int(judge_probe_per_condition),
                "judge_enabled": bool(judge_enabled_flag),
                "judge_model": str(judge_model),
                "prompt_templates": list(prompt_templates),
                "icl_variants": list(icl_variants),
                "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
            },
            "snapshot_reports": snapshot_reports,
            "run_snapshot_reports": run_snapshot_reports,
            "deterministic_sanity": deterministic_sanity,
            "blockers": blockers,
        }

    for language, payload in language_payloads.items():
        script_name = str(payload["script"])
        eval_rows = list(payload["eval_rows"])
        if max_eval_rows > 0:
            eval_rows = eval_rows[: int(max_eval_rows)]
        candidate_pool = payload["candidate_pool"]
        icl_presets = payload["icl_presets"]

        print(
            f"[phase0a] language={language} eval_rows={len(eval_rows)} unique_icl={payload['icl_unique_count']} repeats={payload['icl_repeats_added']}",
            flush=True,
        )

        for prompt_template in prompt_templates:
            for icl_variant in icl_variants:
                for n in n_values:
                    if n == 0 and icl_variant != "helpful":
                        continue

                    if n == 0:
                        helpful_examples: list[dict[str, str]] = []
                    else:
                        preset = icl_presets.get(str(n), [])
                        if not isinstance(preset, list) or len(preset) != n:
                            helpful_examples = candidate_pool[:n]
                        else:
                            helpful_examples = list(preset)

                    variant_seed = _stable_seed(
                        split_seed,
                        language,
                        prompt_template,
                        icl_variant,
                        n,
                    )
                    prompt_examples = materialize_icl_variant(
                        variant=icl_variant,
                        n=n,
                        helpful_examples=helpful_examples,
                        candidate_pool=candidate_pool,
                        rng_seed=variant_seed,
                    )

                    per_item_rows: list[dict[str, float]] = []
                    sample_outputs: list[dict[str, str]] = []
                    raw_audit_examples: list[dict[str, str | bool]] = []

                    effective_batch = max(1, int(batch_size))
                    for batch_start in range(0, len(eval_rows), effective_batch):
                        batch_rows = eval_rows[batch_start : batch_start + effective_batch]
                        prompts = [
                            build_prompt(
                                prompt_template=prompt_template,
                                query=str(row["ood"]),
                                examples=prompt_examples,
                                script_name=script_name,
                                separator=prompt_separator,
                            )
                            for row in batch_rows
                        ]
                        golds = [normalize_text(_target(row)) for row in batch_rows]

                        model_prompts, used_chat_template = _render_model_prompts(tokenizer, prompts)
                        encoded = tokenizer(
                            model_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=not used_chat_template,
                        ).to(device)
                        input_width = int(encoded.input_ids.shape[1])

                        with torch.inference_mode():
                            generated = model.generate(
                                **encoded,
                                max_new_tokens=int(max_new_tokens),
                                do_sample=False,
                                use_cache=True,
                                eos_token_id=generation_stop_ids,
                                pad_token_id=int(generation_pad_id),
                            )

                        new_tokens_batch = generated[:, input_width:]
                        decoded_batch = tokenizer.batch_decode(new_tokens_batch, skip_special_tokens=True)

                        for j, (row, gold, decoded) in enumerate(zip(batch_rows, golds, decoded_batch)):
                            pred = extract_transliteration_candidate(
                                decoded,
                                script_name=script_name,
                            )
                            raw_audit = analyze_generation_text(decoded, pred)
                            row_token_ids = new_tokens_batch[j]
                            actual_len = int((row_token_ids != int(generation_pad_id)).sum().item())
                            row_metrics = {
                                "exact_match": exact_match(pred, gold),
                                "akshara_CER": akshara_cer(pred, gold),
                                "script_valid": script_valid(pred, script_name),
                                "standalone_answer": standalone_answer(pred),
                                "empty_or_refusal": empty_or_refusal(pred),
                                "hit_max_new_tokens": float(
                                    int(actual_len >= int(max_new_tokens))
                                ),
                                "raw_strict_word_only": float(int(raw_audit["strict_word_only"])),
                                "has_leading_text": float(int(raw_audit["has_leading_text"])),
                                "has_trailing_text": float(int(raw_audit["has_trailing_text"])),
                            }
                            per_item_rows.append(row_metrics)

                            if (
                                len(raw_audit_examples) < 3
                                and (
                                    raw_audit["has_leading_text"]
                                    or raw_audit["has_trailing_text"]
                                    or not raw_audit["strict_word_only"]
                                )
                            ):
                                raw_audit_examples.append(
                                    {
                                        "source": str(row["ood"]),
                                        "gold": gold,
                                        "raw_output": normalize_text(decoded),
                                        "prediction": pred,
                                        "strict_word_only": bool(raw_audit["strict_word_only"]),
                                        "has_leading_text": bool(raw_audit["has_leading_text"]),
                                        "has_trailing_text": bool(raw_audit["has_trailing_text"]),
                                    }
                                )

                            global_idx = batch_start + j
                            if global_idx < int(judge_probe_per_condition):
                                sample_outputs.append(
                                    {
                                        "source": str(row["ood"]),
                                        "gold": gold,
                                        "prediction": pred,
                                        "raw_output": normalize_text(decoded),
                                    }
                                )

                    summary = summarize_rows(per_item_rows)
                    summary_row = {
                        "language": language,
                        "prompt_template": prompt_template,
                        "icl_variant": icl_variant,
                        "N": n,
                        "prompt_separator": prompt_separator,
                        **summary,
                        "n_eval": len(per_item_rows),
                        "samples": sample_outputs,
                        "raw_audit_examples": raw_audit_examples,
                    }
                    all_summary_rows.append(summary_row)
                    print(
                        "[phase0a] "
                        f"language={language} template={prompt_template} variant={icl_variant} "
                        f"N={n} exact={summary['exact_match_rate']:.3f} cer={summary['akshara_CER_mean']:.3f} "
                        f"raw_strict={summary.get('raw_strict_word_only_rate', 0.0):.3f} "
                        f"lead={summary.get('leading_text_rate', 0.0):.3f} "
                        f"trail={summary.get('trailing_text_rate', 0.0):.3f}",
                        flush=True,
                    )
                    if raw_audit_examples:
                        sample = raw_audit_examples[0]
                        print(
                            "[phase0a][raw-audit] "
                            f"language={language} template={prompt_template} variant={icl_variant} N={n} "
                            f"source={sample['source']!r} raw={sample['raw_output']!r} extracted={sample['prediction']!r} "
                            f"leading={sample['has_leading_text']} trailing={sample['has_trailing_text']}",
                            flush=True,
                        )

                    probe = probe_prompt_visibility(
                        tokenizer=tokenizer,
                        query=str(eval_rows[0]["ood"]),
                        examples=prompt_examples,
                        script_name=script_name,
                        local_window=local_window,
                        prompt_template=prompt_template,
                        separator=prompt_separator,
                    )
                    token_probe_rows.append(
                        {
                            "language": language,
                            "prompt_template": prompt_template,
                            "icl_variant": icl_variant,
                            "prompt_separator": prompt_separator,
                            "N": n,
                            **probe,
                        }
                    )

    summary_index = {
        (row["language"], row["prompt_template"], row["icl_variant"], row["N"]): row
        for row in all_summary_rows
    }

    judge_curve_rows: list[dict[str, Any]] = []
    if judge_enabled_flag and int(judge_probe_per_condition) > 0:
        for summary_row in all_summary_rows:
            labels: dict[str, int] = {
                "exact": 0,
                "acceptable_variant": 0,
                "script_correct_but_wrong": 0,
                "invalid_or_non_answer": 0,
            }
            decision_source_counts: dict[str, int] = {}
            samples = list(summary_row.get("samples", []))
            for sample in samples:
                verdict = judge_transliteration(
                    source=str(sample.get("source", "")),
                    reference=str(sample.get("gold", "")),
                    output=str(sample.get("prediction", "")),
                    language=str(summary_row["language"]),
                    model=str(judge_model),
                    api_key=google_api_key or None,
                )
                label = str(verdict.get("label", "script_correct_but_wrong"))
                labels[label] = labels.get(label, 0) + 1
                source_key = str(verdict.get("decision_source", "unknown"))
                decision_source_counts[source_key] = decision_source_counts.get(source_key, 0) + 1

            n_judged = len(samples)
            exact_rate = float(labels.get("exact", 0) / n_judged) if n_judged else 0.0
            acceptable_rate = (
                float((labels.get("exact", 0) + labels.get("acceptable_variant", 0)) / n_judged)
                if n_judged
                else 0.0
            )
            judge_curve_rows.append(
                {
                    "language": summary_row["language"],
                    "prompt_template": summary_row["prompt_template"],
                    "icl_variant": summary_row["icl_variant"],
                    "N": int(summary_row["N"]),
                    "n_judged": n_judged,
                    "judge_enabled": True,
                    "judge_model": str(judge_model),
                    "judge_exact_rate": exact_rate,
                    "judge_acceptable_rate": acceptable_rate,
                    "judge_script_wrong_rate": float(labels.get("script_correct_but_wrong", 0) / n_judged)
                    if n_judged
                    else 0.0,
                    "judge_invalid_rate": float(labels.get("invalid_or_non_answer", 0) / n_judged)
                    if n_judged
                    else 0.0,
                    "det_exact_rate": float(summary_row["exact_match_rate"]),
                    "det_cer_mean": float(summary_row["akshara_CER_mean"]),
                    "decision_source_counts": decision_source_counts,
                }
            )
    else:
        for summary_row in all_summary_rows:
            judge_curve_rows.append(
                {
                    "language": summary_row["language"],
                    "prompt_template": summary_row["prompt_template"],
                    "icl_variant": summary_row["icl_variant"],
                    "N": int(summary_row["N"]),
                    "n_judged": 0,
                    "judge_enabled": False,
                    "judge_model": str(judge_model),
                    "judge_exact_rate": None,
                    "judge_acceptable_rate": None,
                    "judge_script_wrong_rate": None,
                    "judge_invalid_rate": None,
                    "det_exact_rate": float(summary_row["exact_match_rate"]),
                    "det_cer_mean": float(summary_row["akshara_CER_mean"]),
                    "decision_source_counts": {"disabled": 1},
                }
            )

    data_caveats = {
        language: {
            "icl_unique_count": int(payload["icl_unique_count"]),
            "icl_repeats_added": int(payload["icl_repeats_added"]),
            "max_N": int(max(n_values)),
            "prompt_templates": list(prompt_templates),
            "icl_variants": list(icl_variants),
            "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
        }
        for language, payload in language_payloads.items()
    }

    baseline_template = "canonical"
    baseline_variant = "helpful"

    rescue_rows: list[dict[str, Any]] = []
    rescue_pass_languages: list[str] = []
    for language in sorted(language_payloads.keys()):
        zero = summary_index[(language, baseline_template, baseline_variant, 0)]
        four = summary_index[(language, baseline_template, baseline_variant, 4)]
        rescue_ok = (
            four["exact_match_rate"] > zero["exact_match_rate"]
            and four["akshara_CER_mean"] < zero["akshara_CER_mean"]
            and four["script_validity_rate"] >= zero["script_validity_rate"]
        )
        if rescue_ok:
            rescue_pass_languages.append(language)
        rescue_rows.append(
            {
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "exact_delta": four["exact_match_rate"] - zero["exact_match_rate"],
                "cer_delta": four["akshara_CER_mean"] - zero["akshara_CER_mean"],
                "script_delta": four["script_validity_rate"] - zero["script_validity_rate"],
                "pass": rescue_ok,
            }
        )

    degradation_rows: list[dict[str, Any]] = []
    degradation_pass_languages: list[str] = []
    max_n = max(n_values)
    for language in sorted(language_payloads.keys()):
        rows = [
            summary_index[(language, baseline_template, baseline_variant, n)]
            for n in n_values
            if n > 0
        ]
        best_exact_row = max(rows, key=lambda r: r["exact_match_rate"])
        best_cer_row = min(rows, key=lambda r: r["akshara_CER_mean"])
        at_max = summary_index[(language, baseline_template, baseline_variant, max_n)]
        exact_drop = best_exact_row["exact_match_rate"] - at_max["exact_match_rate"]
        cer_increase = at_max["akshara_CER_mean"] - best_cer_row["akshara_CER_mean"]
        degradation_ok = best_exact_row["N"] < max_n and (exact_drop >= 0.10 or cer_increase >= 0.10)
        if degradation_ok:
            degradation_pass_languages.append(language)
        degradation_rows.append(
            {
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "peak_N_exact": int(best_exact_row["N"]),
                "peak_exact": float(best_exact_row["exact_match_rate"]),
                "peak_N_cer": int(best_cer_row["N"]),
                "best_cer": float(best_cer_row["akshara_CER_mean"]),
                f"exact_drop_at_{max_n}": float(exact_drop),
                f"cer_increase_at_{max_n}": float(cer_increase),
                "max_N": int(max_n),
                "pass": degradation_ok,
            }
        )

    visibility_rows = [
        row
        for row in token_probe_rows
        if row["prompt_template"] == baseline_template
        and row.get("icl_variant") == baseline_variant
        and row["N"] in {64, max_n}
    ]
    visibility_ok = any(bool(row["exceeds_window"]) for row in visibility_rows)

    strict_output_rows = [
        summary_index[(language, baseline_template, baseline_variant, n)]
        for language in sorted(language_payloads.keys())
        for n in n_values
        if (language, baseline_template, baseline_variant, n) in summary_index
    ]
    strict_output_ok = all(
        float(row.get("leading_text_rate", 1.0)) <= 0.05
        and float(row.get("trailing_text_rate", 1.0)) <= 0.05
        and float(row.get("standalone_answer_rate", 0.0)) >= 0.95
        for row in strict_output_rows
    ) if strict_output_rows else False

    if judge_enabled_flag:
        sanity_cases: list[dict[str, str]] = []
        base_rows: list[tuple[str, dict[str, str]]] = []
        for language, payload in language_payloads.items():
            for row in payload["eval_rows"][:3]:
                base_rows.append((language, row))

        case_counter = 0
        for language, row in base_rows:
            source = str(row["ood"])
            gold = get_target_text(row)
            sanity_cases.append(
                {
                    "case_id": f"case_{case_counter:02d}",
                    "language": language,
                    "source": source,
                    "reference": gold,
                    "output": gold,
                    "expected_label": "exact",
                }
            )
            case_counter += 1
            sanity_cases.append(
                {
                    "case_id": f"case_{case_counter:02d}",
                    "language": language,
                    "source": source,
                    "reference": gold,
                    "output": source,
                    "expected_label": "invalid_or_non_answer",
                }
            )
            case_counter += 1
            sanity_cases.append(
                {
                    "case_id": f"case_{case_counter:02d}",
                    "language": language,
                    "source": source,
                    "reference": gold,
                    "output": "",
                    "expected_label": "invalid_or_non_answer",
                }
            )
            case_counter += 1

        for language, payload in language_payloads.items():
            row_a = payload["eval_rows"][0]
            row_b = payload["eval_rows"][1]
            sanity_cases.append(
                {
                    "case_id": f"case_{case_counter:02d}",
                    "language": language,
                    "source": str(row_a["ood"]),
                    "reference": get_target_text(row_a),
                    "output": get_target_text(row_b),
                    "expected_label": "script_correct_but_wrong",
                }
            )
            case_counter += 1

        sanity_cases = sanity_cases[:20]
        judge_sanity = run_judge_sanity_packet(
            sanity_cases,
            model=str(judge_model),
            api_key=google_api_key or None,
        )
    else:
        judge_sanity = {
            "n_cases": 0,
            "label_accuracy": 0.0,
            "decision_source_counts": {"disabled": 1},
            "rows": [],
            "skipped": True,
            "reason": "Judge disabled in run config.",
        }

    transcoder_smoke: dict[str, Any]
    transcoder_release = "gemma-scope-2-1b-it-transcoders"
    transcoder_sae_id = "transcoder/layer_17_width_262k_l0_medium"
    transcoder_repo = "google/gemma-scope-2-1b-it"
    transcoder_repo_path = transcoder_sae_id
    try:
        from sae_lens import SAE

        loaded = SAE.from_pretrained(
            release=transcoder_release,
            sae_id=transcoder_sae_id,
            device="cpu",
        )
        sae_obj = loaded[0] if isinstance(loaded, tuple) else loaded
        transcoder_smoke = {
            "ok": True,
            "load_mode": "sae_from_pretrained",
            "sae_class": type(sae_obj).__name__,
            "release": transcoder_release,
            "sae_id": transcoder_sae_id,
        }
    except Exception as exc:  # pragma: no cover - network/model dependent
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                repo_id=transcoder_repo,
                allow_patterns=[f"{transcoder_repo_path}/*"],
                token=hf_token or os.environ.get("HF_TOKEN") or None,
            )
            config_file = Path(local_dir) / transcoder_repo_path / "config.json"
            params_file = Path(local_dir) / transcoder_repo_path / "params.safetensors"
            examples_file = Path(local_dir) / transcoder_repo_path / "examples.safetensors"
            transcoder_smoke = {
                "ok": config_file.exists() and params_file.exists(),
                "load_mode": "hf_snapshot_download",
                "release": transcoder_release,
                "repo_id": transcoder_repo,
                "sae_id": transcoder_sae_id,
                "config_exists": config_file.exists(),
                "params_exists": params_file.exists(),
                "examples_exists": examples_file.exists(),
                "source_error": str(exc),
            }
        except Exception as fallback_exc:
            transcoder_smoke = {
                "ok": False,
                "release": transcoder_release,
                "repo_id": transcoder_repo,
                "sae_id": transcoder_sae_id,
                "error": str(exc),
                "fallback_error": str(fallback_exc),
            }

    rescue_ok = len(rescue_pass_languages) >= 1
    degradation_ok = len(degradation_pass_languages) >= 1
    judge_ok = (not judge_enabled_flag) or (judge_sanity["label_accuracy"] >= 0.6)

    problem_statement_ok = rescue_ok and degradation_ok and deterministic_sanity["ok"]
    mechanistic_readiness_ok = (
        problem_statement_ok
        and visibility_ok
        and strict_output_ok
        and bool(transcoder_smoke.get("ok", False))
    )
    go = mechanistic_readiness_ok

    return {
        "status": "complete",
        "created_at_utc": _now(),
        "run_config": {
            "split_seed": int(split_seed),
            "n_candidate": int(n_candidate),
            "n_eval": int(n_eval),
            "max_new_tokens": int(max_new_tokens),
            "n_values": list(n_values),
            "batch_size": int(batch_size),
            "judge_probe_per_condition": int(judge_probe_per_condition),
            "judge_enabled": bool(judge_enabled_flag),
            "judge_model": str(judge_model),
            "prompt_templates": list(prompt_templates),
            "icl_variants": list(icl_variants),
            "language_codes": list(language_codes),
                "snapshots_dir": str(snapshots_dir),
        },
        "snapshot_reports": snapshot_reports,
        "run_snapshot_reports": run_snapshot_reports,
        "deterministic_sanity": deterministic_sanity,
        "behavior_summary": all_summary_rows,
        "token_probe": token_probe_rows,
        "data_caveats": data_caveats,
        "rescue_check": {
            "pass": rescue_ok,
            "passing_languages": rescue_pass_languages,
            "rows": rescue_rows,
        },
        "degradation_check": {
            "pass": degradation_ok,
            "passing_languages": degradation_pass_languages,
            "rows": degradation_rows,
        },
        "problem_statement_check": {
            "pass": problem_statement_ok,
            "criteria": [
                "low-N rescue present in at least one language",
                "high-N degradation present in at least one language",
                "deterministic evaluation sanity passes",
            ],
        },
        "visibility_check": {
            "pass": visibility_ok,
            "rows": visibility_rows,
            "local_window": local_window,
        },
        "strict_output_check": {
            "pass": strict_output_ok,
            "rows": strict_output_rows,
            "criteria": {
                "max_leading_text_rate": 0.05,
                "max_trailing_text_rate": 0.05,
                "min_standalone_answer_rate": 0.95,
            },
        },
        "judge_curve": judge_curve_rows,
        "judge_sanity": {
            "pass": judge_ok,
            **judge_sanity,
        },
        "transcoder_smoke": transcoder_smoke,
        "mechanistic_readiness_check": {
            "pass": mechanistic_readiness_ok,
            "criteria": [
                "problem statement passes",
                "prompt visibility crosses local window",
                "raw outputs are sufficiently strict for clean evaluation",
                "transcoder smoke load succeeds",
            ],
        },
        "go_no_go": "GO" if go else "NO_GO",
    }


@app.function(image=image, timeout=21600, volumes={"/artifacts": artifacts_volume})
def run_phase0a_seed_to_volume(
    *,
    config_path: str = "research/config/phase0a_run_config_verification.json",
    seed: int = 42,
    run_name: str = "phase0a_final_v1",
    hf_token: str = "",
    google_api_key: str = "",
) -> str:
    import sys

    repo_root = Path(os.environ.get("PHASE0_REPO_ROOT", "/repo"))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from research.modules.infra.phase0a_run_config import load_phase0a_run_config

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = load_phase0a_run_config(cfg_path)

    out_root = Path("/artifacts") / str(run_name) / f"seed{int(seed)}"
    out_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_root / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "started_at_utc": started_at,
                "seed": int(seed),
                "config_path": str(config_path),
                "run_name": str(run_name),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "run_log.txt").write_text(
        f"started_at_utc={started_at}\nseed={int(seed)}\nconfig_path={config_path}\n",
        encoding="utf-8",
    )
    artifacts_volume.commit()

    result = run_phase0a.remote(**cfg.to_modal_kwargs(seed=int(seed), google_api_key=google_api_key, hf_token=hf_token))

    packet_path = out_root / f"phase0a_packet_results_seed{int(seed)}.json"
    packet_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_root / f"table_phase0_behavior_seed{int(seed)}.csv", result.get("behavior_summary", []))
    _write_csv(out_root / f"table_phase0_token_probe_seed{int(seed)}.csv", result.get("token_probe", []))
    _write_csv(out_root / f"table_phase0_judge_curve_seed{int(seed)}.csv", result.get("judge_curve", []))
    _write_csv(
        out_root / f"table_phase0_judge_sanity_seed{int(seed)}.csv",
        result.get("judge_sanity", {}).get("rows", []),
    )
    summary = {
        "status": result.get("status"),
        "go_no_go": result.get("go_no_go"),
        "problem_statement_check": result.get("problem_statement_check", {}).get("pass"),
        "mechanistic_readiness_check": result.get("mechanistic_readiness_check", {}).get("pass"),
        "packet_path": str(packet_path),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_root / "run_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_root / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    artifacts_volume.commit()
    return str(out_root)


def _load_key_from_api_txt(name: str) -> str:
    for candidate in (PROJECT_ROOT / "api.txt", PROJECT_ROOT / "research" / "api.txt"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.local_entrypoint()
def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "") or _load_key_from_api_txt("HF_TOKEN")
    google_key = os.environ.get("GOOGLE_API_KEY", "") or _load_key_from_api_txt("GOOGLE_API_KEY")
    max_eval_rows = int(os.environ.get("PHASE0_MAX_EVAL_ROWS", "0") or "0")
    split_seed = int(os.environ.get("PHASE0_SPLIT_SEED", "42") or "42")
    n_candidate = int(os.environ.get("PHASE0_N_CANDIDATE", "300") or "300")
    n_eval = int(os.environ.get("PHASE0_N_EVAL", "50") or "50")
    max_new_tokens = int(os.environ.get("PHASE0_MAX_NEW_TOKENS", "32") or "32")
    n_values_csv = os.environ.get("PHASE0_N_VALUES", "0,4,8,16,32,48,64,96,128,192,256")
    batch_size = int(os.environ.get("PHASE0_BATCH_SIZE", "16") or "16")
    judge_enabled = _parse_bool(os.environ.get("PHASE0_JUDGE_ENABLED", "false"), default=False)
    judge_probe_per_condition = int(
        os.environ.get("PHASE0_JUDGE_PROBE_PER_CONDITION", "0") or "0"
    )
    judge_model = os.environ.get("PHASE0_JUDGE_MODEL", "gemini-2.0-flash-lite")
    snapshots_dir = os.environ.get("PHASE0_SNAPSHOTS_DIR", "research/results/phase0/snapshots")
    language_codes_csv = os.environ.get("PHASE0_LANGUAGE_CODES", "hin,tel")
    prompt_templates_csv = os.environ.get(
        "PHASE0_PROMPT_TEMPLATES",
        "canonical,output_only,task_tagged",
    )
    icl_variants_csv = os.environ.get(
        "PHASE0_ICL_VARIANTS",
        "helpful,random,shuffled_targets,corrupted_targets",
    )

    result = run_phase0a.remote(
        hf_token=hf_token,
        google_api_key=google_key,
        max_new_tokens=max_new_tokens,
        max_eval_rows=max_eval_rows,
        split_seed=split_seed,
        n_candidate=n_candidate,
        n_eval=n_eval,
        n_values_csv=n_values_csv,
        batch_size=batch_size,
        judge_enabled=judge_enabled,
        judge_probe_per_condition=judge_probe_per_condition,
        judge_model=judge_model,
        snapshots_dir=snapshots_dir,
        language_codes_csv=language_codes_csv,
        prompt_templates_csv=prompt_templates_csv,
        icl_variants_csv=icl_variants_csv,
    )

    out_dir = PROJECT_ROOT / os.environ.get("PHASE0_OUTPUT_DIR", "research/results/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"seed{split_seed}"

    out_json = out_dir / f"phase0a_packet_results_{seed_suffix}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "phase0a_packet_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if result.get("status") == "complete":
        _write_csv(
            out_dir / f"table_phase0_rescue_summary_{seed_suffix}.csv",
            result.get("behavior_summary", []),
        )
        _write_csv(
            out_dir / f"table_phase0_token_counts_{seed_suffix}.csv",
            result.get("token_probe", []),
        )
        _write_csv(
            out_dir / f"table_phase0_eval_stack_validation_{seed_suffix}.csv",
            result.get("judge_sanity", {}).get("rows", []),
        )
        _write_csv(
            out_dir / f"table_phase0_judge_curve_{seed_suffix}.csv",
            result.get("judge_curve", []),
        )
        # Backward-compatible latest aliases
        _write_csv(out_dir / "table_phase0_rescue_summary.csv", result.get("behavior_summary", []))
        _write_csv(out_dir / "table_phase0_token_counts.csv", result.get("token_probe", []))
        _write_csv(
            out_dir / "table_phase0_eval_stack_validation.csv",
            result.get("judge_sanity", {}).get("rows", []),
        )
        _write_csv(
            out_dir / "table_phase0_judge_curve.csv",
            result.get("judge_curve", []),
        )

    print(json.dumps({
        "status": result.get("status"),
        "go_no_go": result.get("go_no_go"),
        "stop_reason": result.get("stop_reason"),
        "run_config": result.get("run_config", {}),
        "results_path": str(out_json),
    }, indent=2))
