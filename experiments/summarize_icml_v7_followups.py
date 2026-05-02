#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_ROOT / "outputs/icml_v7_followup_summaries_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/icml_v7_followup_summaries_2026-04-09.md"


def wilson_interval(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    if n <= 0:
        return {"low": float("nan"), "high": float("nan")}
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    adj = z * math.sqrt((p * (1 - p) + (z * z) / (4.0 * n)) / n) / denom
    return {"low": centre - adj, "high": centre + adj}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    telugu_patch = load_json(
        PROJECT_ROOT
        / "research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"
    )
    telugu_base = next(r for r in telugu_patch["summary_rows"] if r["intervention"] == "baseline_no_patch")
    telugu_chosen = next(r for r in telugu_patch["summary_rows"] if r["intervention"] == "chosen_mean_shift")

    prompt_comp = load_json(
        PROJECT_ROOT
        / "research/results/autoresearch/prompt_composition_ablation_v1/1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json"
    )

    safety = load_json(
        PROJECT_ROOT / "research/results/autoresearch/hindi_patch_safety_audit_v1/1b/seed42/hindi_1b_patch_safety_audit.json"
    )

    crossfamily_rows: List[Dict[str, Any]] = []
    for model in ["qwen2.5-1.5b", "qwen2.5-3b", "llama3.2-1b", "llama3.2-3b"]:
        for lang in ["hin", "tel"]:
            p = (
                PROJECT_ROOT
                / f"research/results/autoresearch/cross_model_behavioral_v1/{model}/aksharantar_{lang}_latin/seed42/nicl64/cross_model_behavioral.json"
            )
            payload = load_json(p)
            helpful = payload["summary"]["icl_helpful"]
            crossfamily_rows.append(
                {
                    "model": model,
                    "language": lang,
                    "n": helpful["n_items"],
                    "helpful_first_entry_correct": helpful["mean_first_entry_correct"],
                    "helpful_akshara_cer": helpful["mean_akshara_cer"],
                    "helpful_exact_bank_copy_rate": helpful["exact_bank_copy_rate"],
                    "helpful_fuzzy_bank_copy_rate": helpful["fuzzy_bank_copy_rate"],
                }
            )

    safety_rows: List[Dict[str, Any]] = []
    for domain in ["english", "hindi"]:
        baseline = safety["summaries"][domain]["baseline_no_patch"]
        chosen = safety["summaries"][domain]["chosen_mean_shift"]
        n = int(baseline["n_items"])
        base_k = int(round(float(baseline["exact_match_rate"]) * n))
        chosen_k = int(round(float(chosen["exact_match_rate"]) * n))
        safety_rows.append(
            {
                "domain": domain,
                "n": n,
                "baseline_exact_match_rate": baseline["exact_match_rate"],
                "baseline_exact_match_wilson": wilson_interval(base_k, n),
                "chosen_exact_match_rate": chosen["exact_match_rate"],
                "chosen_exact_match_wilson": wilson_interval(chosen_k, n),
                "baseline_script_valid_rate": baseline["script_valid_rate"],
                "chosen_script_valid_rate": chosen["script_valid_rate"],
                "baseline_mean_best_joint_logprob": baseline["mean_best_joint_logprob"],
                "chosen_mean_best_joint_logprob": chosen["mean_best_joint_logprob"],
                "baseline_mean_best_first_prob": baseline["mean_best_first_prob"],
                "chosen_mean_best_first_prob": chosen["mean_best_first_prob"],
            }
        )

    payload = {
        "telugu_patch": {
            "site": telugu_patch["site"],
            "selected_alpha": telugu_patch["selected_alpha"],
            "baseline": {
                "n_items": telugu_base["n_items"],
                "full_exact_match": telugu_base["full_exact_match"],
                "full_akshara_cer": telugu_base["full_akshara_cer"],
                "continuation_exact_match": telugu_base["continuation_exact_match"],
                "continuation_akshara_cer": telugu_base["continuation_akshara_cer"],
                "bank_copy_rate": telugu_base["bank_copy_rate"],
                "first_divergence_gap": telugu_base["first_divergence_gap"],
            },
            "chosen_mean_shift": {
                "n_items": telugu_chosen["n_items"],
                "full_exact_match": telugu_chosen["full_exact_match"],
                "full_akshara_cer": telugu_chosen["full_akshara_cer"],
                "continuation_exact_match": telugu_chosen["continuation_exact_match"],
                "continuation_akshara_cer": telugu_chosen["continuation_akshara_cer"],
                "bank_copy_rate": telugu_chosen["bank_copy_rate"],
                "first_divergence_gap": telugu_chosen["first_divergence_gap"],
                "delta_full_cer_improvement": telugu_chosen["delta_full_cer_improvement"],
                "delta_continuation_cer_improvement": telugu_chosen["delta_continuation_cer_improvement"],
                "delta_bank_copy_rate": telugu_chosen["delta_bank_copy_rate"],
                "delta_first_divergence_gap": telugu_chosen["delta_first_divergence_gap"],
            },
        },
        "prompt_composition_ablation": {
            key: {
                "n_items": val["n_items"],
                "mean_exact_match": val["mean_exact_match"],
                "mean_akshara_cer": val["mean_akshara_cer"],
                "mean_first_entry_correct": val["mean_first_entry_correct"],
                "exact_bank_copy_rate": val["exact_bank_copy_rate"],
                "fuzzy_bank_copy_rate": val["fuzzy_bank_copy_rate"],
            }
            for key, val in prompt_comp["summary_by_condition"].items()
            if key
            in {
                "icl_helpful",
                "icl_helpful_similarity_desc",
                "icl_helpful_similarity_asc",
                "icl_corrupt",
            }
        },
        "safety_audit": safety_rows,
        "crossfamily": crossfamily_rows,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ICML v7 follow-up summaries (2026-04-09)",
        "",
        "## Telugu static patch (1B, 191-item held-out panel)",
        f"- Site: `{telugu_patch['site']['name']}`; selected alpha: **{telugu_patch['selected_alpha']}**",
        f"- Full exact match: **{telugu_base['full_exact_match']['mean']:.3f} -> {telugu_chosen['full_exact_match']['mean']:.3f}**",
        f"- Full CER improvement delta: **{telugu_chosen['delta_full_cer_improvement']['mean']:+.4f}** with 95% CI **[{telugu_chosen['delta_full_cer_improvement']['ci_low']:+.4f}, {telugu_chosen['delta_full_cer_improvement']['ci_high']:+.4f}]**",
        f"- Continuation CER improvement delta: **{telugu_chosen['delta_continuation_cer_improvement']['mean']:+.4f}** with 95% CI **[{telugu_chosen['delta_continuation_cer_improvement']['ci_low']:+.4f}, {telugu_chosen['delta_continuation_cer_improvement']['ci_high']:+.4f}]**",
        f"- Bank-copy delta: **{telugu_chosen['delta_bank_copy_rate']['mean']:+.3f}**",
        f"- First-divergence gap delta: **{telugu_chosen['delta_first_divergence_gap']['mean']:+.3f}** with 95% CI **[{telugu_chosen['delta_first_divergence_gap']['ci_low']:+.3f}, {telugu_chosen['delta_first_divergence_gap']['ci_high']:+.3f}]**",
        "",
        "## Telugu prompt-composition ablation (1B, 200-item panel)",
    ]
    for key in ["icl_helpful", "icl_helpful_similarity_desc", "icl_helpful_similarity_asc", "icl_corrupt"]:
        row = prompt_comp["summary_by_condition"][key]
        lines.append(
            f"- `{key}`: E_ak={row['mean_first_entry_correct']:.3f}, CER={row['mean_akshara_cer']:.3f}, exact bank-copy={row['exact_bank_copy_rate']:.3f}, fuzzy bank-copy={row['fuzzy_bank_copy_rate']:.3f}"
        )
    lines.extend([
        "",
        "## Hindi patch safety audit (small off-task cloze checks)",
    ])
    for row in safety_rows:
        lines.append(
            f"- `{row['domain']}` (n={row['n']}): EM {row['baseline_exact_match_rate']:.3f} [{row['baseline_exact_match_wilson']['low']:.3f}, {row['baseline_exact_match_wilson']['high']:.3f}] -> {row['chosen_exact_match_rate']:.3f} [{row['chosen_exact_match_wilson']['low']:.3f}, {row['chosen_exact_match_wilson']['high']:.3f}]; first-token prob {row['baseline_mean_best_first_prob']:.3f} -> {row['chosen_mean_best_first_prob']:.3f}"
        )
    lines.extend([
        "",
        "## Cross-family helpful-condition continuation proxies",
    ])
    for row in crossfamily_rows:
        lines.append(
            f"- `{row['model']}` `{row['language']}`: E_ak_help={row['helpful_first_entry_correct']:.3f}, CER_help={row['helpful_akshara_cer']:.3f}, exact bank-copy={row['helpful_exact_bank_copy_rate']:.3f}, fuzzy bank-copy={row['helpful_fuzzy_bank_copy_rate']:.3f}"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
