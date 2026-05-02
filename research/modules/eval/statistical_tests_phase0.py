from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_lo": 0.0,
            "ci95_hi": 0.0,
        }
    m = float(mean(values))
    if len(values) == 1:
        s = 0.0
    else:
        s = float(pstdev(values))
    sem = float(s / math.sqrt(max(1, len(values))))
    ci = 1.96 * sem
    return {
        "n": float(len(values)),
        "mean": m,
        "std": s,
        "sem": sem,
        "ci95_lo": m - ci,
        "ci95_hi": m + ci,
    }


def _binom_two_sided_p(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    # exact two-sided p-value for Binomial(n, 0.5)
    p_obs = math.comb(n, k) * (0.5**n)
    p = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (0.5**n)
        if pi <= p_obs + 1e-15:
            p += pi
    return float(min(1.0, p))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Phase 0A statistical tests across seed packets.")
    ap.add_argument("--input-dir", default="research/results/phase0")
    ap.add_argument("--glob", default="phase0a_packet_results_seed*.json")
    ap.add_argument("--baseline-template", default="canonical")
    ap.add_argument("--baseline-variant", default="helpful")
    ap.add_argument("--out-json", default="research/results/phase0/phase0_stat_tests.json")
    ap.add_argument("--out-csv", default="research/results/phase0/table_phase0_stat_tests.csv")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No packets found in {input_dir} matching {args.glob}")

    # seed -> key -> row
    behavior_by_seed: dict[int, dict[tuple[str, str, str, int], dict[str, Any]]] = {}
    seeds: list[int] = []

    for fp in files:
        payload = _read_json(fp)
        seed = int(payload.get("run_config", {}).get("split_seed", -1))
        if seed < 0:
            continue
        seeds.append(seed)
        rows_map: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for row in payload.get("behavior_summary", []):
            key = (
                str(row.get("language", "")),
                str(row.get("prompt_template", "canonical")),
                str(row.get("icl_variant", "helpful")),
                int(row.get("N", 0)),
            )
            rows_map[key] = row
        behavior_by_seed[seed] = rows_map

    unique_seeds = sorted(set(seeds))
    if not unique_seeds:
        raise RuntimeError("No valid seed packets found")

    baseline_template = str(args.baseline_template)
    baseline_variant = str(args.baseline_variant)

    # descriptive metric summaries across seeds for all conditions
    all_keys: set[tuple[str, str, str, int]] = set()
    for rows in behavior_by_seed.values():
        all_keys.update(rows.keys())

    descriptive_rows: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        language, prompt_template, icl_variant, n = key
        exact_vals: list[float] = []
        cer_vals: list[float] = []
        script_vals: list[float] = []
        for seed in unique_seeds:
            row = behavior_by_seed.get(seed, {}).get(key)
            if not row:
                continue
            exact_vals.append(float(row.get("exact_match_rate", 0.0)))
            cer_vals.append(float(row.get("akshara_CER_mean", 0.0)))
            script_vals.append(float(row.get("script_validity_rate", 0.0)))

        s_exact = _stats(exact_vals)
        s_cer = _stats(cer_vals)
        s_script = _stats(script_vals)
        descriptive_rows.append(
            {
                "test_family": "descriptive",
                "language": language,
                "prompt_template": prompt_template,
                "icl_variant": icl_variant,
                "N": int(n),
                "n_seed": int(s_exact["n"]),
                "exact_mean": s_exact["mean"],
                "exact_ci95_lo": s_exact["ci95_lo"],
                "exact_ci95_hi": s_exact["ci95_hi"],
                "cer_mean": s_cer["mean"],
                "cer_ci95_lo": s_cer["ci95_lo"],
                "cer_ci95_hi": s_cer["ci95_hi"],
                "script_mean": s_script["mean"],
                "script_ci95_lo": s_script["ci95_lo"],
                "script_ci95_hi": s_script["ci95_hi"],
            }
        )

    # Rescue + degradation tests on baseline condition
    test_rows: list[dict[str, Any]] = []
    languages = sorted({k[0] for k in all_keys})

    for language in languages:
        rescue_exact: list[float] = []
        rescue_cer: list[float] = []

        deg_exact_drop: list[float] = []
        deg_cer_increase: list[float] = []

        for seed in unique_seeds:
            rows = behavior_by_seed.get(seed, {})
            key0 = (language, baseline_template, baseline_variant, 0)
            key4 = (language, baseline_template, baseline_variant, 4)
            row0 = rows.get(key0)
            row4 = rows.get(key4)
            if row0 and row4:
                rescue_exact.append(
                    float(row4.get("exact_match_rate", 0.0)) - float(row0.get("exact_match_rate", 0.0))
                )
                rescue_cer.append(
                    float(row4.get("akshara_CER_mean", 0.0)) - float(row0.get("akshara_CER_mean", 0.0))
                )

            # degradation stats
            baseline_rows = [
                r for (lang, tmpl, var, n), r in rows.items()
                if lang == language and tmpl == baseline_template and var == baseline_variant and n > 0
            ]
            if not baseline_rows:
                continue
            max_n = max(int(k[3]) for k in rows if k[0] == language and k[1] == baseline_template and k[2] == baseline_variant)
            at_max = rows.get((language, baseline_template, baseline_variant, max_n))
            if at_max is None:
                continue
            best_exact = max(float(r.get("exact_match_rate", 0.0)) for r in baseline_rows)
            best_cer = min(float(r.get("akshara_CER_mean", 0.0)) for r in baseline_rows)
            deg_exact_drop.append(best_exact - float(at_max.get("exact_match_rate", 0.0)))
            deg_cer_increase.append(float(at_max.get("akshara_CER_mean", 0.0)) - best_cer)

        # Rescue exact: expect >0
        non_tie_exact = [v for v in rescue_exact if abs(v) > 1e-12]
        k_exact = sum(1 for v in non_tie_exact if v > 0)
        p_exact = _binom_two_sided_p(k_exact, len(non_tie_exact))

        non_tie_cer = [v for v in rescue_cer if abs(v) > 1e-12]
        k_cer = sum(1 for v in non_tie_cer if v < 0)
        p_cer = _binom_two_sided_p(k_cer, len(non_tie_cer))

        s_re_exact = _stats(rescue_exact)
        s_re_cer = _stats(rescue_cer)

        test_rows.append(
            {
                "test_family": "rescue",
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "metric": "exact_delta_N4_minus_N0",
                "n_seed": int(s_re_exact["n"]),
                "effect_mean": s_re_exact["mean"],
                "effect_ci95_lo": s_re_exact["ci95_lo"],
                "effect_ci95_hi": s_re_exact["ci95_hi"],
                "sign_test_p": p_exact,
            }
        )
        test_rows.append(
            {
                "test_family": "rescue",
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "metric": "cer_delta_N4_minus_N0",
                "n_seed": int(s_re_cer["n"]),
                "effect_mean": s_re_cer["mean"],
                "effect_ci95_lo": s_re_cer["ci95_lo"],
                "effect_ci95_hi": s_re_cer["ci95_hi"],
                "sign_test_p": p_cer,
            }
        )

        # degradation tests: expect >0
        non_tie_drop = [v for v in deg_exact_drop if abs(v) > 1e-12]
        k_drop = sum(1 for v in non_tie_drop if v > 0)
        p_drop = _binom_two_sided_p(k_drop, len(non_tie_drop))

        non_tie_cer_inc = [v for v in deg_cer_increase if abs(v) > 1e-12]
        k_cer_inc = sum(1 for v in non_tie_cer_inc if v > 0)
        p_cer_inc = _binom_two_sided_p(k_cer_inc, len(non_tie_cer_inc))

        s_drop = _stats(deg_exact_drop)
        s_cer_inc = _stats(deg_cer_increase)

        test_rows.append(
            {
                "test_family": "degradation",
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "metric": "exact_drop_best_minus_maxN",
                "n_seed": int(s_drop["n"]),
                "effect_mean": s_drop["mean"],
                "effect_ci95_lo": s_drop["ci95_lo"],
                "effect_ci95_hi": s_drop["ci95_hi"],
                "sign_test_p": p_drop,
            }
        )
        test_rows.append(
            {
                "test_family": "degradation",
                "language": language,
                "prompt_template": baseline_template,
                "icl_variant": baseline_variant,
                "metric": "cer_increase_maxN_minus_best",
                "n_seed": int(s_cer_inc["n"]),
                "effect_mean": s_cer_inc["mean"],
                "effect_ci95_lo": s_cer_inc["ci95_lo"],
                "effect_ci95_hi": s_cer_inc["ci95_hi"],
                "sign_test_p": p_cer_inc,
            }
        )

    payload = {
        "seeds": unique_seeds,
        "baseline_template": baseline_template,
        "baseline_variant": baseline_variant,
        "descriptive_rows": descriptive_rows,
        "test_rows": test_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    _write_csv(out_csv, [*test_rows, *descriptive_rows])

    print(str(out_json))
    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
