#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_ROOT / "outputs/icml_v9_tail_count_summary_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/icml_v9_tail_count_summary_2026-04-09.md"
BOOTSTRAP_SEED = 0
N_BOOT = 10000


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty input")
    idx = max(0, min(len(sorted_vals) - 1, int(q * len(sorted_vals))))
    return float(sorted_vals[idx])


def bootstrap_mean_ci(vals: list[float], *, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_helpful_tail(path: Path) -> dict[str, float | int]:
    obj = load_json(path)
    rows = [row for row in obj["item_rows"] if str(row["condition"]) == "icl_helpful"]
    first_ak = [float(row["first_entry_correct"]) for row in rows]
    tail_vals = [
        float(row["continuation_fidelity"])
        for row in rows
        if row.get("continuation_fidelity") is not None and not math.isnan(float(row["continuation_fidelity"]))
    ]
    ci_lo, ci_hi = bootstrap_mean_ci(tail_vals)
    return {
        "n_total": int(len(rows)),
        "n_tail_eligible": int(len(tail_vals)),
        "helpful_first_akshara_correct_rate": float(sum(first_ak) / len(first_ak)),
        "tail_cer_mean": float(sum(tail_vals) / len(tail_vals)),
        "tail_cer_bootstrap95_lo": float(ci_lo),
        "tail_cer_bootstrap95_hi": float(ci_hi),
    }


def main() -> None:
    paths = {
        "Hindi": PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json",
        "Marathi": PROJECT_ROOT / "research/results/autoresearch/loop2_vm_controls/expansion_core64_seed42/raw/1b/aksharantar_mar_latin/nicl64/neutral_filler_recency_controls.json",
        "Telugu": PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json",
    }
    payload = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOT,
        "rows": {lang: summarize_helpful_tail(path) for lang, path in paths.items()},
        "source_paths": {lang: str(path) for lang, path in paths.items()},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ICML v9 continuation-tail eligibility summary (2026-04-09)",
        "",
        "Continuation-tail CER is computed only on helpful-condition items whose first akshara is already correct.",
        "",
    ]
    for lang in ["Hindi", "Marathi", "Telugu"]:
        row = payload["rows"][lang]
        lines.append(
            f"- {lang}: eligible {row['n_tail_eligible']}/{row['n_total']} items; "
            f"helpful first-akshara correctness = {row['helpful_first_akshara_correct_rate']:.3f}; "
            f"tail CER = {row['tail_cer_mean']:.3f}; bootstrap 95% CI [{row['tail_cer_bootstrap95_lo']:.3f}, {row['tail_cer_bootstrap95_hi']:.3f}]"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
