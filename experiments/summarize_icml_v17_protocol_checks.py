#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.modules.eval.metrics import normalize_text, segment_aksharas  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_hindi_localization() -> dict:
    path = (
        PROJECT_ROOT
        / "research/results/autoresearch/hindi_patch_panel_lasttoken_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_causal_patch_panel.json"
    )
    obj = _load_json(path)
    rows = [
        r
        for r in (obj.get("summary_rows") or [])
        if r.get("recipient_condition") == "icl_helpful" and r.get("donor_condition") == "zs"
    ]
    ranked = sorted(rows, key=lambda r: float(r.get("delta_mean_gap_latin", float("-inf"))), reverse=True)
    item_rows = obj.get("item_rows") or []
    first_item = item_rows[0] if item_rows else {}
    return {
        "artifact": str(path.relative_to(PROJECT_ROOT)),
        "patch_position_mode": obj.get("patch_position_mode"),
        "top_sites": [
            {
                "layer": int(r["layer"]),
                "component": str(r["component"]),
                "delta_mean_gap_latin": float(r["delta_mean_gap_latin"]),
                "delta_top1_target_rate": float(r["delta_top1_target_rate"]),
                "n_items": int(r["n_items"]),
            }
            for r in ranked[:5]
        ],
        "position_example": {
            "query_position_by_condition": first_item.get("query_position_by_condition"),
            "last_position_by_condition": first_item.get("last_position_by_condition"),
            "patch_position_by_condition": first_item.get("patch_position_by_condition"),
        },
    }


def summarize_telugu_protocol() -> dict:
    path = (
        PROJECT_ROOT
        / "research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json"
    )
    obj = _load_json(path)
    item_rows = obj.get("item_rows") or []
    skipped = obj.get("skipped_rows") or []
    after_first_count = 0
    for row in item_rows:
        prefix = normalize_text(row.get("shared_prefix_text", ""))
        gold = normalize_text(row.get("word_target", ""))
        prefix_aks = segment_aksharas(prefix)
        gold_aks = segment_aksharas(gold)
        if prefix_aks and gold_aks and prefix_aks[0] == gold_aks[0]:
            after_first_count += 1
    skip_reasons: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("reason", "unknown"))
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    return {
        "artifact": str(path.relative_to(PROJECT_ROOT)),
        "selected_alpha": float(obj.get("selected_alpha", float("nan"))),
        "selected_site": obj.get("site"),
        "n_items_evaluated": int(len(item_rows)),
        "n_items_skipped": int(len(skipped)),
        "skip_reasons": skip_reasons,
        "shared_prefix_contains_gold_first_akshara": {
            "count": int(after_first_count),
            "denominator": int(len(item_rows)),
            "rate": float(after_first_count / len(item_rows)) if item_rows else None,
        },
        "shared_prefix_token_length": {
            "min": min(int(r["shared_prefix_len_tokens"]) for r in item_rows) if item_rows else None,
            "max": max(int(r["shared_prefix_len_tokens"]) for r in item_rows) if item_rows else None,
            "mean": (
                sum(int(r["shared_prefix_len_tokens"]) for r in item_rows) / len(item_rows)
                if item_rows
                else None
            ),
        },
    }


def summarize_telugu_temperature_item_level() -> dict:
    path = (
        PROJECT_ROOT
        / "research/results/autoresearch/telugu_temperature_sweep_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_temperature_sweep.json"
    )
    obj = _load_json(path)
    item_rows = obj.get("item_rows") or []
    out: dict[str, dict] = {}
    for temp_key in ["temp_0.0", "temp_0.2", "temp_0.7"]:
        exact_any = []
        fuzzy_any = []
        exact_mean_per_item = []
        for row in item_rows:
            helpful = row["temperatures"][temp_key]["icl_helpful"]["samples"]
            exact_vals = [float(s["generation"]["exact_bank_copy"]) for s in helpful]
            fuzzy_vals = [float(s["generation"]["fuzzy_bank_copy"]) for s in helpful]
            exact_any.append(1.0 if any(v > 0.5 for v in exact_vals) else 0.0)
            fuzzy_any.append(1.0 if any(v > 0.5 for v in fuzzy_vals) else 0.0)
            exact_mean_per_item.append(sum(exact_vals) / len(exact_vals))
        out[temp_key] = {
            "n_items": int(len(item_rows)),
            "item_level_any_exact_bank_copy": float(sum(exact_any) / len(exact_any)) if exact_any else None,
            "item_level_any_fuzzy_bank_copy": float(sum(fuzzy_any) / len(fuzzy_any)) if fuzzy_any else None,
            "mean_per_item_exact_bank_copy_rate": float(sum(exact_mean_per_item) / len(exact_mean_per_item)) if exact_mean_per_item else None,
        }
    return {
        "artifact": str(path.relative_to(PROJECT_ROOT)),
        "item_level_summary": out,
    }


def main() -> int:
    payload = {
        "hindi_localization": summarize_hindi_localization(),
        "telugu_protocol": summarize_telugu_protocol(),
        "telugu_temperature_item_level": summarize_telugu_temperature_item_level(),
    }

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "icml_v17_protocol_checks_2026-04-10.json"
    md_path = out_dir / "icml_v17_protocol_checks_2026-04-10.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    h = payload["hindi_localization"]
    t = payload["telugu_protocol"]
    temp = payload["telugu_temperature_item_level"]["item_level_summary"]
    lines = [
        "# ICML v17 protocol checks (2026-04-10)",
        "",
        "## Hindi localization",
        f"- Patch-position mode: `{h['patch_position_mode']}`",
        f"- Best helpful<-zs site: `L{h['top_sites'][0]['layer']} {h['top_sites'][0]['component']}` with Δgap {h['top_sites'][0]['delta_mean_gap_latin']:+.3f} on n={h['top_sites'][0]['n_items']}",
        f"- Next best sites: `L{h['top_sites'][1]['layer']} {h['top_sites'][1]['component']}` ({h['top_sites'][1]['delta_mean_gap_latin']:+.3f}), `L{h['top_sites'][2]['layer']} {h['top_sites'][2]['component']}` ({h['top_sites'][2]['delta_mean_gap_latin']:+.3f})",
        f"- Example last-token alignment: patch positions = {h['position_example']['patch_position_by_condition']}; query positions = {h['position_example']['query_position_by_condition']}",
        "",
        "## Telugu oracle-positioned patch protocol",
        f"- Selected site: `{t['selected_site']['name']}` with α={t['selected_alpha']:.2f}",
        f"- Evaluated items: {t['n_items_evaluated']} ; skipped: {t['n_items_skipped']} ({t['skip_reasons']})",
        f"- Shared prefix contains the gold first akshara on {t['shared_prefix_contains_gold_first_akshara']['count']}/{t['shared_prefix_contains_gold_first_akshara']['denominator']} usable items",
        f"- Shared-prefix token length: min={t['shared_prefix_token_length']['min']}, max={t['shared_prefix_token_length']['max']}, mean={t['shared_prefix_token_length']['mean']:.3f}",
        "",
        "## Telugu temperature sweep item-level view",
        f"- T=0.0 any exact bank copy across samples: {temp['temp_0.0']['item_level_any_exact_bank_copy']:.3f}",
        f"- T=0.2 any exact bank copy across samples: {temp['temp_0.2']['item_level_any_exact_bank_copy']:.3f}",
        f"- T=0.7 any exact bank copy across samples: {temp['temp_0.7']['item_level_any_exact_bank_copy']:.3f}",
        f"- T=0.0 / 0.2 / 0.7 any fuzzy bank copy: {temp['temp_0.0']['item_level_any_fuzzy_bank_copy']:.3f} / {temp['temp_0.2']['item_level_any_fuzzy_bank_copy']:.3f} / {temp['temp_0.7']['item_level_any_fuzzy_bank_copy']:.3f}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
