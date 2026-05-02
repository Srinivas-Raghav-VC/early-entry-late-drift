#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "research" / "results" / "autoresearch" / "hindi_mlp_subspace_patch_v1" / "1b" / "aksharantar_hin_latin" / "nicl64" / "hindi_1b_mlp_subspace_patch.json"
DEFAULT_OUT_PREFIX = PROJECT_ROOT / "outputs" / "1b_hindi_mlp_subspace_patch_2026-03-29"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _bootstrap_mean(xs: List[float], *, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boots = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


def _iter_matching_item_interventions(payload: Dict[str, Any], *, recipient: str, donor: str, selector_kind: str, k: int) -> Iterable[Dict[str, Any]]:
    for item in payload.get("item_rows", []):
        for intr in item.get("interventions", []):
            if (
                intr.get("recipient_condition") == recipient
                and intr.get("donor_condition") == donor
                and intr.get("selector_kind") == selector_kind
                and int(intr.get("k", -1)) == int(k)
            ):
                yield {"item": item, "intr": intr}


def _find_smallest_k(rows: List[Dict[str, Any]], *, threshold: float) -> int | None:
    good = [row for row in rows if row.get("fraction_of_dense_gap_latin") is not None and row["fraction_of_dense_gap_latin"] >= threshold]
    if not good:
        return None
    return int(min(row["k"] for row in good))


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 1B Hindi L25 MLP subspace patch summary")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(summary["executive_summary"])
    lines.append("")
    lines.append("## No-op sanity check")
    lines.append("")
    no_op = summary["no_op_sanity"]
    lines.append(f"- max absolute no-op `delta_mean_gap_latin`: `{no_op['max_abs_delta_gap_latin']:.6f}`")
    lines.append(f"- max absolute no-op `delta_top1_target_rate`: `{no_op['max_abs_delta_top1']:.6f}`")
    lines.append("")
    lines.append("## Helpful <- ZS selected-topk curve")
    lines.append("")
    lines.append("| k | selected dgapLatin | 95% CI | selected dTop1 | random mean | frac dense |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["helpful_curve"]:
        rand = row.get("random_mean_delta_gap_latin")
        frac = row.get("fraction_of_dense_gap_latin")
        lines.append(
            f"| {row['k']} | {row['selected_delta_mean_gap_latin']:.3f} | [{row['bootstrap_gap_ci_lo']:.3f}, {row['bootstrap_gap_ci_hi']:.3f}] | {row['selected_delta_top1_target_rate']:.3f} | {('NA' if rand is None else f'{rand:.3f}')} | {('NA' if frac is None else f'{frac:.3f}')} |"
        )
    lines.append("")
    lines.append("## ZS <- Helpful selected-topk curve")
    lines.append("")
    lines.append("| k | selected dgapLatin | 95% CI | selected dTop1 | random mean | frac dense |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["zs_curve"]:
        rand = row.get("random_mean_delta_gap_latin")
        frac = row.get("fraction_of_dense_gap_latin")
        lines.append(
            f"| {row['k']} | {row['selected_delta_mean_gap_latin']:.3f} | [{row['bootstrap_gap_ci_lo']:.3f}, {row['bootstrap_gap_ci_hi']:.3f}] | {row['selected_delta_top1_target_rate']:.3f} | {('NA' if rand is None else f'{rand:.3f}')} | {('NA' if frac is None else f'{frac:.3f}')} |"
        )
    lines.append("")
    lines.append("## Concentration milestones")
    lines.append("")
    lines.append(f"- smallest k reaching >=50% of dense helpful<-zs gap: `{summary['milestones']['helpful_frac50_k']}`")
    lines.append(f"- smallest k reaching >=80% of dense helpful<-zs gap: `{summary['milestones']['helpful_frac80_k']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.extend(summary["interpretation_bullets"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Hindi L25 MLP subspace patch artifact.")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--out-prefix", type=str, default=str(DEFAULT_OUT_PREFIX))
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    comparison_rows = list(payload.get("comparison_rows", []))
    helpful_curve = [row for row in comparison_rows if row["recipient_condition"] == "icl_helpful" and row["donor_condition"] == "zs"]
    helpful_curve = sorted(helpful_curve, key=lambda row: int(row["k"]))
    zs_curve = [row for row in comparison_rows if row["recipient_condition"] == "zs" and row["donor_condition"] == "icl_helpful"]
    zs_curve = sorted(zs_curve, key=lambda row: int(row["k"]))

    for curve_idx, curve in enumerate((helpful_curve, zs_curve)):
        for row in curve:
            recipient = row["recipient_condition"]
            donor = row["donor_condition"]
            k = int(row["k"])
            item_deltas_gap = [
                float(bundle["intr"]["delta"]["target_minus_latin_logit"])
                for bundle in _iter_matching_item_interventions(
                    payload,
                    recipient=recipient,
                    donor=donor,
                    selector_kind="selected_topk",
                    k=k,
                )
            ]
            item_deltas_top1 = [
                float(bundle["intr"]["delta"]["top1_is_target"])
                for bundle in _iter_matching_item_interventions(
                    payload,
                    recipient=recipient,
                    donor=donor,
                    selector_kind="selected_topk",
                    k=k,
                )
            ]
            gap_ci = _bootstrap_mean(item_deltas_gap, seed=curve_idx * 1000 + k)
            top1_ci = _bootstrap_mean(item_deltas_top1, seed=curve_idx * 1000 + k + 17)
            row["bootstrap_gap_ci_lo"] = gap_ci["ci_lo"]
            row["bootstrap_gap_ci_hi"] = gap_ci["ci_hi"]
            row["bootstrap_top1_ci_lo"] = top1_ci["ci_lo"]
            row["bootstrap_top1_ci_hi"] = top1_ci["ci_hi"]

    no_op_rows = [row for row in payload.get("summary_rows", []) if row.get("selector_kind") == "self_noop"]
    no_op_gap = max((abs(float(row["delta_mean_gap_latin"])) for row in no_op_rows), default=float("nan"))
    no_op_top1 = max((abs(float(row["delta_top1_target_rate"])) for row in no_op_rows), default=float("nan"))

    helpful_frac50_k = _find_smallest_k(helpful_curve, threshold=0.5)
    helpful_frac80_k = _find_smallest_k(helpful_curve, threshold=0.8)

    exec_summary = (
        f"The new experiment tests whether the validated Hindi failure signal at the last-token layer-25 MLP is concentrated in a relatively small coordinate subset. "
        f"Coordinates were ranked using only the held-out selection split, then evaluated on the separate 30-item eval split with selected-topk patches, random-coordinate controls, dense patches, and self-patch no-op checks."
    )

    interpretation = [
        "- If small-k selected patches recover a large fraction of the dense helpful<-zs effect while random patches stay weak, that supports **causal concentration** inside the L25 MLP output rather than a purely diffuse state shift.",
        "- If the same selected coordinates produce opposite-sign harm under `zs <- helpful`, that strengthens the bottleneck interpretation.",
        "- If only very large k values work, the effect is more likely broad and distributed than feature-like.",
    ]

    summary = {
        "input": str(args.input),
        "executive_summary": exec_summary,
        "no_op_sanity": {
            "max_abs_delta_gap_latin": no_op_gap,
            "max_abs_delta_top1": no_op_top1,
        },
        "helpful_curve": helpful_curve,
        "zs_curve": zs_curve,
        "milestones": {
            "helpful_frac50_k": helpful_frac50_k,
            "helpful_frac80_k": helpful_frac80_k,
        },
        "interpretation_bullets": interpretation,
    }

    out_prefix = Path(args.out_prefix)
    _write_json(out_prefix.with_suffix(".json"), summary)
    out_prefix.with_suffix(".md").write_text(render_markdown(summary), encoding="utf-8")
    print(out_prefix.with_suffix(".json"))
    print(out_prefix.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
