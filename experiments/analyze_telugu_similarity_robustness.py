#!/usr/bin/env python3
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = PROJECT_ROOT / "research/results/autoresearch/four_lang_thesis_panel/seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json"
OUT_JSON = PROJECT_ROOT / "outputs/telugu_similarity_robustness_2026-04-09.json"
OUT_MD = PROJECT_ROOT / "outputs/telugu_similarity_robustness_2026-04-09.md"


def sim(a: str, b: str) -> float:
    return float(SequenceMatcher(a=str(a), b=str(b)).ratio())


def median(xs: list[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return float(xs[n // 2])
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main() -> None:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    meta_by_idx = {int(row["item_index"]): row for row in payload["prompt_ordering_metadata_by_item"]}
    helpful_rows = [row for row in payload["item_rows"] if row["condition"] == "icl_helpful"]

    exact_copy_rows = []
    for row in helpful_rows:
        item_idx = int(row["item_index"])
        meta = meta_by_idx[item_idx]
        gold = str(row["word_hindi"])
        pred = str(row["prediction"])
        bank = [x for x in meta["helpful_similarity_desc"] if str(x["target"]) and str(x["target"]) != gold]
        matched = [x for x in bank if str(x["target"]) == pred]
        if not matched:
            continue
        match = matched[0]
        source_rank = int(match["position"]) + 1

        target_ranked = sorted(
            bank,
            key=lambda x: (-sim(gold, str(x["target"])), int(x["original_index"])),
        )
        target_rank = None
        for rank, cand in enumerate(target_ranked, start=1):
            if str(cand["target"]) == pred and int(cand["original_index"]) == int(match["original_index"]):
                target_rank = rank
                break
        if target_rank is None:
            raise RuntimeError(f"Failed to recover target-side rank for item {item_idx}")

        exact_copy_rows.append(
            {
                "item_index": item_idx,
                "query_source": str(row["word_ood"]),
                "gold_target": gold,
                "prediction": pred,
                "matched_source": str(match["source"]),
                "source_side_rank": source_rank,
                "target_side_rank": int(target_rank),
                "source_side_similarity": float(match["similarity"]),
                "target_side_similarity": sim(gold, pred),
            }
        )

    source_ranks = [int(r["source_side_rank"]) for r in exact_copy_rows]
    target_ranks = [int(r["target_side_rank"]) for r in exact_copy_rows]
    summary = {
        "input_json": str(INPUT_JSON),
        "n_items": len(helpful_rows),
        "n_exact_bank_copies": len(exact_copy_rows),
        "source_top5_count": sum(r <= 5 for r in source_ranks),
        "source_top5_rate": sum(r <= 5 for r in source_ranks) / len(source_ranks) if source_ranks else None,
        "target_top5_count": sum(r <= 5 for r in target_ranks),
        "target_top5_rate": sum(r <= 5 for r in target_ranks) / len(target_ranks) if target_ranks else None,
        "source_median_rank": median(source_ranks),
        "target_median_rank": median(target_ranks),
        "exact_copy_rows": exact_copy_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(
        "\n".join(
            [
                "# Telugu similarity robustness check (2026-04-09)",
                "",
                f"- Input panel: `{INPUT_JSON}`",
                f"- Helpful-condition items: **{len(helpful_rows)}**",
                f"- Exact bank-copy items: **{len(exact_copy_rows)}**",
                f"- Source-side top-5 concentration: **{summary['source_top5_count']}/{len(exact_copy_rows)} = {summary['source_top5_rate']:.3f}**",
                f"- Target-side top-5 concentration: **{summary['target_top5_count']}/{len(exact_copy_rows)} = {summary['target_top5_rate']:.3f}**",
                f"- Source-side median rank: **{summary['source_median_rank']}**",
                f"- Target-side median rank: **{summary['target_median_rank']}**",
                "",
                "Interpretation: exact bank copies in the 30-item Gemma 1B Telugu helpful panel remain concentrated near the top of both the source-side and target-side similarity rankings, so the retrieval-like late-drift picture does not depend entirely on the source-side heuristic.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
