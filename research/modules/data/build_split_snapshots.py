from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.modules.data.aksharantar_loader import load_aksharantar_rows


def build_snapshot(
    *,
    pair: str,
    source_language: str,
    output_script_name: str,
    split_seed: int = 42,
    n_candidate: int = 300,
    n_eval: int = 50,
    n_values: list[int] | None = None,
) -> dict:
    n_values = sorted(n_values or [2, 4, 8, 16, 32, 48, 64, 96, 128])
    rows = load_aksharantar_rows(
        pair_code=pair,
        target_script=output_script_name,
        seed=split_seed,
        min_rows=max(n_candidate + n_eval + 32, 400),
    )

    candidate_pool = rows[:n_candidate]
    eval_rows = rows[n_candidate : n_candidate + n_eval]

    icl_presets: dict[str, list[dict[str, str]]] = {}
    for n in n_values:
        if n > len(candidate_pool):
            raise ValueError(f"Requested N={n} but candidate_pool has only {len(candidate_pool)} rows")
        icl_presets[str(n)] = candidate_pool[:n]

    return {
        "pair": f"aksharantar_{pair}_latin",
        "source_language": source_language,
        "input_script_name": "Latin",
        "output_script_name": output_script_name,
        "split_seed": split_seed,
        "candidate_pool": candidate_pool,
        "eval_rows": eval_rows,
        "icl_presets": icl_presets,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Phase-0 style split snapshots from Aksharantar.")
    ap.add_argument("--pair", required=True, help="Language code (hin/tel/ben/tam/mar)")
    ap.add_argument("--source-language", required=True)
    ap.add_argument("--output-script", required=True)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-candidate", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_snapshot(
        pair=args.pair,
        source_language=args.source_language,
        output_script_name=args.output_script,
        split_seed=args.split_seed,
        n_candidate=args.n_candidate,
        n_eval=args.n_eval,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
