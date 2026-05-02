from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.modules.data.aksharantar_zip_loader import (
    build_unique_snapshot_from_rows,
    load_unique_aksharantar_rows,
)


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build unique Phase 0A snapshots from Aksharantar zips.")
    ap.add_argument("--codes", default="hin,tel", help="Comma-separated language codes")
    ap.add_argument("--out-dir", default="research/results/phase0/snapshots")
    ap.add_argument("--cache-dir", default=".cache/aksharantar")
    ap.add_argument("--seeds", default="42", help="Comma-separated split seeds")
    ap.add_argument("--n-candidate", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument(
        "--n-values",
        default="2,4,8,16,32,48,64,96,128,192,256",
        help="Comma-separated ICL N presets to materialize",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
    seeds = parse_int_list(args.seeds)
    n_values = parse_int_list(args.n_values)

    outputs: list[str] = []

    preloaded: dict[str, tuple[list[dict[str, str]], dict[str, object]]] = {}
    for code in codes:
        rows, report = load_unique_aksharantar_rows(code=code, cache_dir=Path(args.cache_dir))
        preloaded[code] = (rows, report)

    for seed in seeds:
        for code in codes:
            rows, report = preloaded[code]
            snapshot = build_unique_snapshot_from_rows(
                code=code,
                rows=rows,
                quality_report=report,
                split_seed=int(seed),
                n_candidate=int(args.n_candidate),
                n_eval=int(args.n_eval),
                n_values=n_values,
            )
            out_path = (
                out_dir
                / f"aksharantar_{code}_latin_unique_seed{seed}_ncand{args.n_candidate}_neval{args.n_eval}.json"
            )
            out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(str(out_path))
            print(str(out_path))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "n_candidate": int(args.n_candidate),
                "n_eval": int(args.n_eval),
                "n_values": n_values,
                "snapshots": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
