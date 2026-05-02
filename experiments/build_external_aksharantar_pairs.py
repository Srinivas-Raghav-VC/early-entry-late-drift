#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.modules.data.aksharantar_zip_loader import load_unique_aksharantar_rows

PAIR_META = {
    "hin": {
        "pair_id": "aksharantar_hin_latin",
        "name": "ai4bharat/Aksharantar[hin-en]",
        "url": "https://huggingface.co/datasets/ai4bharat/Aksharantar",
        "license": "cc",
        "version_date": "2023-08-31",
    },
    "tel": {
        "pair_id": "aksharantar_tel_latin",
        "name": "ai4bharat/Aksharantar[tel-en]",
        "url": "https://huggingface.co/datasets/ai4bharat/Aksharantar",
        "license": "cc",
        "version_date": "2023-08-31",
    },
    "tam": {
        "pair_id": "aksharantar_tam_latin",
        "name": "ai4bharat/Aksharantar[tam-en]",
        "url": "https://huggingface.co/datasets/ai4bharat/Aksharantar",
        "license": "cc",
        "version_date": "2023-08-31",
    },
    "ben": {
        "pair_id": "aksharantar_ben_latin",
        "name": "ai4bharat/Aksharantar[ben-en]",
        "url": "https://huggingface.co/datasets/ai4bharat/Aksharantar",
        "license": "cc",
        "version_date": "2023-08-31",
    },
    "mar": {
        "pair_id": "aksharantar_mar_latin",
        "name": "ai4bharat/Aksharantar[mar-en]",
        "url": "https://huggingface.co/datasets/ai4bharat/Aksharantar",
        "license": "cc",
        "version_date": "2023-08-31",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize external Aksharantar JSONL files for selected language codes.")
    ap.add_argument("--codes", nargs="+", default=["ben", "mar", "tam"], help="Language codes to build")
    ap.add_argument("--row-limit", type=int, default=2000)
    ap.add_argument("--cache-dir", type=str, default=".cache/aksharantar")
    ap.add_argument("--out-root", type=str, default="Draft_Results/data/transliteration")
    ap.add_argument("--build-note", type=str, default="Materialized for Loop 2/3 language-expansion control verification.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for code in args.codes:
        if code not in PAIR_META:
            raise ValueError(f"Unsupported code: {code}")
        meta = PAIR_META[code]
        rows, report = load_unique_aksharantar_rows(code=code, cache_dir=cache_dir)
        subset = rows[: max(1, int(args.row_limit))]
        out_path = out_root / f"{meta['pair_id']}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in subset:
                payload = {
                    "english": row["english"],
                    "source": row["ood"],
                    "target": row["target"],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_payload = {
            "name": meta["name"],
            "url": meta["url"],
            "license": meta["license"],
            "version_date": meta["version_date"],
            "row_count": len(subset),
            "build_note": args.build_note,
            "source_report": report,
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"pair_id": meta["pair_id"], "rows_written": len(subset), "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
