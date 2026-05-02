#!/usr/bin/env python3
from __future__ import annotations

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
}


def main() -> int:
    cache_dir = Path('.cache/aksharantar')
    out_root = Path('Draft_Results/data/transliteration')
    out_root.mkdir(parents=True, exist_ok=True)

    for code, meta in PAIR_META.items():
        rows, report = load_unique_aksharantar_rows(code=code, cache_dir=cache_dir)
        subset = rows[:2000]
        out_path = out_root / f"{meta['pair_id']}.jsonl"
        with out_path.open('w', encoding='utf-8') as f:
            for row in subset:
                payload = {
                    'english': row['english'],
                    'source': row['ood'],
                    'target': row['target'],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        meta_path = out_path.with_suffix(out_path.suffix + '.meta.json')
        meta_payload = {
            'name': meta['name'],
            'url': meta['url'],
            'license': meta['license'],
            'version_date': meta['version_date'],
            'row_count': len(subset),
            'build_note': 'Subset materialized for Loop 1 cross-scale smoke/full premise-gate autoresearch.',
            'source_report': report,
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'pair_id': meta['pair_id'], 'rows_written': len(subset), 'out': str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
