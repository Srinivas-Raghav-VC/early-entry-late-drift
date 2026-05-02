from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FLORESTranslationDataset:
    name: str
    source_lang: str
    target_lang: str
    rows: List[Dict[str, str]]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "").strip())


def load_flores200_en_te(*, sample_size: int = 600, seed: int = 42) -> FLORESTranslationDataset:
    """
    Load FLORES-200 English->Telugu pairs using HF datasets if available.

    Returns normalized rows with fields:
      - english: english text (metadata key)
      - source: english sentence
      - target: telugu sentence

    This format is compatible with the pair record ingestion schema.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets package is required to load FLORES-200") from e

    # FLORES often stores these columns as sentence_eng_Latn / sentence_tel_Telu
    ds = load_dataset("facebook/flores", "all", split="dev")

    src_key = "sentence_eng_Latn"
    tgt_key = "sentence_tel_Telu"
    if src_key not in ds.column_names or tgt_key not in ds.column_names:
        raise RuntimeError(
            f"FLORES columns missing. Needed {src_key}/{tgt_key}, got {ds.column_names}"
        )

    rows: List[Dict[str, str]] = []
    for row in ds:
        src = _nfc(row.get(src_key, ""))
        tgt = _nfc(row.get(tgt_key, ""))
        if not src or not tgt:
            continue
        rows.append({"english": src, "source": src, "target": tgt})

    rng = random.Random(int(seed))
    rng.shuffle(rows)
    if int(sample_size) > 0:
        rows = rows[: int(sample_size)]

    return FLORESTranslationDataset(
        name="flores200_en_te",
        source_lang="eng_Latn",
        target_lang="tel_Telu",
        rows=rows,
    )
