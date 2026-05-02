from __future__ import annotations

"""Modal launcher for the Hindi causal patch panel.

This is intentionally narrow and review-facing: it runs the same script as the
VM launcher, but in an anonymous Modal workspace, and returns the compact summary
needed to decide whether same-length helpful/corrupt controls should be folded
into the paper.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "crc-hindi-patch-panel"
REMOTE_WORKSPACE = Path("/workspace")
REMOTE_ARTIFACTS = Path("/artifacts")
LOCAL_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",
        "numpy",
        "pandas",
        "datasets",
        "huggingface_hub",
    )
    .add_local_dir(str(LOCAL_ROOT / "Draft_Results"), remote_path=str(REMOTE_WORKSPACE / "Draft_Results"))
    .add_local_dir(str(LOCAL_ROOT / "experiments"), remote_path=str(REMOTE_WORKSPACE / "experiments"))
    .add_local_dir(str(LOCAL_ROOT / "research" / "modules"), remote_path=str(REMOTE_WORKSPACE / "research" / "modules"))
)

artifacts = modal.Volume.from_name("crc-workshop-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("gemma-hf-cache", create_if_missing=True)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    same_length_rows = [
        row
        for row in payload.get("summary_rows", [])
        if {row.get("recipient_condition"), row.get("donor_condition")} == {"icl_helpful", "icl_corrupt"}
    ]
    top_same_length = sorted(
        same_length_rows,
        key=lambda row: float(row.get("delta_mean_gap_latin", float("-inf"))),
        reverse=True,
    )[:12]
    return {
        "artifact": str(path),
        "experiment": payload.get("experiment"),
        "model": payload.get("model"),
        "pair": payload.get("pair"),
        "max_items": payload.get("max_items"),
        "patch_position_mode": payload.get("patch_position_mode"),
        "patches": payload.get("patches"),
        "top_same_length_rows": top_same_length,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=8 * 60 * 60,
    cpu=4,
    memory=32768,
    volumes={
        str(REMOTE_ARTIFACTS): artifacts,
        "/cache/huggingface": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_panel(
    *,
    max_items: int = 30,
    patch_pairs: str = "default,same_length",
    patch_position_mode: str = "last_token",
    results_name: str = "hindi_patch_panel_same_length_v1",
) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    out_dir = REMOTE_ARTIFACTS / results_name / "1b" / "aksharantar_hin_latin" / "nicl64"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "experiments/hindi_1b_causal_patch_panel.py",
        "--model",
        "1b",
        "--pair",
        "aksharantar_hin_latin",
        "--seed",
        "42",
        "--n-icl",
        "64",
        "--n-select",
        "300",
        "--n-eval",
        "200",
        "--max-items",
        str(int(max_items)),
        "--layers",
        "20,23,24,25",
        "--components",
        "layer_output,attention_output,mlp_output",
        "--patch-position-mode",
        str(patch_position_mode),
        "--patch-pairs",
        str(patch_pairs),
        "--device",
        "cuda",
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REMOTE_WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = out_dir / "modal_run.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        artifacts.commit()
        return {
            "ok": False,
            "returncode": proc.returncode,
            "log_tail": proc.stdout[-4000:],
            "log_path": str(log_path),
        }
    out_path = out_dir / "hindi_1b_causal_patch_panel.json"
    summary = _load_summary(out_path)
    summary.update({"ok": True, "log_path": str(log_path)})
    artifacts.commit()
    return summary


@app.local_entrypoint()
def main(
    max_items: int = 30,
    patch_pairs: str = "default,same_length",
    patch_position_mode: str = "last_token",
    results_name: str = "hindi_patch_panel_same_length_v1",
) -> None:
    result = run_panel.remote(
        max_items=max_items,
        patch_pairs=patch_pairs,
        patch_position_mode=patch_position_mode,
        results_name=results_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
