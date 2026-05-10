from __future__ import annotations

"""Modal launcher for the bounded Telugu MLP-channel crossover test."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "crc-telugu-mlp-crossover"
REMOTE_WORKSPACE = Path("/workspace")
REMOTE_ARTIFACTS = Path("/artifacts")
LOCAL_ROOT = Path(__file__).resolve().parents[2]

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
    .add_local_dir(str(LOCAL_ROOT / "lib"), remote_path=str(REMOTE_WORKSPACE / "lib"))
    .add_local_dir(str(LOCAL_ROOT / "experiments"), remote_path=str(REMOTE_WORKSPACE / "experiments"))
    .add_local_dir(str(LOCAL_ROOT / "research" / "modules"), remote_path=str(REMOTE_WORKSPACE / "research" / "modules"))
)

artifacts = modal.Volume.from_name("crc-workshop-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("gemma-hf-cache", create_if_missing=True)


def _compact_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = {row["intervention"]: row for row in payload.get("summary_rows", [])}
    chosen = summary.get("chosen_channel_shift", {})
    baseline = summary.get("baseline_no_patch", {})
    return {
        "artifact": str(path),
        "experiment": payload.get("experiment"),
        "model": payload.get("model"),
        "pair": payload.get("pair"),
        "max_items": payload.get("max_items"),
        "select_max_items": payload.get("select_max_items"),
        "recipient": payload.get("recipient"),
        "donor": payload.get("donor"),
        "layer": payload.get("layer"),
        "selected": payload.get("selected"),
        "baseline": {
            "continuation_exact_match": baseline.get("continuation_exact_match", {}).get("mean"),
            "continuation_akshara_cer": baseline.get("continuation_akshara_cer", {}).get("mean"),
            "bank_copy_rate": baseline.get("bank_copy_rate", {}).get("mean"),
            "first_divergence_gap": baseline.get("first_divergence_gap", {}).get("mean"),
        },
        "chosen": {
            "continuation_exact_match": chosen.get("continuation_exact_match", {}).get("mean"),
            "continuation_akshara_cer": chosen.get("continuation_akshara_cer", {}).get("mean"),
            "bank_copy_rate": chosen.get("bank_copy_rate", {}).get("mean"),
            "delta_continuation_cer_improvement": chosen.get("delta_continuation_cer_improvement", {}).get("mean"),
            "delta_first_divergence_gap": chosen.get("delta_first_divergence_gap", {}).get("mean"),
        },
        "random_aggregate": payload.get("random_aggregate"),
        "comparison_rows": payload.get("comparison_rows"),
        "skipped_count": len(payload.get("skipped_rows", [])),
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10 * 60 * 60,
    cpu=4,
    memory=32768,
    volumes={
        str(REMOTE_ARTIFACTS): artifacts,
        "/cache/huggingface": hf_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_crossover(
    *,
    max_items: int = 100,
    select_max_items: int = 100,
    k_grid: str = "2,4,8,16,32,64,128",
    alpha_grid: str = "0.25,0.5,1.0,1.5,2.0",
    n_random: int = 3,
    results_name: str = "telugu_mlp_channel_crossover_v1",
) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    out_dir = REMOTE_ARTIFACTS / results_name / "1b" / "aksharantar_tel_latin" / "seed42" / "nicl64"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "experiments/telugu_1b_mlp_channel_crossover.py",
        "--model",
        "1b",
        "--pair",
        "aksharantar_tel_latin",
        "--seed",
        "42",
        "--n-icl",
        "64",
        "--n-select",
        "300",
        "--n-eval",
        "200",
        "--select-max-items",
        str(int(select_max_items)),
        "--max-items",
        str(int(max_items)),
        "--recipient",
        "icl_helpful",
        "--donor",
        "zs",
        "--layer",
        "25",
        "--offband-layer",
        "10",
        "--k-grid",
        str(k_grid),
        "--alpha-grid",
        str(alpha_grid),
        "--n-random",
        str(int(n_random)),
        "--external-only",
        "--require-external-sources",
        "--device",
        "cuda",
        "--out-root",
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
    artifacts.commit()
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "log_tail": proc.stdout[-6000:],
            "log_path": str(log_path),
        }
    out_path = out_dir / "telugu_1b_mlp_channel_crossover.json"
    summary = _compact_summary(out_path)
    summary.update({"ok": True, "log_path": str(log_path)})
    return summary


@app.local_entrypoint()
def main(
    max_items: int = 100,
    select_max_items: int = 100,
    k_grid: str = "2,4,8,16,32,64,128",
    alpha_grid: str = "0.25,0.5,1.0,1.5,2.0",
    n_random: int = 3,
    results_name: str = "telugu_mlp_channel_crossover_v1",
) -> None:
    result = run_crossover.remote(
        max_items=max_items,
        select_max_items=select_max_items,
        k_grid=k_grid,
        alpha_grid=alpha_grid,
        n_random=n_random,
        results_name=results_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
