from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.modules.data.aksharantar_zip_loader import (
    build_unique_snapshot_from_rows,
    load_unique_aksharantar_rows,
)
from research.modules.infra.phase0a_run_config import (
    Phase0ARunConfig,
    load_phase0a_run_config,
    write_default_phase0a_run_config,
)
from research.modules.modal.modal_phase0_packet import app as phase0_modal_app
from research.modules.modal.modal_phase0_packet import run_phase0a


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_snapshots(cfg: Phase0ARunConfig, cache_dir: Path) -> list[Path]:
    out_dir = Path(cfg.paths.snapshots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preloaded: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {}
    for code in cfg.language_codes:
        rows, report = load_unique_aksharantar_rows(code=code, cache_dir=cache_dir)
        preloaded[code] = (rows, report)

    out_paths: list[Path] = []
    need_per_seed = int(cfg.n_candidate) + int(cfg.n_eval)

    if cfg.cross_seed_disjoint:
        for code in cfg.language_codes:
            rows, report = preloaded[code]
            total_needed = need_per_seed * len(cfg.seeds)
            if len(rows) < total_needed:
                raise RuntimeError(
                    f"{code}: insufficient rows for cross-seed-disjoint split "
                    f"({len(rows)} < {total_needed})"
                )

            shuffled = list(rows)
            rng = random.Random(17_000 + sum(ord(ch) for ch in code))
            rng.shuffle(shuffled)

            for idx, seed in enumerate(cfg.seeds):
                chunk = shuffled[idx * need_per_seed : (idx + 1) * need_per_seed]
                snapshot = build_unique_snapshot_from_rows(
                    code=code,
                    rows=chunk,
                    quality_report=report,
                    split_seed=int(seed),
                    n_candidate=int(cfg.n_candidate),
                    n_eval=int(cfg.n_eval),
                    n_values=list(cfg.n_values),
                )
                out_path = (
                    out_dir
                    / f"aksharantar_{code}_latin_unique_seed{seed}_ncand{cfg.n_candidate}_neval{cfg.n_eval}.json"
                )
                out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                out_paths.append(out_path)
    else:
        for seed in cfg.seeds:
            for code in cfg.language_codes:
                rows, report = preloaded[code]
                snapshot = build_unique_snapshot_from_rows(
                    code=code,
                    rows=rows,
                    quality_report=report,
                    split_seed=int(seed),
                    n_candidate=int(cfg.n_candidate),
                    n_eval=int(cfg.n_eval),
                    n_values=list(cfg.n_values),
                )
                out_path = (
                    out_dir
                    / f"aksharantar_{code}_latin_unique_seed{seed}_ncand{cfg.n_candidate}_neval{cfg.n_eval}.json"
                )
                out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                out_paths.append(out_path)

    manifest = {
        "name": cfg.name,
        "seeds": cfg.seeds,
        "language_codes": cfg.language_codes,
        "n_candidate": cfg.n_candidate,
        "n_eval": cfg.n_eval,
        "n_values": cfg.n_values,
        "cross_seed_disjoint": bool(cfg.cross_seed_disjoint),
        "snapshots": [str(p) for p in out_paths],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_paths


def _run_seed(cfg: Phase0ARunConfig, *, seed: int, hf_token: str, google_api_key: str) -> dict[str, Any]:
    kwargs = cfg.to_modal_kwargs(seed=seed, google_api_key=google_api_key, hf_token=hf_token)
    return run_phase0a.remote(**kwargs)


def _write_seed_outputs(cfg: Phase0ARunConfig, *, seed: int, result: dict[str, Any]) -> Path:
    out_root = Path(cfg.paths.output_dir)
    packets_dir = out_root / "packets"
    tables_dir = out_root / "tables"
    packets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_json = packets_dir / f"phase0a_packet_results_seed{seed}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if result.get("status") == "complete":
        _write_csv(tables_dir / f"table_phase0_behavior_seed{seed}.csv", result.get("behavior_summary", []))
        _write_csv(tables_dir / f"table_phase0_token_probe_seed{seed}.csv", result.get("token_probe", []))
        _write_csv(tables_dir / f"table_phase0_judge_curve_seed{seed}.csv", result.get("judge_curve", []))
        _write_csv(
            tables_dir / f"table_phase0_judge_sanity_seed{seed}.csv",
            result.get("judge_sanity", {}).get("rows", []),
        )

    return out_json


def _run_postprocess(cfg: Phase0ARunConfig) -> None:
    py = sys.executable
    out_root = Path(cfg.paths.output_dir)
    packets_dir = out_root / "packets"
    tables_dir = out_root / "tables"
    stats_dir = out_root / "stats"
    plots_dir = out_root / "plots"

    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.aggregate_after_run:
        subprocess.run(
            [
                py,
                str(PROJECT_ROOT / "research/modules/eval/aggregate_phase0_seeds.py"),
                "--input-dir",
                str(packets_dir),
                "--glob",
                "phase0a_packet_results_seed*.json",
                "--out",
                str(tables_dir / "table_phase0_seed_aggregate.csv"),
            ],
            check=True,
        )

    if cfg.stats_after_run:
        subprocess.run(
            [
                py,
                str(PROJECT_ROOT / "research/modules/eval/statistical_tests_phase0.py"),
                "--input-dir",
                str(packets_dir),
                "--glob",
                "phase0a_packet_results_seed*.json",
                "--baseline-template",
                str(cfg.baseline_prompt_template),
                "--baseline-variant",
                str(cfg.baseline_icl_variant),
                "--out-json",
                str(stats_dir / "phase0_stat_tests.json"),
                "--out-csv",
                str(stats_dir / "table_phase0_stat_tests.csv"),
            ],
            check=True,
        )

    if cfg.plots_after_run:
        packet_files = sorted(packets_dir.glob("phase0a_packet_results_seed*.json"))
        for packet in packet_files:
            stem = packet.stem
            seed_tag = stem.replace("phase0a_packet_results_", "")
            seed_plot_dir = plots_dir / seed_tag
            seed_plot_dir.mkdir(parents=True, exist_ok=True)
            for prompt_template in cfg.prompt_templates:
                for icl_variant in cfg.icl_variants:
                    subprocess.run(
                        [
                            py,
                            str(PROJECT_ROOT / "research/modules/eval/plot_phase0_curves.py"),
                            "--input",
                            str(packet),
                            "--out-dir",
                            str(seed_plot_dir),
                            "--prompt-template",
                            str(prompt_template),
                            "--icl-variant",
                            str(icl_variant),
                        ],
                        check=True,
                    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Phase 0A experiment matrix from config manifest.")
    ap.add_argument("--config", default="research/config/phase0a_run_config.json")
    ap.add_argument("--cache-dir", default=".cache/aksharantar")
    ap.add_argument("--write-default-config", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="Build snapshots/manifest only")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)

    if args.write_default_config:
        out = write_default_phase0a_run_config(cfg_path)
        print(str(out))
        return 0

    cfg = load_phase0a_run_config(cfg_path)
    print(json.dumps({"loaded_config": cfg.to_payload()}, ensure_ascii=False, indent=2))

    out_root = Path(cfg.paths.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config_resolved.json").write_text(
        json.dumps(cfg.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "run_config_source.json").write_text(
        cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    _ensure_snapshots(cfg, cache_dir=Path(args.cache_dir))
    if args.build_only:
        print("[phase0a-config] build-only complete")
        return 0

    hf_token = os.environ.get("HF_TOKEN", "")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")

    with phase0_modal_app.run():
        for seed in cfg.seeds:
            print(f"[phase0a-config] running seed={seed}", flush=True)
            result = _run_seed(cfg, seed=int(seed), hf_token=hf_token, google_api_key=google_api_key)
            out_json = _write_seed_outputs(cfg, seed=int(seed), result=result)
            print(
                json.dumps(
                    {
                        "seed": int(seed),
                        "status": result.get("status"),
                        "go_no_go": result.get("go_no_go"),
                        "stop_reason": result.get("stop_reason"),
                        "result_json": str(out_json),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    _run_postprocess(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
