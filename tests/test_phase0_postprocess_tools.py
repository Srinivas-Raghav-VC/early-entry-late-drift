import json
import subprocess
import sys
from pathlib import Path


def _write_packet(path: Path, seed: int, exact0: float, exact4: float, exact256: float) -> None:
    payload = {
        "run_config": {"split_seed": seed},
        "behavior_summary": [
            {
                "language": "Hindi",
                "prompt_template": "canonical",
                "icl_variant": "helpful",
                "N": 0,
                "exact_match_rate": exact0,
                "akshara_CER_mean": 0.8,
                "script_validity_rate": 0.8,
                "empty_or_refusal_rate": 0.0,
            },
            {
                "language": "Hindi",
                "prompt_template": "canonical",
                "icl_variant": "helpful",
                "N": 4,
                "exact_match_rate": exact4,
                "akshara_CER_mean": 0.4,
                "script_validity_rate": 0.9,
                "empty_or_refusal_rate": 0.0,
            },
            {
                "language": "Hindi",
                "prompt_template": "canonical",
                "icl_variant": "helpful",
                "N": 256,
                "exact_match_rate": exact256,
                "akshara_CER_mean": 0.7,
                "script_validity_rate": 0.85,
                "empty_or_refusal_rate": 0.0,
            },
        ],
        "judge_curve": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_aggregate_and_stats_tools(tmp_path: Path) -> None:
    _write_packet(tmp_path / "phase0a_packet_results_seed11.json", 11, 0.10, 0.35, 0.12)
    _write_packet(tmp_path / "phase0a_packet_results_seed42.json", 42, 0.12, 0.40, 0.10)

    agg_out = tmp_path / "agg.csv"
    subprocess.run(
        [
            sys.executable,
            "research/modules/eval/aggregate_phase0_seeds.py",
            "--input-dir",
            str(tmp_path),
            "--glob",
            "phase0a_packet_results_seed*.json",
            "--out",
            str(agg_out),
        ],
        check=True,
    )
    assert agg_out.exists()

    stats_json = tmp_path / "stats.json"
    stats_csv = tmp_path / "stats.csv"
    subprocess.run(
        [
            sys.executable,
            "research/modules/eval/statistical_tests_phase0.py",
            "--input-dir",
            str(tmp_path),
            "--glob",
            "phase0a_packet_results_seed*.json",
            "--out-json",
            str(stats_json),
            "--out-csv",
            str(stats_csv),
        ],
        check=True,
    )
    assert stats_json.exists()
    assert stats_csv.exists()

    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    assert payload["seeds"] == [11, 42]
    assert payload["test_rows"]
