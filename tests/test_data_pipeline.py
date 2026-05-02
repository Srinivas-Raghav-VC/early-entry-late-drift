from pathlib import Path

from research.modules.eval.verification_packet import (
    run_deterministic_metric_sanity,
    validate_snapshot,
)


HIN_SNAPSHOT = Path(
    "Draft_Results/paper2_fidelity_calibrated/split_snapshots/"
    "aksharantar_hin_latin_split_seed42_nicl16_nselect300_neval50.json"
)
TEL_SNAPSHOT = Path(
    "Draft_Results/paper2_fidelity_calibrated/split_snapshots/"
    "aksharantar_tel_latin_split_seed42_nicl16_nselect300_neval50.json"
)


def test_snapshot_validation_hindi_telugu() -> None:
    for path in (HIN_SNAPSHOT, TEL_SNAPSHOT):
        report = validate_snapshot(path)
        assert report["schema"]["ok"]
        assert report["disjointness"]["ok"]
        assert report["script_validity"]["ok"]


def test_deterministic_metric_sanity_packet() -> None:
    report = run_deterministic_metric_sanity()
    assert report["ok"]
    assert report["accuracy"] == 1.0
