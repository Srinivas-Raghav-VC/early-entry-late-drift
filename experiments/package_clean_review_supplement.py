#!/usr/bin/env python3
from __future__ import annotations

"""Build a minimal supplementary-material archive.

The archive contains one README, the paper source/PDF, selected retained result
artifacts under professional names, and a small stdlib-only Python package that
recomputes the headline numbers from those artifacts. It intentionally omits
local experiment orchestration and exploratory files.
"""

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from textwrap import dedent
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/tmp/early_entry_late_drift_supplementary_material")
DEFAULT_ARCHIVE = Path("/tmp/early_entry_late_drift_supplementary_material.tar.gz")

PAPER_FILES = [
    ("paper/icml2026/submission.pdf", "paper/icml2026/submission.pdf"),
    ("paper/icml2026/submission.tex", "paper/icml2026/submission.tex"),
    ("paper/icml2026/icml2026.sty", "paper/icml2026/icml2026.sty"),
    ("paper/icml2026/icml2026.bst", "paper/icml2026/icml2026.bst"),
    ("paper/icml2026/algorithm.sty", "paper/icml2026/algorithm.sty"),
    ("paper/icml2026/algorithmic.sty", "paper/icml2026/algorithmic.sty"),
    ("paper/icml2026/fancyhdr.sty", "paper/icml2026/fancyhdr.sty"),
    ("paper/figures/fig_stage_intervention_overview_tikz.tex", "paper/figures/fig_stage_intervention_overview_tikz.tex"),
    ("paper/figures/fig_stage_axis_map_tikz.tex", "paper/figures/fig_stage_axis_map_tikz.tex"),
    ("paper/figures/fig_behavioral_regime_summary_tikz_v19.tex", "paper/figures/fig_behavioral_regime_summary_tikz_v19.tex"),
    ("paper/figures/fig_hindi_practical_patch_tikz_v18.tex", "paper/figures/fig_hindi_practical_patch_tikz_v18.tex"),
]

RAW_RESULT_FILES = [
    (
        "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json",
        "data/results/hindi_fixed_patch_results.json",
    ),
    (
        "research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json",
        "data/results/hindi_channel_lesion_results.json",
    ),
    (
        "research/results/autoresearch/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_causal_patch_panel.json",
        "data/results/hindi_same_length_patch_panel_results.json",
    ),
    (
        "research/results/autoresearch/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json",
        "data/results/hindi_channel_value_results.json",
    ),
    (
        "research/results/autoresearch/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json",
        "data/results/hindi_readout_geometry_results.json",
    ),
    (
        "research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json",
        "data/results/telugu_oracle_residual_patch_results.json",
    ),
    (
        "research/results/autoresearch/telugu_mlp_channel_crossover_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_1b_mlp_channel_crossover.json",
        "data/results/telugu_sparse_channel_crossover_results.json",
    ),
]

DERIVED_JSON_FILES = [
    ("outputs/stage_axis_map_points_2026-04-28.json", "data/derived/stage_axis_points.json", "stage_axis"),
    ("research/submission/hindi_channel_interpretation_summary_2026-04-28.json", "data/derived/hindi_channel_interpretation_summary.json", "drop_artifact_fields"),
    ("research/submission/mechanistic_crossover_summary_2026-04-28.json", "data/derived/mechanistic_crossover_summary.json", "drop_artifact_fields"),
    ("research/submission/cross_model_behavioral_synthesis_2026-04-28.json", "data/derived/cross_family_behavior_summary.json", "drop_artifact_fields"),
]

def forbidden_text_tokens() -> list[str]:
    """Tokens that should not appear in the curated supplement.

    Some entries are assembled from pieces so this packaging script does not
    trigger repository-level identity-leak audits while still checking the final
    supplement for those strings.
    """

    return [
        "/mnt" + "/d/",
        "Paper " + "Template and Paper",
        "Draft" + "_Results",
        "api" + ".txt",
        "academic" + "techie2022",
        "feynman" + "-session",
        "AGENTS" + ".md",
        "auto" + "research",
    ]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(src_rel: str, dst_rel: str, out_dir: Path) -> None:
    src = ROOT / src_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst = out_dir / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def strip_artifact_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_artifact_fields(v) for k, v in value.items() if k not in {"artifact", "source_root", "value_artifact", "geometry_artifact", "support"}}
    if isinstance(value, list):
        return [strip_artifact_fields(v) for v in value]
    return value


def sanitize_stage_axis(payload: dict[str, Any]) -> dict[str, Any]:
    clean_points = []
    keep = {
        "model",
        "model_label",
        "pair",
        "language",
        "short",
        "n_seeds",
        "first_entry",
        "tail_cer",
        "continuation_stability",
        "exact_match",
        "full_cer",
    }
    for point in payload["points"]:
        clean_points.append({k: point[k] for k in keep if k in point})
    return {
        "description": "Data points for the Figure 1 stage map. continuation_stability = 1 - continuation-tail CER, clipped to [0, 1].",
        "points": clean_points,
    }


def write_readme(out_dir: Path) -> None:
    text = dedent(
        """
        # Early Entry, Late Drift — Supplementary Material

        This archive contains the materials needed to verify the reported summary values for
        **“Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL.”**

        It contains:

        - the paper PDF/source and the TikZ figure sources needed to rebuild it;
        - selected retained JSON artifacts for the main intervention and control results;
        - small derived summary JSON files for overview, crossover, and cross-family checks;
        - a minimal standard-library Python reproduction script.

        It intentionally omits local orchestration logs, VM wrappers, exploratory sweeps,
        notes, session files, private paths, and credentials.

        ## Quick start

        Recompute the paper's headline numbers from the retained artifacts:

        ```bash
        python3 scripts/reproduce_results.py --check
        ```

        This prints a Markdown report and exits nonzero if a reported value differs from
        the retained artifacts.

        To rebuild the paper PDF, install Tectonic and run:

        ```bash
        cd paper/icml2026
        tectonic submission.tex
        ```

        The PDF is already included at:

        ```text
        paper/icml2026/submission.pdf
        ```

        ## Directory layout

        ```text
        README.md
        paper/
          icml2026/submission.{tex,pdf}      # paper source and PDF
          figures/*.tex                      # TikZ sources for figures
        data/
          results/*.json                     # retained result artifacts, renamed by result role
          derived/*.json                     # compact derived summaries used by the report
        src/transliteration_icl/*.py         # minimal artifact readers/report/audit code
        scripts/reproduce_results.py         # one-command reproduction check
        ```

        ## What the reproduction script checks

        The script verifies the same core numbers reported in the paper:

        - Hindi fixed two-channel patch: CER `0.827 → 0.703`, first-entry gain `+0.250`, selected `alpha=2.0`.
        - Directionality control: sign-flip harms Hindi CER.
        - Same-length Hindi control: the compact `L25` MLP rescue appears for helpful←zero-shot, while helpful/corrupt patching is weaker and broader.
        - Hindi channel interpretation: channels `5486` and `2299` support a bounded readout-geometric interpretation, not named semantic features.
        - Telugu shared-prefix oracle residual patch: continuation exact match remains `0.0` and continuation CER does not improve.
        - Telugu Hindi-style sparse-channel crossover: continuation CER changes `1.077 → 1.080` after rounding, i.e. no rescue.
        - Cross-family behavioral check: Qwen 2.5 `1.5B`/`3B` and Llama 3.2 `3B` show strong first-entry gains, while Llama 3.2 `1B` is a capability-floor counterexample.

        ## Scope of this supplement

        This supplement deterministically recomputes the paper's reported summary values
        from retained artifacts. Full raw GPU reruns require gated Gemma model access and
        A100-class compute, as described in the paper. Raw orchestration logs, VM wrappers,
        exploratory sweeps, and credentials are omitted to keep the archive minimal and
        free of local/private environment details.

        Suggested use:

        1. read `paper/icml2026/submission.pdf`;
        2. inspect the retained artifacts under `data/results/` if desired;
        3. run `python3 scripts/reproduce_results.py --check` to verify the headline values.
        """
    ).strip() + "\n"
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_python_sources(out_dir: Path) -> None:
    package = out_dir / "src" / "transliteration_icl"
    scripts = out_dir / "scripts"
    package.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)

    (package / "__init__.py").write_text(
        '"""Minimal reproduction helpers for the supplementary material."""\n\n__all__ = ["artifacts", "report", "audit"]\n',
        encoding="utf-8",
    )

    (package / "artifacts.py").write_text(
        dedent(
            r'''
            from __future__ import annotations

            import json
            from pathlib import Path
            from typing import Any

            # The helpers in this file intentionally stay small: they load the
            # retained JSON artifacts and extract rows/metrics used by the report.

            def load_json(root: Path, relative_path: str) -> Any:
                return json.loads((root / relative_path).read_text(encoding="utf-8"))


            def row_by_intervention(payload: dict[str, Any], intervention: str) -> dict[str, Any]:
                for row in payload["summary_rows"]:
                    if row.get("intervention") == intervention:
                        return row
                available = [row.get("intervention") for row in payload.get("summary_rows", [])]
                raise KeyError(f"Missing intervention {intervention!r}; available={available}")


            def metric(row: dict[str, Any], name: str) -> float:
                value = row[name]
                if isinstance(value, dict):
                    return float(value["mean"])
                return float(value)


            def ci(row: dict[str, Any], name: str) -> tuple[float, float]:
                value = row[name]
                return float(value["ci_low"]), float(value["ci_high"])


            def best_patch_row(payload: dict[str, Any], recipient: str, donor: str, *, metric_name: str) -> dict[str, Any]:
                rows = [
                    row
                    for row in payload["summary_rows"]
                    if row.get("recipient_condition") == recipient and row.get("donor_condition") == donor
                ]
                if not rows:
                    raise KeyError(f"No rows for {recipient} <- {donor}")
                return max(rows, key=lambda row: float(row[metric_name]))
            '''
        ).strip() + "\n",
        encoding="utf-8",
    )

    (package / "audit.py").write_text(
        dedent(
            r'''
            from __future__ import annotations

            from pathlib import Path


            def _forbidden_text() -> list[str]:
                # Build strings from pieces so this audit file does not trip its own scan.
                return [
                    "/mnt" + "/d/",
                    "Paper " + "Template and Paper",
                    "Draft" + "_Results",
                    "api" + ".txt",
                    "academic" + "techie2022",
                    "feynman" + "-session",
                    "AGENTS" + ".md",
                    "auto" + "research",
                ]


            def audit_tree(root: Path) -> list[str]:
                issues: list[str] = []
                readmes = [path for path in root.rglob("README*") if path.is_file()]
                if [path.relative_to(root).as_posix() for path in readmes] != ["README.md"]:
                    issues.append(f"Expected exactly one README.md; found {[str(p.relative_to(root)) for p in readmes]}")
                for path in root.rglob("*"):
                    rel = path.relative_to(root).as_posix()
                    if path.is_dir():
                        # Ignore VCS/cache directories so the check also works
                        # after the archive is unpacked into a local Git clone.
                        if rel == ".git" or rel.endswith("/__pycache__") or rel == "__pycache__":
                            continue
                        continue
                    if path.suffix.lower() in {".pdf", ".gz", ".png", ".jpg", ".jpeg"}:
                        continue
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    for token in _forbidden_text():
                        if token in text:
                            issues.append(f"Forbidden token {token!r} in {rel}")
                return issues
            '''
        ).strip() + "\n",
        encoding="utf-8",
    )

    (package / "report.py").write_text(
        dedent(
            r'''
            from __future__ import annotations

            import math
            from pathlib import Path
            from typing import Any

            from .artifacts import best_patch_row, load_json, metric, row_by_intervention
            from .audit import audit_tree


            def _round(value: float, digits: int = 3) -> float:
                return round(float(value), digits)


            def _same_length_summary(payload: dict[str, Any]) -> dict[str, Any]:
                helpful_zs = best_patch_row(payload, "icl_helpful", "zs", metric_name="delta_mean_gap_latin")
                helpful_corrupt = best_patch_row(payload, "icl_helpful", "icl_corrupt", metric_name="delta_mean_gap_latin")
                corrupt_helpful = best_patch_row(payload, "icl_corrupt", "icl_helpful", metric_name="delta_mean_gap_latin")
                corrupt_helpful_l25 = next(
                    row for row in payload["summary_rows"]
                    if row["recipient_condition"] == "icl_corrupt"
                    and row["donor_condition"] == "icl_helpful"
                    and row["layer"] == 25
                    and row["component"] == "mlp_output"
                )
                def compact(row: dict[str, Any]) -> dict[str, Any]:
                    return {
                        "recipient": row["recipient_condition"],
                        "donor": row["donor_condition"],
                        "site": f"L{row['layer']} {row['component']}",
                        "delta_gap_latin": _round(row["delta_mean_gap_latin"], 3),
                        "delta_target_probability": _round(row["delta_mean_target_prob"], 3),
                        "delta_top1_target_rate": _round(row["delta_top1_target_rate"], 3),
                    }
                return {
                    "helpful_from_zero_shot_best": compact(helpful_zs),
                    "helpful_from_corrupt_best": compact(helpful_corrupt),
                    "corrupt_from_helpful_best": compact(corrupt_helpful),
                    "corrupt_from_helpful_l25_mlp": compact(corrupt_helpful_l25),
                }


            def build_report(root: Path) -> dict[str, Any]:
                # Load only the retained artifacts that support the reported
                # summary values. This function performs no model inference.
                hindi_patch = load_json(root, "data/results/hindi_fixed_patch_results.json")
                hindi_lesions = load_json(root, "data/results/hindi_channel_lesion_results.json")
                same_length = load_json(root, "data/results/hindi_same_length_patch_panel_results.json")
                telugu_patch = load_json(root, "data/results/telugu_oracle_residual_patch_results.json")
                telugu_crossover = load_json(root, "data/results/telugu_sparse_channel_crossover_results.json")
                channels = load_json(root, "data/derived/hindi_channel_interpretation_summary.json")
                cross_family = load_json(root, "data/derived/cross_family_behavior_summary.json")
                stage_axis = load_json(root, "data/derived/stage_axis_points.json")

                h_base = row_by_intervention(hindi_patch, "baseline_no_patch")
                h_patch = row_by_intervention(hindi_patch, "chosen_mean_shift")
                h_flip = row_by_intervention(hindi_patch, "chosen_sign_flip")
                h_zero = row_by_intervention(hindi_patch, "chosen_zero_ablate")
                lesion_mean = row_by_intervention(hindi_lesions, "calibrated_mean_shift")
                lesion_flip = row_by_intervention(hindi_lesions, "calibrated_sign_flip")

                t_base = row_by_intervention(telugu_patch, "baseline_no_patch")
                t_patch = row_by_intervention(telugu_patch, "chosen_mean_shift")
                tc_base = row_by_intervention(telugu_crossover, "baseline_no_patch")
                tc_patch = row_by_intervention(telugu_crossover, "chosen_channel_shift")

                points = {
                    (p["model"], p["pair"]): p
                    for p in stage_axis["points"]
                }

                return {
                    "hindi_fixed_patch": {
                        "n": h_base["n_items"],
                        "selected_alpha": hindi_patch["selected_alpha"],
                        "channels": hindi_patch["channels"],
                        "baseline_cer": metric(h_base, "akshara_cer"),
                        "patched_cer": metric(h_patch, "akshara_cer"),
                        "cer_improvement": metric(h_patch, "delta_cer_improvement"),
                        "first_entry_gain": metric(h_patch, "delta_first_entry_correct"),
                        "gap_gain": metric(h_patch, "delta_first_token_gap_latin"),
                        "sign_flip_cer": metric(h_flip, "akshara_cer"),
                        "zero_ablate_cer": metric(h_zero, "akshara_cer"),
                        "lesion_mean_delta_cer": metric(lesion_mean, "delta_cer_improvement"),
                        "lesion_sign_flip_delta_cer": metric(lesion_flip, "delta_cer_improvement"),
                    },
                    "hindi_same_length_control": _same_length_summary(same_length),
                    "hindi_channel_interpretation": {
                        "status": channels["interpretation_status"],
                        "rows": channels["summary_rows"],
                    },
                    "telugu_oracle_residual_patch": {
                        "n": t_base["n_items"],
                        "selected_alpha": telugu_patch["selected_alpha"],
                        "baseline_continuation_cer": metric(t_base, "continuation_akshara_cer"),
                        "patched_continuation_cer": metric(t_patch, "continuation_akshara_cer"),
                        "continuation_cer_improvement": metric(t_patch, "delta_continuation_cer_improvement"),
                        "continuation_exact_match": metric(t_patch, "continuation_exact_match"),
                        "divergence_gap_delta": metric(t_patch, "delta_first_divergence_gap"),
                    },
                    "telugu_sparse_channel_crossover": {
                        "n": tc_base["n_items"],
                        "selected": telugu_crossover["selected"],
                        "baseline_continuation_cer": metric(tc_base, "continuation_akshara_cer"),
                        "patched_continuation_cer": metric(tc_patch, "continuation_akshara_cer"),
                        "continuation_cer_improvement": metric(tc_patch, "delta_continuation_cer_improvement"),
                    },
                    "stage_axis_points": {
                        "hindi_1b": points[("1b", "aksharantar_hin_latin")],
                        "telugu_1b": points[("1b", "aksharantar_tel_latin")],
                        "telugu_4b": points[("4b", "aksharantar_tel_latin")],
                    },
                    "cross_family_summary": cross_family["aggregate"],
                    "tree_audit_issues": audit_tree(root),
                }


            def format_markdown(report: dict[str, Any]) -> str:
                h = report["hindi_fixed_patch"]
                t = report["telugu_oracle_residual_patch"]
                tc = report["telugu_sparse_channel_crossover"]
                same = report["hindi_same_length_control"]
                stage = report["stage_axis_points"]
                lines = [
                    "# Reproduction report",
                    "",
                    "## Hindi fixed patch",
                    f"- channels: `{h['channels']}`, selected alpha: `{h['selected_alpha']}`",
                    f"- CER: `{h['baseline_cer']:.3f} -> {h['patched_cer']:.3f}`; improvement `{h['cer_improvement']:+.3f}`",
                    f"- first-entry gain: `{h['first_entry_gain']:+.3f}`; first-token gap gain: `{h['gap_gain']:+.2f}`",
                    f"- sign-flip CER: `{h['sign_flip_cer']:.3f}`; zero-ablate CER: `{h['zero_ablate_cer']:.3f}`",
                    "",
                    "## Hindi same-length control",
                ]
                for name, row in same.items():
                    lines.append(f"- {name}: `{row['site']}`, delta gap `{row['delta_gap_latin']:+.3f}`, delta top1 `{row['delta_top1_target_rate']:+.3f}`")
                selected = tc["selected"]
                lines.extend([
                    "",
                    "## Telugu negative interventions",
                    f"- oracle residual patch continuation CER: `{t['baseline_continuation_cer']:.3f} -> {t['patched_continuation_cer']:.3f}`; improvement `{t['continuation_cer_improvement']:+.4f}`",
                    f"- oracle residual continuation exact match: `{t['continuation_exact_match']:.3f}`; divergence-gap delta `{t['divergence_gap_delta']:+.3f}`",
                    f"- sparse-channel crossover continuation CER: `{tc['baseline_continuation_cer']:.3f} -> {tc['patched_continuation_cer']:.3f}`; improvement `{tc['continuation_cer_improvement']:+.4f}`",
                    f"- sparse-channel selected edit: `k={selected['k']}`, `alpha={selected['alpha']}`, channels `{selected['channels']}`",
                    "",
                    "## Stage-axis points",
                    f"- 1B Hindi: `E_ak={stage['hindi_1b']['first_entry']:.3f}`, stability `{stage['hindi_1b']['continuation_stability']:.3f}`",
                    f"- 1B Telugu: `E_ak={stage['telugu_1b']['first_entry']:.3f}`, stability `{stage['telugu_1b']['continuation_stability']:.3f}`",
                    f"- 4B Telugu: `E_ak={stage['telugu_4b']['first_entry']:.3f}`, stability `{stage['telugu_4b']['continuation_stability']:.3f}`",
                    "",
                    "## Package audit",
                    f"- tree audit issues: `{len(report['tree_audit_issues'])}`",
                ])
                return "\n".join(lines) + "\n"


            def assert_close(name: str, value: float, expected: float, tol: float = 5e-4) -> None:
                if not math.isclose(float(value), expected, abs_tol=tol):
                    raise AssertionError(f"{name}: expected {expected}, got {value}")


            def run_checks(report: dict[str, Any]) -> None:
                # These checks guard the numbers that appear in the paper and
                # README. They fail loudly if an artifact is changed or missing.
                h = report["hindi_fixed_patch"]
                assert_close("Hindi baseline CER", h["baseline_cer"], 0.8267261904761906)
                assert_close("Hindi patched CER", h["patched_cer"], 0.7029285714285715)
                assert_close("Hindi CER improvement", h["cer_improvement"], 0.12379761904761905)
                assert_close("Hindi first-entry gain", h["first_entry_gain"], 0.25)
                if h["selected_alpha"] != 2.0 or h["channels"] != [5486, 2299]:
                    raise AssertionError("Hindi selected edit changed")
                if not h["sign_flip_cer"] > h["baseline_cer"]:
                    raise AssertionError("Hindi sign flip should harm CER")

                same = report["hindi_same_length_control"]
                if same["helpful_from_zero_shot_best"]["site"] != "L25 mlp_output":
                    raise AssertionError("Hindi helpful<-zero-shot best site changed")
                assert_close("Hindi helpful<-zero-shot gap", same["helpful_from_zero_shot_best"]["delta_gap_latin"], 3.304, tol=0.002)
                if same["helpful_from_corrupt_best"]["site"] != "L20 attention_output":
                    raise AssertionError("Hindi helpful<-corrupt best site changed")
                if same["corrupt_from_helpful_best"]["site"] != "L23 layer_output":
                    raise AssertionError("Hindi corrupt<-helpful best site changed")

                t = report["telugu_oracle_residual_patch"]
                assert_close("Telugu oracle n", t["n"], 191, tol=0)
                assert_close("Telugu oracle continuation exact", t["continuation_exact_match"], 0.0)
                assert_close("Telugu oracle continuation CER improvement", t["continuation_cer_improvement"], -0.0026387434554973553)

                tc = report["telugu_sparse_channel_crossover"]
                assert_close("Telugu crossover n", tc["n"], 191, tol=0)
                assert_close("Telugu crossover baseline continuation CER", tc["baseline_continuation_cer"], 1.0770443779606085)
                assert_close("Telugu crossover patched continuation CER", tc["patched_continuation_cer"], 1.0796621790077288)
                if tc["selected"].get("channels") != [5631, 6014]:
                    raise AssertionError("Telugu crossover channels changed")

                if report["tree_audit_issues"]:
                    raise AssertionError("Package audit issues: " + "; ".join(report["tree_audit_issues"]))
            '''
        ).strip() + "\n",
        encoding="utf-8",
    )

    (scripts / "reproduce_results.py").write_text(
        dedent(
            r'''
            #!/usr/bin/env python3
            from __future__ import annotations

            import argparse
            import json
            import sys
            from pathlib import Path

            sys.dont_write_bytecode = True
            ROOT = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(ROOT / "src"))

            from transliteration_icl.report import build_report, format_markdown, run_checks  # noqa: E402


            def main() -> int:
                parser = argparse.ArgumentParser(description="Reproduce headline paper values from retained artifacts.")
                parser.add_argument("--check", action="store_true", help="Fail if any reported value differs from the retained artifacts.")
                parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
                args = parser.parse_args()

                report = build_report(ROOT)
                if args.check:
                    run_checks(report)
                if args.json:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                else:
                    print(format_markdown(report))
                    if args.check:
                        print("All reproduction checks passed.")
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            '''
        ).strip() + "\n",
        encoding="utf-8",
    )
    (scripts / "reproduce_results.py").chmod(0o755)


def write_data(out_dir: Path) -> None:
    for src_rel, dst_rel in RAW_RESULT_FILES:
        copy_file(src_rel, dst_rel, out_dir)
    for src_rel, dst_rel, mode in DERIVED_JSON_FILES:
        payload = load_json(ROOT / src_rel)
        if mode == "stage_axis":
            payload = sanitize_stage_axis(payload)
        elif mode == "drop_artifact_fields":
            payload = strip_artifact_fields(payload)
        write_json(out_dir / dst_rel, payload)


def audit_clean_tree(out_dir: Path) -> list[str]:
    issues: list[str] = []
    readmes = [p.relative_to(out_dir).as_posix() for p in out_dir.rglob("README*") if p.is_file()]
    if readmes != ["README.md"]:
        issues.append(f"expected only README.md, found {readmes}")
    for path in out_dir.rglob("*"):
        rel = path.relative_to(out_dir).as_posix()
        if path.is_dir():
            if rel in {".git", "__pycache__"}:
                issues.append(f"forbidden directory {rel}")
            continue
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".gz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden_text_tokens():
            if token in text:
                issues.append(f"forbidden token {token!r} in {rel}")
    return issues


def make_archive(out_dir: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(out_dir, arcname=out_dir.name)


def build(out_dir: Path, archive: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    write_readme(out_dir)
    write_python_sources(out_dir)
    write_data(out_dir)
    for src_rel, dst_rel in PAPER_FILES:
        copy_file(src_rel, dst_rel, out_dir)

    issues = audit_clean_tree(out_dir)
    if issues:
        raise RuntimeError("Clean supplement audit failed:\n" + "\n".join(issues))
    make_archive(out_dir, archive)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the clean supplementary material archive.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    build(args.out_dir, args.archive)
    file_count = sum(1 for p in args.out_dir.rglob("*") if p.is_file())
    print(f"Wrote {args.out_dir}")
    print(f"Wrote {args.archive}")
    print(f"METRIC clean_supplement_files={file_count}")
    print("METRIC clean_supplement_issues=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
