# Reproducing the ICML 2026 workshop submission artifacts

This repository is prepared for double-blind review. **Anonymous review note:** do **not** add an identifying GitHub URL, username, local VM hostname, or author name to the submitted PDF or supplementary archive. If a code link is provided during review, use an anonymous mirror.

The paper is intentionally organized around a small number of reproducibility artifacts. Raw GPU experiments are expensive, so the compact archive includes the final paper source/PDF, generated figure sources, selected retained JSON artifacts for the main causal checks, and concise summaries under `research/submission/`.

## Compact anonymous archive note

The compact archive is designed for review, not for storing every exploratory sweep. It includes the artifacts needed to inspect the main claims, but it does not include bulky raw sweep stores or historical local `outputs/` summaries. Full raw-sweep regeneration requires the original local/VM result store; the review archive instead includes the generated TikZ figures, selected intervention JSON artifacts, and compact summary files.

## Environment

### CPU-only local checks

Use these for paper/package/reproducibility checks in the compact archive:

```bash
python3 -m py_compile \
  experiments/submission_readiness_audit.py \
  experiments/hindi_1b_causal_patch_panel.py \
  experiments/final_submission_audit.py

bash autoresearch.sh final_submission_audit
bash autoresearch.sh readiness_audit
pytest -q \
  tests/test_eval_pipeline.py \
  tests/test_output_extraction.py \
  tests/test_prompt_variants.py \
  tests/test_run_config.py
```

Expected outputs:

- final submission audit JSON printed to stdout with `final_submission_blockers=0`
- readiness audit JSON printed to stdout with `critical_submission_blockers=0`

### GPU experiments

The mechanistic experiments require gated model access for Gemma 3 and a CUDA GPU. They were run on A100-class GPUs. The scripts support both the local/VM launchers in `experiments/run_vm_*.sh` and Modal-style execution if the same command is run inside a GPU container with Hugging Face authentication.

Required model access:

- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`

For Modal, configure an anonymous review-safe workspace snapshot and a secret named `huggingface-secret` exposing `HF_TOKEN`. Do not upload private notes, named logs, or non-anonymous remotes in a review artifact.

## Paper build

Public anonymous source:

```text
paper/icml2026/submission.tex
```

Public anonymous PDF:

```text
paper/icml2026/submission.pdf
```

Build and metadata-sanitize the public paper package:

```bash
bash autoresearch.sh package_submission_artifacts
```

Or build manually with Tectonic:

```bash
cd paper/icml2026
tectonic submission.tex
```

Expected output:

```text
paper/icml2026/submission.pdf
```

The local working-copy source is mirrored from `Paper Template and Paper/Paper/icml2026/gemma_1b_icl_paper_submission_8plus2_icml2026_final.tex`, but reviewers should use the clean `paper/` tree.

## Figure and table mapping

### Figure 1: overview / stage map

The left panel is a compact static TikZ overview linking behavioral regime, tested state handle, and intervention outcome:

```text
paper/figures/fig_stage_intervention_overview_tikz.tex
```

The data-driven right panel is included as generated TikZ:

```text
paper/figures/fig_stage_axis_map_tikz.tex
```

Full raw-sweep regeneration of the right panel requires the broad multi-seed result store, which is not included in the compact archive. The included TikZ source is the review artifact.

Expected headline values include `1B Hindi` at roughly `(E_ak=0.40, stability=0.22)`, `1B Telugu` at roughly `(0.67, 0.23)`, and `4B Telugu` at roughly `(0.98, 0.64)`.

### Figure 2: behavioral regime heatmap

The paper uses the retained TikZ figure:

```text
paper/figures/fig_behavioral_regime_summary_tikz_v19.tex
```

The source values are baked into the retained TikZ source. Full raw-sweep regeneration requires the broad multi-language result store, which is not included in the compact archive.

Optional regeneration path when the full result store is available:

```bash
uv run --with altair --with pandas --with vl-convert-python \
  python3 experiments/plot_final_paper_figures.py
```

### Table 1: stage-sensitive diagnostic axes

Table 1 is derived from matched seed-42 30-item diagnostic panels. The compact archive includes the paper table and generated figure sources; full regeneration of the broad diagnostic panel requires the raw sweep store, which is not included in the compact archive.

### Figure 3: Hindi fixed intervention

Main retained artifact:

```text
research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json
```

Related intervention/lesion artifact:

```text
research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json
```

Expected headline:

- baseline CER ≈ `0.827`
- chosen fixed patch CER ≈ `0.703`
- sign-flip CER ≈ `0.929`
- first-entry delta ≈ `+0.250`

The retained JSON artifact above is included in the compact archive and is the source for the reported held-out patch values.

### Hindi channel interpretation / readout geometry

Minimal channel-characterization summary:

```text
research/submission/hindi_channel_interpretation_summary_2026-04-28.md
```

Regenerate it from retained artifacts:

```bash
python3 experiments/summarize_hindi_channel_interpretation.py
```

Inputs:

- `research/results/autoresearch/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json`
- `research/results/autoresearch/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json`

Expected headline: channels `5486` and `2299` are not named as semantic features. The defensible characterization is readout-geometric: helpful high-shot prompts increase both coordinates relative to zero-shot, and moving them back toward the zero-shot mean improves the gold-first-token-vs-Latin margin.

### Mechanistic crossover check

The bounded intervention-family crossover summary is:

```text
research/submission/mechanistic_crossover_summary_2026-04-28.md
```

Regenerate it after fetching Modal artifacts:

```bash
python3 experiments/summarize_mechanistic_crossover.py
```

Telugu compact-channel crossover Modal run:

```bash
# Full run was launched detached to survive local session interruptions.
CROSSOVER_MODAL_DETACH=1 bash autoresearch.sh telugu_mlp_crossover_full

# Fetch after Modal completion.
bash autoresearch.sh telugu_mlp_crossover_fetch
```

Expected headline: the Hindi fixed two-channel edit improves held-out full CER, while both tested Telugu static edit families fail to improve continuation CER under shared-prefix conditioning. The sparse Telugu crossover changes continuation CER from about `1.077` to `1.080`, i.e. no rescue.

### Telugu oracle / negative intervention result

Main retained artifact:

```text
research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json
```

Expected headline:

- usable oracle-conditioned items: `191 / 200`
- continuation exact match remains `0.0`
- continuation CER improvement for the chosen mean shift is approximately `-0.003`

The retained JSON artifact above is included in the compact archive and is the source for the reported Telugu negative-intervention values.

### Appendix: cross-family checks and prompt composition

Cross-family synthesis:

```text
research/submission/cross_model_behavioral_synthesis_2026-04-28.md
```

Regenerate the bounded cross-model analysis table from retained artifacts:

```bash
python3 experiments/summarize_cross_model_behavioral.py
```

Expected headline: Qwen 2.5 `1.5B`/`3B` and Llama 3.2 `3B` show large helpful-prompt first-entry gains on both Hindi and Telugu, Telugu remains harder on continuation-sensitive proxies, and Llama 3.2 `1B` is a capability-floor counterexample. Treat these rows as behavioral robustness evidence, not shared-circuit evidence. The compact archive includes the synthesis summary, not the full cross-model raw result store.

## Same-length Hindi helpful/corrupt control hook

The Hindi causal patch panel supports explicit same-length prompt-state patch pairs:

```bash
PATCH_PAIRS=default,same_length \
PATCH_POSITION_MODE=last_token \
RESULTS_ROOT_NAME=hindi_patch_panel_same_length_v1 \
bash experiments/run_vm_hindi_1b_patch_panel.sh
```

Equivalent direct command inside any authenticated CUDA environment:

```bash
python3 experiments/hindi_1b_causal_patch_panel.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --seed 42 \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 200 \
  --max-items 30 \
  --layers 20,23,24,25 \
  --components layer_output,attention_output,mlp_output \
  --patch-position-mode last_token \
  --patch-pairs default,same_length \
  --device cuda \
  --out research/results/autoresearch/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64
```

This control is designed to test whether the Hindi localization is only a prompt-length artifact. The paper should not overclaim this control unless the corresponding artifact is included and summarized.

## Anonymous code artifact check

Before uploading to an anonymous review mirror such as anonymous.4open.science, run:

```bash
bash autoresearch.sh anonymous_repo_audit
```

Expected output:

```text
METRIC anonymous_repo_blockers=0
METRIC anonymous_repo_warnings=0
```

For the supplementary-material upload, prefer the curated clean package:

```bash
bash autoresearch.sh package_clean_supplement
```

This writes:

```text
/tmp/early_entry_late_drift_supplementary_material.tar.gz
```

The curated package contains one README, the paper source/PDF, selected retained JSON artifacts under professional names, derived summaries, and a minimal standard-library reproduction script. It omits exploratory sweeps, local orchestration wrappers, notes, and internal logs.

A broader tracked-tree export can still be made for internal archival purposes:

```bash
git archive --format=tar.gz --output /tmp/transliteration-icl-anonymous.tar.gz HEAD
bash autoresearch.sh final_submission_audit
bash autoresearch.sh public_archive_surface_audit
```

Do not upload local `.git/` metadata, private Modal profile names, local VM addresses, Hugging Face tokens, or named paper artifacts outside an anonymous mirror.

## End-to-end sanity check

Run the compact submission-safety audits:

```bash
bash autoresearch.sh readiness_audit
bash autoresearch.sh workshop_risk_audit
bash autoresearch.sh anonymous_repo_audit
bash autoresearch.sh channel_characterization_audit
bash autoresearch.sh framing_calibration_audit
```

The audits are not scientific proofs. They check for common submission hazards: missing claim ledger, missing data-driven overview map, missing reproduction README, missing same-length/crossover summaries, under-characterized Hindi channels, framing overclaim, and obvious identity leaks.
