# Reproducing Paper Results

This file maps each paper figure and table to the commands and artifacts needed to inspect or regenerate it.

The archive includes the final paper source/PDF, TikZ figure sources, selected retained JSON artifacts for the main causal checks, and concise summaries under `research/submission/`. Large raw sweep stores are intentionally excluded; the retained artifacts are sufficient to verify the reported headline values.

## Quick verification (no GPU required)

```bash
pytest -q \
  tests/test_eval_pipeline.py \
  tests/test_output_extraction.py \
  tests/test_prompt_variants.py \
  tests/test_run_config.py
```

All four test modules pass with no external dependencies beyond the standard library.

## GPU experiment environment

The mechanistic experiments require CUDA and gated model access.

**Models required** (request access on Hugging Face):
- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`

**Hardware:** experiments were run on A100-class GPUs.

**Authentication:** set `HF_TOKEN` in your environment or use `huggingface-cli login`.

---

## Figure 1: behavioral stage map

### Left panel (static TikZ overview)

```text
paper/figures/fig_stage_intervention_overview_tikz.tex
```

### Right panel (data-driven stage map)

The TikZ source is included as a retained artifact:

```text
paper/figures/fig_stage_axis_map_tikz.tex
```

Expected headline values: `1B Hindi` at roughly `(E_ak=0.40, stability=0.22)`, `1B Telugu` at `(0.67, 0.23)`, `4B Telugu` at `(0.98, 0.64)`.

The generated TikZ is the retained artifact — it is already included. Full regeneration from the raw sweep store (not included in this archive) can be triggered with:

```bash
python3 experiments/figures/render_stage_axis_map.py
```

Running this script in the compact archive prints a notice and exits without overwriting anything.

---

## Figure 2: behavioral regime heatmap

```text
paper/figures/fig_behavioral_regime_summary_tikz_v19.tex
```

Source values are baked into the TikZ. Full regeneration requires the broad multi-language result store (not included).

Optional regeneration when the full result store is available:

```bash
uv run --with altair --with pandas --with vl-convert-python \
  python3 experiments/figures/plot_paper_figures.py
```

---

## Table 1: stage-sensitive diagnostic axes

Table 1 is derived from matched seed-42 30-item diagnostic panels. Full regeneration requires the raw sweep store (not included). The paper table values are in `paper/icml2026/submission.tex`.

---

## Figure 3: Hindi fixed-patch intervention

**Main artifact:**

```text
research/results/hindi_practical_patch_eval_review200_v1/
  1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json
```

**Intervention/lesion artifact:**

```text
research/results/hindi_intervention_eval_review200_v1/
  1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json
```

Expected headline values:
- baseline CER ≈ `0.827`
- chosen fixed patch CER ≈ `0.703`
- sign-flip CER ≈ `0.929`
- first-entry delta ≈ `+0.250`

**Regenerate** (requires GPU + gated model access):

```bash
python3 experiments/hindi/practical_patch_eval.py \
  --model 1b \
  --pair aksharantar_hin_latin \
  --seed 42 \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 200 \
  --device cuda \
  --out research/results/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64
```

---

## Hindi channel interpretation

**Summary:**

```text
research/submission/hindi_channel_interpretation_summary_2026-04-28.md
```

**Regenerate from retained artifacts** (no GPU required):

```bash
python3 experiments/summaries/hindi_channels.py
```

Inputs:
- `research/results/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json`
- `research/results/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json`

Expected headline: channels `5486` and `2299` are characterized as readout-geometric, not semantic. Helpful high-shot prompts increase both coordinates relative to zero-shot; moving them back toward the zero-shot mean improves the gold-first-token-vs-Latin margin.

---

## Telugu mechanistic crossover check

**Summary:**

```text
research/submission/mechanistic_crossover_summary_2026-04-28.md
```

**Main artifact:**

```text
research/results/telugu_mlp_channel_crossover_v1/
  1b/aksharantar_tel_latin/seed42/nicl64/telugu_1b_mlp_channel_crossover.json
```

**Regenerate from retained artifacts** (no GPU required):

```bash
python3 experiments/summaries/telugu_crossover.py
```

Expected headline: the Hindi fixed two-channel edit improves held-out full CER; both tested Telugu static edit families fail to improve continuation CER. The sparse Telugu crossover changes continuation CER from about `1.077` to `1.080` — no rescue.

**Full rerun** (requires GPU + Modal):

```bash
modal run experiments/telugu/modal_crossover.py \
  --max-items 200 \
  --select-max-items 100 \
  --k-grid 2,4,8,16,32,64,128 \
  --alpha-grid 0.25,0.5,1.0,1.5,2.0 \
  --n-random 3 \
  --results-name telugu_mlp_channel_crossover_v1
```

---

## Telugu oracle / negative intervention result

**Main artifact:**

```text
research/results/telugu_continuation_practical_patch_eval_review200_v1/
  1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json
```

Expected headline:
- usable oracle-conditioned items: `191 / 200`
- continuation exact match: `0.0`
- continuation CER improvement for the chosen mean shift: approximately `-0.003`

**Regenerate** (requires GPU + gated model access):

```bash
python3 experiments/telugu/practical_patch_eval.py \
  --model 1b \
  --pair aksharantar_tel_latin \
  --seed 42 \
  --n-icl 64 \
  --n-select 300 \
  --n-eval 200 \
  --device cuda \
  --out research/results/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64
```

---

## Appendix: cross-family behavioral checks

**Summary:**

```text
research/submission/cross_model_behavioral_synthesis_2026-04-28.md
```

The pre-computed summary is the retained artifact — it is already included. The raw per-model result store is not included in this archive. Running the summarize script in the compact archive prints a notice and exits without overwriting anything:

```bash
python3 experiments/summaries/cross_model_behavioral.py
```

Expected headline: Qwen 2.5 1.5B/3B and Llama 3.2 3B show large helpful-prompt first-entry gains on both Hindi and Telugu. Llama 3.2 1B is a capability-floor counterexample. These rows are behavioral robustness evidence, not shared-circuit evidence.

---

## Hindi same-length patch control

```bash
python3 experiments/hindi/causal_patch_panel.py \
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
  --out research/results/hindi_patch_panel_same_length_v1/1b/aksharantar_hin_latin/nicl64
```

This control tests whether the Hindi localization is a prompt-length artifact. The retained artifact at `research/results/hindi_patch_panel_same_length_v1/...` is included.

---

## Paper source

```bash
cd paper/icml2026
tectonic submission.tex
```

Produces `paper/icml2026/submission.pdf`. The pre-built PDF is already included.
