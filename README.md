# Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL

Research code for the paper **"Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL"** (ICML 2026 Workshop on Compositional Learning).

## Paper

| File | Description |
|------|-------------|
| `paper/icml2026/submission.pdf` | Workshop submission PDF |
| `paper/icml2026/submission.tex` | LaTeX source |
| `paper/complearn2026/submission.pdf` | CompLearn-framed variant PDF |
| `paper/figures/` | TikZ figure sources |

## Main Results

- **Hindi (Gemma 3 1B):** early target-entry failure. A fixed two-channel MLP patch reduces held-out CER from **0.827 → 0.703** on 200 evaluation items.
- **Telugu (Gemma 3 1B):** later continuation-drift failure. No improvement found within the tested full-state mean-shift family (CER change ≈ −0.003).
- **Takeaway:** stage-specific diagnosis matters — the same intervention family that rescues Hindi does not rescue Telugu.

## Repository Layout

```
experiments/
  hindi/              Hindi-specific experiment scripts
  telugu/             Telugu-specific experiment scripts
  figures/            Figure generation utilities
  summaries/          Summary regeneration scripts (read retained artifacts)
research/
  modules/
    behavior/         ICL variant construction (helpful, random, shuffled, corrupted)
    data/             Data loaders and Aksharantar split utilities
    eval/             Metrics (CER, exact-match), output extraction, judge wrapper
    infra/            Run configuration loader
    modal/            Modal-based batch inference packet
    prompts/          Prompt template library (canonical, output_only, task_tagged)
  config/             JSON run configurations
  results/            Retained JSON artifacts for the main causal checks
  submission/         Compact paper-support summaries
lib/                  Shared model-patching utilities, data pipeline, and
                      behavioral-control run protocol
tests/                Deterministic correctness checks (no GPU required)
paper/                Anonymous paper source, PDF, and TikZ figures
```

## Requirements

**Deterministic tests and figure utilities — no GPU:**

```
Python >= 3.10   (standard library only for tests)
```

**GPU experiments:**

```
torch >= 2.1
transformers >= 4.40
```

With gated Hugging Face access to `google/gemma-3-1b-it` and `google/gemma-3-4b-it`. Experiments were run on A100-class GPUs.

**Optional figure regeneration** (`plot_final_paper_figures.py`): `altair`, `pandas`, `vl-convert-python`.

## Quick Check

Run the deterministic test suite (no GPU, no external dependencies):

```bash
pytest -q \
  tests/test_eval_pipeline.py \
  tests/test_output_extraction.py \
  tests/test_prompt_variants.py \
  tests/test_run_config.py
```

## Reproducing Results

See [`README_REPRODUCE.md`](README_REPRODUCE.md) for the figure-by-figure command map, including retained artifact paths and GPU rerun commands.

## Experiment Scripts

Scripts are organized by language and purpose under `experiments/`:

**Hindi (`experiments/hindi/`)**

| Script | Description |
|--------|-------------|
| `practical_patch_eval.py` | Held-out two-channel patch evaluation — main result (Figure 3) |
| `causal_patch_panel.py` | Causal patch sweep across layers, components, and positions |
| `intervention_eval.py` | Channel intervention and lesion analysis |
| `channel_value_stats.py` | MLP channel value statistics for identified channels |
| `channel_readout_geometry.py` | Readout-geometry analysis for channels 5486 and 2299 |
| `mlp_channel_panel.py` | Shared channel-selection and patching utilities (used by other Hindi scripts) |

**Telugu (`experiments/telugu/`)**

| Script | Description |
|--------|-------------|
| `practical_patch_eval.py` | Oracle negative-result evaluation (continuation-drift) |
| `mlp_channel_crossover.py` | MLP-channel crossover experiment |
| `modal_crossover.py` | Modal-based full crossover sweep (GPU cloud) |

**Figures (`experiments/figures/`)**

| Script | Description |
|--------|-------------|
| `render_stage_axis_map.py` | Regenerate Figure 1b TikZ (requires full sweep store) |
| `plot_paper_figures.py` | Regenerate all paper figures (requires full sweep store + altair) |

**Summaries (`experiments/summaries/`)**

| Script | Description |
|--------|-------------|
| `hindi_channels.py` | Regenerate Hindi channel interpretation summary from retained artifacts |
| `telugu_crossover.py` | Regenerate Telugu mechanistic crossover summary from retained artifact |
| `cross_model_behavioral.py` | Regenerate cross-family behavioral synthesis (requires full cross-model store) |

## Data

Transliteration data (Aksharantar pairs) is included under `lib/data/transliteration/` for Hindi, Bengali, Marathi, Tamil, and Telugu (Latin-script targets). The full Aksharantar dataset is available at [AI4Bharat/Aksharantar](https://github.com/AI4Bharat/Aksharantar).

## Retained Artifacts

Seven JSON result files from the main causal experiments are included under `research/results/`. These are sufficient to verify the paper's headline CER values without re-running the GPU experiments.

## Anonymity

This repository is prepared for double-blind review. Do not add an identifying GitHub URL, username, or institution to any submitted artifact. Use an anonymous mirror (e.g., anonymous.4open.science) if a code link is required in the submission.
