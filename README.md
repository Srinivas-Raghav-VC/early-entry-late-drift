# Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL

Research code for the paper **"Early Entry, Late Drift: Stage-Specific Failure and Intervention in Multilingual Transliteration ICL"**.

## Start here

- `paper/icml2026/submission.pdf` — anonymous workshop submission PDF
- `paper/icml2026/submission.tex` — clean public paper source
- `README_REPRODUCE.md` — claim-to-command reproduction map
- `experiments/hindi_1b_practical_patch_eval.py` — held-out Hindi patch evaluation
- `experiments/telugu_continuation_practical_patch_eval.py` — held-out Telugu continuation patch evaluation
- `experiments/reverify_end_to_end_artifacts.py` — end-to-end claim recheck script
- `tests/` — small objective checks for the reusable stack

## Main result shape

- Hindi behaves like an **early target-start failure** and a fixed two-channel held-out patch reduces CER from **0.827 to 0.703** on the main 200-item evaluation.
- Telugu behaves like a **later continuation-drift failure** and does **not** show rescue within the tested full-state mean-shift family.
- The paper's top-level framing is therefore a **multilingual behavioral regime map plus two bounded causal/internal-state case studies**, not a single universal mechanism claim.

## Repo layout

- `experiments/` — paper-facing experiment and analysis scripts
- `research/modules/` — reusable data, prompt, eval, and infra helpers
- `tests/` — lightweight checks
- `Draft_Results/` — small retained legacy-support subset still used by active scripts
- `research/submission/` — compact paper-support summaries
- `research/results/autoresearch/` — selected retained JSON artifacts needed by the reproduction map
- `research/` — project notes, specs, and journals

## Quick check

```bash
pytest -q \
  tests/test_eval_pipeline.py \
  tests/test_output_extraction.py \
  tests/test_prompt_variants.py \
  tests/test_run_config.py
```

## Note

Large raw result sweeps, local paper build trees, caches, and generated logs are intentionally not tracked in Git. The anonymous public tree keeps only the compact paper source/PDF, selected JSON artifacts, and scripts needed to audit the claims.
