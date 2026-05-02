# Cross-model behavioral synthesis (2026-04-28)

Scope: single seed-42, 200-item, 64-shot non-Gemma behavioral checks. These checks are behavioral robustness evidence, not mechanistic sharing evidence.

## Summary table

| model | lang | first-entry zs→help | EM help−corrupt | CER corrupt−help | fuzzy bank help |
|---|---|---:|---:|---:|---:|
| qwen2.5-1.5b | Hindi | 0.285→0.930 | +0.170 | +0.216 | 0.085 |
| qwen2.5-1.5b | Telugu | 0.005→0.995 | +0.110 | +0.391 | 0.120 |
| qwen2.5-3b | Hindi | 0.215→0.965 | +0.150 | +0.139 | 0.025 |
| qwen2.5-3b | Telugu | 0.000→0.900 | +0.115 | +0.243 | 0.065 |
| llama3.2-1b | Hindi | 0.000→0.005 | +0.000 | +0.002 | 0.000 |
| llama3.2-1b | Telugu | 0.450→0.005 | +0.000 | +0.002 | 0.000 |
| llama3.2-3b | Hindi | 0.010→0.995 | +0.060 | +0.053 | 0.065 |
| llama3.2-3b | Telugu | 0.060→1.000 | +0.065 | +0.238 | 0.130 |

## Interpretation

The non-Gemma rows support the behavioral decomposition but also bound it. Qwen 2.5 1.5B/3B and Llama 3.2 3B mostly solve target entry under helpful prompting, while Telugu remains materially harder end-to-end and retains nonzero fuzzy bank-copy. Llama 3.2 1B is a useful counterexample: it fails both languages, so the Gemma 1B Hindi/Telugu split should not be described as universal.

## Claim boundaries

### Established
- Several non-Gemma instruction models show large helpful-prompt improvements in first-entry on Hindi/Telugu.
- Llama 3.2 1B is a capability-floor counterexample rather than a replication of the Gemma 1B split.

### Supported But Provisional
- The early-entry vs continuation-difficulty decomposition is not obviously Gemma-only.
- Telugu remains harder end-to-end than first-entry accuracy alone suggests in stronger non-Gemma models.

### Not Claimed
- No shared circuit or shared channel mechanism across model families.
- No multi-seed cross-family estimate; these are single-seed robustness checks.

