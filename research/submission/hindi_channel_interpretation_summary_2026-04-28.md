# Hindi channel interpretation summary (2026-04-28)

Status: **supported but bounded**. The evidence below characterizes a local readout direction; it does not name a monosemantic feature or a complete circuit.

Artifacts:

- `research/results/autoresearch/hindi_channel_value_audit_v1/1b/aksharantar_hin_latin/nicl64/hindi_1b_channel_value_audit.json`
- `research/results/autoresearch/hindi_channel_readout_geometry_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_channel_readout_geometry_audit.json`

## Main claim

Channels 5486 and 2299 are not assigned human-semantic labels. They are a small high-shot state/readout direction: helpful prompts move the coordinates upward relative to zero-shot, and because their down-projection columns have negative local dot product with the target-vs-Latin readout gradient, moving those coordinates back toward zero-shot increases the gold-first-token margin.

## Extracted rows

| channel | helpful − zs value | grad·column | predicted Δgap | actual singleton Δgap | corr(pred, actual) | Latin-collapse actual Δgap |
|---:|---:|---:|---:|---:|---:|---:|
| 5486 | +11.14 | -0.212 | +2.36 | +2.38 | 0.97 | +2.48 |
| 2299 | +8.00 | -0.233 | +1.87 | +1.64 | 0.89 | +1.70 |

Interpretation: helpful high-shot prompting increases both coordinates by roughly 8--11 units relative to zero-shot. The local readout gradient is negative along both down-projection columns, so the zero-shot-minus-helpful shift has a positive first-order effect on the target-vs-Latin margin. The first-order estimate closely tracks actual singleton channel replacement, especially for channel 5486.

## Item-level characterization

This is still **not** a semantic feature label. The item-level audit is useful mainly for ruling in a prompt-state/readout interpretation and ruling out a clean monosemantic story. Channel 2299 is more associated with Latin top-1 pressure than 5486; channel 5486 is broader and appears in both base-success and Latin-collapse contexts.

| channel | corr(value, target−Latin gap) | corr(value, Latin top-1) | largest-value contexts |
|---:|---:|---:|---|
| 5486 | -0.22 | +0.16 | aadarahi→आदरही (base_success, top1=आ); aachhya→आछ्या (latin_collapse, top1=a); aagyakarita→आज्ञाकारिता (base_success, top1=आ) |
| 2299 | -0.61 | +0.51 | aachhya→आछ्या (latin_collapse, top1=a); aabharit→आभरित (base_success, top1=आ); aaeeeesee→आईईसी (latin_collapse, top1=a) |
