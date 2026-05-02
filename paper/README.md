# Paper artifact

This directory is the clean anonymous paper package for review.

- `icml2026/submission.pdf` — built, metadata-sanitized primary Mech Interp submission PDF.
- `icml2026/submission.tex` — canonical anonymous LaTeX source for the primary submission.
- `complearn2026/submission.pdf` — built, metadata-sanitized CompLearn-framed variant.
- `complearn2026/submission.tex` — anonymous LaTeX source for the CompLearn-framed variant.
- `figures/` — only the figure sources needed by the paper sources.

## Submission format decision

Use the current artifact as the **long paper** submission. Do not compress to a
short paper unless the venue explicitly requires it: the controls, claim ledger,
bounded negative result, and appendix summaries are part of the evidence that
makes the paper reviewable. A short paper would be cleaner-looking but would cut
against the methodological safeguards.

## Build and upload checklist

From the repository root, always use the packaging command before upload:

```bash
bash autoresearch.sh package_submission_artifacts
```

This rebuilds `paper/icml2026/submission.pdf` and strips nonessential PDF
metadata. For the CompLearn-framed variant, run:

```bash
bash autoresearch.sh package_complearn_submission_artifacts
```

This writes `paper/complearn2026/submission.pdf`. A raw manual `tectonic
submission.tex` build is useful for debugging, but it repopulates PDF build
metadata and should not be the final upload step.

Before uploading the PDF, run:

```bash
bash autoresearch.sh final_submission_audit
```

For the supplementary code/material upload, prefer the curated clean package over
a raw repository archive:

```bash
bash autoresearch.sh package_clean_supplement
```

This writes:

```text
/tmp/early_entry_late_drift_supplementary_material.tar.gz
```

That archive contains one README, the paper source/PDF, selected retained JSON
artifacts under professional names, and a minimal standard-library Python
reproduction script. It intentionally omits exploratory local working files,
exploratory sweeps, local orchestration wrappers, notes, and internal logs.

Use `paper/icml2026/submission.pdf` as the Mech Interp paper upload. Use
`paper/complearn2026/submission.pdf` as the CompLearn paper upload. Use the
curated tarball above as the supplementary material/code artifact if the target
OpenReview form provides a supplementary-material field; otherwise use an
anonymous repository initialized from the extracted clean supplement.
