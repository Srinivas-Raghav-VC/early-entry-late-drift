# Mechanistic crossover summary (2026-04-28)

The crossover matrix is meant to reduce the apples-to-oranges objection. It does not prove a universal early-vs-late law; it records which edit families rescue or fail under held-out evaluation.

| language | stage | edit family | n | status | primary before→after | delta |
|---|---|---|---:|---|---:|---:|
| Hindi | early target entry | fixed sparse MLP-channel shift | 200 | works | 0.827→0.703 | +0.124 [+0.087,+0.163] |
| Telugu | late continuation | shared-prefix full-residual mean shift | 191 | no rescue | 1.074→1.076 | -0.003 [-0.008,+0.000] |
| Telugu | late continuation | fixed sparse MLP-channel shift | 191 | no rescue | 1.077→1.080 | -0.003 [-0.008,+0.000] |

Notes:
- Hindi / fixed sparse MLP-channel shift: Held-out fixed two-channel shift improves full-word CER and first-entry metrics; sign flip harms in the source artifact.
- Telugu / shared-prefix full-residual mean shift: Oracle-conditioned static residual shift does not improve continuation generation under this tested edit family.
- Telugu / fixed sparse MLP-channel shift: The selected Hindi-style compact channel shift does not rescue Telugu continuation; the selected split gain was near zero and held-out continuation CER slightly worsened.
