# RP closed-form production snapshot

Generated: 2026-07-29

This manifest identifies the external `X_all_rec.npy` arrays used for the
refreshed RP fidelity and classification results. They were generated from the
working-tree production path `Reconstruct_RP(method="direct2ds")` with
metadata-free diagonal centring, two-dimensional MDS, relative-translation
alignment, closed-form orientation and scale-invariant signed selection.

The numerical fold summaries are in `rp_closed_form_regen.json`. The earlier
grid/operator-range snapshot is separately documented in
`results/_baseline_grid720/PROVENANCE.md`; the two snapshots must not be treated
as a one-factor orientation experiment.

| protocol | fold | shape | SHA-256 |
|---|---:|---|---|
| LOTO | 0 | `(2724, 3, 128)` | `772e52f3f4beabd2bb6e567a796dd0f3a6360df80b32a88fbae035c09bfcf71a` |
| LOTO | 1 | `(2724, 3, 128)` | `93f18c418aa29b01d926f57018ac64a62a1a93b7753b42e01fd0ea6cec9ea1b4` |
| LOTO | 2 | `(2724, 3, 128)` | `a5f4061ae1e2aa32b6188681f6e25a1a807d3c2b414d0dff98733785dace88e9` |
| LOSO | 0 | `(2645, 3, 128)` | `43afd4d397a5bfab374fa1d419d0b2d1ccdbdba0671d3b2f39dd31104097da6f` |
| LOSO | 1 | `(2788, 3, 128)` | `dde4a3b5ccc1d8f222038581efec42655e7679da771c99a49d3a0ba1c7122dcb` |
| LOSO | 2 | `(2739, 3, 128)` | `87742dbc816f9a06d17756ee27a53c8506833dc5b6ce3581eedb9ff85d71dbf2` |

The two protocols contain the same 8,172 windows under different fold
partitions, so the six rows contain 16,344 stored entries but only 8,172 unique
source windows.
