# RP 2-D Decoder Family & Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two new signed-RP decoders (RP-GEO-2D, RP-DIRECT-2D) plus their shared 2-D overlap-based reconstruction, and run a reconstruction-quality ablation against the current RP-Jaccard-1D baseline to determine which RP decoder is most interesting.

**Architecture:** All variants share one parametrised code path — `distance_matrix(source)` → `embed_2d` → either 1-D projection (+mean-append, legacy) or 2-D O(2) overlap alignment (+overlap reconstruction +signed selection). Variants differ only in the distance source fed to MDS and in the recovery mode. A single ablation runner enumerates the variant configs over one fold's windows and emits a Markdown+JSON report with fidelity + geometry metrics. Nothing touches the committed pipeline defaults — the current decoder (`method="1d"`) stays the pipeline default; the new family is opt-in and evaluated by a standalone script, exactly like the existing `scripts/_ablation_rp_recon.py`.

**Tech Stack:** Python 3.10/3.11, NumPy, SciPy (`scipy.sparse.csgraph.dijkstra`), OpenCV (`cv2`) for image loading. No new dependencies. No pytest — verification is via assert-based `scripts/_check_*.py` self-checks run under the WSL venv.

## Global Constraints

- **No new dependencies.** Reuse `_classical_mds`, `varRP2`, `weigthed_graphRP`, `calculate_shortest_path_matrix` from `src/recurrence_plots.py`.
- **Non-destructive.** Do not change any pipeline default, any committed `.npy`, any `results/**` JSON, or the `method="1d"` path. New work is opt-in and read-only w.r.t. existing artefacts.
- **States convention:** `s_i = (x_i, x_{i+1})`, RP is `(N-1) × (N-1)`, reconstruction returns `N` samples. WISDM `TIME_STEPS = 128`.
- **No pytest.** Tests are assert-based self-checks in `scripts/_check_*.py`, invoked with `python -m scripts.<name>`.
- **Run environment (established this session):** the WSL venv with NumPy is `/home/adrianogarcia/tfg/.venv/bin/python`, but the edited `src/` lives on the Windows checkout. Always run from the `/mnt/c` checkout root so cwd wins on `sys.path`:
  ```bash
  wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts.<name> <args>"
  ```
- **Import convention:** `from src.X import Y`, `from scripts.X import Y` (editable install; never sys.path-hack).
- **Default operating point:** `rho = 0.60` → `PERCENTILE = 60.0` (matches `scripts/_ablation_rp_recon.py`).
- **`eps = 1e-10`** for all normalised-error denominators.

**Variant matrix (the 6-row ablation from the spec):**

| Name | source | dims | recover | signed |
|---|---|---:|---|---|
| RP-Jaccard-1D (baseline) | jaccard | 1 | append-mean | yes |
| RP-GEO-1D | geo | 1 | append-mean | yes |
| RP-GEO-2D | geo | 2 | overlap | no |
| RP-DIRECT-2D | direct | 2 | overlap | no |
| RP-DIRECT-2D-S | direct | 2 | overlap | yes |
| RP-DIRECT-2D-SR *(optional, YAGNI)* | direct | 2 | overlap+refine | yes |

Variant 6 (`-SR`, non-linear refinement) is **out of scope for the first pass** — the spec itself marks it optional. Add it only if the ablation shows `-S` is the winner and a reviewer asks for the iterative refinement. It is not planned below.

---

### Task 1: Distance sources + 2-D embedding + geometry metrics

**Files:**
- Create: `src/rp_decoders.py`
- Create: `scripts/_check_rp_decoders.py`
- Reference (read, do not modify): `src/recurrence_plots.py:431-515` (`CreateCostMatrix`, `weigthed_graphRP`, `calculate_shortest_path_matrix`, `_classical_mds`, `reconstruct_time_series`)

**Interfaces:**
- Consumes: `src.recurrence_plots._classical_mds`, `weigthed_graphRP`, `calculate_shortest_path_matrix`, `CreateCostMatrix`.
- Produces:
  - `symmetrized_magnitude(R_obs: np.ndarray) -> np.ndarray` — `(M,M)`, `D_ij=(|R_ij|+|R_ji|)/2`, `D_ii=0`.
  - `distance_matrix(R_obs: np.ndarray, source: str, rho: float = 0.60) -> np.ndarray` — `source in {"direct","geo","jaccard"}`, returns the `(M,M)` matrix fed to MDS.
  - `embed_2d(dmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — returns `(Y, pos_eigvals)` where `Y` is `(M,2)` and `pos_eigvals` is the descending array of positive eigenvalues of the double-centred matrix.
  - `evr2(pos_eigvals: np.ndarray) -> float` — `(λ1+λ2)/Σλ⁺`.
  - `mds_stress(dmat: np.ndarray, Y: np.ndarray) -> float` — Kruskal stress-1.

- [ ] **Step 1: Write the failing self-check**

Create `scripts/_check_rp_decoders.py`:

```python
"""Self-check for the 2-D RP decoder family (src/rp_decoders.py).

Assert-based (no pytest). Run:
    python -m scripts._check_rp_decoders
"""
import numpy as np

from src.rp_decoders import (
    distance_matrix,
    embed_2d,
    evr2,
    mds_stress,
    symmetrized_magnitude,
)


def _abs_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return abs(float(a @ b) / d) if d > 0 else 0.0


# --- Task 1: distance sources, embedding, geometry metrics ---
t = np.linspace(0.0, 4.0 * np.pi, 128)
x = np.sin(t) + 0.3 * np.sin(3.0 * t)
states = np.stack([x[:-1], x[1:]], axis=1)                       # (127, 2)
D_true = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=2)

# signed RP with the L2 magnitude and an arbitrary sign pattern (quantisation-free)
R_obs = D_true.copy()

D = symmetrized_magnitude(R_obs)
assert D.shape == (127, 127) and np.allclose(np.diag(D), 0.0)
assert np.allclose(D, D.T), "symmetrised magnitude must be symmetric"

for src in ("direct", "geo", "jaccard"):
    dm = distance_matrix(R_obs, source=src, rho=0.60)
    assert dm.shape == (127, 127), (src, dm.shape)

Y, ev = embed_2d(distance_matrix(R_obs, source="direct"))
assert Y.shape == (127, 2)
assert evr2(ev) > 0.99, f"EVR2 for a true 2-D config should be ~1, got {evr2(ev):.4f}"
assert mds_stress(distance_matrix(R_obs, source="direct"), Y) < 1e-6, "direct MDS stress must be ~0"
print("[ok] Task 1: distance sources + embedding + geometry metrics")
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `ModuleNotFoundError: No module named 'src.rp_decoders'`.

- [ ] **Step 3: Implement `src/rp_decoders.py` (Task 1 portion)**

```python
"""Two-dimensional signed-RP decoder family and reconstruction metrics.

All variants share this module. They differ only in the distance matrix handed
to MDS (``distance_matrix(source=...)``) and in the recovery mode (1-D
projection + mean-append, legacy; or 2-D O(2) overlap alignment). The current
pipeline decoder (recurrence_plots.reconstruct_time_series + fix_rotationscale)
is unaffected and remains the default; this module is opt-in and evaluated by
scripts/_ablation_rp_2d.py.

State convention: s_i = (x_i, x_{i+1}); RP is (N-1)x(N-1); reconstruction -> N.
"""
import numpy as np

from src.recurrence_plots import (
    CreateCostMatrix,
    _classical_mds,
    calculate_shortest_path_matrix,
    weigthed_graphRP,
)

EPS = 1e-10


def symmetrized_magnitude(R_obs):
    """(M,M) signed RP -> symmetric magnitude matrix with zero diagonal.

    PNG quantisation can make R_ij and R_ji differ slightly; averaging the two
    magnitudes restores the symmetry the encoder guarantees in principle.
    """
    A = np.abs(np.asarray(R_obs, dtype=float))
    D = 0.5 * (A + A.T)
    np.fill_diagonal(D, 0.0)
    return D


def _continuous_geodesic(D, rho):
    """Continuous-weighted graph geodesic: keep edges below the rho-quantile of
    the off-diagonal magnitudes, weight each retained edge by its magnitude
    D_ij, then take all-pairs shortest paths. Mirrors the Jaccard path's
    graph->shortest-path structure but preserves the observed distances."""
    iu = np.triu_indices_from(D, k=1)
    tau = np.quantile(D[iu], rho)
    W = np.where(D <= tau, D, 0.0)   # 0 = no edge (dijkstra treats 0 as absent for sparse)
    np.fill_diagonal(W, 0.0)
    return calculate_shortest_path_matrix(W)


def distance_matrix(R_obs, source, rho=0.60):
    """Build the (M,M) matrix fed to MDS.

    source="direct" : symmetric magnitudes (== L2 state distances, up to noise)
    source="geo"    : continuous-weighted graph geodesic distances
    source="jaccard": legacy Jaccard-dissimilarity graph geodesic (baseline)
    """
    D = symmetrized_magnitude(R_obs)
    if source == "direct":
        return D
    if source == "geo":
        return _continuous_geodesic(D, rho)
    if source == "jaccard":
        # weigthed_graphRP expects the signed RP and applies its own |.| + threshold.
        return calculate_shortest_path_matrix(weigthed_graphRP(R_obs, 100.0 * rho))
    raise ValueError(f"unknown source: {source!r}")


def embed_2d(dmat):
    """Classical MDS to 2-D. Returns (Y (M,2), positive eigenvalues descending)."""
    n = dmat.shape[0]
    D2 = np.square(dmat)
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = eigvals[eigvals > 0]
    L = np.clip(eigvals[:2], a_min=0.0, a_max=None)
    Y = eigvecs[:, :2] * np.sqrt(L)
    return Y, pos


def evr2(pos_eigvals):
    """Fraction of geometry explained by the two leading positive dimensions."""
    s = float(np.sum(pos_eigvals))
    if s <= 0:
        return 0.0
    return float(np.sum(pos_eigvals[:2]) / s)


def mds_stress(dmat, Y):
    """Kruskal stress-1 between input distances and embedded distances."""
    iu = np.triu_indices_from(dmat, k=1)
    d = dmat[iu]
    dh = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)[iu]
    denom = float(np.sum(d ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(np.sum((d - dh) ** 2) / denom))
```

Note on `embed_2d` vs `_classical_mds`: `embed_2d` reimplements the double-centring so it can also return the positive-eigenvalue spectrum for `evr2`. It is intentionally the same math as `recurrence_plots._classical_mds(D, 2)`; the Task 2 check asserts they agree up to an orthogonal transform.

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `[ok] Task 1: distance sources + embedding + geometry metrics`

- [ ] **Step 5: Commit**

```bash
git add src/rp_decoders.py scripts/_check_rp_decoders.py
git commit -m "feat(rp): distance sources + 2-D embedding + geometry metrics"
```

---

### Task 2: O(2) overlap alignment + N-sample reconstruction

**Files:**
- Modify: `src/rp_decoders.py` (append functions)
- Modify: `scripts/_check_rp_decoders.py` (append Task 2 checks)

**Interfaces:**
- Consumes: `embed_2d` (Task 1).
- Produces:
  - `align_states(Y: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]` — apply orthogonal `Q`, δ-offset correct, return `(a, b, E_overlap_norm)` where `a`,`b` are `(M,)`.
  - `orientation_candidates(n_theta: int = 720) -> list[np.ndarray]` — the `2*n_theta` matrices `Q_{θ,η}` spanning O(2).
  - `reconstruct_signal_from_states(a: np.ndarray, b: np.ndarray) -> np.ndarray` — `(M,)`,`(M,)` → `(M+1,)` via the overlap average.

- [ ] **Step 1: Append the failing self-check**

Append to `scripts/_check_rp_decoders.py`:

```python
# --- Task 2: overlap alignment + reconstruction (geometric core, all sources) ---
from src.rp_decoders import (
    align_states,
    orientation_candidates,
    reconstruct_signal_from_states,
)

Qs = orientation_candidates(n_theta=720)
assert len(Qs) == 1440
assert all(np.allclose(Q.T @ Q, np.eye(2), atol=1e-9) for Q in Qs), "all Q must be orthogonal"

for src in ("direct", "geo"):
    Y, _ = embed_2d(distance_matrix(R_obs, source=src))
    best = min((align_states(Y, Q) for Q in Qs), key=lambda ab: ab[2])
    a, b, e_ov = best
    xhat = reconstruct_signal_from_states(a, b)
    assert xhat.shape == (128,), (src, xhat.shape)
    r = _abs_corr(xhat, x)
    assert r > 0.99, f"[{src}] geometric-core corr too low: {r:.4f}"
    assert e_ov < 1e-3, f"[{src}] overlap error should be ~0 on clean input: {e_ov:.2e}"
print("[ok] Task 2: overlap alignment + reconstruction")
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `ImportError: cannot import name 'align_states'` (Task 1 line still prints `[ok]` first).

- [ ] **Step 3: Append the implementation to `src/rp_decoders.py`**

```python
def orientation_candidates(n_theta=720):
    """The 2*n_theta matrices Q_{θ,η} = rot(θ)·diag(1,η), η∈{+1,-1}, spanning O(2)."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    Qs = []
    for eta in (1.0, -1.0):
        for th in thetas:
            c, s = np.cos(th), np.sin(th)
            Qs.append(np.array([[c, -s * eta], [s, c * eta]]))
    return Qs


def align_states(Y, Q):
    """Apply Q, correct the MDS translation offset along the overlap, and return
    aligned state coordinates (a, b) plus the normalised overlap error.

    a_i approximates x_i, b_i approximates x_{i+1}; the overlap constraint is
    b_i ≈ a_{i+1}. δ is the mean offset between the two, split symmetrically.
    """
    Z = Y @ Q                                   # (M, 2)
    d = Z[:-1, 1] - Z[1:, 0]                     # b_i - a_{i+1}, i=0..M-2
    delta = -float(np.mean(d)) if d.size else 0.0
    a = Z[:, 0] - 0.5 * delta
    b = Z[:, 1] + 0.5 * delta
    num = float(np.sum((b[:-1] - a[1:]) ** 2))
    den = float(np.sum(a ** 2 + b ** 2)) + EPS
    return a, b, num / den


def reconstruct_signal_from_states(a, b):
    """Overlap reconstruction: x1=a1; x_i=(b_{i-1}+a_i)/2; xN=b_{M}. Length M+1."""
    M = a.shape[0]
    x = np.empty(M + 1)
    x[0] = a[0]
    x[1:M] = 0.5 * (b[:-1] + a[1:])
    x[M] = b[M - 1]
    return x
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: both `[ok] Task 1 ...` and `[ok] Task 2 ...`.

- [ ] **Step 5: Commit**

```bash
git add src/rp_decoders.py scripts/_check_rp_decoders.py
git commit -m "feat(rp): O(2) overlap alignment + N-sample reconstruction"
```

---

### Task 3: Signed-orientation selection + unified single-channel decoder

**Files:**
- Modify: `src/rp_decoders.py` (append)
- Modify: `scripts/_check_rp_decoders.py` (append Task 3 checks)
- Reference (read): `src/recurrence_plots.py:546-589` (`fix_rotationscale`), `src/recurrence_plots.py:24-56` (`varRP2`)

**Interfaces:**
- Consumes: `distance_matrix`, `embed_2d`, `orientation_candidates`, `align_states`, `reconstruct_signal_from_states`; `recurrence_plots.varRP2`, `recurrence_plots.reconstruct_time_series`, `recurrence_plots.fix_rotationscale`.
- Produces:
  - `signed_rp_error(R_obs: np.ndarray, x_scaled: np.ndarray) -> float` — normalised Frobenius error `‖R_obs − varRP2(x_scaled)‖_F / (‖R_obs‖_F + eps)`.
  - `decode_channel(R_obs, minmax_c, source, dims=2, recover="overlap", signed=True, rho=0.60, n_theta=720, top_k=8) -> np.ndarray` — returns the reconstructed channel of length `TIME_STEPS` in physical units, matching the axis convention of `recurrence_plots.Reconstruct_RP` outputs.

`minmax_c` is `(max_c, min_c)` for the channel, matching the per-channel entries that `_ablation_rp_recon._minmax` produces.

- [ ] **Step 1: Append the failing self-check**

Append to `scripts/_check_rp_decoders.py`:

```python
# --- Task 3: signed selection + unified single-channel decoder ---
from src.rp_decoders import decode_channel, signed_rp_error

# Build a clean signed RP (physical units) for one channel from a known signal.
from src.recurrence_plots import varRP2
xc = 2.0 * np.sin(t) - 0.5 * np.cos(2.0 * t)      # length 128
R_signed = varRP2(xc, TIME_STEPS=128)              # (127,127) signed, physical units
minmax_c = (float(xc.max()), float(xc.min()))

for src in ("direct", "geo"):
    rec = decode_channel(R_signed, minmax_c, source=src, dims=2,
                         recover="overlap", signed=True)
    assert rec.shape == (128,), (src, rec.shape)
    r = _abs_corr(rec, xc)
    assert r > 0.95, f"[{src}] signed 2-D decode corr too low: {r:.4f}"

# signed error must be small for the true signal, large for a shuffled one.
e_true = signed_rp_error(R_signed, np.interp(xc, (xc.min(), xc.max()), (xc.min(), xc.max())))
e_bad = signed_rp_error(R_signed, np.random.default_rng(0).permutation(xc))
assert e_true < e_bad, f"signed error not discriminative: {e_true:.3f} !< {e_bad:.3f}"

# 1-D legacy path still reachable through the unified entry.
rec1 = decode_channel(R_signed, minmax_c, source="jaccard", dims=1,
                      recover="append", signed=True)
assert rec1.shape == (128,)
print("[ok] Task 3: signed selection + unified decoder")
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `ImportError: cannot import name 'decode_channel'`.

- [ ] **Step 3: Append the implementation to `src/rp_decoders.py`**

```python
from src.recurrence_plots import (          # placed with the other imports at top of file
    fix_rotationscale,
    reconstruct_time_series,
    varRP2,
)

TIME_STEPS = 128


def signed_rp_error(R_obs, x_scaled):
    """Normalised Frobenius distance between the observed signed RP and the
    signed RP regenerated from a candidate (physical-unit) reconstruction."""
    Rhat = varRP2(np.asarray(x_scaled, dtype=float), TIME_STEPS=len(x_scaled))
    num = float(np.linalg.norm(R_obs - Rhat))
    den = float(np.linalg.norm(R_obs)) + EPS
    return num / den


def _rescale(x, minmax_c):
    """Map a unit-free reconstruction onto the channel's physical [min,max]."""
    mx, mn = minmax_c
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < EPS:
        return np.full_like(x, 0.5 * (mn + mx))
    return np.interp(x, (lo, hi), (mn, mx))


def decode_channel(R_obs, minmax_c, source, dims=2, recover="overlap",
                   signed=True, rho=0.60, n_theta=720, top_k=8):
    """Unified single-channel RP decoder. Returns a length-TIME_STEPS channel in
    physical units, axis-compatible with recurrence_plots.Reconstruct_RP output.

    dims=1, recover="append": legacy 1-D projection + fix_rotationscale (baseline
      and RP-GEO-1D). dims=2, recover="overlap": 2-D O(2) alignment + overlap
      reconstruction; signed=True adds the signed-RP orientation selection.
    """
    dmat = distance_matrix(R_obs, source=source, rho=rho)

    if dims == 1:
        # Reuse the legacy 1-D machinery: project to the max-variance MDS axis,
        # then mean-append + (optional) signed flip. reconstruct_time_series does
        # its own double-centring on the passed matrix, so hand it `dmat`.
        proj = reconstruct_time_series(dmat, ep=0.0)          # (M,1)
        rec, _ = fix_rotationscale(R_obs, proj, 0, [ [minmax_c[0]], [minmax_c[1]] ],
                                   TIME_STEPS=TIME_STEPS,
                                   use_orientation=signed, append_value="mean")
        return np.asarray(rec)[:TIME_STEPS]

    # 2-D path.
    Y, _ = embed_2d(dmat)
    cands = []
    for Q in orientation_candidates(n_theta=n_theta):
        a, b, e_ov = align_states(Y, Q)
        cands.append((e_ov, a, b))
    cands.sort(key=lambda c: c[0])

    if not signed:
        _, a, b = cands[0]
        x = reconstruct_signal_from_states(a, b)
        return _rescale(x, minmax_c)[:TIME_STEPS]

    # Signed selection (lambda-free lexicographic): among the top_k lowest
    # overlap-error orientations, pick the one whose regenerated signed RP is
    # closest to the observed one.
    # ponytail: top_k prefilter caps the O(n_theta) signed-RP regenerations at
    # top_k per channel; widen top_k only if the winner is unstable across folds.
    best = None
    for e_ov, a, b in cands[:top_k]:
        x = _rescale(reconstruct_signal_from_states(a, b), minmax_c)[:TIME_STEPS]
        e_sig = signed_rp_error(R_obs, x)
        if best is None or e_sig < best[0]:
            best = (e_sig, x)
    return best[1]
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `[ok] Task 1 ...`, `[ok] Task 2 ...`, `[ok] Task 3 ...`.

- [ ] **Step 5: Commit**

```bash
git add src/rp_decoders.py scripts/_check_rp_decoders.py
git commit -m "feat(rp): signed-orientation selection + unified single-channel decoder"
```

---

### Task 4: Reconstruction/overlap metrics + 3-channel window decode

**Files:**
- Modify: `src/rp_decoders.py` (append)
- Modify: `scripts/_check_rp_decoders.py` (append Task 4 checks)

**Interfaces:**
- Consumes: `decode_channel`, `signed_rp_error`, `symmetrized_magnitude`, `align_states`, `embed_2d`, `distance_matrix`, `evr2`, `mds_stress`.
- Produces:
  - `overlap_error_abs(a: np.ndarray, b: np.ndarray) -> float` — `mean|b_i − a_{i+1}|` (the L1 diagnostic from the spec's "métricas internas").
  - `decode_window(img_rgb: np.ndarray, minmax, source, dims=2, recover="overlap", signed=True, rho=0.60) -> tuple[np.ndarray, dict]` — returns `(rec (TIME_STEPS,3), metrics_dict)` where `metrics_dict` has keys `e_rp`, `evr2`, `stress` averaged over the 3 channels. `minmax` is `[[max0,max1,max2],[min0,min1,min2]]` as produced by `_ablation_rp_recon._minmax`. `img_rgb` is the decoded RGB image `(M,M,3)` uint8.

- [ ] **Step 1: Append the failing self-check**

Append to `scripts/_check_rp_decoders.py`:

```python
# --- Task 4: metrics + 3-channel window decode ---
from src.rp_decoders import decode_window, overlap_error_abs
from src.recurrence_plots import NormalizeMatrix

sig = {0: x, 1: np.cos(t), 2: 0.5 * np.sin(2.0 * t)}
chans, maxs, mins = [], [], []
for c in range(3):
    xs = sig[c]
    Rn = NormalizeMatrix(varRP2(xs, TIME_STEPS=128))
    chans.append(np.round(Rn * 255).astype(np.uint8))
    maxs.append(float(xs.max())); mins.append(float(xs.min()))
img = np.stack(chans, axis=2)                     # (127,127,3)
minmax = [maxs, mins]

rec, metrics = decode_window(img, minmax, source="direct", dims=2,
                             recover="overlap", signed=True)
assert rec.shape == (128, 3), rec.shape
assert np.all(np.isfinite(rec))
assert set(metrics) == {"e_rp", "evr2", "stress"}
assert 0.0 <= metrics["evr2"] <= 1.0

ab = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 9.0])  # last b intentionally off
assert abs(overlap_error_abs(*ab) - (abs(2-2)+abs(3-3))/2) < 1e-12
print("[ok] Task 4: metrics + 3-channel window decode")
print("all checks passed")
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: `ImportError: cannot import name 'decode_window'`.

- [ ] **Step 3: Append the implementation to `src/rp_decoders.py`**

```python
def overlap_error_abs(a, b):
    """L1 overlap diagnostic: mean|b_i - a_{i+1}| over shared samples."""
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(b[:-1] - a[1:])))


def decode_window(img_rgb, minmax, source, dims=2, recover="overlap",
                  signed=True, rho=0.60):
    """Decode a 3-channel RP image to (TIME_STEPS,3) plus geometry metrics.

    Metrics (averaged over channels): e_rp (signed-RP reproduction error),
    evr2 (fraction of geometry in 2 dims), stress (MDS stress-1).
    """
    maxs, mins = minmax
    chans, e_rp, evrs, stresses = [], [], [], []
    for c in range(3):
        R_obs = np.interp(img_rgb[:, :, c].astype(float), (0, 255), (mins[c], maxs[c]))
        rec = decode_channel(R_obs, (maxs[c], mins[c]), source=source, dims=dims,
                             recover=recover, signed=signed, rho=rho)
        chans.append(rec)
        e_rp.append(signed_rp_error(R_obs, rec))
        dmat = distance_matrix(R_obs, source=source, rho=rho)
        Y, pos = embed_2d(dmat)
        evrs.append(evr2(pos))
        stresses.append(mds_stress(dmat, Y))
    rec = np.stack(chans, axis=1)                       # (TIME_STEPS, 3)
    metrics = dict(e_rp=float(np.mean(e_rp)),
                   evr2=float(np.mean(evrs)),
                   stress=float(np.mean(stresses)))
    return rec, metrics
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._check_rp_decoders"
```
Expected: all four `[ok]` lines + `all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rp_decoders.py scripts/_check_rp_decoders.py
git commit -m "feat(rp): reconstruction metrics + 3-channel window decode"
```

---

### Task 5: Ablation runner over one fold + Markdown/JSON report

**Files:**
- Create: `scripts/_ablation_rp_2d.py`
- Reference (read, mirror): `scripts/_ablation_rp_recon.py` (data loading, `_minmax`, `_pearson`)

**Interfaces:**
- Consumes: `src.rp_decoders.decode_window`, `src.rp_decoders.overlap_error_abs`, `src.config.DEFAULT_DATA_DIR`.
- Produces: writes `results/_stats/rp_2d_ablation.md` and `results/_stats/rp_2d_ablation.json`. No importable API.

- [ ] **Step 1: Write the runner (this task's deliverable IS the script; its "test" is a small-N smoke run)**

Create `scripts/_ablation_rp_2d.py`:

```python
#!/usr/bin/env python3
"""RP decoder ablation: which RP decoder reconstructs best?

Runs the variant matrix over LOTO fold-0 test windows at rho=0.60 and reports
pointwise fidelity (MAE, median MAE, MSE, Pearson) plus geometry diagnostics
(E_RP, E_overlap, EVR2, MDS stress) and per-window wall-clock. Reconstruction
only — classifier accuracy is a separate, optional phase (see plan Task 6).

    python -m scripts._ablation_rp_2d --n 120
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_DATA_DIR
from src.rp_decoders import decode_window, distance_matrix, embed_2d, align_states, orientation_candidates

VARIANTS = {
    "RP-Jaccard-1D": dict(source="jaccard", dims=1, recover="append", signed=True),
    "RP-GEO-1D":     dict(source="geo",     dims=1, recover="append", signed=True),
    "RP-GEO-2D":     dict(source="geo",     dims=2, recover="overlap", signed=False),
    "RP-DIRECT-2D":  dict(source="direct",  dims=2, recover="overlap", signed=False),
    "RP-DIRECT-2D-S":dict(source="direct",  dims=2, recover="overlap", signed=True),
}
PERCENTILE_RHO = 0.60


def _minmax(window):
    return [[float(np.max(window[:, c])) for c in range(3)],
            [float(np.min(window[:, c])) for c in range(3)]]


def _pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _overlap_for_window(img, minmax, cfg):
    """E_overlap (L1) for a 2-D variant; N/A (nan) for 1-D variants."""
    if cfg["dims"] != 2:
        return np.nan
    maxs, mins = minmax
    vals = []
    for c in range(3):
        R_obs = np.interp(img[:, :, c].astype(float), (0, 255), (mins[c], maxs[c]))
        Y, _ = embed_2d(distance_matrix(R_obs, source=cfg["source"], rho=PERCENTILE_RHO))
        cands = [align_states(Y, Q) for Q in orientation_candidates()]
        a, b, _ = min(cands, key=lambda ab: ab[2])
        vals.append(float(np.mean(np.abs(b[:-1] - a[1:]))))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    base = f"{DEFAULT_DATA_DIR}tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    plots = f"{DEFAULT_DATA_DIR}plots/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    orig = np.load(base + "validation_data.npy")
    y_val = np.load(base + "y_validation_data.npy")
    sj_val = np.load(base + "sj_validation_data.npy")
    m = min(args.n, len(orig))

    acc = {k: dict(mae=[], mse=[], pear=[], e_rp=[], e_ov=[], evr2=[], stress=[], t=[]) for k in VARIANTS}

    for i in range(m):
        label = int(np.argmax(y_val[i]))
        sj = sj_val[i][0]
        img = cv2.imread(f"{plots}{label}/{sj}x{i}.png")
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mm = _minmax(orig[i])
        for name, cfg in VARIANTS.items():
            t0 = time.perf_counter()
            rec, metrics = decode_window(img, mm, source=cfg["source"], dims=cfg["dims"],
                                         recover=cfg["recover"], signed=cfg["signed"],
                                         rho=PERCENTILE_RHO)
            dt = time.perf_counter() - t0
            a = acc[name]
            a["mae"].append(float(np.mean(np.abs(orig[i] - rec))))
            a["mse"].append(float(np.mean((orig[i] - rec) ** 2)))
            a["pear"].append(_pearson(orig[i], rec))
            a["e_rp"].append(metrics["e_rp"])
            a["e_ov"].append(_overlap_for_window(img, mm, cfg))
            a["evr2"].append(metrics["evr2"])
            a["stress"].append(metrics["stress"])
            a["t"].append(dt)

    def agg(v):
        v = np.asarray(v, dtype=float)
        return float(np.nanmean(v)) if v.size else float("nan")

    rows = {}
    for name, a in acc.items():
        rows[name] = dict(
            MAE=agg(a["mae"]), MedianMAE=float(np.nanmedian(a["mae"])) if a["mae"] else float("nan"),
            MSE=agg(a["mse"]), Pearson=agg(a["pear"]), E_RP=agg(a["e_rp"]),
            E_overlap=agg(a["e_ov"]), EVR2=agg(a["evr2"]), Stress=agg(a["stress"]),
            Time_ms=1000.0 * agg(a["t"]))

    out_dir = Path("results/_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rp_2d_ablation.json").write_text(
        json.dumps(dict(n_windows=m, rho=PERCENTILE_RHO, variants=rows), indent=2))

    cols = ["MAE", "MedianMAE", "MSE", "Pearson", "E_RP", "E_overlap", "EVR2", "Stress", "Time_ms"]
    lines = [f"# RP 2-D decoder ablation (LOTO fold-0, {m} windows, rho=0.60)", "",
             "| Method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, r in rows.items():
        cells = [name] + [("N/A" if np.isnan(r[c]) else f"{r[c]:.4f}") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (out_dir / "rp_2d_ablation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a tiny N to verify it executes and writes both files**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._ablation_rp_2d --n 4"
```
Expected: a Markdown table printed with 5 variant rows, and `results/_stats/rp_2d_ablation.{md,json}` written. `RP-Jaccard-1D` shows `E_overlap = N/A`. If the data paths are missing (fold not generated on this machine), regenerate the LOTO 3-fold RP artefacts first with `python -m scripts.generate_images --image-type RP --sampling loto --n-folds 3 --create-numpies` and `python -m scripts.generate_time_series --image-type RP --sampling loto --n-folds 3`.

- [ ] **Step 3: Full ablation run**

Run:
```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales && /home/adrianogarcia/tfg/.venv/bin/python -m scripts._ablation_rp_2d --n 400"
```
Expected: fidelity numbers for all 5 variants. Sanity: `RP-DIRECT-2D-S` should have the lowest MAE if the encoder is L2 (the spec's hypothesis); `RP-Jaccard-1D` reproduces the existing baseline MAE from `scripts/_ablation_rp_recon.py` (`full` config) within noise.

- [ ] **Step 4: Commit**

```bash
git add scripts/_ablation_rp_2d.py results/_stats/rp_2d_ablation.md results/_stats/rp_2d_ablation.json
git commit -m "feat(rp): 2-D decoder ablation runner + fidelity/geometry report"
```

---

### Task 6 *(optional — decide after Task 5 results)*: Classification-utility column

Only run this if the fidelity ablation leaves the "most interesting" decoder genuinely ambiguous and downstream KNN accuracy is needed to break the tie. It is heavier: it reconstructs the full fold per candidate variant and evaluates KNN via the existing `evaluate_run` machinery.

**Files:**
- Modify: `scripts/_ablation_rp_2d.py` (add `--knn` flag)
- Reference (read): `src/evaluator.py` (`evaluate_run`, `_load_reconstruction`, axis normalisation), `scripts/knnvalidation.py` (adapter construction), `CLAUDE.md` "Validator architecture".

**Approach (documented, not fully coded here because it depends on Task 5 findings):**
- For each candidate variant, decode the entire fold to an `X_all_<stype>`-shaped array `(N, TIME_STEPS, 3)` and persist it to a scratch path (do NOT overwrite the committed `tseries/.../X_all_*.npy`).
- Reuse the KNN `ClassifierAdapter` and call `evaluate_run` pointing at the scratch reconstruction, with `--force` into a scratch `results/` prefix so no committed JSON is clobbered.
- Add the resulting accuracy as the `Accuracy_KNN` column to `rp_2d_ablation.{md,json}`.

- [ ] **Step 1:** Read `src/evaluator.py` and `scripts/knnvalidation.py`; confirm the exact adapter signature and the reconstruction-array axis contract (`_load_reconstruction` normalises `(N,C,T)`→`(N,T,C)`).
- [ ] **Step 2:** Implement `--knn` in `scripts/_ablation_rp_2d.py` per the approach above, writing recon + results under a scratch prefix.
- [ ] **Step 3:** Run `... -m scripts._ablation_rp_2d --n 400 --knn` and confirm the `Accuracy_KNN` column appears and the committed `tseries/`/`results/` trees are untouched (`git status` clean except the ablation report).
- [ ] **Step 4:** Commit.

---

### Task 7 *(paper)*: "Alternative signed-RP reconstruction strategies" section

**Files:**
- Modify: `doc/main.tex` (add subsection after `sec:rp_reconstruction_method`, and the comparison table)
- Verify: `python scripts/_check_refs.py`; rebuild with `/compile-latex-wsl`.

**Interfaces:** none (LaTeX prose). The section text is supplied verbatim in the user's spec ("Texto para incluir en el artículo" → *Alternative signed-RP reconstruction strategies*). Fill the comparison table (`Método | MAE | Mediana MAE | MSE | Pearson | Error RP | Error overlap | Accuracy KNN | Tiempo`) from `results/_stats/rp_2d_ablation.json` produced in Task 5 (+ Task 6 for the KNN column).

- [ ] **Step 1:** Paste the spec's English subsection text after `sec:rp_reconstruction_method`. Add the four internal-metric definitions (E_RP, E_overlap, EVR2, Stress) where the section references them.
- [ ] **Step 2:** Insert the comparison table populated from `rp_2d_ablation.json` (leave the KNN/Tiempo cells only if Task 6 ran; otherwise state the table is reconstruction-fidelity only and mark KNN as future work).
- [ ] **Step 3:** Reconcile the existing claim: the earlier "future work" framing of the 2-D decoder ([main.tex:530](../../../doc/main.tex)) must now say it is *evaluated* here, and the novelty-table cell should reflect the winning variant.
- [ ] **Step 4:** Run `python scripts/_check_refs.py` → `BROKEN: (none)`; then `/compile-latex-wsl` → `EXIT=0`.
- [ ] **Step 5:** Commit.

---

## Self-Review

**1. Spec coverage:**
- Two new decoders RP-GEO-2D / RP-DIRECT-2D → Task 1 (`distance_matrix` sources) + Tasks 2–4 (shared 2-D path). ✓
- Symmetrised magnitude `D_ij` → Task 1 `symmetrized_magnitude`. ✓
- Continuous-weighted geodesic vs direct → Task 1 `distance_matrix(source=...)`. ✓
- O(2) alignment with δ-offset + normalised overlap error → Task 2 `align_states`. ✓
- N-sample overlap reconstruction → Task 2 `reconstruct_signal_from_states`. ✓
- Signed-orientation selection (λ-free lexicographic default; λ-combined noted as optional) → Task 3 `decode_channel`. ✓
- Internal metrics E_RP, E_overlap, EVR2, Stress → Tasks 1 & 4. ✓
- Comparison table + recommended ablation (variants 1–5) → Task 5; variant 6 `-SR` explicitly deferred (YAGNI, spec-optional). ✓
- Accuracy KNN + Tiempo columns → Time in Task 5, KNN in optional Task 6. ✓
- Paper section (user-supplied text) → Task 7. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Task 6 is intentionally approach-level (depends on Task 5 outcome) and is flagged optional, not a hidden placeholder in the core path.

**3. Type consistency:** `decode_channel`/`decode_window` signatures match between definition (Tasks 3–4) and callers (Task 5). `minmax` layout `[[maxs],[mins]]` is consistent with `_ablation_rp_recon._minmax` and with `decode_window`'s unpacking. `align_states` returns `(a,b,E)` used identically in Task 2 check, Task 3 `decode_channel`, and Task 5 `_overlap_for_window`. `orientation_candidates()` returns `2*n_theta` matrices — asserted in Task 2, consumed in Tasks 3 & 5.

**Ponytail notes:** single module + single runner (no per-variant duplication); the O(2) alignment uses a θ-grid (closed form noted as upgrade path); signed selection is prefiltered to `top_k` overlap candidates (ceiling named); variant 6 dropped until proven necessary; classifier eval deferred behind a flag so the fast fidelity answer comes first.
