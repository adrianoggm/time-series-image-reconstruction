"""Self-check for the PRODUCTION RP decoder path: recurrence_plots.Reconstruct_RP
with method="direct2ds" (DIRECT-2D-S), metadata-free (rp_range=None -> diagonal
centring) and with closed-form orientation. Also checks the operator-range oracle
(reconstruct_tseries._build_rp_range) stays reachable for the ablation.

Synthetic-only (no pipeline data needed). Assert-based, no pytest. Run:
    python -m scripts._check_rp_production
"""
import numpy as np

from src.recurrence_plots import NormalizeMatrix, Reconstruct_RP, varRP2
from src.reconstruct_tseries import _build_rp_range

TIME_STEPS = 128


def _abs_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return abs(float(a @ b) / d) if d > 0 else 0.0


def main():
    t = np.linspace(0.0, 4.0 * np.pi, TIME_STEPS)

    # 1. Known 3-channel window WITH a DC offset per channel, so the signal-range
    #    scale bug (input decode range = signal amplitude instead of RP-matrix
    #    range) would manifest as an affine distortion if it were still present.
    w = np.empty((TIME_STEPS, 3))
    w[:, 0] = np.sin(t) + 5.0
    w[:, 1] = np.cos(t) + 3.0
    w[:, 2] = 0.5 * np.sin(2.0 * t) + 8.0
    print("[ok] built synthetic 3-channel window with DC offsets", w.shape)

    # 2. Encode to a uint8 RP image exactly as the encoder does: per channel
    #    NormalizeMatrix(varRP2(window)) -> round(*255) -> stack.
    chans = []
    for c in range(3):
        Rc = varRP2(w[:, c], TIME_STEPS=TIME_STEPS)
        chans.append(np.round(NormalizeMatrix(Rc) * 255).astype(np.uint8))
    img = np.stack(chans, axis=2)  # (TIME_STEPS-1, TIME_STEPS-1, 3)
    assert img.shape == (TIME_STEPS - 1, TIME_STEPS - 1, 3), img.shape
    assert img.dtype == np.uint8
    print("[ok] encoded synthetic window to uint8 RP image", img.shape)

    # 3. dictionary = [[maxs],[mins]] (signal range, for OUTPUT rescale). Production
    #    needs NOTHING else: rp_range=None -> diagonal centring.
    dictionary = {
        0: [
            [float(w[:, c].max()) for c in range(3)],
            [float(w[:, c].min()) for c in range(3)],
        ]
    }
    print("[ok] built dictionary (signal range only — no RP-matrix metadata)")

    # 4. Production DIRECT-2D-S decode through Reconstruct_RP (the same entry point
    #    and the same arguments reconstruct_tseries._process_sample now passes:
    #    method="direct2ds", rp_range=None -> metadata-free diagonal inversion).
    rec, ninv = Reconstruct_RP(img, dictionary, 0, method="direct2ds")
    assert rec.shape == (3, TIME_STEPS), rec.shape
    assert np.all(np.isfinite(rec)), "direct2ds reconstruction contains non-finite values"

    corrs = []
    for c in range(3):
        r = _abs_corr(rec[c], w[:, c])
        corrs.append(r)
        assert r > 0.95, f"channel {c}: |corr| too low ({r:.4f}), expected near-exact reconstruction"
    print(f"[ok] metadata-free production decode: shape {rec.shape}, |corr| per channel = "
          f"{[round(c, 4) for c in corrs]}")

    # 5. Operator-range ORACLE variant (ablation only): passing the per-window
    #    (Rmin,Rmax) metadata must stay reachable and must agree with the
    #    metadata-free decode to within the 8-bit quantisation floor.
    rp_range = _build_rp_range(w[None, ...], TIME_STEPS)
    assert 0 in rp_range and len(rp_range[0]) == 3
    rec_oracle, _ = Reconstruct_RP(img, dictionary, 0, method="direct2ds", rp_range=rp_range)
    assert rec_oracle.shape == (3, TIME_STEPS), rec_oracle.shape
    rng = np.ptp(w, axis=0)
    nmae = float(np.mean(np.abs(rec_oracle.T - rec.T) / rng))
    assert nmae < 0.02, f"metadata-free vs oracle NMAE too large: {nmae:.4f}"
    print(f"[ok] operator-range oracle still reachable; agreement NMAE={nmae:.4f}")

    # 6. Legacy method="1d" path must stay reachable and shape-compatible
    #    (backward compat) — fidelity is NOT asserted for this path.
    rec_1d, ninv_1d = Reconstruct_RP(img, dictionary, 0, method="1d")
    assert rec_1d.shape == (3, TIME_STEPS), rec_1d.shape
    assert np.all(np.isfinite(rec_1d))
    print(f"[ok] legacy 1d production decode still returns shape {rec_1d.shape} (backward compat)")

    print("all checks passed")


if __name__ == "__main__":
    main()
