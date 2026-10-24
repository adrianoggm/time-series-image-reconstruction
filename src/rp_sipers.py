"""Reconstruction of a signal from its unthresholded RP by Sipers, Borm & Peeters (Phys. Lett. A 381, 2017).

Their method assumes a periodic signal and a circular time-delay embedding X(t) = (x(t), ..., x(t+(M-1)tau)):
the 2D Fourier coefficients of the squared URP then factor into c_p conj(c_q) <T_p, T_q> (their Prop. 3.1,
Eqs. 6-8), the rank-one matrix W = w w* of Fourier coefficients follows from Thm. 3.4 (Eq. 15), and an
eigendecomposition gives w up to a sign (Prop. 3.3). This is the discrete version for N samples, with
M = 2 and tau = 1/N, i.e. the benchmark's adjacent states s_n = (x_n, x_{n+1}) taken circularly (n mod N).
The benchmark's own RP is not circular, so this decoder is evaluated on the circular variant of the same
encoding: N states instead of N - 1, the extra one being the wrap state (x_{N-1}, x_0).

Node set: frequencies with |c_k|^2 > tol * max |c_l|^2. Sipers et al. use tol = 0.01 without optimising
it (their Sect. 4); on the benchmark windows, which are broadband, that discards real content even in
floating point, and tol = 1e-4 gave the lowest 8-bit error in a sweep (scripts/_sipers_pilot.py --sweep).
"""
import numpy as np

TOL = 1e-4


def circular_rp(x: np.ndarray) -> np.ndarray:
    """Signed RP of the manuscript's Eq. (1) on circular adjacent states; its magnitude is the circular URP."""
    x = np.asarray(x, dtype=float)
    s = np.column_stack([x, np.roll(x, -1)])
    delta = s[:, None, :] - s[None, :, :]
    d = np.linalg.norm(delta, axis=-1)
    proj = delta.sum(axis=-1) / np.sqrt(2)
    return np.where((d == 0) | (proj >= d * np.cos(np.pi / 4) - 1e-12), 1.0, -1.0) * d


def sipers_decode(D: np.ndarray, M: int = 2, delay: int = 1, tol: float = TOL) -> np.ndarray:
    """Zero-mean signal, up to sign and scale, from an (N, N) circular URP with embedding (M, delay)."""
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    F = np.fft.fft2(D ** 2)
    k = np.arange(1, N)                                         # non-zero frequencies, mod N
    It = -F[np.ix_(k, (-k) % N)] / (2.0 * N * N)                # c_p conj(c_q) T_pq, their Eq. (8)
    T = np.exp(2j * np.pi * np.subtract.outer(k, k)[..., None] * delay * np.arange(M) / N).sum(-1)
    power = np.real(np.diag(It)) / M                            # |c_k|^2
    keep = power > tol * power.max()                            # symmetric in +-k, since |c_k| = |c_-k|
    kk, It, T = k[keep], It[np.ix_(keep, keep)], T[np.ix_(keep, keep)]
    S = np.abs(T)
    Ihat = np.where(S > 1e-12, It * np.conj(T) / np.where(S > 1e-12, S, 1.0), 0.0)   # W o S
    W = M * Ihat @ np.diag(1.0 / np.diag(Ihat)) @ Ihat / (S @ S)                     # their Eq. (15)
    W = 0.5 * (W + W.conj().T)
    lam, U = np.linalg.eigh(W)
    u = U[:, -1]
    J = np.array([np.flatnonzero(kk == (N - q) % N)[0] for q in kk])                # k -> -k
    c = np.zeros(N, dtype=complex)
    c[kk] = np.sqrt(np.vdot(u, np.conj(u)[J])) * np.sqrt(max(lam[-1], 0.0)) * u   # their Prop. 3.3
    return np.real(N * np.fft.ifft(c))


def to_range(x: np.ndarray, mm) -> np.ndarray:
    """Map a unit-free reconstruction onto the channel's (max, min)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5 * (mm[0] + mm[1]))
    return np.interp(x, (lo, hi), (mm[1], mm[0]))


def decode_unsigned(D: np.ndarray, mm) -> np.ndarray:
    """Sipers et al. on the circular URP; the sign is left as returned, as the URP cannot fix it."""
    return to_range(sipers_decode(np.abs(D)), mm)


def decode_signed(R: np.ndarray, mm) -> np.ndarray:
    """Sipers et al. on |R| of the circular signed RP; the sign picks x or its reflection, as in our decoder."""
    R = np.asarray(R, dtype=float)
    xh = sipers_decode(0.5 * (np.abs(R) + np.abs(R).T))
    best = None
    for cand in (to_range(xh, mm), to_range(-xh, mm)):
        Rc = circular_rp(cand)
        e = np.linalg.norm(Rc / np.linalg.norm(Rc) - R / np.linalg.norm(R))
        best = (e, cand) if best is None or e < best[0] else best
    return best[1]
