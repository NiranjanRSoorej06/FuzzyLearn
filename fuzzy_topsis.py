"""Fuzzy TOPSIS, Chen (2000) variant only.

Formulas must be checked against the original Chen (2000) paper, Fuzzy Sets
and Systems; not yet verified by the team.

This module must not share normalisation or distance formulas with the
crisp min-max TOPSIS in topsis.py -- the two are deliberately different
variants and are only compared at the closeness-coefficient level.
"""

import warnings

import numpy as np

FUZZY_POSITIVE_IDEAL = (1.0, 1.0, 1.0)
FUZZY_NEGATIVE_IDEAL = (0.0, 0.0, 0.0)


def degenerate(T):
    """Collapse a triangular fuzzy matrix to crisp values (l = u = c).

    Used for the controlled E0 (crisp-as-degenerate-fuzzy) vs E1 (genuinely
    fuzzy) comparison: everything except the spread is held fixed.
    """
    T = np.array(T, dtype=float, copy=True)
    T[..., 0] = T[..., 1]
    T[..., 2] = T[..., 1]
    return T


def _admissible(N):
    return (N[:, :, 0] <= N[:, :, 1] + 1e-9).all() and (N[:, :, 1] <= N[:, :, 2] + 1e-9).all()


def normalise(T, benefit):
    """Chen (2000) normalisation of a triangular fuzzy matrix.

    T has shape (m, n, 3) holding (l, c, u). Benefit criteria divide by
    c* = max over alternatives of u. Cost criteria use
    (a-/c_ij, a-/b_ij, a-/a_ij) with a- = min over alternatives of l.
    """
    T = np.asarray(T, dtype=float)
    m, n, _ = T.shape
    benefit = np.asarray(benefit, dtype=bool)
    N = np.zeros_like(T)
    for j in range(n):
        if benefit[j]:
            cstar = T[:, j, 2].max()
            if cstar == 0:
                warnings.warn("Zero c* normalising benefit criterion index %d" % j)
                cstar = 1.0
            N[:, j, :] = T[:, j, :] / cstar
        else:
            aminus = T[:, j, 0].min()
            denom = T[:, j, ::-1]  # (c, b, a) order, Chen's cost formula
            zero = denom == 0
            if zero.any():
                warnings.warn("Zero denominator normalising cost criterion index %d" % j)
            safe_denom = np.where(zero, 1.0, denom)
            N[:, j, :] = np.where(zero, 0.0, aminus / safe_denom)
    if not _admissible(N):
        raise ValueError("Fuzzy admissibility violated after normalisation (l <= c <= u)")
    return N


def vertex_distance(a, b):
    """Chen's vertex distance between two triangular fuzzy numbers."""
    return np.sqrt(((a - b) ** 2).sum(axis=-1) / 3.0)


def topsis_chen(T, w, benefit):
    """Chen (2000) fuzzy TOPSIS closeness coefficient.

    Returns (closeness, N, V, D_plus, D_minus): the normalised matrix,
    weighted matrix and summed vertex distances, for reporting.
    """
    N = normalise(T, benefit)
    w = np.asarray(w, dtype=float)
    V = N * w[None, :, None]
    d_pos = vertex_distance(V, np.array(FUZZY_POSITIVE_IDEAL)).sum(axis=1)
    d_neg = vertex_distance(V, np.array(FUZZY_NEGATIVE_IDEAL)).sum(axis=1)
    denom = d_pos + d_neg
    zero = denom == 0
    if zero.any():
        warnings.warn(
            "D+ + D- is zero for alternative(s) at index %s; closeness set to 0.5"
            % np.where(zero)[0].tolist()
        )
    safe_denom = np.where(zero, 1.0, denom)
    closeness = np.where(zero, 0.5, d_neg / safe_denom)
    if not np.all(np.isfinite(closeness)):
        raise ValueError("Non-finite closeness coefficients produced")
    return closeness, N, V, d_pos, d_neg
