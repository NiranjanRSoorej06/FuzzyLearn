"""Crisp TOPSIS (Hwang & Yoon, 1981) exactly as defined in the assignment brief.

Alternatives are the 10 improvement actions in data/actions.csv; criteria are
difficulty, importance, resource gap (benefit) and implementation effort
(cost). This module only implements the min-max / Euclidean-distance TOPSIS
variant; it must not share normalisation or distance formulas with the Chen
(2000) fuzzy variant in fuzzy_topsis.py.
"""

import warnings

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "alt_id", "topic", "barrier_type", "action",
    "diff_l", "diff_c", "diff_u", "imp_l", "imp_c", "imp_u",
    "resource_gap", "effort_hours",
    "source_difficulty_importance", "source_resource_effort", "data_status",
]
TRIANGLE_FUZZY_COLUMNS = ["diff_l", "diff_c", "diff_u", "imp_l", "imp_c", "imp_u"]
TRIANGLE_TRIPLES = [("diff_l", "diff_c", "diff_u"), ("imp_l", "imp_c", "imp_u")]


def load_actions(path):
    """Load and validate the actions dataset. Raises ValueError with every
    problem found (validation does not stop at the first error)."""
    df = pd.read_csv(path)
    errors = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % ", ".join(missing))

    if df[REQUIRED_COLUMNS].isna().any().any():
        bad_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
        errors.append("Missing values in columns: %s" % ", ".join(bad_cols))

    for lo, mid, hi in TRIANGLE_TRIPLES:
        bad = df[~((df[lo] <= df[mid]) & (df[mid] <= df[hi]))]
        if len(bad):
            errors.append(
                "%s <= %s <= %s violated for alt_id: %s"
                % (lo, mid, hi, ", ".join(bad["alt_id"].astype(str)))
            )

    for col in TRIANGLE_FUZZY_COLUMNS:
        bad = df[(df[col] < 0) | (df[col] > 100)]
        if len(bad):
            errors.append(
                "%s out of range [0,100] for alt_id: %s"
                % (col, ", ".join(bad["alt_id"].astype(str)))
            )

    if df["alt_id"].duplicated().any():
        dups = sorted(set(df["alt_id"][df["alt_id"].duplicated()]))
        errors.append("Duplicate alt_id values: %s" % ", ".join(dups))

    if "data_status" not in df.columns or df["data_status"].isna().any() or (df["data_status"].astype(str).str.strip() == "").any():
        errors.append("data_status missing for one or more rows")

    if errors:
        raise ValueError("Validation failed: " + "; ".join(errors))

    return df


def orient_normalise(X, benefit):
    """Min-max normalisation, oriented so larger is always better.

    Benefit criteria: (x - min) / (max - min). Cost criteria:
    (max - x) / (max - min). A constant column has a zero denominator; that
    column is set to 0 and a warning is emitted. Returns (R, constant_mask)
    so callers can inspect which columns were constant.
    """
    X = np.asarray(X, dtype=float)
    benefit = np.asarray(benefit, dtype=bool)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    constant = span == 0
    if constant.any():
        warnings.warn(
            "Constant column(s) at index %s have zero range; normalised to 0."
            % np.where(constant)[0].tolist()
        )
    safe_span = np.where(constant, 1.0, span)
    R = np.where(benefit, (X - lo) / safe_span, (hi - X) / safe_span)
    R[:, constant] = 0.0
    return R, constant


def equal_weights(n):
    return np.full(n, 1.0 / n)


def check_weights(w):
    """Validate a weight vector: non-negative, finite, sums to 1 (tol 1e-9)."""
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite, got %r" % w.tolist())
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative, got %r" % w.tolist())
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1, got sum=%.12f" % w.sum())
    return w


def critic_weights(X, benefit):
    """CRITIC weights (Diakoulaki et al., 1995).

    C_j = sigma_j * sum_k(1 - rho_jk), on the min-max normalised matrix,
    sample standard deviation (ddof=1), normalised to sum to 1.
    """
    R, _ = orient_normalise(X, benefit)
    sigma = R.std(axis=0, ddof=1)
    rho = np.corrcoef(R, rowvar=False)
    info = sigma * (1 - rho).sum(axis=1)
    return info / info.sum()


def topsis_crisp(X, w, benefit):
    """Crisp TOPSIS closeness coefficient.

    Returns (closeness, R, V, D_plus, D_minus) so reports can show the
    normalised matrix, weighted matrix and distances as intermediate steps.
    """
    w = check_weights(w)
    R, _ = orient_normalise(X, benefit)
    V = R * w
    v_pos = V.max(axis=0)
    v_neg = V.min(axis=0)
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    denom = d_pos + d_neg
    zero = denom == 0
    if zero.any():
        warnings.warn(
            "D+ + D- is zero for alternative(s) at index %s; closeness set to 0.5"
            % np.where(zero)[0].tolist()
        )
    safe_denom = np.where(zero, 1.0, denom)
    closeness = np.where(zero, 0.5, d_neg / safe_denom)
    return closeness, R, V, d_pos, d_neg


def rank_desc(scores):
    """Dense/min ranking, higher score = better (lower rank number).

    Ties share the smallest rank among the tied group (pandas 'min' method),
    e.g. scores [0.9, 0.7, 0.7] -> ranks [1, 2, 2]. This is the tie rule used
    throughout this project's TOPSIS outputs.
    """
    return pd.Series(scores).rank(ascending=False, method="min").astype(int).to_numpy()


def spearman(a, b):
    """Spearman rank correlation between two score/rank vectors."""
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def top_overlap(a, b, k=3):
    """Size of the overlap between the top-k alternatives of a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ta = set(np.argsort(-a, kind="stable")[:k])
    tb = set(np.argsort(-b, kind="stable")[:k])
    return len(ta & tb)
