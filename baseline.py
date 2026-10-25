"""Weighted-score baseline used as a non-fuzzy comparison for the rankings."""

DEFAULT_WEIGHTS = (0.35, 0.30, 0.35)

# Alternative weight combinations explored for the 11.2 requirement
# ("try at least two alternative weight combinations"). These are
# assumptions, not empirically established values.
WEIGHT_VARIANTS = {
    "resources_heavy": (0.25, 0.25, 0.50),
    "confusion_heavy": (0.25, 0.50, 0.25),
    "failure_heavy": (0.50, 0.25, 0.25),
}


def weighted_score(F, C, R, weights=DEFAULT_WEIGHTS):
    """S = wF*F + wC*C + wR*R."""
    wF, wC, wR = weights
    return float(wF * F + wC * C + wR * R)


def rank_baseline(df, weights=DEFAULT_WEIGHTS, score_col="baseline_score",
                  rank_col="baseline_rank"):
    """Add the baseline score column and a 1-based rank (descending, ties by topic).

    The row order of the input is preserved; only the two new columns are added,
    so the caller's ordering (for example fuzzy-score order) is not disturbed.
    """
    out = df.copy()
    out[score_col] = out.apply(
        lambda r: weighted_score(
            r["failure_rate"], r["confusion"], r["resource_inadequacy"], weights
        ),
        axis=1,
    )
    ordered = out.sort_values([score_col, "topic"], ascending=[False, True])
    rank_map = {idx: i + 1 for i, idx in enumerate(ordered.index)}
    out[rank_col] = out.index.map(rank_map)
    return out
