DEFAULT_WEIGHTS = (0.35, 0.30, 0.35)


def weighted_score(F, C, R, weights=DEFAULT_WEIGHTS):
    wF, wC, wR = weights
    return float(wF * F + wC * C + wR * R)


def rank_baseline(df, weights=DEFAULT_WEIGHTS, score_col="baseline_score", rank_col="baseline_rank"):
    import pandas as pd
    out = df.copy()
    out[score_col] = out.apply(
        lambda r: weighted_score(r["failure_rate"], r["confusion"], r["resource_inadequacy"], weights),
        axis=1,
    )
    out[rank_col] = out[score_col].rank(ascending=False, method="dense").astype(int)
    # deterministic tie-break by topic name
    out = out.sort_values([score_col, "topic"], ascending=[False, True]).reset_index(drop=True)
    out[rank_col] = range(1, len(out) + 1)
    return out
