"""Shared scoring pipeline: validated dataframe -> ranked results.

Keeps ranking and recommendation lookup separate from the fuzzy inference
code, as required by the specification.
"""

import pandas as pd

from fuzzy_engine import priority
from baseline import weighted_score, rank_baseline, WEIGHT_VARIANTS
from recommendations import suggest_action, NEEDS_REVIEW
from explanations import explain


def score_dataframe(df, weights=None):
    """Score every row of a validated topics dataframe.

    Returns a dataframe sorted by fuzzy score (descending, ties broken by
    topic name) with columns:

    rank, course, topic, failure_rate, confusion, resource_inadequacy,
    barrier_type, source, fuzzy_score, priority_label, suggested_action,
    explanation, memberships, fired_rules,
    baseline_score, baseline_rank,
    plus one baseline_score_<name>/baseline_rank_<name> pair per weight variant.
    """
    if weights is None:
        weights = (0.35, 0.30, 0.35)

    rows = []
    for _, r in df.iterrows():
        F = float(r["failure_rate"])
        C = float(r["confusion"])
        R = float(r["resource_inadequacy"])
        res = priority(F, C, R)
        action = suggest_action(r["barrier_type"])
        if action is None:
            action = NEEDS_REVIEW
        rows.append({
            "course": r["course"],
            "topic": r["topic"],
            "failure_rate": F,
            "confusion": C,
            "resource_inadequacy": R,
            "barrier_type": r["barrier_type"],
            "source": r["source"],
            "fuzzy_score": res["score"],
            "priority_label": res["label"],
            "suggested_action": action,
            "memberships": res["memberships"],
            "fired_rules": res["fired_rules"],
            "explanation": explain(
                res["score"], res["label"], res["memberships"], res["fired_rules"]
            ),
        })
    out = pd.DataFrame(rows)

    out["baseline_score"] = out.apply(
        lambda r: weighted_score(r["failure_rate"], r["confusion"], r["resource_inadequacy"], weights),
        axis=1,
    )

    for name, w in WEIGHT_VARIANTS.items():
        out[f"baseline_score_{name}"] = out.apply(
            lambda r, w=w: weighted_score(r["failure_rate"], r["confusion"], r["resource_inadequacy"], w),
            axis=1,
        )

    # Fuzzy ranking: descending score, deterministic tie-break by topic name.
    out = out.sort_values(["fuzzy_score", "topic"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)

    # Baseline ranking with the same tie-break policy.
    out = rank_baseline(out)

    for name in WEIGHT_VARIANTS:
        col = f"baseline_score_{name}"
        sorted_by = out.sort_values([col, "topic"], ascending=[False, True]).reset_index(drop=True)
        rank_map = {row["topic"]: i + 1 for i, row in sorted_by.iterrows()}
        out[f"baseline_rank_{name}"] = out["topic"].map(rank_map)

    ordered = [
        "rank", "course", "topic", "failure_rate", "confusion",
        "resource_inadequacy", "barrier_type", "source",
        "fuzzy_score", "priority_label", "suggested_action",
        "baseline_score", "baseline_rank",
    ] + [c for c in out.columns if c.startswith("baseline_score_") or c.startswith("baseline_rank_")] + [
        "explanation", "memberships", "fired_rules",
    ]
    return out[ordered]
