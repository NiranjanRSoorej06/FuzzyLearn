"""Produce rankings, comparisons, sensitivity analysis and figures for the
TOPSIS prioritisation of Data Structures teaching-resource improvement
actions. All inputs are synthetic-anchored (see data/actions.csv); nothing
here is a recommendation about a real course.

Run:  python topsis_evaluation.py
Writes results/ranking.csv, results/comparisons.csv, results/sensitivity.csv
and figures under results/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topsis import (
    load_actions, equal_weights, critic_weights, topsis_crisp, rank_desc,
    spearman, top_overlap,
)
from fuzzy_topsis import degenerate, topsis_chen
from mamdani_adapter import score_actions as mamdani_scores

ACTIONS_PATH = "data/actions.csv"
RESULTS_DIR = "results"
CRITERIA = ["difficulty", "importance", "resource_gap", "effort_hours"]
BENEFIT = [True, True, True, False]
CAPTION_SUFFIX = "(synthetic-anchored data)"

WEIGHT_SCENARIOS = {
    "equal": None,  # filled in with equal_weights(4) in main()
    "difficulty-heavy": [0.4, 0.2, 0.2, 0.2],
    "importance-heavy": [0.2, 0.4, 0.2, 0.2],
    "resource-heavy": [0.2, 0.2, 0.4, 0.2],
    "effort-heavy": [0.2, 0.2, 0.2, 0.4],
    "CRITIC": None,  # filled in with critic_weights(Xc, BENEFIT) in main()
}


def build_matrices(df):
    Xc = df[["diff_c", "imp_c", "resource_gap", "effort_hours"]].to_numpy(dtype=float)
    T = np.stack(
        [
            df[["diff_l", "diff_c", "diff_u"]].to_numpy(dtype=float),
            df[["imp_l", "imp_c", "imp_u"]].to_numpy(dtype=float),
            np.repeat(df[["resource_gap"]].to_numpy(dtype=float), 3, axis=1),
            np.repeat(df[["effort_hours"]].to_numpy(dtype=float), 3, axis=1),
        ],
        axis=1,
    )
    return Xc, T


def build_ranking(df, Xc, T, w):
    cc_crisp, R, V, dp, dm = topsis_crisp(Xc, w, BENEFIT)
    cc_deg, *_ = topsis_chen(degenerate(T), w, BENEFIT)
    cc_fuzzy, *_ = topsis_chen(T, w, BENEFIT)
    baseline_sum = (R * w).sum(axis=1)
    mam = mamdani_scores(df)

    out = df[["alt_id", "topic", "action", "data_status"]].copy()
    out["topsis_crisp_score"] = cc_crisp.round(4)
    out["topsis_crisp_rank"] = rank_desc(cc_crisp)
    out["chen_degenerate_score"] = cc_deg.round(4)
    out["chen_degenerate_rank"] = rank_desc(cc_deg)
    out["chen_fuzzy_score"] = cc_fuzzy.round(4)
    out["chen_fuzzy_rank"] = rank_desc(cc_fuzzy)
    out["equal_weight_sum_score"] = baseline_sum.round(4)
    out["equal_weight_sum_rank"] = rank_desc(baseline_sum)
    out["difficulty_only_rank"] = rank_desc(df["diff_c"].to_numpy(dtype=float))
    out["mamdani_score"] = mam.round(1)
    out["mamdani_rank"] = rank_desc(mam)
    return out, dict(cc_crisp=cc_crisp, cc_deg=cc_deg, cc_fuzzy=cc_fuzzy, baseline_sum=baseline_sum, mam=mam)


def build_comparisons(scores):
    pairs = [
        ("crisp vs Chen-degenerate", scores["cc_crisp"], scores["cc_deg"]),
        ("Chen-degenerate vs Chen-fuzzy (extension)", scores["cc_deg"], scores["cc_fuzzy"]),
        ("crisp vs Chen-fuzzy", scores["cc_crisp"], scores["cc_fuzzy"]),
        ("crisp vs equal-weight sum", scores["cc_crisp"], scores["baseline_sum"]),
        ("crisp vs difficulty-only", scores["cc_crisp"], scores["diff_only"]),
        ("crisp vs Mamdani", scores["cc_crisp"], scores["mam"]),
        ("Chen-fuzzy vs Mamdani", scores["cc_fuzzy"], scores["mam"]),
    ]
    rows = []
    for name, a, b in pairs:
        rows.append({
            "comparison": name,
            "spearman_rho": round(spearman(a, b), 4),
            "top3_overlap": top_overlap(a, b),
        })
    return pd.DataFrame(rows)


def build_sensitivity(df, Xc, T, scenarios):
    base_crisp, *_ = topsis_crisp(Xc, scenarios["equal"], BENEFIT)
    base_fuzzy, *_ = topsis_chen(T, scenarios["equal"], BENEFIT)
    rows = []
    rank_matrix = {}
    for name, w in scenarios.items():
        cc, *_ = topsis_crisp(Xc, w, BENEFIT)
        cf, *_ = topsis_chen(T, w, BENEFIT)
        top3 = list(df["alt_id"].to_numpy()[np.argsort(-cc, kind="stable")[:3]])
        top3f = list(df["alt_id"].to_numpy()[np.argsort(-cf, kind="stable")[:3]])
        rows.append({
            "scenario": name,
            "weights": [round(float(x), 4) for x in w],
            "crisp_top3": top3,
            "crisp_rho_vs_equal": round(spearman(cc, base_crisp), 4),
            "fuzzy_top3": top3f,
            "fuzzy_rho_vs_equal": round(spearman(cf, base_fuzzy), 4),
        })
        rank_matrix[name] = rank_desc(cc)
    return pd.DataFrame(rows), pd.DataFrame(rank_matrix, index=df["alt_id"])


def plot_ranked_closeness(ranking, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = ranking.sort_values("topsis_crisp_score", ascending=False)
    ax.bar(ordered["alt_id"], ordered["topsis_crisp_score"], color="#4C72B0")
    ax.set_xlabel("Improvement action")
    ax.set_ylabel("Crisp TOPSIS closeness coefficient")
    ax.set_title("Ranked crisp TOPSIS closeness %s" % CAPTION_SUFFIX)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity_heatmap(rank_matrix, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = rank_matrix.to_numpy().T
    im = ax.imshow(data, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(rank_matrix.index)))
    ax.set_xticklabels(rank_matrix.index)
    ax.set_yticks(range(len(rank_matrix.columns)))
    ax.set_yticklabels(rank_matrix.columns)
    ax.set_xlabel("Improvement action")
    ax.set_ylabel("Weight scenario")
    ax.set_title("Crisp TOPSIS rank by weight scenario %s" % CAPTION_SUFFIX)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, int(data[i, j]), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="rank (1 = top priority)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fuzzy_vs_degenerate(ranking, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = ranking.sort_values("chen_degenerate_score", ascending=False)
    x = np.arange(len(ordered))
    width = 0.35
    ax.bar(x - width / 2, ordered["chen_degenerate_score"], width, label="Chen degenerate (E0)", color="#4C72B0")
    ax.bar(x + width / 2, ordered["chen_fuzzy_score"], width, label="Chen fuzzy (E1)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["alt_id"])
    ax.set_xlabel("Improvement action")
    ax.set_ylabel("Closeness coefficient")
    ax.set_title("Chen fuzzy vs degenerate closeness %s" % CAPTION_SUFFIX)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_actions(ACTIONS_PATH)
    Xc, T = build_matrices(df)
    w_eq = equal_weights(len(CRITERIA))

    ranking, scores = build_ranking(df, Xc, T, w_eq)
    scores["diff_only"] = df["diff_c"].to_numpy(dtype=float)
    ranking.to_csv(f"{RESULTS_DIR}/ranking.csv", index=False)

    comparisons = build_comparisons(scores)
    comparisons.to_csv(f"{RESULTS_DIR}/comparisons.csv", index=False)

    scenarios = dict(WEIGHT_SCENARIOS)
    scenarios["equal"] = w_eq
    scenarios["CRITIC"] = critic_weights(Xc, BENEFIT)
    sensitivity, rank_matrix = build_sensitivity(df, Xc, T, scenarios)
    sensitivity.to_csv(f"{RESULTS_DIR}/sensitivity.csv", index=False)

    plot_ranked_closeness(ranking, f"{RESULTS_DIR}/ranked_closeness.png")
    plot_sensitivity_heatmap(rank_matrix, f"{RESULTS_DIR}/sensitivity_heatmap.png")
    plot_fuzzy_vs_degenerate(ranking, f"{RESULTS_DIR}/fuzzy_vs_degenerate.png")

    with open(f"{RESULTS_DIR}/figure_captions.txt", "w") as f:
        f.write(
            "ranked_closeness.png -- Ranked crisp TOPSIS closeness coefficients "
            "for the 10 improvement actions, equal weights %s.\n" % CAPTION_SUFFIX
        )
        f.write(
            "sensitivity_heatmap.png -- Crisp TOPSIS rank of each action across "
            "six weight scenarios (equal, difficulty-heavy, importance-heavy, "
            "resource-heavy, effort-heavy, CRITIC) %s.\n" % CAPTION_SUFFIX
        )
        f.write(
            "fuzzy_vs_degenerate.png -- Chen (2000) fuzzy TOPSIS closeness "
            "compared with the degenerate (crisp-as-fuzzy) baseline, equal "
            "weights %s.\n" % CAPTION_SUFFIX
        )

    print("=" * 70)
    print("FuzzyLearn TOPSIS evaluation -- synthetic-anchored data")
    print("=" * 70)
    print("\nEqual weights:", np.round(w_eq, 4))
    print("CRITIC weights:", np.round(scenarios["CRITIC"], 4))
    print("\nRanking (top 3 by crisp TOPSIS):")
    print(ranking.sort_values("topsis_crisp_rank").head(3)[
        ["alt_id", "topic", "topsis_crisp_score", "topsis_crisp_rank",
         "chen_fuzzy_score", "chen_fuzzy_rank", "mamdani_score", "mamdani_rank"]
    ].to_string(index=False))
    print("\nComparisons:")
    print(comparisons.to_string(index=False))
    print("\nOutputs written to %s/" % RESULTS_DIR)
    print("\nIndependent validation: none (synthetic-anchored data)")


if __name__ == "__main__":
    main()
