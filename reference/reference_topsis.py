"""Independent reference calculation for the FuzzyLearn TOPSIS project.

Purpose: an implementation that does NOT share code with the repo, used to
(1) check the professor's sample example, (2) generate data/actions.csv
reproducibly, (3) produce the numbers quoted in the documents, and (4) give
the repo implementation something to be tested against.

Run:  python reference_topsis.py [path_to_repo_for_mamdani_comparison]
Only needs numpy and pandas.
"""
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Source data (published values, Goldman et al. 2008, SIGCSE '08, Table
#    of Delphi ratings: mean (SD) on a 1-10 scale)
# ---------------------------------------------------------------------------
# id, topic (as named in the paper), barrier_type (team assumption),
# importance mean, importance SD, difficulty mean, difficulty SD
ALTERNATIVES = [
    ("A1", "Recursion, tracing and designing", "visualization", 7.8, 2.4, 9.2, 0.9),
    ("A2", "Memory model, references, pointers", "visualization", 7.5, 1.7, 8.9, 0.7),
    ("A3", "Recursive algorithms", "application", 6.3, 2.7, 8.1, 0.8),
    ("A4", "Recurrence relations", "unclear_explanation", 8.0, 1.4, 7.7, 1.0),
    ("A5", "Order of growth", "unclear_explanation", 8.1, 2.0, 8.3, 1.1),
    ("A6", "Algorithm analysis", "process_demo", 5.5, 2.2, 7.5, 1.1),
    ("A7", "Algorithm correctness", "self_check", 4.5, 2.6, 8.6, 1.0),
    ("A8", "Recursive definitions", "relationships", 8.7, 1.3, 8.0, 1.1),
    ("A9", "Abstraction / pattern recognition and use", "application", 8.8, 0.4, 9.0, 0.4),
    ("A10", "Primitive vs. reference variables", "self_check", 8.5, 2.4, 7.0, 0.8),
]
ACTIONS = {
    "visualization": "Interactive visualization or animation",
    "application": "Worked examples and graded practice",
    "unclear_explanation": "Revised notes and step-by-step explanations",
    "process_demo": "Short instructional video",
    "self_check": "Self-assessment quiz with explanations",
    "relationships": "Concept map or diagram",
}
SEED = 2026
CRITERIA = ["difficulty", "importance", "resource_gap", "effort_hours"]
BENEFIT = [True, True, True, False]  # effort: lower is better


def build_data():
    rng = np.random.default_rng(SEED)
    resource_gap = rng.integers(20, 91, len(ALTERNATIVES))   # SYNTHETIC
    effort_hours = rng.integers(6, 61, len(ALTERNATIVES))    # SYNTHETIC
    rows = []
    for (aid, topic, barrier, im, isd, dm, dsd), rg, eh in zip(
            ALTERNATIVES, resource_gap, effort_hours):
        def tfn(mean, sd):
            lo = max(mean - sd, 1.0) * 10
            hi = min(mean + sd, 10.0) * 10
            return round(lo, 1), round(mean * 10, 1), round(hi, 1)
        dl, dc, du = tfn(dm, dsd)
        il, ic, iu = tfn(im, isd)
        rows.append(dict(
            alt_id=aid, topic=topic, barrier_type=barrier,
            action=f"{ACTIONS[barrier]} - {topic}",
            diff_l=dl, diff_c=dc, diff_u=du,
            imp_l=il, imp_c=ic, imp_u=iu,
            resource_gap=int(rg), effort_hours=int(eh),
            source_difficulty_importance="published (Goldman et al. 2008)",
            source_resource_effort="synthetic (seeded draw, seed=2026)",
            data_status="synthetic-anchored"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Crisp TOPSIS exactly as in the assignment brief (Step 6 + Step 9A)
# ---------------------------------------------------------------------------
def minmax(X, benefit):
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    rng_ = np.where(hi - lo == 0, np.nan, hi - lo)
    R = np.where(benefit, (X - lo) / rng_, (hi - X) / rng_)
    return np.nan_to_num(R, nan=0.0)   # constant column -> 0 (flag in report)


def topsis_crisp(X, w, benefit):
    R = minmax(X, np.array(benefit))
    V = R * np.asarray(w)
    vp, vm = V.max(0), V.min(0)
    dp = np.sqrt(((V - vp) ** 2).sum(1))
    dm = np.sqrt(((V - vm) ** 2).sum(1))
    return dm / (dp + dm), R, V, dp, dm


# ---------------------------------------------------------------------------
# 3. Fuzzy TOPSIS, Chen (2000) variant (VERIFY against the original paper)
#    triangular matrix T has shape (m, n, 3) holding (l, c, u)
# ---------------------------------------------------------------------------
def vertex_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(-1) / 3.0)


def topsis_chen(T, w, benefit):
    T = np.asarray(T, float)
    m, n, _ = T.shape
    N = np.zeros_like(T)
    for j in range(n):
        if benefit[j]:
            cstar = T[:, j, 2].max()
            N[:, j, :] = T[:, j, :] / cstar
        else:
            aminus = T[:, j, 0].min()
            # Chen: (a-/c_ij, a-/b_ij, a-/a_ij); T[:, j, ::-1] = (c, b, a)
            N[:, j, :] = aminus / T[:, j, ::-1]
    assert (N[:, :, 0] <= N[:, :, 1] + 1e-12).all() and (N[:, :, 1] <= N[:, :, 2] + 1e-12).all()
    V = N * np.asarray(w)[None, :, None]
    fpis = np.ones(3)
    fnis = np.zeros(3)
    dp = vertex_distance(V, fpis).sum(1)
    dm = vertex_distance(V, fnis).sum(1)
    return dm / (dp + dm), N, V, dp, dm


# ---------------------------------------------------------------------------
# 4. Weights: equal, CRITIC (Diakoulaki et al., 1995)
# ---------------------------------------------------------------------------
def critic_weights(X, benefit):
    R = minmax(X, np.array(benefit))
    sd = R.std(0, ddof=1)
    corr = np.corrcoef(R, rowvar=False)
    info = sd * (1 - corr).sum(1)
    return info / info.sum()


# ---------------------------------------------------------------------------
# 5. Ranking helpers
# ---------------------------------------------------------------------------
def rank_desc(score):
    return pd.Series(score).rank(ascending=False, method="min").astype(int).values


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def top_overlap(a, b, k=3):
    ta = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    tb = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ta & tb)


def main(repo=None):
    # --- check against the assignment's own sample (Table III / IV) ---
    X = np.array([[8, 100, 20000], [6, 80, 10000], [4, 40, 15000]], float)
    cc, R, V, dp, dm = topsis_crisp(X, [0.4, 0.4, 0.2], [True, True, False])
    print("SAMPLE CHECK (brief Table IV): closeness =", np.round(cc, 4),
          "(expected 0.7388 0.6179 0.1483)")
    print("  D+ =", np.round(dp, 4), " D- =", np.round(dm, 4))
    assert np.allclose(np.round(cc, 4), [0.7388, 0.6179, 0.1483])

    # --- project data ---
    df = build_data()
    df.to_csv("../data/actions.csv", index=False)
    n = len(CRITERIA)
    Xc = df[["diff_c", "imp_c", "resource_gap", "effort_hours"]].values.astype(float)
    w_eq = np.full(n, 1.0 / n)

    cc_crisp, R, V, dp, dm = topsis_crisp(Xc, w_eq, BENEFIT)

    T = np.stack([
        df[["diff_l", "diff_c", "diff_u"]].values,
        df[["imp_l", "imp_c", "imp_u"]].values,
        np.stack([df.resource_gap] * 3, 1),
        np.stack([df.effort_hours] * 3, 1)], axis=1).astype(float)
    Td = T.copy()
    Td[:, :, 0] = Td[:, :, 1]   # degenerate: l=c
    Td[:, :, 2] = Td[:, :, 1]   # degenerate: u=c

    cc_fuzzy, *_ = topsis_chen(T, w_eq, BENEFIT)
    cc_chen_crisp, *_ = topsis_chen(Td, w_eq, BENEFIT)

    out = df[["alt_id", "topic"]].copy()
    out["TOPSIS_crisp"] = cc_crisp.round(4)
    out["rank_crisp"] = rank_desc(cc_crisp)
    out["Chen_degenerate"] = cc_chen_crisp.round(4)
    out["rank_chen_deg"] = rank_desc(cc_chen_crisp)
    out["Chen_fuzzy"] = cc_fuzzy.round(4)
    out["rank_chen_fuzzy"] = rank_desc(cc_fuzzy)
    out["baseline_sum"] = (R * w_eq).sum(1).round(4)
    out["rank_sum"] = rank_desc(out["baseline_sum"])
    out["rank_difficulty_only"] = rank_desc(df.diff_c.values)

    # Mamdani (existing repo engine) on three shared criteria
    if repo:
        sys.dont_write_bytecode = True
        sys.path.insert(0, repo)
        from fuzzy_engine import priority
        mam = np.array([priority(r.diff_c, r.imp_c, r.resource_gap)["score"]
                        for r in df.itertuples()])
        out["Mamdani"] = mam
        out["rank_mamdani"] = rank_desc(mam)

    pd.set_option("display.width", 200)
    print("\nWEIGHTS (equal):", w_eq)
    print("\nNORMALISED crisp matrix R:\n", pd.DataFrame(R.round(4), columns=CRITERIA, index=df.alt_id))
    print("\nRESULTS:\n", out.to_string(index=False))

    print("\nCOMPARISONS (Spearman / top-3 overlap)")
    def cmp(name, a, b):
        print(f"  {name}: rho={spearman(a, b):.4f}, top3={top_overlap(a, b)}/3")
    cmp("crisp TOPSIS vs Chen degenerate", cc_crisp, cc_chen_crisp)
    cmp("Chen degenerate vs Chen fuzzy (EXTENSION)", cc_chen_crisp, cc_fuzzy)
    cmp("crisp TOPSIS vs Chen fuzzy", cc_crisp, cc_fuzzy)
    cmp("crisp TOPSIS vs equal-weight sum", cc_crisp, out.baseline_sum)
    cmp("crisp TOPSIS vs difficulty-only", cc_crisp, df.diff_c)
    if repo:
        cmp("crisp TOPSIS vs Mamdani", cc_crisp, out.Mamdani)
        cmp("Chen fuzzy vs Mamdani", cc_fuzzy, out.Mamdani)

    # sensitivity on weights
    scen = {
        "equal": w_eq,
        "difficulty-heavy": [0.4, 0.2, 0.2, 0.2],
        "importance-heavy": [0.2, 0.4, 0.2, 0.2],
        "resource-heavy": [0.2, 0.2, 0.4, 0.2],
        "effort-heavy": [0.2, 0.2, 0.2, 0.4],
        "CRITIC": critic_weights(Xc, BENEFIT),
    }
    print("\nCRITIC weights:", np.round(scen["CRITIC"], 4))
    print("\nSENSITIVITY (crisp TOPSIS)")
    base = cc_crisp
    tops = {}
    for k, w in scen.items():
        c, *_ = topsis_crisp(Xc, w, BENEFIT)
        cf, *_ = topsis_chen(T, w, BENEFIT)
        tops[k] = (c, cf)
        top3 = [df.alt_id[i] for i in np.argsort(-c, kind="stable")[:3]]
        top3f = [df.alt_id[i] for i in np.argsort(-cf, kind="stable")[:3]]
        print(f"  {k:17s} w={np.round(w,3)}  crisp top3={top3} rho_vs_equal={spearman(c, base):.3f} | fuzzy top3={top3f}")

    # constant-column / zero-denominator guard demo
    print("\nGuard check: constant column ->", minmax(np.array([[1, 5], [2, 5]], float), np.array([True, True])).tolist())
    out.to_csv("../data/reference_results.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
