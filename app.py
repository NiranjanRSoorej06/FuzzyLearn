"""FuzzyLearn Streamlit dashboard.

Features (spec section 12):
- CSV upload with validation feedback
- Ranked results table (fuzzy + weighted baseline side by side)
- Per-topic explanation generated from the actual fired rules
- Priority chart, sensitivity viewer, membership-function plots
- Fuzzy vs baseline comparison including alternative weight sets
- CSV export including explanations
- Data provenance indicator (real vs synthetic)
- Manual single-topic scoring
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fuzzy_engine import (
    priority, trapmf, GRID, INPUT_PARAMS, OUTPUT_PARAMS, OUTPUT_SETS,
)
from recommendations import BARRIER_ACTIONS, NEEDS_REVIEW, suggest_action
from data_validation import validate_dataframe
from pipeline import score_dataframe
from baseline import WEIGHT_VARIANTS, DEFAULT_WEIGHTS
from evaluation import (
    membership_boundary_checks, rule_coverage_check, output_coverage,
    monotonicity_report, ranking_comparison, worked_example_report,
)

st.set_page_config(page_title="FuzzyLearn", layout="wide")
st.title("FuzzyLearn — Teaching-Resource Prioritization")
st.caption(
    "Explainable Mamdani fuzzy decision support. Suggested improvements are "
    "recommendations, not guaranteed solutions; instructors keep final authority."
)


def mf_plot(params, sets):
    fig, ax = plt.subplots(figsize=(7, 3))
    for name, p in params.items():
        a, b, c, d = p
        ys = []
        for x in GRID:
            arr = np.asarray([x], dtype=float)
            r = np.zeros_like(arr)
            if b > a:
                m = (arr >= a) & (arr <= b)
                r[m] = (arr[m] - a) / (b - a)
            m = (arr >= b) & (arr <= c)
            r[m] = 1.0
            if d > c:
                m = (arr >= c) & (arr <= d)
                r[m] = (d - arr[m]) / (d - c)
            ys.append(float(r[0]))
        ax.plot(GRID, ys, label=name)
    ax.set_xlabel("value")
    ax.set_ylabel("membership")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv("data/topics.csv")
        st.info("Using default dataset `data/topics.csv`.")
    except FileNotFoundError:
        df = None

if df is None:
    st.warning("No dataset available. Upload a CSV matching the required schema.")
    st.stop()

ok, errors = validate_dataframe(df)
if not ok:
    st.error("Validation errors — nothing was scored:")
    for e in errors:
        st.write("- " + e)
    st.stop()

st.success("Dataset passed validation (%d topics)." % len(df))

sources = set(df["source"].astype(str).str.strip())
if sources == {"synthetic"}:
    st.warning(
        "**Provenance: fully synthetic data.** These values exist only to test "
        "the software. They are not observations of real students."
    )
elif "real" in sources:
    st.success(
        "**Provenance: dataset contains real data.** Synthetic rows are labelled "
        "separately in the `source` column."
    )
    st.caption("Real rows should be anonymized topic-level aggregates with appropriate permission.")

ranked = score_dataframe(df)

# --------------------------------------------------------------------- filters
with st.expander("Filters", expanded=False):
    f_courses = st.multiselect("Courses", sorted(ranked["course"].unique()))
    f_barriers = st.multiselect("Barrier types", sorted(ranked["barrier_type"].unique()))
    f_min_score = st.slider("Minimum fuzzy score", 0.0, 100.0, 0.0)

filtered = ranked.copy()
if f_courses:
    filtered = filtered[filtered["course"].isin(f_courses)]
if f_barriers:
    filtered = filtered[filtered["barrier_type"].isin(f_barriers)]
filtered = filtered[filtered["fuzzy_score"] >= f_min_score].reset_index(drop=True)

# --------------------------------------------------------------------- ranked table
st.subheader("Ranked topics")
display_cols = [
    "rank", "course", "topic", "fuzzy_score", "priority_label", "suggested_action",
    "baseline_score", "baseline_rank", "barrier_type", "source",
]
st.dataframe(filtered[display_cols], use_container_width=True)

st.subheader("Priority chart")
st.bar_chart(filtered.set_index("topic")["fuzzy_score"])

# --------------------------------------------------------------------- comparison
st.subheader("Fuzzy vs baseline comparison")
default_cmp = ranking_comparison(ranked, "fuzzy_score", "baseline_score")
st.write(
    "Default weights 0.35F + 0.30C + 0.35R: Spearman rank correlation "
    "**%.4f** with top-%d overlap **%d/3** %s."
    % (default_cmp["spearman"], default_cmp["top3_k"], default_cmp["top3_overlap"],
       default_cmp["top3_topics"])
)

variant_rows = []
for name in WEIGHT_VARIANTS:
    cmp = ranking_comparison(ranked, "fuzzy_score", f"baseline_score_{name}")
    variant_rows.append({
        "weights": name,
        "spearman_vs_fuzzy": round(cmp["spearman"], 4),
        "top3_overlap": cmp["top3_overlap"],
    })
st.dataframe(pd.DataFrame(variant_rows), use_container_width=True)
st.caption(
    "Weight sets are assumptions, not established values. A fuzzy ranking that "
    "differs from the baseline is not thereby better."
)

st.subheader("Worked example (F=65, C=82, R=90)")
we = worked_example_report()
st.write(
    "Score **%.1f** (design reference %.1f, difference %+.2f), label **%s**, "
    "%d rules fired."
    % (we["score"], we["reference"], we["difference"], we["label"], we["n_fired"])
)

# --------------------------------------------------------------------- export
st.subheader("CSV export")
export_cols = [
    "rank", "course", "topic", "failure_rate", "confusion", "resource_inadequacy",
    "barrier_type", "source", "fuzzy_score", "priority_label", "suggested_action",
    "explanation", "baseline_score", "baseline_rank",
]
csv_bytes = filtered[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download ranked CSV (with explanations)",
    data=csv_bytes, file_name="ranked_topics.csv", mime="text/csv",
)

# --------------------------------------------------------------------- explanation
st.subheader("Per-topic explanation")
if len(filtered) == 0:
    st.info("No topics match the current filters.")
else:
    topic_name = st.selectbox("Select topic", filtered["topic"].tolist())
    sel = filtered[filtered["topic"] == topic_name].iloc[0]

    st.markdown(f"**{sel['topic']}** — {sel['priority_label']} priority "
                f"({sel['fuzzy_score']:.1f}), rank {sel['rank']}")
    st.write(f"F={sel['failure_rate']:.0f}, C={sel['confusion']:.0f}, "
             f"R={sel['resource_inadequacy']:.0f}")
    st.info(sel["explanation"])

    st.write("Membership degrees:")
    mem_df = pd.DataFrame(sel["memberships"]).T
    st.dataframe(mem_df.style.format("{:.2f}"), use_container_width=True)

    st.write("Strongest fired rules:")
    for fr in sel["fired_rules"][:3]:
        a = fr["antecedents"]
        st.write(
            f"- F={a['F']} AND C={a['C']} AND R={a['R']} → {fr['consequent']} "
            f"(strength {fr['strength']:.2f})"
        )

    st.write(f"Suggested improvement: **{sel['suggested_action']}** "
             f"(from barrier `{sel['barrier_type']}`)")
    if sel["suggested_action"] == NEEDS_REVIEW:
        st.warning("Barrier type not recognized — needs manual review.")

# --------------------------------------------------------------------- sensitivity
st.subheader("Sensitivity viewer")
with st.expander("Vary one input while holding the others fixed"):
    s_input = st.selectbox("Input to vary", ["failure_rate (F)", "confusion (C)", "resource inadequacy (R)"])
    c1, c2 = st.columns(2)
    with c1:
        s_a = st.slider("Fixed first input", 0.0, 100.0, 50.0)
    with c2:
        s_b = st.slider("Fixed second input", 0.0, 100.0, 50.0)
    which = {"failure_rate (F)": "F", "confusion (C)": "C", "resource inadequacy (R)": "R"}[s_input]
    xs = list(range(0, 101, 2))
    ys = []
    for v in xs:
        F, C, R = (v, s_a, s_b) if which == "F" else (
            (s_a, v, s_b) if which == "C" else (s_a, s_b, v))
        ys.append(priority(F, C, R)["score"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, ys, marker=".")
    ax.set_xlabel(s_input)
    ax.set_ylabel("priority score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------------------------- MF plots
st.subheader("Membership functions")
mf_tab1, mf_tab2 = st.tabs(["Input sets", "Output sets"])
with mf_tab1:
    st.pyplot(mf_plot(INPUT_PARAMS, ["Low", "Medium", "High"]))
    st.caption("Low: trapezoid (0,0,25,50) · Medium: triangle (25,50,75) · "
               "High: trapezoid (50,75,100,100)")
with mf_tab2:
    st.pyplot(mf_plot(OUTPUT_PARAMS, OUTPUT_SETS))
    st.caption("Priority output sets on the same 0–100 scale; labels are chosen "
               "by highest membership at the crisp score.")

# --------------------------------------------------------------------- evaluation
st.subheader("Evaluation summary")
with st.spinner("Running evaluation checks..."):
    mb = membership_boundary_checks()
    rc = rule_coverage_check()
    oc = output_coverage(step=10)
    mo = monotonicity_report(step=5)
st.write(
    "Membership boundaries: %s · Rule coverage: %s (%d rules) · "
    "Output coverage: %d/%d grid points valid, range [%.1f, %.1f] · "
    "Monotonicity: %d/%d checks monotone"
    % (
        "pass" if all(c["pass"] for c in mb) else "FAIL",
        "pass" if rc["ok"] else "FAIL",
        rc["n_rules"],
        oc["n_combinations"] - oc["n_invalid"],
        oc["n_combinations"],
        oc["min"], oc["max"],
        mo["checked"] - mo["n_violations"], mo["checked"],
    )
)
if mo["n_violations"]:
    st.caption(
        "Small local non-monotonicities can occur at membership transitions with "
        "centroid defuzzification; they are documented in the README."
    )

# --------------------------------------------------------------------- manual entry
st.subheader("Manual single-topic scoring")
with st.expander("Score one topic by hand"):
    mf = st.number_input("Failure rate", 0.0, 100.0, 50.0)
    mc = st.number_input("Confusion", 0.0, 100.0, 50.0)
    mr = st.number_input("Resource inadequacy", 0.0, 100.0, 50.0)
    barrier = st.selectbox("Barrier type", sorted(BARRIER_ACTIONS.keys()))
    if st.button("Score"):
        r = priority(mf, mc, mr)
        st.write(f"Score **{r['score']:.1f}**, label **{r['label']}**")
        st.write(f"Suggested improvement: **{suggest_action(barrier)}**")
        for fr in r["fired_rules"][:3]:
            a = fr["antecedents"]
            st.write(
                f"- F={a['F']} AND C={a['C']} AND R={a['R']} → {fr['consequent']} "
                f"(strength {fr['strength']:.2f})"
            )

# --------------------------------------------------------------------- rule table
st.subheader("Rule base (27 rules)")
from rules import RULES as _RULES
st.dataframe(pd.DataFrame(_RULES), use_container_width=True)
