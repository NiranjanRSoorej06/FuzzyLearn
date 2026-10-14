import io
import pandas as pd
import streamlit as st

from fuzzy_engine import priority
from baseline import weighted_score, rank_baseline
from recommendations import suggest_action, is_valid_barrier
from data_validation import validate_dataframe


st.set_page_config(page_title="FuzzyLearn", layout="wide")
st.title("FuzzyLearn — Teaching Resource Prioritization")

uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv("data/topics.csv")
        st.info("Using default dataset data/topics.csv")
    except FileNotFoundError:
        df = None

if df is not None:
    ok, errors = validate_dataframe(df)
    if not ok:
        st.error("Validation errors:")
        for e in errors:
            st.write("- " + e)
    else:
        # provenance
        sources = set(df["source"].astype(str).str.strip())
        if sources == {"synthetic"}:
            st.warning("Dataset is fully synthetic.")
        elif "real" in sources:
            st.success("Dataset contains real data; synthetic rows labelled separately.")

        # compute
        rows = []
        for _, r in df.iterrows():
            F = float(r["failure_rate"]); C = float(r["confusion"]); R = float(r["resource_inadequacy"])
            res = priority(F, C, R)
            rows.append({
                "course": r["course"],
                "topic": r["topic"],
                "fuzzy_score": res["score"],
                "priority_label": res["label"],
                "suggested_action": suggest_action(r["barrier_type"]) or "Needs review",
                "memberships": res["memberships"],
                "fired_rules": res["fired_rules"],
                "failure_rate": F,
                "confusion": C,
                "resource_inadequacy": R,
                "barrier_type": r["barrier_type"],
            })
        out = pd.DataFrame(rows)
        out = rank_baseline(out.rename(columns={}), ) if False else out
        # baseline
        out["baseline_score"] = out.apply(lambda r: weighted_score(r["failure_rate"], r["confusion"], r["resource_inadequacy"]), axis=1)
        out = out.sort_values(["fuzzy_score", "topic"], ascending=[False, True]).reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        baseline_sorted = out.sort_values(["baseline_score", "topic"], ascending=[False, True]).reset_index(drop=True)
        base_rank_map = {row["topic"]: i + 1 for i, row in baseline_sorted.iterrows()}
        out["baseline_rank"] = out["topic"].map(base_rank_map)

        display = out[["rank", "course", "topic", "fuzzy_score", "priority_label", "suggested_action", "baseline_score", "baseline_rank"]]
        st.subheader("Ranked topics")
        st.dataframe(display, use_container_width=True)

        st.subheader("Priority chart")
        chart_df = display.set_index("topic")["fuzzy_score"]
        st.bar_chart(chart_df)

        # comparison
        st.subheader("Fuzzy vs baseline comparison")
        st.write("Spearman correlation:",
                 display[["fuzzy_score", "baseline_score"]].rank().corr().iloc[0, 1])

        # export
        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button("Download ranked CSV", data=csv_bytes, file_name="ranked_topics.csv", mime="text/csv")

        # explanation
        st.subheader("Per-topic explanation")
        topic_name = st.selectbox("Select topic", out["topic"].tolist())
        sel = out[out["topic"] == topic_name].iloc[0]
        st.write(f"**{sel['topic']}** — {sel['priority_label']} ({sel['fuzzy_score']})")
        st.write(f"F={sel['failure_rate']}, C={sel['confusion']}, R={sel['resource_inadequacy']}")
        mem = sel["memberships"]
        st.write("Memberships:")
        st.json(mem)
        st.write("Top fired rules:")
        for fr in sel["fired_rules"][:3]:
            st.write(f"- F={fr['antecedents']['F']}, C={fr['antecedents']['C']}, R={fr['antecedents']['R']} → {fr['consequent']} (strength {fr['strength']:.2f})")

        # manual entry
        with st.expander("Manual single-topic scoring"):
            mf = st.number_input("Failure rate", 0.0, 100.0, 50.0)
            mc = st.number_input("Confusion", 0.0, 100.0, 50.0)
            mr = st.number_input("Resource inadequacy", 0.0, 100.0, 50.0)
            if st.button("Score"):
                r = priority(mf, mc, mr)
                st.write(r["score"], r["label"])
