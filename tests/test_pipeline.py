import pandas as pd
import pytest

from pipeline import score_dataframe
from data_validation import validate_dataframe
from recommendations import BARRIER_ACTIONS


@pytest.fixture
def topics_df():
    return pd.DataFrame([
        {
            "course": "DS",
            "topic": f"T{i}",
            "failure_rate": f,
            "confusion": c,
            "resource_inadequacy": r,
            "barrier_type": bt,
            "source": "synthetic",
        }
        for i, (f, c, r, bt) in enumerate([
            (65, 82, 90, "visualization"),
            (10, 12, 8, "navigation"),
            (48, 70, 75, "application"),
            (20, 30, 35, "unclear_explanation"),
            (100, 100, 100, "self_check"),
        ])
    ])


def test_score_dataframe_validates_and_ranks(topics_df):
    assert validate_dataframe(topics_df)[0]
    ranked = score_dataframe(topics_df)
    assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
    assert ranked["fuzzy_score"].is_monotonic_decreasing
    assert ranked.iloc[0]["topic"] == "T4"  # all-100 row must rank first
    assert ranked.iloc[-1]["topic"] == "T1"  # all-low row must rank last


def test_score_dataframe_columns(topics_df):
    ranked = score_dataframe(topics_df)
    for col in [
        "rank", "course", "topic", "failure_rate", "confusion", "resource_inadequacy",
        "barrier_type", "source", "fuzzy_score", "priority_label", "suggested_action",
        "explanation", "baseline_score", "baseline_rank",
        "baseline_score_resources_heavy", "baseline_rank_resources_heavy",
    ]:
        assert col in ranked.columns, col


def test_score_dataframe_actions_are_valid(topics_df):
    ranked = score_dataframe(topics_df)
    assert set(ranked["suggested_action"]) <= set(BARRIER_ACTIONS.values())


def test_score_dataframe_explanation_mentions_rules(topics_df):
    ranked = score_dataframe(topics_df)
    for _, row in ranked.iterrows():
        assert "Priority" in row["explanation"]
        assert "rules fired" in row["explanation"] or "No rules fired" in row["explanation"]


def test_score_dataframe_is_deterministic(topics_df):
    a = score_dataframe(topics_df)
    b = score_dataframe(topics_df)
    assert list(a["fuzzy_score"]) == list(b["fuzzy_score"])
    assert list(a["topic"]) == list(b["topic"])


def test_shipped_dataset_scores():
    df = pd.read_csv("data/topics.csv")
    ok, errors = validate_dataframe(df)
    assert ok, errors
    ranked = score_dataframe(df)
    assert len(ranked) == 15
    assert ranked["fuzzy_score"].between(0, 100).all()
