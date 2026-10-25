import pytest

from explanations import explain, explanation_table, INPUT_MEANING


MEMBERSHIPS = {
    "F": {"Low": 0.0, "Medium": 0.4, "High": 0.6},
    "C": {"Low": 0.0, "Medium": 0.0, "High": 1.0},
    "R": {"Low": 0.0, "Medium": 0.0, "High": 1.0},
}
FIRED = [
    {"antecedents": {"F": "High", "C": "High", "R": "High"},
     "consequent": "Very High", "strength": 0.6},
    {"antecedents": {"F": "Medium", "C": "High", "R": "High"},
     "consequent": "High", "strength": 0.4},
]


def test_explain_contains_score_label_and_rules():
    text = explain(79.4, "High", MEMBERSHIPS, FIRED)
    assert "79.4" in text
    assert "High" in text
    assert "Very High" in text
    assert "0.60" in text


def test_explain_uses_actual_memberships():
    text = explain(50.0, "Medium", MEMBERSHIPS, FIRED)
    assert "failure rate" in text
    assert "Medium (0.40)" in text
    assert "High (0.60)" in text


def test_explain_no_rules():
    text = explain(10.0, "Low", MEMBERSHIPS, [])
    assert "No rules fired" in text


def test_explain_respects_top_n():
    text = explain(79.4, "High", MEMBERSHIPS, FIRED, top_n=1)
    assert "Very High" in text
    assert "F is Medium" not in text


def test_all_inputs_named():
    text = explain(50.0, "Medium", MEMBERSHIPS, FIRED)
    for key in ("F", "C", "R"):
        assert INPUT_MEANING[key] in text


def test_explanation_table_length():
    import pandas as pd
    from pipeline import score_dataframe
    df = pd.DataFrame([
        {"course": "DS", "topic": "A", "failure_rate": 50, "confusion": 50,
         "resource_inadequacy": 50, "barrier_type": "visualization", "source": "synthetic"},
        {"course": "DS", "topic": "B", "failure_rate": 10, "confusion": 10,
         "resource_inadequacy": 10, "barrier_type": "navigation", "source": "synthetic"},
    ])
    ranked = score_dataframe(df)
    rows = explanation_table(ranked)
    assert len(rows) == 2
    assert rows[0]["topic"] in ("A", "B")
    assert all(r["explanation"] for r in rows)
