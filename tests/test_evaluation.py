import numpy as np
import pandas as pd
import pytest

import evaluation as ev
from data_validation import validate_dataframe
from pipeline import score_dataframe


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv("data/topics.csv")
    assert validate_dataframe(d)[0]
    return d


@pytest.fixture(scope="module")
def report(df):
    rep, ranked = ev.run_all(df, step=5)
    return rep, ranked


def test_membership_boundaries_pass(report):
    rep, _ = report
    failures = [c for c in rep["membership_boundaries"] if not c["pass"]]
    assert not failures


def test_rule_coverage_pass(report):
    rep, _ = report
    assert rep["rule_coverage"]["ok"]
    assert rep["rule_coverage"]["n_rules"] == 27


def test_output_coverage_all_valid(report):
    rep, _ = report
    oc = rep["output_coverage"]
    assert oc["n_invalid"] == 0
    assert oc["min"] >= 0.0
    assert oc["max"] <= 100.0
    assert oc["n_combinations"] > 1000


def test_worked_example_reference(report):
    rep, _ = report
    we = rep["worked_example"]
    assert abs(we["difference"]) < 1.0
    assert we["label"] == "High"
    assert we["n_fired"] == 2


def test_ranking_comparison(report):
    rep, ranked = report
    cmp = ev.ranking_comparison(ranked)
    assert -1.0 <= cmp["spearman"] <= 1.0
    assert cmp["spearman"] > 0.7  # fuzzy and linear baseline should broadly agree
    assert 0 <= cmp["top3_overlap"] <= 3


def test_baseline_variants_present(report):
    rep, _ = report
    variants = rep["ranking_variants"]
    assert "resources_heavy" in variants
    assert "confusion_heavy" in variants
    assert "failure_heavy" in variants
    for v in variants.values():
        assert -1.0 <= v["spearman"] <= 1.0


def test_limiter_example(report):
    rep, _ = report
    le = rep["limiter_example"]
    # resources look adequate, so the rule base limits priority well below "all high"
    all_high = ev.priority(85, 85, 85)["score"]
    assert le["score"] < all_high
    assert le["score"] < 60.0


def test_expert_comparison_absent_column(report):
    rep, _ = report
    assert rep["expert_comparison"]["available"] is False


def test_expert_comparison_with_column():
    ranked = score_dataframe(pd.read_csv("data/topics.csv"))
    ranked["expert_rank"] = list(range(1, len(ranked) + 1))
    ec = ev.expert_comparison(ranked)
    assert ec["available"] is True
    assert abs(ec["fuzzy_spearman"] - 1.0) < 1e-9  # ranks already match fuzzy order


def test_data_validation_probes(report):
    rep, _ = report
    dv = rep["data_validation"]
    assert dv["dataset_valid"]
    assert dv["all_probes_reject"]


def test_sensitivity_curves_shape(report):
    rep, _ = report
    curves = rep["sensitivity"]
    assert set(curves) == {
        "others_low", "others_medium", "others_high", "resources_low_F_C_high",
    }
    for by_input in curves.values():
        for points in by_input.values():
            assert len(points) >= 21
            assert all(0 <= p["score"] <= 100 for p in points)


def test_sensitivity_increases_with_input():
    curves = ev.sensitivity_curves(step=5)
    points = curves["others_high"]["F"]
    assert points[-1]["score"] > points[0]["score"]


def test_format_report_renders(report):
    rep, _ = report
    text = ev.format_report(rep)
    assert "FuzzyLearn evaluation report" in text
    assert "Monotonicity" in text
    assert "Worked example" in text


def test_monotonicity_report_runs():
    mo = ev.monotonicity_report(step=10)
    assert mo["checked"] > 0
    assert isinstance(mo["n_violations"], int)


def test_run_all_matches_direct_scoring(df):
    rep, ranked = ev.run_all(df, step=10)
    direct = score_dataframe(df)
    assert list(ranked["topic"]) == list(direct["topic"])
    assert np.allclose(ranked["fuzzy_score"], direct["fuzzy_score"])
