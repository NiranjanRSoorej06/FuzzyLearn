import math

import pytest

from fuzzy_engine import (
    priority, defuzzify, infer, trapmf,
    OUTPUT_SETS,
)
from rules import RULES, validate_rules


def test_rule_coverage():
    ok, msg = validate_rules(RULES)
    assert ok, msg


def test_each_combination_exactly_once():
    keys = [(r["failure_set"], r["confusion_set"], r["resource_set"]) for r in RULES]
    assert len(keys) == 27
    assert len(set(keys)) == 27


def test_worked_example_score_and_label():
    r = priority(65, 82, 90)
    assert r["score"] == pytest.approx(79.4, abs=0.5)
    assert r["label"] == "High"


def test_worked_example_fires_two_rules_with_expected_strengths():
    r = priority(65, 82, 90)
    fired = r["fired_rules"]
    assert len(fired) == 2
    top = fired[0]
    assert top["consequent"] == "Very High"
    assert top["strength"] == pytest.approx(0.6)
    second = fired[1]
    assert second["consequent"] == "High"
    assert second["strength"] == pytest.approx(0.4)


def test_memberships_in_worked_example():
    r = priority(65, 82, 90)
    m = r["memberships"]
    assert m["F"]["Low"] == pytest.approx(0.0)
    assert m["F"]["Medium"] == pytest.approx(0.4)
    assert m["F"]["High"] == pytest.approx(0.6)
    assert m["C"]["High"] == pytest.approx(1.0)
    assert m["R"]["High"] == pytest.approx(1.0)


def test_output_coverage():
    for F in range(0, 101, 10):
        for C in range(0, 101, 10):
            for R in range(0, 101, 10):
                r = priority(F, C, R)
                assert 0 <= r["score"] <= 100
                assert math.isfinite(r["score"])


def test_extremes():
    low = priority(0, 0, 0)
    high = priority(100, 100, 100)
    assert low["score"] < high["score"]
    assert low["label"] in OUTPUT_SETS
    assert high["label"] in OUTPUT_SETS


def test_fired_rules_returned():
    r = priority(100, 100, 100)
    assert any(x["strength"] > 0 for x in r["fired_rules"])


def test_zero_total_fallback():
    agg = __import__("numpy").zeros(101)
    assert defuzzify(agg) == 50.0


def test_label_deterministic():
    r1 = priority(50, 50, 50)
    r2 = priority(50, 50, 50)
    assert r1["label"] == r2["label"]
    assert r1["score"] == r2["score"]


def test_out_of_range_input_rejected():
    with pytest.raises(ValueError):
        priority(101, 50, 50)
    with pytest.raises(ValueError):
        priority(50, -1, 50)
    with pytest.raises(ValueError):
        priority("high", 50, 50)


def test_infer_returns_grid_and_aggregated():
    res = infer(65, 82, 90)
    assert len(res["grid"]) == 101
    assert len(res["aggregated"]) == 101
    assert res["aggregated"].max() <= 1.0


def test_trapmf_array_and_scalar_agree():
    import numpy as np
    xs = np.arange(0, 101, dtype=float)
    vec = trapmf(xs, 0, 0, 25, 50)
    sca = [float(trapmf(float(x), 0, 0, 25, 50)) for x in xs]
    assert np.allclose(vec, sca)


def test_monotonicity_broadly_increases():
    """Rising failure rate must not lower the score more than a small epsilon.

    Mamdani centroid defuzzification is not guaranteed numerically monotone at
    membership transitions, so a tiny tolerance and coarse steps are used.
    """
    prev = -1.0
    worst_drop = 0.0
    for v in range(0, 101, 5):
        s = priority(v, 50, 50)["score"]
        worst_drop = min(worst_drop, s - prev)
        prev = s
    assert worst_drop >= -1.0, "worst drop %.3f exceeds 1.0 tolerance" % worst_drop


@pytest.mark.parametrize("fixed", [(20, 20), (50, 50), (80, 80)])
def test_monotonicity_endpoints(fixed):
    assert priority(0, *fixed)["score"] <= priority(100, *fixed)["score"] + 1e-9
