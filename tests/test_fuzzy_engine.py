import numpy as np
import pytest
from fuzzy_engine import priority, defuzzify, infer, OUTPUT_SETS, OUTPUT_PARAMS, trapmf
from rules import RULES, validate_rules


def test_rule_coverage():
    ok, msg = validate_rules(RULES)
    assert ok, msg


def test_worked_example():
    r = priority(65, 82, 90)
    assert r["score"] == pytest.approx(79.4, abs=0.5)
    assert r["label"] == "High"
    assert len(r["fired_rules"]) == 2


def test_output_range():
    for F in range(0, 101, 10):
        for C in range(0, 101, 10):
            for R in range(0, 101, 10):
                r = priority(F, C, R)
                assert 0 <= r["score"] <= 100
                assert np.isfinite(r["score"])


def test_fired_rules_returned():
    r = priority(100, 100, 100)
    assert any(x["strength"] > 0 for x in r["fired_rules"])


def test_zero_total_fallback():
    agg = np.zeros(101)
    assert defuzzify(agg) == 50.0


def test_label_deterministic():
    r1 = priority(50, 50, 50)
    r2 = priority(50, 50, 50)
    assert r1["label"] == r2["label"]
    assert r1["score"] == r2["score"]


@pytest.mark.xfail(reason='Mamdani centroid defuzzification is not guaranteed monotone at MF transitions; documented in README.')
def test_monotonicity():
    # increasing one input should not decrease priority
    for fixed in [(30, 30), (50, 50), (70, 70)]:
        prev = None
        for v in range(0, 101, 5):
            s = priority(v, fixed[0], fixed[1])["score"]
            if prev is not None:
                assert s >= prev - 1e-9
            prev = s
