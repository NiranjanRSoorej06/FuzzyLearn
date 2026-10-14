import pytest
from fuzzy_engine import membership_low, membership_medium, membership_high, fuzzify


def test_boundaries_low():
    assert membership_low(0) == 1.0
    assert membership_low(25) == 1.0
    assert membership_low(50) == 0.0
    assert membership_low(100) == 0.0


def test_boundaries_medium():
    assert membership_medium(25) == 0.0
    assert membership_medium(50) == 1.0
    assert membership_medium(75) == 0.0


def test_boundaries_high():
    assert membership_high(0) == 0.0
    assert membership_high(50) == 0.0
    assert membership_high(75) == 1.0
    assert membership_high(100) == 1.0


def test_example_f_65():
    m = fuzzify(65, 82, 90)["F"]
    assert m["Low"] == 0.0
    assert m["Medium"] == pytest.approx(0.4)
    assert m["High"] == pytest.approx(0.6)


def test_sum_to_one():
    for x in [0, 1, 25, 37, 50, 65, 80, 100]:
        m = fuzzify(x, x, x)["F"]
        assert sum(m.values()) == pytest.approx(1.0)
