import pandas as pd
import pytest

from baseline import weighted_score, rank_baseline, DEFAULT_WEIGHTS, WEIGHT_VARIANTS


def test_default_formula():
    # S = 0.35*65 + 0.30*82 + 0.35*90
    assert weighted_score(65, 82, 90) == pytest.approx(0.35 * 65 + 0.30 * 82 + 0.35 * 90)


def test_zero_and_full():
    assert weighted_score(0, 0, 0) == 0.0
    assert weighted_score(100, 100, 100) == pytest.approx(100.0)


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)


def test_variant_weights_sum_to_one():
    for name, w in WEIGHT_VARIANTS.items():
        assert sum(w) == pytest.approx(1.0), name


def test_rank_baseline_orders_by_score():
    df = pd.DataFrame([
        {"topic": "A", "failure_rate": 10, "confusion": 10, "resource_inadequacy": 10},
        {"topic": "B", "failure_rate": 90, "confusion": 90, "resource_inadequacy": 90},
        {"topic": "C", "failure_rate": 50, "confusion": 50, "resource_inadequacy": 50},
    ])
    out = rank_baseline(df)
    ranks = dict(zip(out["topic"], out["baseline_rank"]))
    assert ranks == {"B": 1, "C": 2, "A": 3}


def test_rank_baseline_preserves_row_order():
    df = pd.DataFrame([
        {"topic": "A", "failure_rate": 10, "confusion": 10, "resource_inadequacy": 10},
        {"topic": "B", "failure_rate": 90, "confusion": 90, "resource_inadequacy": 90},
    ])
    out = rank_baseline(df)
    assert list(out["topic"]) == ["A", "B"]


def test_rank_baseline_tie_breaks_by_topic():
    df = pd.DataFrame([
        {"topic": "Zeta", "failure_rate": 50, "confusion": 50, "resource_inadequacy": 50},
        {"topic": "Alpha", "failure_rate": 50, "confusion": 50, "resource_inadequacy": 50},
    ])
    out = rank_baseline(df)
    ranks = dict(zip(out["topic"], out["baseline_rank"]))
    assert ranks["Alpha"] == 1
    assert ranks["Zeta"] == 2
