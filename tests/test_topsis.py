import warnings

import numpy as np
import pandas as pd
import pytest

from topsis import (
    load_actions, orient_normalise, equal_weights, critic_weights,
    check_weights, topsis_crisp, rank_desc,
)
from fuzzy_topsis import degenerate, normalise, topsis_chen
from mamdani_adapter import score_actions as mamdani_score_actions

ACTIONS_PATH = "data/actions.csv"
REFERENCE_PATH = "data/reference_results.csv"
CRITERIA_COLS = ["diff_c", "imp_c", "resource_gap", "effort_hours"]
BENEFIT = [True, True, True, False]


@pytest.fixture(scope="module")
def df():
    return load_actions(ACTIONS_PATH)


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(REFERENCE_PATH)


@pytest.fixture(scope="module")
def matrices(df):
    Xc = df[CRITERIA_COLS].to_numpy(dtype=float)
    T = np.stack([
        df[["diff_l", "diff_c", "diff_u"]].to_numpy(dtype=float),
        df[["imp_l", "imp_c", "imp_u"]].to_numpy(dtype=float),
        np.repeat(df[["resource_gap"]].to_numpy(dtype=float), 3, axis=1),
        np.repeat(df[["effort_hours"]].to_numpy(dtype=float), 3, axis=1),
    ], axis=1)
    return Xc, T


# --------------------------------------------------------------------- brief sample
def test_brief_sample():
    X = [[8, 100, 20000], [6, 80, 10000], [4, 40, 15000]]
    benefit = [True, True, False]
    w = [0.4, 0.4, 0.2]
    cc, R, V, dp, dm = topsis_crisp(X, w, benefit)
    assert np.allclose(cc, [0.7388, 0.6179, 0.1483], atol=1e-4)
    assert np.allclose(dp, [0.2000, 0.2404, 0.5745], atol=1e-4)
    assert np.allclose(dm, [0.5657, 0.3887, 0.1000], atol=1e-4)
    assert list(rank_desc(cc)) == [1, 2, 3]


# --------------------------------------------------------------------- A1 hand values
def test_a1_hand_values(df, matrices):
    Xc, T = matrices
    w_eq = equal_weights(4)
    cc_crisp, R, *_ = topsis_crisp(Xc, w_eq, BENEFIT)
    a1 = df.index[df["alt_id"] == "A1"][0]
    assert np.allclose(R[a1], [1.0, 0.7674, 1.0, 0.0909], atol=1e-4)
    assert cc_crisp[a1] == pytest.approx(0.6320, abs=1e-4)

    cc_fuzzy, *_ = topsis_chen(T, w_eq, BENEFIT)
    assert cc_fuzzy[a1] == pytest.approx(0.1828, abs=1e-4)


# --------------------------------------------------------------------- reference reproduction
def test_reproduces_reference_crisp(df, matrices, reference):
    Xc, T = matrices
    w_eq = equal_weights(4)
    cc, *_ = topsis_crisp(Xc, w_eq, BENEFIT)
    ref = reference.set_index("alt_id").loc[df["alt_id"]]
    assert np.allclose(cc, ref["TOPSIS_crisp"].to_numpy(), atol=1e-4)
    assert list(rank_desc(cc)) == list(ref["rank_crisp"])


def test_reproduces_reference_chen_degenerate(df, matrices, reference):
    Xc, T = matrices
    w_eq = equal_weights(4)
    cc, *_ = topsis_chen(degenerate(T), w_eq, BENEFIT)
    ref = reference.set_index("alt_id").loc[df["alt_id"]]
    assert np.allclose(cc, ref["Chen_degenerate"].to_numpy(), atol=1e-4)
    assert list(rank_desc(cc)) == list(ref["rank_chen_deg"])


def test_reproduces_reference_chen_fuzzy(df, matrices, reference):
    Xc, T = matrices
    w_eq = equal_weights(4)
    cc, *_ = topsis_chen(T, w_eq, BENEFIT)
    ref = reference.set_index("alt_id").loc[df["alt_id"]]
    assert np.allclose(cc, ref["Chen_fuzzy"].to_numpy(), atol=1e-4)
    assert list(rank_desc(cc)) == list(ref["rank_chen_fuzzy"])


# --------------------------------------------------------------------- weights
def test_equal_weights_sum_to_one():
    w = equal_weights(4)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.25)


def test_critic_weights(matrices):
    Xc, _ = matrices
    w = critic_weights(Xc, BENEFIT)
    assert np.allclose(w, [0.2826, 0.2204, 0.1906, 0.3064], atol=1e-3)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_check_weights_rejects_invalid():
    with pytest.raises(ValueError):
        check_weights([0.5, 0.6])  # doesn't sum to 1
    with pytest.raises(ValueError):
        check_weights([-0.1, 1.1])  # negative
    with pytest.raises(ValueError):
        check_weights([np.nan, 1.0])  # not finite


# --------------------------------------------------------------------- guards
def test_constant_column_normalised_to_zero_with_warning():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    with pytest.warns(UserWarning):
        R, constant = orient_normalise(X, [True, True])
    assert constant[1]
    assert np.allclose(R[:, 1], 0.0)


def test_zero_distance_denominator_handled():
    X = np.array([[5.0, 5.0], [5.0, 5.0]])
    with pytest.warns(UserWarning):
        cc, *_ = topsis_crisp(X, [0.5, 0.5], [True, True])
    assert np.allclose(cc, 0.5)


def test_crisp_scores_finite_and_bounded(matrices):
    Xc, _ = matrices
    cc, *_ = topsis_crisp(Xc, equal_weights(4), BENEFIT)
    assert np.all(np.isfinite(cc))
    assert np.all((cc >= 0) & (cc <= 1))


# --------------------------------------------------------------------- fuzzy admissibility
def test_fuzzy_admissibility_after_normalisation(matrices):
    _, T = matrices
    N = normalise(T, BENEFIT)
    assert (N[:, :, 0] <= N[:, :, 1] + 1e-9).all()
    assert (N[:, :, 1] <= N[:, :, 2] + 1e-9).all()


def test_degenerate_matches_centre_values(matrices):
    Xc, T = matrices
    w_eq = equal_weights(4)
    cc_deg, *_ = topsis_chen(degenerate(T), w_eq, BENEFIT)
    # degenerate fuzzy input normalised under Chen's own rule should match
    # treating the centre values as crisp l=c=u triples throughout.
    Td = degenerate(T)
    assert np.allclose(Td[:, :, 0], Td[:, :, 1])
    assert np.allclose(Td[:, :, 1], Td[:, :, 2])
    assert np.all(np.isfinite(cc_deg))


# --------------------------------------------------------------------- data validation
def test_load_actions_valid():
    df = load_actions(ACTIONS_PATH)
    assert len(df) == 10
    assert "data_status" in df.columns


def test_load_actions_rejects_missing_column(tmp_path):
    df = load_actions(ACTIONS_PATH).drop(columns=["imp_c"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_actions(path)


def test_load_actions_rejects_l_gt_c(tmp_path):
    df = load_actions(ACTIONS_PATH).copy()
    df.loc[0, "diff_l"] = df.loc[0, "diff_c"] + 10
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_actions(path)


def test_load_actions_rejects_out_of_range(tmp_path):
    df = load_actions(ACTIONS_PATH).copy()
    df.loc[0, "diff_c"] = 150
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_actions(path)


def test_load_actions_rejects_duplicate_id(tmp_path):
    df = load_actions(ACTIONS_PATH).copy()
    df.loc[1, "alt_id"] = df.loc[0, "alt_id"]
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_actions(path)


def test_load_actions_rejects_missing_data_status(tmp_path):
    df = load_actions(ACTIONS_PATH).copy()
    df["data_status"] = None
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_actions(path)


# --------------------------------------------------------------------- Mamdani adapter
def test_mamdani_adapter_scores(df):
    scores = mamdani_score_actions(df)
    assert len(scores) == len(df)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 100))
