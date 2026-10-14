import pandas as pd
import pytest
from data_validation import validate_dataframe

VALID = pd.DataFrame([{
    "course": "DS", "topic": "T", "failure_rate": 50, "confusion": 60,
    "resource_inadequacy": 70, "barrier_type": "visualization", "source": "synthetic",
}])


def test_valid():
    ok, errs = validate_dataframe(VALID)
    assert ok and not errs


def test_missing_column():
    df = VALID.drop(columns=["topic"])
    ok, errs = validate_dataframe(df)
    assert not ok


def test_out_of_range():
    df = VALID.copy()
    df["failure_rate"] = 120
    ok, errs = validate_dataframe(df)
    assert not ok


def test_invalid_barrier():
    df = VALID.copy()
    df["barrier_type"] = "foo"
    ok, errs = validate_dataframe(df)
    assert not ok


def test_duplicate_topic():
    df = pd.concat([VALID, VALID], ignore_index=True)
    ok, errs = validate_dataframe(df)
    assert not ok


def test_invalid_source():
    df = VALID.copy()
    df["source"] = "unknown"
    ok, errs = validate_dataframe(df)
    assert not ok
