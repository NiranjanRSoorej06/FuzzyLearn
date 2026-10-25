"""Validation for the topic-level dataset described in section 4 / section 10.2."""

import pandas as pd

REQUIRED_COLUMNS = [
    "course", "topic", "failure_rate", "confusion",
    "resource_inadequacy", "barrier_type", "source",
]
NUMERIC_COLUMNS = ["failure_rate", "confusion", "resource_inadequacy"]
VALID_BARRIER_TYPES = {
    "visualization", "application", "unclear_explanation", "process_demo",
    "self_check", "navigation", "relationships",
}
VALID_SOURCES = {"real", "synthetic"}


def validate_dataframe(df):
    """Validate a topics dataframe.

    Returns (ok, errors) where errors is a list of human-readable strings.
    Every problem is reported; validation does not stop at the first error.
    """
    errors = []

    if not isinstance(df, pd.DataFrame):
        return False, ["Input is not a pandas DataFrame"]
    if df.empty:
        return False, ["Dataset is empty"]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: %s" % ", ".join(missing))
        return False, errors

    for col in REQUIRED_COLUMNS:
        values = df[col]
        blank = values.isna() | (values.astype(str).str.strip() == "")
        if blank.any():
            row_ids = ", ".join(str(i) for i in df.index[blank][:5])
            errors.append("Missing values in column '%s' (rows: %s)" % (col, row_ids))

    for col in NUMERIC_COLUMNS:
        try:
            vals = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            errors.append("Non-numeric values in column '%s'" % col)
            continue
        out_of_range = (vals < 0) | (vals > 100) | vals.isna()
        if out_of_range.any():
            row_ids = ", ".join(str(i) for i in df.index[out_of_range][:5])
            errors.append("Values out of range [0,100] in column '%s' (rows: %s)" % (col, row_ids))

    invalid_bt = sorted(
        set(df["barrier_type"].astype(str).str.strip()) - VALID_BARRIER_TYPES
    )
    if invalid_bt:
        errors.append("Invalid barrier_type values: %s" % ", ".join(invalid_bt))

    invalid_src = sorted(set(df["source"].astype(str).str.strip()) - VALID_SOURCES)
    if invalid_src:
        errors.append("Invalid source values: %s" % ", ".join(invalid_src))

    course_topic = (
        df["course"].astype(str).str.strip() + " :: " + df["topic"].astype(str).str.strip()
    )
    duplicated = course_topic.duplicated(keep=False)
    if duplicated.any():
        dups = sorted(set(course_topic[duplicated]))
        errors.append("Duplicate course-topic rows: %s" % "; ".join(dups))

    return len(errors) == 0, errors


def load_and_validate(path):
    """Load a CSV and validate it. Returns (df, ok, errors)."""
    df = pd.read_csv(path)
    ok, errors = validate_dataframe(df)
    return df, ok, errors
