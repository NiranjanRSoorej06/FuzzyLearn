import pandas as pd

REQUIRED_COLUMNS = [
    "course", "topic", "failure_rate", "confusion",
    "resource_inadequacy", "barrier_type", "source",
]
VALID_BARRIER_TYPES = {
    "visualization", "application", "unclear_explanation", "process_demo",
    "self_check", "navigation", "relationships",
}
VALID_SOURCES = {"real", "synthetic"}


def validate_dataframe(df):
    errors = []
    # columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: %s" % ", ".join(missing))
        return False, errors
    # missing values
    for col in REQUIRED_COLUMNS:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            errors.append("Missing values in column '%s'" % col)
    # numeric ranges
    for col in ["failure_rate", "confusion", "resource_inadequacy"]:
        try:
            vals = pd.to_numeric(df[col], errors="raise")
        except Exception:
            errors.append("Non-numeric values in column '%s'" % col)
            continue
        if ((vals < 0) | (vals > 100)).any():
            errors.append("Values out of range [0,100] in column '%s'" % col)
    # barrier types
    invalid_bt = set(df["barrier_type"].astype(str).str.strip()) - VALID_BARRIER_TYPES
    if invalid_bt:
        errors.append("Invalid barrier_type values: %s" % ", ".join(sorted(invalid_bt)))
    # source
    invalid_src = set(df["source"].astype(str).str.strip()) - VALID_SOURCES
    if invalid_src:
        errors.append("Invalid source values: %s" % ", ".join(sorted(invalid_src)))
    # duplicate topics
    if df["topic"].astype(str).str.strip().duplicated().any():
        errors.append("Duplicate topic rows found")
    # course/topic non-empty already via missing check
    ok = len(errors) == 0
    return ok, errors
