"""Patient-history features built only from columns available at prediction time.

Every feature comes from `off`, `on`, `ledd`, ages and visit order. None uses `target`,
which does not exist in the test set (see `experiments/leakage_check.py` for what happens
when it does).
"""

from __future__ import annotations

import pandas as pd

# Columns never used as model inputs: identifiers, categorical metadata, helper columns.
EXCLUDE = {"Index", "patient_id", "cohort", "gene", "rater_id", "target", "age_first_visit"}

# The 8 raw numeric columns used by the first models (Ridge, plain gradient boosting).
RAW_FEATURES = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]  # fmt: skip


def sort_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Visits in chronological order within each patient (age increases between visits)."""
    return df.sort_values(["patient_id", "age"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags, deltas, expanding/rolling aggregates, visit timing and missingness flags.

    `df` must be sorted with `sort_visits`. The current visit is always excluded from the
    aggregates (`shift(1)`), so a visit only sees the patient's earlier visits.
    """
    if not df["patient_id"].is_monotonic_increasing:
        raise ValueError("sort the visits with sort_visits() first")
    df = df.copy()
    g = df.groupby("patient_id", sort=False)

    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]
        shifted = g[col].shift(1)
        by_patient = shifted.groupby(df["patient_id"])
        df[f"{col}_cumean"] = by_patient.expanding().mean().values
        df[f"{col}_cumax"] = by_patient.expanding().max().values
        df[f"{col}_cumin"] = by_patient.expanding().min().values
        df[f"{col}_custd"] = by_patient.expanding().std().values
        df[f"{col}_roll3"] = by_patient.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll5"] = by_patient.transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Acceleration of the change between visits.
    df["off_lag2_delta"] = df["off_lag1"] - df["off_lag2"]
    df["on_lag2_delta"] = df["on_lag1"] - df["on_lag2"]

    # Position of the visit in the patient's trajectory.
    df["age_delta"] = g["age"].diff()
    df["visit_number"] = g.cumcount() + 1
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"] = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]

    # Missing values are clinical signals: no OFF score means an ON-only visit.
    df["off_missing"] = df["off"].isna().astype(int)
    df["on_missing"] = df["on"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    df["off_count"] = (
        df.groupby("patient_id", sort=False)["off_missing"]
        .transform(lambda x: (1 - x).shift(1).expanding().sum())
        .fillna(0)
    )

    # Interactions between treatment dose and motor scores.
    df["on_off_diff"] = df["on"] - df["off"]
    df["ledd_times_off"] = df["ledd"] * df["off"]
    df["ledd_times_on"] = df["ledd"] * df["on"]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("target")]
