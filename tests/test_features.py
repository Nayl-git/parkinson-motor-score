import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parkinson import patient_folds  # noqa: E402
from parkinson.features import build_features, feature_columns, sort_visits  # noqa: E402


def toy_visits() -> pd.DataFrame:
    """Two patients, visits given out of order on purpose."""
    return pd.DataFrame(
        {
            "Index": [0, 1, 2, 3, 4],
            "patient_id": ["B", "A", "A", "B", "A"],
            "cohort": ["x"] * 5,
            "sexM": [1, 0, 0, 1, 0],
            "gene": ["g"] * 5,
            "age_at_diagnosis": [60.0, 50.0, 50.0, 60.0, 50.0],
            "age": [62.0, 53.0, 51.0, 61.0, 52.0],
            "ledd": [300.0, 500.0, 100.0, 200.0, np.nan],
            "time_since_intake_on": [1.0] * 5,
            "time_since_intake_off": [12.0] * 5,
            "rater_id": ["R1"] * 5,
            "on": [12.0, 14.0, 10.0, 11.0, 13.0],
            "off": [32.0, 30.0, 20.0, np.nan, 25.0],
            "target": [33.0, 31.0, 21.0, 30.0, 26.0],
        }
    )


def test_unsorted_visits_are_refused():
    with pytest.raises(ValueError):
        build_features(toy_visits())


def test_lags_only_look_at_earlier_visits_of_the_same_patient():
    df = build_features(sort_visits(toy_visits()))
    a = df[df["patient_id"] == "A"]
    assert a["age"].tolist() == [51.0, 52.0, 53.0]
    assert np.isnan(a["off_lag1"].iloc[0])  # first visit: no history
    assert a["off_lag1"].tolist()[1:] == [20.0, 25.0]
    assert a["off_cumean"].iloc[2] == pytest.approx(22.5)  # mean of the two earlier visits
    b = df[df["patient_id"] == "B"]
    assert np.isnan(b["off_lag1"].iloc[0])  # never borrows patient A's last visit
    assert b["visit_number"].tolist() == [1, 2]


def test_missingness_flags_and_counts():
    df = build_features(sort_visits(toy_visits()))
    b = df[df["patient_id"] == "B"]
    assert b["off_missing"].tolist() == [1, 0]
    assert b["off_count"].tolist() == [0, 0]  # earlier visits with an OFF score
    a = df[df["patient_id"] == "A"]
    assert a["off_count"].tolist() == [0, 1, 2]


def test_no_feature_uses_the_target():
    with_target = build_features(sort_visits(toy_visits()))
    without = build_features(sort_visits(toy_visits().drop(columns="target")))
    cols = feature_columns(with_target)
    assert "target" not in cols and not any(c.startswith("target") for c in cols)
    assert cols == feature_columns(without)
    pd.testing.assert_frame_equal(with_target[cols], without[cols])


def test_folds_never_share_a_patient():
    df = sort_visits(toy_visits())
    df = pd.concat([df.assign(patient_id=df["patient_id"] + str(i)) for i in range(5)], ignore_index=True)
    for train_idx, val_idx in patient_folds(df, df["target"], df["patient_id"]):
        assert not set(df["patient_id"].iloc[train_idx]) & set(df["patient_id"].iloc[val_idx])
