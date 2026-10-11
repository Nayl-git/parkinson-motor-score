"""Why the "better" models were withdrawn: target leakage.

During the hackathon, features such as `target_lag1` (the target of the patient's previous
visit) lowered the cross-validated RMSE a lot. But `target` does not exist in the test set,
so these features are always empty at prediction time. This script measures the same model
twice on the same patient-grouped folds:

1. validation visits keep `target_lag1` (what the cross-validation reported);
2. validation visits lose it, as test visits do (what the leaderboard would see).

    python experiments/leakage_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parkinson import best_model, patient_folds  # noqa: E402
from parkinson.data import load  # noqa: E402
from parkinson.features import build_features, feature_columns, sort_visits  # noqa: E402


def main() -> None:
    train, _ = load()
    visits = build_features(sort_visits(train))
    visits["target_lag1"] = visits.groupby("patient_id", sort=False)["target"].shift(1)
    y = visits["target"]
    cols = feature_columns(visits) + ["target_lag1"]
    X = visits[cols]
    folds = patient_folds(X, y, visits["patient_id"])
    print(f"target_lag1 available for {visits['target_lag1'].notna().mean():.0%} of training visits")

    seen, hidden = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for train_idx, val_idx in folds:
        model = best_model(0).fit(X.iloc[train_idx], y.iloc[train_idx])
        X_val = X.iloc[val_idx]
        seen[val_idx] = model.predict(X_val)
        seen_as_test = X_val.copy()
        seen_as_test["target_lag1"] = np.nan  # the test set has no target
        hidden[val_idx] = model.predict(seen_as_test)

    def rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - y.values) ** 2)))

    results = {"with_target_lag1": rmse(seen), "target_lag1_missing_like_test": rmse(hidden)}
    print(f"RMSE with target_lag1 in validation : {results['with_target_lag1']:.2f}")
    print(f"RMSE with target_lag1 missing (test): {results['target_lag1_missing_like_test']:.2f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "leakage.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
