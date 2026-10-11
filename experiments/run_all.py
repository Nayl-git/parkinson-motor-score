"""Replays the model progression with patient-grouped cross-validation and writes
`results/metrics.json` and `submissions/honest_model.csv`.

    python experiments/run_all.py            # 10-seed ensemble, about 10 minutes on a laptop
    python experiments/run_all.py --seeds 2  # quicker check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parkinson import best_model, patient_folds  # noqa: E402
from parkinson.data import load  # noqa: E402
from parkinson.features import RAW_FEATURES, build_features, feature_columns, sort_visits  # noqa: E402


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def oof_predictions(model, X, y, folds) -> np.ndarray:
    """Out-of-fold predictions: each visit is predicted by a model that never saw its patient."""
    preds = np.full(len(y), np.nan)
    for train_idx, val_idx in folds:
        fitted = model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[val_idx] = fitted.predict(X.iloc[val_idx])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="models in the final ensemble")
    args = parser.parse_args()
    start = time.time()

    train, test = load()
    visits = sort_visits(train)
    test = sort_visits(test)
    y = visits["target"]
    folds = patient_folds(visits, y, visits["patient_id"])
    results: dict[str, float | int] = {"visits": len(visits), "patients": int(visits["patient_id"].nunique())}

    raw = visits[RAW_FEATURES]
    steps = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge_raw": make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)),
        "hgbr_raw": HistGradientBoostingRegressor(random_state=0),
    }
    for name, model in steps.items():
        results[name] = rmse(y, oof_predictions(model, raw, y, folds))
        print(f"{name:<16} RMSE {results[name]:.2f}")

    visits = build_features(visits)
    test = build_features(test)
    cols = feature_columns(visits)
    X = visits[cols]
    results["features"] = len(cols)
    results["honest_single_seed"] = rmse(y, oof_predictions(best_model(0), X, y, folds))
    print(f"{'honest (seed 0)':<16} RMSE {results['honest_single_seed']:.2f}  ({len(cols)} features)")

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(test))
    for seed in range(args.seeds):
        oof += oof_predictions(best_model(seed), X, y, folds)
        test_pred += best_model(seed).fit(X, y).predict(test[cols])
    oof /= args.seeds
    test_pred /= args.seeds
    results["ensemble_seeds"] = args.seeds
    results["honest_ensemble"] = rmse(y, oof)
    results["test_prediction_mean"] = float(test_pred.mean())
    results["train_target_mean"] = float(y.mean())
    print(f"{'honest ensemble':<16} RMSE {results['honest_ensemble']:.2f}  ({args.seeds} seeds)")
    print(f"mean prediction on test {test_pred.mean():.1f} vs mean target on train {y.mean():.1f}")

    (ROOT / "submissions").mkdir(exist_ok=True)
    test.assign(target=test_pred)[["Index", "target"]].sort_values("Index").to_csv(
        ROOT / "submissions" / "honest_model.csv", index=False
    )
    results["minutes"] = round((time.time() - start) / 60, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
