"""Predicting the unbiased OFF motor score of Parkinson's patients (Bobathon ESILV 2026)."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# Configuration kept after the grid search run during the hackathon.
BEST_HGBR = dict(max_iter=1000, learning_rate=0.01, max_leaf_nodes=63, min_samples_leaf=5)
N_SPLITS = 5


def patient_folds(X, y, groups):
    """5 folds split by patient: a patient is never in training and validation at once."""
    return list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=groups))


def best_model(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **BEST_HGBR)
