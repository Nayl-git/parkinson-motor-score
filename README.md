# Predicting unbiased Parkinson's motor scores

[![CI](https://github.com/Nayl-git/parkinson-motor-score/actions/workflows/ci.yml/badge.svg)](https://github.com/Nayl-git/parkinson-motor-score/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-gradient_boosting-F7931E?logo=scikitlearn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Bobathon_ESILV-20BEFF?logo=kaggle&logoColor=white)

My team's solution to the **Bobathon ESILV** (7–8 October 2026), a two-day machine-learning
hackathon run on Kaggle by [Probabl](https://probabl.ai) (makers of skore) and IBM: for every visit of a Parkinson's patient, predict the **unbiased ("true") OFF
motor score** (MDS-UPDRS part III, 0 to 132).

**Result:** patient-grouped cross-validated RMSE from **16.5** (predicting the mean) to **4.64** (−72%),
with features built only from information available at prediction time. Along the way, a model
that looked twice as good turned out to be **target leakage**, and was withdrawn.

## The problem

The OFF score (motor exam without medication effect) tracks how far the disease has progressed,
but in practice it is biased: rater subjectivity, missing values, fluctuating levodopa levels at
exam time, and OFF exams are often skipped. The organisers built a synthetic multi-cohort dataset
(44,590 training visits from 5,576 patients, 11,013 test visits from unseen patients) with a
"true OFF" target to recover. Visits are irregular; ON, OFF and dose (LEDD) values are often missing.

## Results

Root mean squared error of out-of-fold predictions, 5 folds grouped by patient (a patient is
never in training and validation at once, as in the Kaggle split):

| Step | Model | Features | RMSE |
|---|---|---|---:|
| Baseline | Predict the mean | – | 16.50 |
| Linear | Ridge, median imputation | 8 raw columns | 10.60 |
| Trees | HistGradientBoosting (missing values handled natively) | 8 raw columns | 7.75 |
| **Final** | **HistGradientBoosting, tuned, 10-seed ensemble** | **48 patient-history features** | **4.64** |

The final model predicts a mean of 36.6 on the test visits against 37.5 for the training target:
no sign of a train/test mismatch. Kaggle leaderboard position: *to be added*.

## What made the difference

- **Patient history instead of isolated visits.** Visits are sorted by age within each patient; each
  visit gets its previous OFF, ON and dose values, visit-to-visit changes, expanding and rolling
  statistics over earlier visits, the visit number and the time since diagnosis and first visit.
- **Missingness as a signal.** A missing OFF score means an ON-only visit; explicit flags and a
  count of earlier OFF exams help the trees.
- **Evaluation that matches the test.** Grouping the folds by patient removed the optimism of a
  random split and made cross-validation a reliable guide.

## The leak we caught

Adding the previous visit's **target** (`target_lag1`) made cross-validation look excellent. But the
test set has no target column, so this feature is always empty at prediction time.
`experiments/leakage_check.py` measures the same model on the same folds:

| Validation visits | RMSE |
|---|---:|
| keep `target_lag1` (what cross-validation reported) | 2.54 |
| lose `target_lag1`, as test visits do | 11.90 |

The warning sign during the hackathon was the distribution of test predictions, far below the
training target. We withdrew those models and kept the honest one.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python experiments/run_all.py         # downloads the data, replays every step (~15 min, 10 seeds)
python experiments/leakage_check.py   # the leakage measurement
pytest -q                             # feature tests: no target use, no cross-patient history
```

The data is downloaded from the organisers' [public repository](https://github.com/probabl-ai/bobathon-esilv)
and checked by SHA-256; it is not redistributed here. `run_all.py` writes `results/metrics.json` and
`submissions/honest_model.csv`, identical to our final submission file from the hackathon.

## Repository

```
src/parkinson/features.py      leak-free patient-history features
src/parkinson/data.py          data download and SHA-256 check
src/parkinson/__init__.py      model settings and patient-grouped folds
experiments/run_all.py         baseline → linear → trees → final ensemble
experiments/leakage_check.py   target leakage measurement
results/                       metrics produced by the experiments
tests/                         unit tests of the feature pipeline
.github/workflows/ci.yml       lint, tests, data check and a 1-seed replay on every push
```

## Credits

Team of 4, ESILV. The rules required writing all code in IBM's AI coding assistant, Bob, and
evaluating models with skore; this repository is a cleaned-up version of our scripts, without the
Skore Hub reporting. Challenge, data and scientific context:
[probabl-ai/bobathon-esilv](https://github.com/probabl-ai/bobathon-esilv) (the Kaggle
competition itself is invitation-only).
