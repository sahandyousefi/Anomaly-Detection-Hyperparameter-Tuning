# Roadmap

This file tracks the planned refactor of the project from a single exploratory notebook into a tested, reproducible fraud detection codebase. Items are completed in order. Each item is scoped to one focused commit.

## Target layout

```
Anomaly-Detection-Hyperparameter-Tuning/
|-- .github/workflows/ci.yml
|-- configs/
|   |-- default.yaml
|-- data/
|   |-- README.md                  # download instructions; raw data is not tracked
|-- docs/
|   |-- methodology.md
|   |-- model_card.md
|   |-- figures/
|-- notebooks/
|   |-- 00_original_exploration.ipynb
|   |-- 01_exploratory_analysis.ipynb
|   |-- 02_modeling_walkthrough.ipynb
|-- reports/
|   |-- figures/
|-- src/fraud_detection/
|   |-- __init__.py
|   |-- config.py
|   |-- cli.py
|   |-- data/                      # load.py, split.py, resample.py, synthetic.py
|   |-- features/                  # build.py
|   |-- models/                    # baselines.py, gbm.py, autoencoder.py, sequence.py
|   |-- tuning/                    # optuna_search.py
|   |-- evaluation/                # metrics.py, thresholds.py, calibration.py
|   |-- explain/                   # shap_analysis.py, reason_codes.py
|   |-- monitoring/                # drift.py
|   |-- serving/                   # api.py, schemas.py
|-- tests/
|-- Dockerfile
|-- Makefile
|-- pyproject.toml
|-- LICENSE
|-- README.md
|-- ROADMAP.md
```

## Known issues in the original notebook

These issues drive Phase 1 and must be fixed before any new results are reported.

- Engineered features are created after undersampling, and the split uses `X_resampled`, so none of the engineered features reach the model.
- `is_transfer` and `is_cashout` are computed after label encoding of `type`, so both are always 0.
- Per-account aggregates (`orig_avg_amount`, `orig_tx_std`, `dest_tx_count`, and others) are computed on the full dataset before the split, which leaks test information into training.
- The split is random rather than time-based, although `step` is a time index.
- Metrics are reported on an undersampled test set, which overstates precision relative to the real class ratio.
- The 0.15 threshold is fixed by hand, and no actual hyperparameter search is performed.
- The README states that engineered features improved performance. That claim must be corrected once Phase 1 is complete.

## Phase 0: Project foundation


## Phase 1: Correct data pipeline


## Phase 2: Modeling and evaluation

- [ ] Threshold selection on the validation set only, by cost minimization or a target recall
- [ ] Baseline models: logistic regression and a rule-based baseline reproducing `isFlaggedFraud`
- [ ] XGBoost training module with early stopping on the validation fold
- [ ] LightGBM and CatBoost alternatives behind a common model interface
- [ ] Class weighting (`scale_pos_weight`) compared against undersampling and SMOTE
- [ ] Probability calibration (isotonic and Platt) with reliability plots and Brier score
- [ ] `train.py` CLI entry point driven by config

## Phase 3: Hyperparameter optimization

- [ ] Optuna study for XGBoost with time-series cross-validation and PR-AUC objective
- [ ] Pruning with median pruner and persistent study storage (SQLite)
- [ ] Search spaces for LightGBM and CatBoost
- [ ] Optimization history, parameter importance, and slice plots saved to `reports/figures/`
- [ ] Nested evaluation so the final test fold is never seen during tuning

## Phase 4: Unsupervised and deep learning anomaly detection

- [ ] Isolation Forest and Local Outlier Factor baselines trained on legitimate transactions only
- [ ] PyTorch tabular autoencoder with reconstruction-error scoring
- [ ] Variational autoencoder variant
- [ ] Sequence model (GRU) over each account's transaction history
- [ ] Feed-forward network with focal loss for supervised comparison
- [ ] Hybrid model: autoencoder reconstruction error as a feature for gradient boosting
- [ ] Unified comparison table across supervised, unsupervised, and hybrid approaches

## Phase 5: Explainability

- [ ] `explain/shap_analysis.py`: global SHAP summary, dependence plots, and interaction values
- [ ] Local explanations for individual flagged transactions (waterfall plots)
- [ ] Reason-code generator mapping top SHAP contributions to human-readable alert reasons
- [ ] Permutation importance compared against SHAP rankings
- [ ] Explanation stability check across seeds and folds

## Phase 6: Experiment tracking and reproducibility

- [ ] MLflow tracking for parameters, metrics, artifacts, and models
- [ ] Deterministic seeding across NumPy, scikit-learn, XGBoost, and PyTorch
- [ ] `Makefile` targets for data, train, tune, evaluate, and test
- [ ] DVC or a documented data-versioning approach for the PaySim file

## Phase 7: Serving and monitoring

- [ ] Model artifact packaging with a versioned feature schema
- [ ] FastAPI scoring service returning score, decision, and reason codes
- [ ] Pydantic request and response models with input validation
- [ ] API tests with the FastAPI test client
- [ ] Dockerfile and `docker-compose.yml`
- [ ] Data drift and score drift monitoring (PSI and KS) with a report script
- [ ] Batch scoring script for offline evaluation

## Phase 8: Documentation

- [ ] `docs/methodology.md`: formal problem statement, notation, evaluation protocol, and references to the PaySim, XGBoost, SHAP, Optuna, Isolation Forest, and focal loss papers
- [ ] Rewrite the notebook as a clean walkthrough that imports from the package
- [ ] Model card covering intended use, data, metrics, limitations, and fairness notes
- [ ] Architecture diagram for the pipeline
- [ ] Rewrite README with corrected methodology and results from actual runs on PaySim
- [ ] `CONTRIBUTING.md` and docstrings for the public API

## Completed

Completed items are moved here with the date.

- [x] 2026-11-19: Add `.gitignore`, `LICENSE` (MIT), and `pyproject.toml` with package metadata and pinned dependency ranges
- [x] 2026-11-19: Move the original notebook to `notebooks/00_original_exploration.ipynb` and add `data/README.md` with dataset download instructions
- [x] 2026-11-19: Create `src/fraud_detection/` package layout with `__init__.py`
- [x] 2026-11-19: Add `configs/default.yaml` holding paths, seeds, split settings, and model parameters
- [x] 2026-11-19: Add a config loader (`config.py`) with typed dataclasses
- [x] 2026-11-19: Add a synthetic PaySim-like data generator (`data/synthetic.py`) for tests and CI, since the real dataset is not stored in the repo
- [x] 2026-11-20: Add `ruff` and `pre-commit` configuration
- [x] 2026-11-20: Add GitHub Actions workflow running lint and tests on push
- [x] 2026-11-20: `data/load.py`: schema-validated CSV loader with memory-efficient dtypes
- [x] 2026-11-20: `data/split.py`: time-based train, validation, and test split on `step`
- [x] 2026-11-20: `features/build.py`: transaction-level features (type flags from raw strings, hour of day, merchant flag, amount bins)
- [x] 2026-11-20: `features/build.py`: point-in-time account aggregates computed only from prior transactions, no lookahead
- [x] 2026-11-21: Balance-based features (`errorBalanceOrig`, `errorBalanceDest`) with an ablation flag
- [x] 2026-11-21: Apply resampling to the training fold only, after feature construction
- [x] 2026-11-21: Unit tests for leakage: aggregates must not use future rows, and test rows must never appear in training
- [x] 2026-11-21: Unit tests for feature correctness on hand-built fixtures
- [x] 2026-11-21: `evaluation/metrics.py`: PR-AUC, ROC-AUC, recall at fixed precision, precision at fixed recall
- [x] 2026-11-21: Cost-based evaluation with configurable false-negative and false-positive costs
