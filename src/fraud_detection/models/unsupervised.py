r"""Unsupervised anomaly detectors fitted on legitimate transactions only.

These detectors model the distribution of legitimate behaviour and flag
departures from it, without using fraud labels for fitting. They answer a
different question from the supervised models: whether fraud in PaySim is
separable as an outlier, which matters for fraud patterns absent from the
labelled history.

* Isolation Forest (Liu et al., 2008) isolates points with random axis-aligned
  splits. The anomaly score is a decreasing function of the expected path
  length :math:`\mathbb{E}[h(x)]` over trees, normalised by the average path
  length :math:`c(\psi)` of an unsuccessful search in a binary tree built on
  :math:`\psi` points.
* Local Outlier Factor (Breunig et al., 2000) compares the local reachability
  density of a point with that of its :math:`k` nearest neighbours. It is used
  in novelty mode: the reference set is the legitimate training rows and new
  rows are scored against it.

Fraud labels are used only to drop the positive rows of the training fold.
Missing point-in-time aggregates are imputed with the training median plus a
missingness indicator. For LOF, each feature is mapped through its empirical
quantiles to a standard normal, since PaySim amounts and balances span
several orders of magnitude and would otherwise dominate the distances.
Isolation Forest splits on one feature at a time and needs no scaling.

Raw anomaly scores have no common scale across detectors. Each fitted
detector therefore reports

.. math::

    s(x) = \hat{F}_0\big(a(x)\big)
         = \frac{1}{n_0} \sum_{i=1}^{n_0} \mathbf{1}\{a(x_i) \le a(x)\},

where :math:`a` is the raw anomaly score (higher is more anomalous) and
:math:`x_1, \dots, x_{n_0}` are the legitimate training rows. The score is the
fraction of legitimate training rows that look no more anomalous than
:math:`x`. It lies in :math:`[0, 1]` and preserves the ranking, so ranking
metrics are unchanged. It is not a fraud probability.

References
----------
Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). LOF:
Identifying density-based local outliers. SIGMOD.

Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation forest. ICDM.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from fraud_detection.data.resample import ResampledFold

ISOLATION_FOREST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_samples": 256,
    "max_features": 1.0,
    "n_jobs": -1,
}

LOF_DEFAULTS: dict[str, Any] = {
    "n_neighbors": 35,
    "n_jobs": -1,
}

# LOF memory and time grow with the reference set, so it is subsampled
DEFAULT_MAX_REFERENCE_ROWS = 50_000


@dataclass(frozen=True)
class AnomalyDetector:
    """Fitted preprocessing, detector, and reference anomaly scores.

    Attributes
    ----------
    preprocess
        Fitted transformer applied before the detector.
    detector
        Fitted ``IsolationForest`` or ``LocalOutlierFactor`` (novelty mode).
    reference_scores
        Sorted raw anomaly scores of the legitimate training rows.
    """

    preprocess: Pipeline
    detector: Any
    reference_scores: np.ndarray

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Raw anomaly score, higher meaning more anomalous."""
        return -np.asarray(
            self.detector.score_samples(self.preprocess.transform(X)), dtype="float64"
        )

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Empirical CDF of the anomaly score under legitimate training rows."""
        raw = self.anomaly_scores(X)
        ranks = np.searchsorted(self.reference_scores, raw, side="right")
        return ranks / self.reference_scores.size


def legitimate_rows(
    train: ResampledFold,
    max_rows: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Feature rows of the training fold labelled legitimate.

    Parameters
    ----------
    train
        Training fold. Rebalancing does not matter here because positives are
        dropped, although SMOTE rows are positive and also dropped.
    max_rows
        Optional cap; a uniform random subset is drawn when exceeded.
    seed
        Random state of the subsample.

    Returns
    -------
    pandas.DataFrame
        Legitimate rows, in their original order.
    """
    X = train.X.loc[train.y.to_numpy() == 0]
    if X.empty:
        raise ValueError("The training fold has no legitimate rows.")
    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X = X.iloc[keep]
    return X


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", add_indicator=True)


def make_gaussianizer(n_rows: int, seed: int = 0) -> Pipeline:
    """Unfitted median imputation followed by a per-feature normal-score transform.

    Parameters
    ----------
    n_rows
        Number of rows the transformer will be fitted on; caps the quantiles.
    seed
        Random state of the quantile subsample.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``impute`` and ``gaussianize``.
    """
    return Pipeline(
        [
            ("impute", _imputer()),
            (
                "gaussianize",
                QuantileTransformer(
                    n_quantiles=max(2, min(1000, n_rows)),
                    output_distribution="normal",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_isolation_forest(
    train: ResampledFold,
    params: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    seed: int = 0,
) -> AnomalyDetector:
    """Fit an Isolation Forest on the legitimate training rows.

    Parameters
    ----------
    train
        Training fold; positive rows are discarded.
    params
        Overrides of :data:`ISOLATION_FOREST_DEFAULTS`, passed to
        :class:`sklearn.ensemble.IsolationForest`.
    max_rows
        Optional cap on the number of legitimate rows used.
    seed
        Random state of the subsample and the forest.

    Returns
    -------
    AnomalyDetector
        Fitted detector.
    """
    X = legitimate_rows(train, max_rows, seed)
    preprocess = Pipeline([("impute", _imputer())]).fit(X)
    Z = preprocess.transform(X)
    forest = IsolationForest(
        **{**ISOLATION_FOREST_DEFAULTS, **(params or {})}, random_state=seed
    ).fit(Z)
    reference = np.sort(-forest.score_samples(Z))
    return AnomalyDetector(preprocess, forest, reference)


def fit_local_outlier_factor(
    train: ResampledFold,
    params: Mapping[str, Any] | None = None,
    max_rows: int | None = DEFAULT_MAX_REFERENCE_ROWS,
    seed: int = 0,
) -> AnomalyDetector:
    """Fit a novelty-mode Local Outlier Factor on the legitimate training rows.

    Parameters
    ----------
    train
        Training fold; positive rows are discarded.
    params
        Overrides of :data:`LOF_DEFAULTS`, passed to
        :class:`sklearn.neighbors.LocalOutlierFactor`.
    max_rows
        Cap on the reference set size, since neighbour queries scale with it.
    seed
        Random state of the subsample and the quantile transform.

    Returns
    -------
    AnomalyDetector
        Fitted detector.
    """
    X = legitimate_rows(train, max_rows, seed)
    preprocess = make_gaussianizer(len(X), seed).fit(X)
    lof = LocalOutlierFactor(**{**LOF_DEFAULTS, **(params or {})}, novelty=True)
    lof.fit(preprocess.transform(X))
    # score_samples on the reference rows would count each row as its own neighbour
    reference = np.sort(-lof.negative_outlier_factor_)
    return AnomalyDetector(preprocess, lof, reference)
