import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FeaturesConfig, ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.features.sequence import history_column
from fraud_detection.models.base import fit_model

torch = pytest.importorskip("torch")

from fraud_detection.models.sequence import (  # noqa: E402
    SequenceEncoder,
    _raw_sequences,
    fit_sequence_model,
    history_layout,
)

LENGTH = 3
FEATURES = FeaturesConfig(history_length=LENGTH)
FAST = {"hidden_size": 8, "static_dims": [16], "head_dims": [8], "max_epochs": 6}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(
        n_rows=4_000, fraud_rate=0.03, n_steps=80, n_customers=400, seed=13
    )
    split = time_split(build_features(df, FEATURES), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split,
        feature_columns(FEATURES),
        "isFraud",
        ResamplingConfig("class_weight", 0.2),
    )


def test_layout_is_read_from_columns() -> None:
    roles, length = history_layout(feature_columns(FEATURES))
    assert roles == ["orig", "dest"]
    assert length == LENGTH
    with pytest.raises(ValueError, match="history_length"):
        history_layout(feature_columns())
    partial = [c for c in feature_columns(FEATURES) if c != "dest_hist2_hour_cos"]
    with pytest.raises(ValueError, match="Incomplete"):
        history_layout(partial)


def test_sequences_are_left_aligned_and_end_with_current_row(folds) -> None:
    X = folds.X_test
    values, lengths = _raw_sequences(X, "dest", LENGTH)
    assert values.shape == (len(X), LENGTH + 1, 7)
    assert lengths.min() >= 1
    assert lengths.max() <= LENGTH + 1
    ages = X[[history_column("dest", k, "steps_before") for k in (1, 2, 3)]]
    np.testing.assert_array_equal(lengths, 1 + ages.notna().sum(axis=1))

    rows = np.arange(len(X))
    last = values[rows, lengths - 1]
    np.testing.assert_allclose(last[:, 0], X["log_amount"])
    np.testing.assert_array_equal(last[:, -1], 0.0)
    # The element before the current row is the most recent earlier transaction
    has_history = lengths > 1
    previous = values[rows[has_history], lengths[has_history] - 2]
    expected = X.loc[has_history, history_column("dest", 1, "log_amount")]
    np.testing.assert_allclose(previous[:, 0], expected)


def test_encoder_outputs_finite_tensors(folds) -> None:
    encoder = SequenceEncoder.fit(folds.train.X)
    static, *sequences = encoder(folds.X_validation)
    assert static.shape == (len(folds.X_validation), encoder.n_static)
    assert len(sequences) == 4
    for tensor in (static, sequences[0], sequences[2]):
        assert torch.isfinite(tensor).all()


def test_gru_through_interface(folds) -> None:
    model = fit_model(
        "gru", folds.train, folds.X_validation, folds.y_validation, FAST, seed=0
    )
    scores = model.predict_scores(folds.X_test)
    assert scores.shape == (len(folds.X_test),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert 1 <= len(model.history) <= FAST["max_epochs"]
    # Synthetic fraud is easy to separate; the model must beat the base rate
    assert pr_auc(folds.y_test, scores) > 5 * folds.y_test.mean()


def test_gru_is_reproducible(folds) -> None:
    a = fit_sequence_model(
        folds.train, folds.X_validation, folds.y_validation, FAST, seed=2
    )
    b = fit_sequence_model(
        folds.train, folds.X_validation, folds.y_validation, FAST, seed=2
    )
    np.testing.assert_allclose(
        a.predict_scores(folds.X_test), b.predict_scores(folds.X_test), rtol=1e-6
    )


def test_unknown_parameter_is_rejected(folds) -> None:
    with pytest.raises(ValueError, match="Unknown SequenceParams"):
        fit_sequence_model(
            folds.train, folds.X_validation, folds.y_validation, {"hidden": 4}
        )


def test_validation_without_positives_is_rejected(folds) -> None:
    y = pd.Series(0, index=folds.y_validation.index)
    with pytest.raises(ValueError, match="positives"):
        fit_sequence_model(folds.train, folds.X_validation, y, FAST)
