import numpy as np
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import fit_model

torch = pytest.importorskip("torch")

from fraud_detection.models.autoencoder import (  # noqa: E402
    AutoencoderParams,
    VariationalAutoencoder,
    build_autoencoder,
    fit_autoencoder,
)

FAST = {"hidden_dims": [16], "latent_dim": 4, "max_epochs": 8, "batch_size": 256}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.03, n_steps=80, seed=13)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_architecture_is_symmetric() -> None:
    params = AutoencoderParams(hidden_dims=(32, 16), latent_dim=4, dropout=0.1)
    net = build_autoencoder(20, params)
    enc = [m.out_features for m in net.encoder if isinstance(m, torch.nn.Linear)]
    dec = [m.out_features for m in net.decoder if isinstance(m, torch.nn.Linear)]
    assert enc == [32, 16, 4]
    assert dec == [16, 32, 20]
    # No activation after the bottleneck or the output layer
    assert isinstance(net.encoder[-1], torch.nn.Linear)
    assert isinstance(net.decoder[-1], torch.nn.Linear)
    assert any(isinstance(m, torch.nn.Dropout) for m in net.encoder)


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown autoencoder"):
        AutoencoderParams.from_mapping({"latent": 3})
    assert AutoencoderParams.from_mapping({"hidden_dims": [8]}).hidden_dims == (8,)


def test_training_reduces_holdout_loss(folds) -> None:
    model = fit_autoencoder(folds.train, FAST, seed=0)
    history = model.detector.history
    assert 1 <= len(history) <= FAST["max_epochs"]
    best = min(h["holdout_loss"] for h in history)
    assert best < history[0]["holdout_loss"]
    n_legit = int((folds.train.y == 0).sum())
    assert model.reference_scores.size == round(0.1 * n_legit)


@pytest.mark.parametrize("name", ["autoencoder", "variational_autoencoder"])
def test_autoencoder_through_interface(folds, name) -> None:
    model = fit_model(name, folds.train, folds.X_validation, folds.y_validation, FAST)
    scores = model.predict_scores(folds.X_test)
    assert scores.shape == (len(folds.X_test),)
    assert scores.min() >= 0
    assert scores.max() <= 1
    raw = model.anomaly_scores(folds.X_test)
    assert np.all(raw >= 0)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(scores[order]) >= 0)


@pytest.mark.parametrize("variational", [False, True])
def test_training_is_reproducible(folds, variational) -> None:
    params = {**FAST, "variational": variational}
    a = fit_autoencoder(folds.train, params, seed=3)
    b = fit_autoencoder(folds.train, params, seed=3)
    np.testing.assert_allclose(
        a.anomaly_scores(folds.X_test), b.anomaly_scores(folds.X_test), rtol=1e-6
    )


def test_early_stopping_respects_patience(folds) -> None:
    params = {**FAST, "max_epochs": 40, "patience": 1, "learning_rate": 0.5}
    model = fit_autoencoder(folds.train, params, seed=0)
    assert len(model.detector.history) < 40


def test_invalid_holdout_fraction(folds) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        fit_autoencoder(folds.train, {**FAST, "validation_fraction": 1.0})


def test_vae_architecture_and_heads() -> None:
    params = AutoencoderParams(hidden_dims=(32, 16), latent_dim=4, variational=True)
    net = build_autoencoder(20, params)
    assert isinstance(net, VariationalAutoencoder)
    batch = torch.randn(7, 20)
    mean, log_var = net.encode(batch)
    assert mean.shape == log_var.shape == (7, 4)
    dec = [m.out_features for m in net.decoder if isinstance(m, torch.nn.Linear)]
    assert dec == [16, 32, 20]


def test_vae_row_loss_matches_closed_form() -> None:
    torch.manual_seed(0)
    params = AutoencoderParams(hidden_dims=(8,), latent_dim=2, beta=0.5)
    net = VariationalAutoencoder(5, params).eval()
    batch = torch.randn(3, 5)
    with torch.no_grad():
        mean, log_var = net.encode(batch)
        recon = ((batch - net.decoder(mean)) ** 2).sum(dim=1)
        kl = 0.5 * (mean**2 + log_var.exp() - log_var - 1).sum(dim=1)
        expected = (recon + 0.5 * kl) / 5
        np.testing.assert_allclose(net.row_loss(batch), expected, rtol=1e-6)
        assert torch.all(kl >= 0)


def test_vae_scoring_is_deterministic(folds) -> None:
    model = fit_autoencoder(folds.train, {**FAST, "variational": True}, seed=1)
    np.testing.assert_array_equal(
        model.anomaly_scores(folds.X_test), model.anomaly_scores(folds.X_test)
    )
    best = min(h["holdout_loss"] for h in model.detector.history)
    assert best < model.detector.history[0]["holdout_loss"]
