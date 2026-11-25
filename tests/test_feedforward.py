import numpy as np
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import fit_model

torch = pytest.importorskip("torch")

from fraud_detection.models.feedforward import (  # noqa: E402
    FeedForwardParams,
    fit_feedforward,
)
from fraud_detection.models.neural import (  # noqa: E402
    binary_cross_entropy,
    focal_loss,
)

FAST = {"hidden_dims": [32, 16], "batch_size": 64, "max_epochs": 8}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.03, n_steps=80, seed=17)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_focal_loss_reduces_to_cross_entropy() -> None:
    logits = torch.tensor([-3.0, -0.2, 0.0, 1.5, 4.0])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    torch.testing.assert_close(
        focal_loss(logits, targets, gamma=0.0), binary_cross_entropy(logits, targets)
    )


def test_focal_loss_matches_definition() -> None:
    logits = torch.tensor([-2.0, 0.5, 3.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    p = torch.sigmoid(logits)
    p_y = torch.where(targets == 1, p, 1 - p)
    alpha_y = torch.where(targets == 1, 0.25, 0.75)
    expected = -alpha_y * (1 - p_y) ** 2 * torch.log(p_y)
    torch.testing.assert_close(focal_loss(logits, targets, 2.0, 0.25), expected)


def test_focal_loss_down_weights_easy_rows() -> None:
    targets = torch.tensor([0.0, 0.0])
    logits = torch.tensor([-6.0, 0.0])  # confident and uncertain negatives
    ratio_focal = (
        focal_loss(logits, targets, 2.0)[0] / focal_loss(logits, targets, 2.0)[1]
    )
    ratio_ce = binary_cross_entropy(logits, targets)
    assert ratio_focal < 1e-3 * (ratio_ce[0] / ratio_ce[1])


def test_focal_loss_rejects_invalid_parameters() -> None:
    logits, targets = torch.zeros(2), torch.ones(2)
    with pytest.raises(ValueError, match="gamma"):
        focal_loss(logits, targets, gamma=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        focal_loss(logits, targets, alpha=1.5)


@pytest.mark.parametrize("loss", ["focal", "cross_entropy"])
def test_feedforward_through_interface(folds, loss) -> None:
    params = {**FAST, "loss": loss}
    model = fit_model(
        "feedforward", folds.train, folds.X_validation, folds.y_validation, params
    )
    scores = model.predict_scores(folds.X_test)
    assert scores.shape == (len(folds.X_test),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert pr_auc(folds.y_test, scores) > 5 * folds.y_test.mean()
    best = max(h["validation_pr_auc"] for h in model.history)
    assert best >= model.history[0]["validation_pr_auc"]


def test_architecture_follows_params() -> None:
    from fraud_detection.models.feedforward import FeedForwardClassifier

    net = FeedForwardClassifier(10, FeedForwardParams(hidden_dims=(8, 4), dropout=0.0))
    widths = [m.out_features for m in net.layers if isinstance(m, torch.nn.Linear)]
    assert widths == [8, 4, 1]
    assert net(torch.zeros(3, 10)).shape == (3,)


def test_feedforward_is_reproducible(folds) -> None:
    args = (folds.train, folds.X_validation, folds.y_validation, FAST)
    a, b = fit_feedforward(*args, seed=5), fit_feedforward(*args, seed=5)
    np.testing.assert_allclose(
        a.predict_scores(folds.X_test), b.predict_scores(folds.X_test), rtol=1e-6
    )


def test_unknown_loss_is_rejected(folds) -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        fit_feedforward(
            folds.train, folds.X_validation, folds.y_validation, {"loss": "hinge"}
        )
