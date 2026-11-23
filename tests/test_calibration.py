import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from fraud_detection.evaluation.calibration import (
    CalibratedModel,
    brier_score,
    calibration_report,
    expected_calibration_error,
    fit_calibrator,
    plot_reliability,
    prior_shift_correction,
    reliability_table,
    undersampling_weight,
)
from fraud_detection.models.base import FraudModel


def _draw(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_true = rng.beta(0.5, 10.0, size=n)
    y = (rng.random(n) < p_true).astype(int)
    return y, p_true


def test_brier_score_by_hand() -> None:
    assert brier_score([0, 1, 1], [0.0, 0.5, 1.0]) == pytest.approx(0.25 / 3)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="binary"):
        brier_score([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0, 1], [0.1, 1.2])
    with pytest.raises(ValueError, match="equal size"):
        brier_score([0, 1], [0.1])


def test_prior_shift_correction_divides_odds() -> None:
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(prior_shift_correction(p, 1.0), p)
    corrected = prior_shift_correction(p[1:-1], 4.0)
    odds = p[1:-1] / (1 - p[1:-1])
    np.testing.assert_allclose(corrected / (1 - corrected), odds / 4.0)
    assert prior_shift_correction(p, 4.0)[[0, -1]].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        prior_shift_correction(p, 0.0)


def test_undersampling_weight() -> None:
    # 10 positives, 1000 negatives, undersampled to 1:10 keeps 100 negatives
    assert undersampling_weight(10, 1000, 0.1) == pytest.approx(10.0)


def test_prior_shift_recovers_base_rate() -> None:
    y, p_true = _draw(200_000, seed=0)
    w = 8.0
    # Scores a model would learn after weighting positives by w
    inflated = p_true * w / (p_true * w + 1 - p_true)
    corrected = prior_shift_correction(inflated, w)
    np.testing.assert_allclose(corrected, p_true, atol=1e-12)
    assert abs(corrected.mean() - y.mean()) < 0.002


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_improves_held_out_brier(method: str) -> None:
    y_fit, p_fit = _draw(20_000, seed=1)
    y_eval, p_eval = _draw(20_000, seed=2)
    # Overconfident monotone distortion of the true probability
    distort = np.sqrt
    calibrator = fit_calibrator(y_fit, distort(p_fit), method)
    raw = brier_score(y_eval, distort(p_eval))
    calibrated = calibrator.transform(distort(p_eval))
    assert brier_score(y_eval, calibrated) < raw
    assert expected_calibration_error(y_eval, calibrated) < expected_calibration_error(
        y_eval, distort(p_eval)
    )
    assert np.all(np.diff(calibrated[np.argsort(p_eval)]) >= -1e-12)


def test_platt_is_strictly_monotone() -> None:
    y, p = _draw(5_000, seed=3)
    calibrator = fit_calibrator(y, p, "platt")
    grid = np.linspace(0.001, 0.999, 50)
    assert np.all(np.diff(calibrator.transform(grid)) > 0)


def test_unknown_method_and_single_class() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        fit_calibrator([0, 1], [0.2, 0.8], "beta")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator([0, 0], [0.2, 0.8])


def test_calibrated_model_wraps_base() -> None:
    class Constant:
        def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
            return X["s"].to_numpy()

    y, p = _draw(5_000, seed=4)
    calibrator = fit_calibrator(y, p, "isotonic")
    model = CalibratedModel(Constant(), calibrator)
    assert isinstance(model, FraudModel)
    X = pd.DataFrame({"s": p[:10]})
    np.testing.assert_allclose(model.predict_scores(X), calibrator.transform(p[:10]))


def test_reliability_table_quantile_bins() -> None:
    y, p = _draw(10_000, seed=5)
    table = reliability_table(y, p, n_bins=10)
    assert table["count"].sum() == len(y)
    assert table["positives"].sum() == y.sum()
    assert len(table) == 10
    assert table["count"].max() - table["count"].min() <= 1
    assert (table["mean_predicted"] >= table["lower"]).all()
    assert (table["mean_predicted"] <= table["upper"]).all()


def test_reliability_table_ties_reduce_bins() -> None:
    p = np.r_[np.zeros(900), np.linspace(0.1, 0.9, 100)]
    y = (p > 0.5).astype(int)
    table = reliability_table(y, p, n_bins=10)
    assert len(table) < 10
    assert table["count"].sum() == len(p)


def test_uniform_bins_include_top_edge() -> None:
    table = reliability_table(
        [0, 1, 1], [0.0, 0.95, 1.0], n_bins=10, strategy="uniform"
    )
    assert table["count"].sum() == 3
    assert table.iloc[-1]["count"] == 2


def test_perfect_calibration_has_small_error() -> None:
    y, p = _draw(200_000, seed=6)
    report = calibration_report(y, p)
    assert report["ece"] < 0.005
    assert report["brier_skill"] > 0
    assert report["mean_predicted"] == pytest.approx(report["prevalence"], abs=0.002)


def test_plot_reliability_returns_figure(tmp_path) -> None:
    y, p = _draw(5_000, seed=7)
    fig = plot_reliability({"raw": (y, np.sqrt(p)), "true": (y, p)})
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 3
    out = tmp_path / "reliability.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
