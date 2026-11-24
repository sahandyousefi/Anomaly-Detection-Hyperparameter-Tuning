import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from fraud_detection.tuning.study_plots import (  # noqa: E402
    parameter_importances,
    plot_optimization_history,
    plot_slices,
    save_study_figures,
    trials_frame,
)


def _quadratic(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
    depth = trial.suggest_int("max_depth", 2, 10)
    booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
    trial.report(0.1, step=0)
    if trial.number in {3, 7}:
        raise optuna.TrialPruned()
    return 1.0 - (depth - 6) ** 2 / 100 - 0.01 * lr - 0.001 * (booster == "dart")


@pytest.fixture(scope="module")
def study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name="toy",
    )
    study.optimize(_quadratic, n_trials=20)
    return study


def test_trials_frame_keeps_completed_trials_only(study) -> None:
    frame = trials_frame(study)
    assert len(frame) == 18
    assert {3, 7}.isdisjoint(frame["number"])
    assert {"value", "learning_rate", "max_depth", "booster"} <= set(frame.columns)


def test_importances_are_normalized_and_rank_depth_first(study) -> None:
    importances = parameter_importances(study, seed=0)
    assert set(importances.index) == {"learning_rate", "max_depth", "booster"}
    assert importances.sum() == pytest.approx(1.0)
    assert importances.index[0] == "max_depth"


def test_history_marks_pruned_trials(study) -> None:
    fig = plot_optimization_history(study)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "Pruned (last running mean)" in labels
    assert "Best so far" in labels


def test_slices_use_log_axis_for_log_parameters(study) -> None:
    fig = plot_slices(study, n_cols=2)
    panels = {ax.get_xlabel(): ax for ax in fig.axes if ax.get_xlabel()}
    assert panels["learning_rate"].get_xscale() == "log"
    assert panels["max_depth"].get_xscale() == "linear"
    assert [t.get_text() for t in panels["booster"].get_xticklabels()] == [
        "dart",
        "gbtree",
    ]


def test_save_study_figures_writes_pngs(study, tmp_path) -> None:
    paths = save_study_figures(study, tmp_path / "figures")
    assert set(paths) == {"history", "slices", "importance"}
    for path in paths.values():
        assert path.name.startswith("toy_")
        assert path.stat().st_size > 0


def test_importance_skipped_with_single_trial(tmp_path) -> None:
    study = optuna.create_study(direction="maximize", study_name="single")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    paths = save_study_figures(study, tmp_path)
    assert set(paths) == {"history", "slices"}
    with pytest.raises(ValueError, match="two completed"):
        parameter_importances(study)


def test_empty_study_raises() -> None:
    study = optuna.create_study()
    with pytest.raises(ValueError, match="no completed"):
        plot_optimization_history(study)
    assert trials_frame(study).empty
    assert isinstance(trials_frame(study), pd.DataFrame)
