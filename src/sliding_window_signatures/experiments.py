"""Runs the paper's tables and figures, in faithful or corrected mode."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sliding_window_signatures.baselines import (
    LinearForecastResult,
    calendar_real_baselines,
    calendar_synthetic_baselines,
    reference_real_baselines,
    synthetic_baselines,
)
from sliding_window_signatures.data import SAMPLES_PER_DAY, PaperDataset
from sliding_window_signatures.model import (
    RidgeForecastResult,
    TemporalSplit,
    fit_ridge_forecaster,
)
from sliding_window_signatures.signatures import SignatureFeatures, sequential_sliding_signatures
from sliding_window_signatures.simulation import SyntheticDemand, simulate_synthetic_demand

ExperimentMode = Literal["faithful", "corrected"]
Progress = Callable[[str], None]

TRAIN_END = np.datetime64("2014-01-01T00:00:00")
VALIDATION_END = np.datetime64("2015-01-01T00:00:00")


def run_synthetic_table(
    dataset: PaperDataset,
    *,
    mode: ExperimentMode = "faithful",
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Reproduce Table 1 with a nine-day window and order four."""
    report = progress or _ignore_progress
    _validate_mode(mode)
    report("Simulating the Table 1 target (alpha=0.005, sigma=1000, seed=141)")
    synthetic = simulate_synthetic_demand(
        dataset.temperature,
        dataset.demand,
        alpha=0.005,
        noise_standard_deviation=1_000.0,
        seed=141,
    )
    report("Computing nine-day signatures through level four")
    features = sequential_sliding_signatures(
        dataset.temperature,
        window=9 * SAMPLES_PER_DAY,
        order=4,
    )
    split = split_for_features(dataset, features, mode=mode)
    penalties = np.logspace(-3, 3, 50)

    report("Selecting separate ridge penalties for RMSE and MAPE")
    ridge_rmse = fit_ridge_forecaster(
        features,
        synthetic.values,
        delay=2 * SAMPLES_PER_DAY,
        split=split,
        penalties=penalties,
        selection_metric="rmse",
    )
    ridge_mape = fit_ridge_forecaster(
        features,
        synthetic.values,
        delay=2 * SAMPLES_PER_DAY,
        split=split,
        penalties=penalties,
        selection_metric="mape",
    )

    train_count, validation_count = paper_year_counts(dataset)
    if mode == "faithful":
        baselines = synthetic_baselines(
            dataset.temperature,
            synthetic.values,
            alpha=0.005,
            train_stop=train_count + validation_count,
        )
    else:
        baselines = calendar_synthetic_baselines(
            dataset.temperature,
            synthetic.values,
            alpha=0.005,
            end_indices=features.end_indices,
            split=split,
        )

    rows = [
        _baseline_row(label, baselines[key], dataset)
        for label, key in (
            ("LR(T)", "LR(T)"),
            ("LR(T, T^2)", "LR(T,T^2)"),
            ("LR(smoothed T)", "LR(T_alpha)"),
        )
    ]
    rows.append(_ridge_row("RidgeSig", ridge_rmse, ridge_mape, dataset))
    rows.append(
        _baseline_row(
            "Oracle LR(smoothed T, smoothed T^2)",
            baselines["LR(T_alpha,T_alpha^2)"],
            dataset,
        )
    )
    return pd.DataFrame(rows)


def run_real_table(
    dataset: PaperDataset,
    *,
    mode: ExperimentMode = "faithful",
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Reproduce Table 2 with the nine-day, level-six RidgeSig model."""
    report = progress or _ignore_progress
    _validate_mode(mode)
    window = 9 * SAMPLES_PER_DAY
    delay = 7 * SAMPLES_PER_DAY
    report("Computing real-data signatures for w=9 days and N=6")
    features = sequential_sliding_signatures(dataset.temperature, window=window, order=6)
    split = split_for_features(dataset, features, mode=mode)
    penalties = np.logspace(-1, 2, 20)

    report("Selecting separate real-data ridge penalties for RMSE and MAPE")
    ridge_rmse = fit_ridge_forecaster(
        features,
        dataset.demand,
        delay=delay,
        split=split,
        penalties=penalties,
        selection_metric="rmse",
    )
    ridge_mape = fit_ridge_forecaster(
        features,
        dataset.demand,
        delay=delay,
        split=split,
        penalties=penalties,
        selection_metric="mape",
    )

    train_count, validation_count = paper_year_counts(dataset)
    if mode == "faithful":
        baselines = reference_real_baselines(
            dataset.temperature,
            dataset.demand,
            alpha=0.005,
            train_count=train_count,
            validation_count=validation_count,
            delay=delay,
            window=window,
        )
    else:
        baselines = calendar_real_baselines(
            dataset.temperature,
            dataset.demand,
            alpha=0.005,
            delay=delay,
            end_indices=features.end_indices,
            split=split,
        )

    rows = [
        _baseline_row("LR(smoothed T, smoothed T^2)", baselines["LR(T_alpha,T_alpha^2)"], dataset),
        _baseline_row(
            "LR(T, T^2, smoothed T, smoothed T^2)",
            baselines["LR(T,T^2,T_alpha,T_alpha^2)"],
            dataset,
        ),
        _baseline_row("LR(Y[t-7 days])", baselines["Y_lag"], dataset),
        _baseline_row(
            "LR(T features, Y[t-7 days])",
            baselines["LR(T,T^2,T_alpha,T_alpha^2,Y_lag)"],
            dataset,
        ),
        _ridge_row("RidgeSig", ridge_rmse, ridge_mape, dataset),
    ]
    return pd.DataFrame(rows)


def run_window_sweep(
    dataset: PaperDataset,
    *,
    mode: ExperimentMode = "faithful",
    window_days: Sequence[int] = tuple(range(2, 33)),
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Reproduce Figure 2's memory-strength/window-size experiment."""
    report = progress or _ignore_progress
    _validate_mode(mode)
    targets = {
        alpha: simulate_synthetic_demand(
            dataset.temperature,
            dataset.demand,
            alpha=alpha,
            noise_standard_deviation=500.0,
            seed=141,
        ).values
        for alpha in (0.005, 0.05)
    }
    penalties = np.logspace(-3, 3, 50)
    rows: list[dict[str, float | int | str]] = []
    for days in window_days:
        if days < 2:
            raise ValueError("Figure 2 windows must be at least the two-day delay")
        report(f"Figure 2: computing N=5 signatures for w={days} days")
        features = sequential_sliding_signatures(
            dataset.temperature,
            window=days * SAMPLES_PER_DAY,
            order=5,
        )
        split = split_for_features(dataset, features, mode=mode)
        for alpha, target in targets.items():
            result = fit_ridge_forecaster(
                features,
                target,
                delay=2 * SAMPLES_PER_DAY,
                split=split,
                penalties=penalties,
                selection_metric="rmse",
            )
            rows.append(
                {
                    "alpha": alpha,
                    "window_days": days,
                    "order": 5,
                    "best_penalty": result.best_penalty,
                    "validation_rmse_mw": result.validation_score,
                    "test_rmse_mw": result.metrics.rmse,
                    "mode": mode,
                }
            )
    return pd.DataFrame(rows)


def run_order_sweep(
    dataset: PaperDataset,
    *,
    mode: ExperimentMode = "faithful",
    window_days: Sequence[int] = tuple(range(2, 33)),
    orders: Sequence[int] = tuple(range(2, 8)),
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Reproduce Figure 3's truncation-order/window-size experiment."""
    report = progress or _ignore_progress
    _validate_mode(mode)
    if not orders:
        raise ValueError("orders must not be empty")
    maximum_order = max(orders)
    if min(orders) < 1:
        raise ValueError("orders must be positive")
    target = simulate_synthetic_demand(
        dataset.temperature,
        dataset.demand,
        alpha=0.005,
        noise_standard_deviation=500.0,
        seed=141,
    ).values
    penalties = np.logspace(-1, 3, 20)
    rows: list[dict[str, float | int | str]] = []
    for days in window_days:
        if days < 2:
            raise ValueError("Figure 3 windows must be at least the two-day delay")
        report(f"Figure 3: computing N={maximum_order} signatures for w={days} days")
        all_levels = sequential_sliding_signatures(
            dataset.temperature,
            window=days * SAMPLES_PER_DAY,
            order=maximum_order,
        )
        split = split_for_features(dataset, all_levels, mode=mode)
        for order in orders:
            features = all_levels.through_order(order)
            result = fit_ridge_forecaster(
                features,
                target,
                delay=2 * SAMPLES_PER_DAY,
                split=split,
                penalties=penalties,
                selection_metric="rmse",
            )
            rows.append(
                {
                    "window_days": days,
                    "order": order,
                    "fitted_feature_count": features.values.shape[1],
                    "best_penalty": result.best_penalty,
                    "validation_rmse_mw": result.validation_score,
                    "test_rmse_mw": result.metrics.rmse,
                    "mode": mode,
                }
            )
    return pd.DataFrame(rows)


def run_real_forecasts(
    dataset: PaperDataset,
    *,
    mode: ExperimentMode = "faithful",
    progress: Progress | None = None,
) -> tuple[RidgeForecastResult, LinearForecastResult]:
    """Fit RidgeSig and the strongest baseline on the same rows for Figure 4."""
    report = progress or _ignore_progress
    _validate_mode(mode)
    window = 9 * SAMPLES_PER_DAY
    delay = 7 * SAMPLES_PER_DAY
    report("Figure 4: computing signatures and fitting RidgeSig")
    features = sequential_sliding_signatures(dataset.temperature, window=window, order=6)
    split = split_for_features(dataset, features, mode=mode)
    ridge = fit_ridge_forecaster(
        features,
        dataset.demand,
        delay=delay,
        split=split,
        penalties=np.logspace(-1, 2, 20),
        selection_metric="rmse",
    )
    aligned_baselines = calendar_real_baselines(
        dataset.temperature,
        dataset.demand,
        alpha=0.005,
        delay=delay,
        end_indices=features.end_indices,
        split=split,
    )
    return ridge, aligned_baselines["LR(T,T^2,T_alpha,T_alpha^2,Y_lag)"]


def make_figure5_synthetic_data(dataset: PaperDataset) -> SyntheticDemand:
    """Return the alpha=0.005, sigma=1000 series displayed in Figure 5."""
    return simulate_synthetic_demand(
        dataset.temperature,
        dataset.demand,
        alpha=0.005,
        noise_standard_deviation=1_000.0,
        seed=141,
    )


def run_smoke_experiment(
    dataset: PaperDataset,
    *,
    observations: int = 6_000,
) -> Mapping[str, float | int]:
    """Run a quick end-to-end check. The numbers are not paper quality."""
    if not 1_000 <= observations <= len(dataset):
        raise ValueError("observations must be between 1,000 and the dataset length")
    synthetic = simulate_synthetic_demand(
        dataset.temperature[:observations],
        dataset.demand[:observations],
        alpha=0.05,
        noise_standard_deviation=500.0,
        seed=141,
    )
    features = sequential_sliding_signatures(
        dataset.temperature[:observations],
        window=2 * SAMPLES_PER_DAY,
        order=3,
    )
    split = TemporalSplit(
        train_stop=int(features.values.shape[0] * 0.5),
        validation_stop=int(features.values.shape[0] * 0.75),
        sample_count=features.values.shape[0],
    )
    result = fit_ridge_forecaster(
        features,
        synthetic.values,
        delay=2 * SAMPLES_PER_DAY,
        split=split,
        penalties=np.logspace(-2, 2, 9),
        selection_metric="rmse",
    )
    return {
        "observations": observations,
        "signature_rows": features.values.shape[0],
        "signature_features": features.values.shape[1],
        "best_penalty": result.best_penalty,
        "test_rmse_mw": result.metrics.rmse,
        "test_mape_percent": result.metrics.mape_percent,
    }


def split_for_features(
    dataset: PaperDataset,
    features: SignatureFeatures,
    *,
    mode: ExperimentMode,
) -> TemporalSplit:
    """Build either the notebook's split or the strict calendar split."""
    _validate_mode(mode)
    if mode == "faithful":
        train_count, validation_count = paper_year_counts(dataset)
        return TemporalSplit.from_reference_counts(
            train_count=train_count,
            validation_count=validation_count,
            sample_count=features.values.shape[0],
        )
    return TemporalSplit.from_calendar_boundaries(
        dates=dataset.dates,
        end_indices=features.end_indices,
    )


def paper_year_counts(dataset: PaperDataset) -> tuple[int, int]:
    """Return the 2012-2013 and 2014 row counts used by the notebook."""
    train_count = int(np.count_nonzero(dataset.dates < TRAIN_END))
    validation_count = int(
        np.count_nonzero((dataset.dates >= TRAIN_END) & (dataset.dates < VALIDATION_END))
    )
    return train_count, validation_count


def _ridge_row(
    label: str,
    rmse_result: RidgeForecastResult,
    mape_result: RidgeForecastResult,
    dataset: PaperDataset,
) -> dict[str, float | str]:
    if not np.array_equal(rmse_result.end_indices, mape_result.end_indices):
        raise ValueError("RMSE- and MAPE-selected models must share a test horizon")
    return {
        "Model": label,
        "RMSE (MW)": rmse_result.metrics.rmse,
        "MAPE (%)": mape_result.metrics.mape_percent,
        "Selected lambda (RMSE)": rmse_result.best_penalty,
        "Selected lambda (MAPE)": mape_result.best_penalty,
        "Test start": _date_string(dataset, int(rmse_result.end_indices[0])),
        "Test end": _date_string(dataset, int(rmse_result.end_indices[-1])),
    }


def _baseline_row(
    label: str,
    result: LinearForecastResult,
    dataset: PaperDataset,
) -> dict[str, float | str]:
    return {
        "Model": label,
        "RMSE (MW)": result.metrics.rmse,
        "MAPE (%)": result.metrics.mape_percent,
        "Selected lambda (RMSE)": np.nan,
        "Selected lambda (MAPE)": np.nan,
        "Test start": _date_string(dataset, int(result.end_indices[0])),
        "Test end": _date_string(dataset, int(result.end_indices[-1])),
    }


def _date_string(dataset: PaperDataset, index: int) -> str:
    return np.datetime_as_string(dataset.dates[index], unit="m")


def _validate_mode(mode: str) -> None:
    if mode not in {"faithful", "corrected"}:
        raise ValueError("mode must be 'faithful' or 'corrected'")


def _ignore_progress(_: str) -> None:
    return None
