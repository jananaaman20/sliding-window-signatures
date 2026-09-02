"""A fair comparison of rolling-window and fading-memory signatures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import iisignature
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sliding_window_signatures.data import SAMPLES_PER_DAY, PaperDataset
from sliding_window_signatures.fading_memory import sequential_fading_memory_signatures
from sliding_window_signatures.model import (
    RidgeForecastResult,
    TemporalSplit,
    calculate_metrics,
    fit_ridge_forecaster,
    tune_ridge_penalty,
)
from sliding_window_signatures.signatures import sequential_sliding_signatures

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Progress = Callable[[str], None]

COMPARISON_START = np.datetime64("2013-01-01T00:00:00")
TRAIN_END = np.datetime64("2014-01-01T00:00:00")
VALIDATION_END = np.datetime64("2015-01-01T00:00:00")


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Settings for the comparison. Both methods get the same feature count."""

    order: int = 6
    delay_days: int = 7
    memory_horizon_days: tuple[float, ...] = tuple(range(7, 38))
    ridge_penalties: tuple[float, ...] = tuple(np.logspace(-6, 6, 49))
    standardize_features: bool = True
    bootstrap_repetitions: int = 2_000
    bootstrap_block_days: int = 7
    bootstrap_seed: int = 25_070_370


@dataclass(frozen=True, slots=True)
class AlignedFeatures:
    """The smallest feature object the ridge functions accept."""

    values: FloatArray
    end_indices: IntArray


@dataclass(frozen=True, slots=True)
class SignatureComparison:
    """Every table produced by one comparison run."""

    sweep: pd.DataFrame
    metrics: pd.DataFrame
    seasonal_metrics: pd.DataFrame
    forecasts: pd.DataFrame
    runtime: pd.DataFrame
    bootstrap: dict[str, float | int | str]


def run_signature_memory_comparison(
    dataset: PaperDataset,
    *,
    config: ComparisonConfig | None = None,
    progress: Progress | None = None,
) -> SignatureComparison:
    """Compare rolling-window and EFM signatures on exactly the same rows.

    2012 is history only, 2013 trains, 2014 picks the memory horizon and the
    ridge penalty, and 2015 is read once at the end. Everything else is shared
    between the two methods: order six, 120 fitted coefficients once pure-time
    words are dropped, the same seven-day increment, the same ridge grid, and
    standardization fitted on training rows only.

    Order six is taken from the rolling-window paper, not chosen here. That
    paper did look at 2015 when picking it, so 2015 was not used for any of the
    settings chosen here, but it was used for the order.

    Both methods take one memory setting ``H``. The rolling window keeps ``H``
    days. Fading memory uses ``lambda = log(100) / H``, so at level one
    something ``H`` days old still has one percent of its weight. Both divide
    the time coordinate by ``H``.
    """
    settings = config or ComparisonConfig()
    report = progress or _ignore_progress
    _validate_config(settings)

    start_index = _exact_date_index(dataset.dates, COMPARISON_START)
    common_end_indices = np.arange(start_index, len(dataset), dtype=np.int64)
    split = TemporalSplit.from_calendar_boundaries(
        dates=dataset.dates,
        end_indices=common_end_indices,
        train_end=np.datetime_as_string(TRAIN_END),
        validation_end=np.datetime_as_string(VALIDATION_END),
    )
    delay = settings.delay_days * SAMPLES_PER_DAY
    penalties = np.asarray(settings.ridge_penalties, dtype=np.float64)

    sweep_rows: list[dict[str, float | int | str]] = []
    selected_features: dict[str, AlignedFeatures] = {}
    selected_scores: dict[str, float] = {}

    for horizon_days in settings.memory_horizon_days:
        window_intervals = _whole_day_intervals(horizon_days, "rolling window")
        report(f"Rolling signature: validating a {horizon_days:g}-day memory horizon")
        started = perf_counter()
        raw_features = sequential_sliding_signatures(
            dataset.temperature,
            window=window_intervals,
            order=settings.order,
        )
        construction_seconds = perf_counter() - started
        features = _align_rolling_features(
            raw_features.values,
            raw_features.end_indices,
            start_index,
        )
        _require_common_indices(features, common_end_indices)
        tuning = tune_ridge_penalty(
            features,
            dataset.demand,
            delay=delay,
            split=split,
            penalties=penalties,
            selection_metric="rmse",
            standardize_features=settings.standardize_features,
        )
        row = _sweep_row(
            method="Rolling Window Signature",
            parameter_name="window_days",
            parameter_days=horizon_days,
            level_one_half_life_days=np.nan,
            tuning_penalty=tuning.best_penalty,
            validation_rmse=tuning.validation_score,
            construction_seconds=construction_seconds,
            features=features,
            order=settings.order,
        )
        sweep_rows.append(row)
        _keep_if_best(
            "Rolling Window Signature",
            features,
            tuning.validation_score,
            selected_features,
            selected_scores,
        )

    for horizon_days in settings.memory_horizon_days:
        half_life_days = horizon_days * np.log(2.0) / np.log(100.0)
        half_life_intervals = half_life_days * SAMPLES_PER_DAY
        report(f"Fading memory signature: validating a {horizon_days:g}-day 1% horizon")
        started = perf_counter()
        raw_features = sequential_fading_memory_signatures(
            dataset.temperature,
            half_life_intervals=half_life_intervals,
            order=settings.order,
            time_scale_intervals=horizon_days * SAMPLES_PER_DAY,
            start_index=start_index,
        )
        construction_seconds = perf_counter() - started
        features = AlignedFeatures(raw_features.values, raw_features.end_indices)
        _require_common_indices(features, common_end_indices)
        tuning = tune_ridge_penalty(
            features,
            dataset.demand,
            delay=delay,
            split=split,
            penalties=penalties,
            selection_metric="rmse",
            standardize_features=settings.standardize_features,
        )
        row = _sweep_row(
            method="Fading Memory Signature",
            parameter_name="level_one_1pct_horizon_days",
            parameter_days=horizon_days,
            level_one_half_life_days=half_life_days,
            tuning_penalty=tuning.best_penalty,
            validation_rmse=tuning.validation_score,
            construction_seconds=construction_seconds,
            features=features,
            order=settings.order,
        )
        sweep_rows.append(row)
        _keep_if_best(
            "Fading Memory Signature",
            features,
            tuning.validation_score,
            selected_features,
            selected_scores,
        )

    sweep = pd.DataFrame(sweep_rows)
    selected_rows = (
        sweep.sort_values(["method", "validation_rmse_mw", "parameter_days"])
        .groupby("method", as_index=False)
        .first()
    )

    fitted: dict[str, RidgeForecastResult] = {}
    fit_seconds: dict[str, float] = {}
    for method in ("Rolling Window Signature", "Fading Memory Signature"):
        report(f"Final fit on 2013-2014: {method}")
        started = perf_counter()
        fitted[method] = fit_ridge_forecaster(
            selected_features[method],
            dataset.demand,
            delay=delay,
            split=split,
            penalties=penalties,
            selection_metric="rmse",
            standardize_features=settings.standardize_features,
        )
        fit_seconds[method] = perf_counter() - started

    rolling = fitted["Rolling Window Signature"]
    fading = fitted["Fading Memory Signature"]
    if not np.array_equal(rolling.end_indices, fading.end_indices):
        raise RuntimeError("final forecasts do not share exactly the same test timestamps")

    metrics = _metrics_table(dataset, selected_rows, fitted)
    forecasts = _forecast_table(dataset, rolling, fading)
    seasonal = _seasonal_metrics(forecasts)
    runtime = _runtime_table(selected_rows, selected_features, fit_seconds, settings.order)
    bootstrap = paired_block_bootstrap_rmse_difference(
        rolling.y_true,
        rolling.y_predicted,
        fading.y_predicted,
        block_length=settings.bootstrap_block_days * SAMPLES_PER_DAY,
        repetitions=settings.bootstrap_repetitions,
        seed=settings.bootstrap_seed,
    )
    bootstrap.update(
        {
            "comparison": "Fading Memory Signature minus Rolling Window Signature",
            "block_days": settings.bootstrap_block_days,
        }
    )
    return SignatureComparison(
        sweep=sweep,
        metrics=metrics,
        seasonal_metrics=seasonal,
        forecasts=forecasts,
        runtime=runtime,
        bootstrap=bootstrap,
    )


def paired_block_bootstrap_rmse_difference(
    y_true: FloatArray,
    rolling_prediction: FloatArray,
    fading_prediction: FloatArray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    """Put a confidence interval on the paired RMSE difference.

    Resampling whole weeks rather than single half-hours keeps the daily and
    weekly patterns of the errors intact.
    """
    actual = np.asarray(y_true, dtype=np.float64)
    rolling = np.asarray(rolling_prediction, dtype=np.float64)
    fading = np.asarray(fading_prediction, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != rolling.shape or actual.shape != fading.shape:
        raise ValueError("bootstrap arrays must be one-dimensional and equally sized")
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(rolling).all()
        or not np.isfinite(fading).all()
    ):
        raise ValueError("bootstrap arrays must contain only finite values")
    if isinstance(block_length, bool) or not 1 <= block_length <= actual.size:
        raise ValueError("block_length must be between one and the number of observations")
    if isinstance(repetitions, bool) or repetitions < 100:
        raise ValueError("repetitions must be an integer of at least 100")

    rolling_squared_error = np.square(rolling - actual)
    fading_squared_error = np.square(fading - actual)
    observed = float(
        np.sqrt(np.mean(fading_squared_error)) - np.sqrt(np.mean(rolling_squared_error))
    )

    rng = np.random.default_rng(seed)
    block_offsets = np.arange(block_length, dtype=np.int64)
    blocks_needed = int(np.ceil(actual.size / block_length))
    differences = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        starts = rng.integers(0, actual.size, size=blocks_needed)
        indices = ((starts[:, None] + block_offsets) % actual.size).ravel()[: actual.size]
        differences[repetition] = np.sqrt(np.mean(fading_squared_error[indices])) - np.sqrt(
            np.mean(rolling_squared_error[indices])
        )

    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "observed_rmse_difference_mw": observed,
        "ci_95_lower_mw": float(lower),
        "ci_95_upper_mw": float(upper),
        "bootstrap_resample_share_fading_lower_rmse": float(np.mean(differences < 0.0)),
        "bootstrap_repetitions": repetitions,
        "block_length_observations": block_length,
        "seed": seed,
    }


def _align_rolling_features(
    values: FloatArray,
    end_indices: IntArray,
    start_index: int,
) -> AlignedFeatures:
    first_row = int(np.searchsorted(end_indices, start_index, side="left"))
    return AlignedFeatures(
        values=np.asarray(values[first_row:], dtype=np.float64),
        end_indices=np.asarray(end_indices[first_row:], dtype=np.int64),
    )


def _keep_if_best(
    method: str,
    features: AlignedFeatures,
    score: float,
    selected_features: dict[str, AlignedFeatures],
    selected_scores: dict[str, float],
) -> None:
    if method not in selected_scores or score < selected_scores[method]:
        selected_scores[method] = score
        selected_features[method] = features


def _sweep_row(
    *,
    method: str,
    parameter_name: str,
    parameter_days: float,
    level_one_half_life_days: float,
    tuning_penalty: float,
    validation_rmse: float,
    construction_seconds: float,
    features: AlignedFeatures,
    order: int,
) -> dict[str, float | int | str]:
    return {
        "method": method,
        "parameter_name": parameter_name,
        "parameter_days": parameter_days,
        "level_one_half_life_days": level_one_half_life_days,
        "order": order,
        "fitted_feature_count": features.values.shape[1],
        "selected_ridge_penalty": tuning_penalty,
        "validation_rmse_mw": validation_rmse,
        "feature_construction_seconds": construction_seconds,
    }


def _metrics_table(
    dataset: PaperDataset,
    selected_rows: pd.DataFrame,
    fitted: dict[str, RidgeForecastResult],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in ("Rolling Window Signature", "Fading Memory Signature"):
        selection = selected_rows[selected_rows["method"] == method].iloc[0]
        result = fitted[method]
        rows.append(
            {
                "method": method,
                "selected_parameter": str(selection["parameter_name"]),
                "selected_parameter_days": float(selection["parameter_days"]),
                "level_one_half_life_days": float(selection["level_one_half_life_days"]),
                "order": int(selection["order"]),
                "fitted_feature_count": int(selection["fitted_feature_count"]),
                "selected_ridge_penalty": result.best_penalty,
                "validation_rmse_mw": result.validation_score,
                "test_rmse_mw": result.metrics.rmse,
                "test_mae_mw": result.metrics.mae,
                "test_mape_percent": result.metrics.mape_percent,
                "test_start": np.datetime_as_string(dataset.dates[result.end_indices[0]], unit="m"),
                "test_end": np.datetime_as_string(dataset.dates[result.end_indices[-1]], unit="m"),
                "test_observations": result.end_indices.size,
            }
        )
    return pd.DataFrame(rows)


def _forecast_table(
    dataset: PaperDataset,
    rolling: RidgeForecastResult,
    fading: RidgeForecastResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": dataset.dates[rolling.end_indices],
            "observed_mw": rolling.y_true,
            "rolling_window_prediction_mw": rolling.y_predicted,
            "fading_memory_prediction_mw": fading.y_predicted,
            "rolling_window_error_mw": rolling.y_predicted - rolling.y_true,
            "fading_memory_error_mw": fading.y_predicted - fading.y_true,
        }
    )


def _seasonal_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(forecasts["datetime"])
    season_by_month = {
        12: "Winter (DJF)",
        1: "Winter (DJF)",
        2: "Winter (DJF)",
        3: "Spring (MAM)",
        4: "Spring (MAM)",
        5: "Spring (MAM)",
        6: "Summer (JJA)",
        7: "Summer (JJA)",
        8: "Summer (JJA)",
        9: "Autumn (SON)",
        10: "Autumn (SON)",
        11: "Autumn (SON)",
    }
    seasons = np.asarray([season_by_month[month] for month in dates.month])
    rows: list[dict[str, float | int | str]] = []
    for season in ("Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"):
        mask = seasons == season
        actual = forecasts.loc[mask, "observed_mw"].to_numpy()
        for method, column in (
            ("Rolling Window Signature", "rolling_window_prediction_mw"),
            ("Fading Memory Signature", "fading_memory_prediction_mw"),
        ):
            prediction = forecasts.loc[mask, column].to_numpy()
            metrics = calculate_metrics(actual, prediction)
            rows.append(
                {
                    "season": season,
                    "method": method,
                    "observations": int(np.count_nonzero(mask)),
                    "rmse_mw": metrics.rmse,
                    "mae_mw": metrics.mae,
                    "mape_percent": metrics.mape_percent,
                }
            )
    return pd.DataFrame(rows)


def _runtime_table(
    selected_rows: pd.DataFrame,
    selected_features: dict[str, AlignedFeatures],
    fit_seconds: dict[str, float],
    order: int,
) -> pd.DataFrame:
    raw_state_width = int(iisignature.siglength(2, order))
    rows: list[dict[str, float | int | str]] = []
    for method in ("Rolling Window Signature", "Fading Memory Signature"):
        selection = selected_rows[selected_rows["method"] == method].iloc[0]
        features = selected_features[method]
        rows.append(
            {
                "method": method,
                "selected_parameter_days": float(selection["parameter_days"]),
                "feature_construction_seconds": float(selection["feature_construction_seconds"]),
                "ridge_tune_and_refit_seconds": fit_seconds[method],
                "feature_rows": features.values.shape[0],
                "fitted_feature_count": features.values.shape[1],
                "feature_matrix_mib": features.values.nbytes / (1024.0**2),
                "online_signature_state_kib": raw_state_width * 8 / 1024.0,
            }
        )
    return pd.DataFrame(rows)


def _require_common_indices(features: AlignedFeatures, expected: IntArray) -> None:
    if not np.array_equal(features.end_indices, expected):
        raise RuntimeError("candidate features do not cover the common comparison rows")


def _whole_day_intervals(days: float, label: str) -> int:
    intervals = days * SAMPLES_PER_DAY
    rounded = round(intervals)
    if not np.isclose(intervals, rounded) or rounded < 1:
        raise ValueError(f"{label} must be a positive multiple of one half-hour")
    return rounded


def _exact_date_index(dates: NDArray[np.datetime64], date: np.datetime64) -> int:
    index = int(np.searchsorted(dates, date, side="left"))
    if index >= dates.size or dates[index] != date:
        raise ValueError(f"dataset does not contain the required timestamp {date}")
    return index


def _validate_config(config: ComparisonConfig) -> None:
    if config.order < 1:
        raise ValueError("order must be positive")
    if config.delay_days < 1:
        raise ValueError("delay_days must be positive")
    if not config.memory_horizon_days or min(config.memory_horizon_days) <= 0:
        raise ValueError("memory_horizon_days must contain positive values")
    penalties = np.asarray(config.ridge_penalties, dtype=np.float64)
    if penalties.ndim != 1 or penalties.size < 1 or np.any(penalties <= 0):
        raise ValueError("ridge_penalties must contain positive values")
    if config.bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be at least 100")
    if config.bootstrap_block_days < 1:
        raise ValueError("bootstrap_block_days must be positive")


def _ignore_progress(_: str) -> None:
    return None
