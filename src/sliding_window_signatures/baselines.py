"""Linear-regression baselines used in Tables 1 and 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression

from sliding_window_signatures.model import ForecastMetrics, TemporalSplit, calculate_metrics
from sliding_window_signatures.simulation import exponential_smoothing

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LinearForecastResult:
    """Held-out forecasts from an ordinary linear-regression baseline."""

    y_true: FloatArray
    y_predicted: FloatArray
    end_indices: IntArray
    metrics: ForecastMetrics


def synthetic_baselines(
    temperature: ArrayLike,
    synthetic_demand: ArrayLike,
    *,
    alpha: float,
    train_stop: int,
) -> Mapping[str, LinearForecastResult]:
    """Fit the four synthetic-data baselines from Table 1."""
    temp, demand = _validated_pair(temperature, synthetic_demand)
    smoothed = exponential_smoothing(temp, alpha)
    designs = {
        "LR(T)": temp.reshape(-1, 1),
        "LR(T,T^2)": np.column_stack((temp, np.square(temp))),
        "LR(T_alpha)": smoothed.reshape(-1, 1),
        "LR(T_alpha,T_alpha^2)": np.column_stack((smoothed, np.square(smoothed))),
    }
    train = np.arange(0, train_stop, dtype=np.int64)
    test = np.arange(train_stop, temp.size, dtype=np.int64)
    return {
        name: _fit_linear_forecast(design, demand, train_indices=train, test_indices=test)
        for name, design in designs.items()
    }


def calendar_synthetic_baselines(
    temperature: ArrayLike,
    synthetic_demand: ArrayLike,
    *,
    alpha: float,
    end_indices: ArrayLike,
    split: TemporalSplit,
) -> Mapping[str, LinearForecastResult]:
    """Score the synthetic baselines on the signature model's own test rows."""
    temp, demand = _validated_pair(temperature, synthetic_demand)
    indices = np.asarray(end_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size != split.sample_count:
        raise ValueError("end_indices must match the temporal split")
    smoothed = exponential_smoothing(temp, alpha)
    designs = {
        "LR(T)": temp[indices].reshape(-1, 1),
        "LR(T,T^2)": np.column_stack((temp[indices], np.square(temp[indices]))),
        "LR(T_alpha)": smoothed[indices].reshape(-1, 1),
        "LR(T_alpha,T_alpha^2)": np.column_stack((smoothed[indices], np.square(smoothed[indices]))),
    }
    aligned_target = demand[indices]
    train_rows = np.arange(0, split.validation_stop, dtype=np.int64)
    test_rows = np.arange(split.validation_stop, split.sample_count, dtype=np.int64)
    return {
        name: _fit_linear_forecast(
            design,
            aligned_target,
            train_indices=train_rows,
            test_indices=test_rows,
            original_indices=indices,
        )
        for name, design in designs.items()
    }


def reference_real_baselines(
    temperature: ArrayLike,
    demand: ArrayLike,
    *,
    alpha: float,
    train_count: int,
    validation_count: int,
    delay: int,
    window: int,
) -> Mapping[str, LinearForecastResult]:
    """Recreate the mismatched row alignments behind the paper's Table 2.

    The published rows do not all use the same test period. Use
    :func:`calendar_real_baselines` when the rows need to be comparable.
    """
    temp, target = _validated_pair(temperature, demand)
    train_and_validation_count = train_count + validation_count
    smoothed = exponential_smoothing(temp, alpha)
    temperature_designs = {
        "LR(T_alpha,T_alpha^2)": np.column_stack((smoothed, np.square(smoothed))),
        "LR(T,T^2,T_alpha,T_alpha^2)": np.column_stack(
            (temp, np.square(temp), smoothed, np.square(smoothed))
        ),
    }
    train = np.arange(0, train_and_validation_count, dtype=np.int64)
    test = np.arange(train_and_validation_count, target.size, dtype=np.int64)
    results = {
        name: _fit_linear_forecast(design, target, train_indices=train, test_indices=test)
        for name, design in temperature_designs.items()
    }

    lag_end_indices = np.arange(delay, target.size, dtype=np.int64)
    lagged_design = np.column_stack(
        (
            temp[lag_end_indices],
            np.square(temp[lag_end_indices]),
            smoothed[lag_end_indices],
            np.square(smoothed[lag_end_indices]),
            target[lag_end_indices - delay],
        )
    )
    lagged_target = target[lag_end_indices]

    # This one-window offset is what reproduces the paper's 3,714 MW row.
    lag_train_rows = np.arange(window, train_and_validation_count + window, dtype=np.int64)
    lag_test_rows = np.arange(
        train_and_validation_count + window, lagged_target.size, dtype=np.int64
    )
    lag_result = _fit_linear_forecast(
        lagged_design,
        lagged_target,
        train_indices=lag_train_rows,
        test_indices=lag_test_rows,
        original_indices=lag_end_indices,
    )
    results["LR(T,T^2,T_alpha,T_alpha^2,Y_lag)"] = lag_result

    # The saved notebook scores this naive row on 2014-2015, not just 2015.
    naive_end_indices = np.arange(train_count, target.size, dtype=np.int64)
    naive_true = target[naive_end_indices]
    naive_predicted = target[naive_end_indices - delay]
    results["Y_lag"] = LinearForecastResult(
        y_true=naive_true,
        y_predicted=naive_predicted,
        end_indices=naive_end_indices,
        metrics=calculate_metrics(naive_true, naive_predicted),
    )
    return results


def calendar_real_baselines(
    temperature: ArrayLike,
    demand: ArrayLike,
    *,
    alpha: float,
    delay: int,
    end_indices: ArrayLike,
    split: TemporalSplit,
) -> Mapping[str, LinearForecastResult]:
    """Score every real-data baseline on the same calendar-aligned rows."""
    temp, target = _validated_pair(temperature, demand)
    indices = np.asarray(end_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size != split.sample_count:
        raise ValueError("end_indices must match the temporal split")
    if indices.min() < delay or indices.max() >= target.size:
        raise ValueError("end_indices do not support the requested lag")

    smoothed = exponential_smoothing(temp, alpha)
    designs = {
        "LR(T_alpha,T_alpha^2)": np.column_stack((smoothed[indices], np.square(smoothed[indices]))),
        "LR(T,T^2,T_alpha,T_alpha^2)": np.column_stack(
            (
                temp[indices],
                np.square(temp[indices]),
                smoothed[indices],
                np.square(smoothed[indices]),
            )
        ),
        "LR(T,T^2,T_alpha,T_alpha^2,Y_lag)": np.column_stack(
            (
                temp[indices],
                np.square(temp[indices]),
                smoothed[indices],
                np.square(smoothed[indices]),
                target[indices - delay],
            )
        ),
    }
    train_rows = np.arange(0, split.validation_stop, dtype=np.int64)
    test_rows = np.arange(split.validation_stop, split.sample_count, dtype=np.int64)
    aligned_target = target[indices]
    results = {
        name: _fit_linear_forecast(
            design,
            aligned_target,
            train_indices=train_rows,
            test_indices=test_rows,
            original_indices=indices,
        )
        for name, design in designs.items()
    }

    naive_true = target[indices[test_rows]]
    naive_predicted = target[indices[test_rows] - delay]
    results["Y_lag"] = LinearForecastResult(
        y_true=naive_true,
        y_predicted=naive_predicted,
        end_indices=indices[test_rows],
        metrics=calculate_metrics(naive_true, naive_predicted),
    )
    return results


def _fit_linear_forecast(
    design: FloatArray,
    target: FloatArray,
    *,
    train_indices: IntArray,
    test_indices: IntArray,
    original_indices: IntArray | None = None,
) -> LinearForecastResult:
    if train_indices.size < 1 or test_indices.size < 1:
        raise ValueError("baseline train and test partitions must be non-empty")
    if train_indices.max() >= target.size or test_indices.max() >= target.size:
        raise ValueError("baseline indices exceed the available rows")
    model = LinearRegression().fit(design[train_indices], target[train_indices])
    y_true = target[test_indices]
    y_predicted = model.predict(design[test_indices])
    end_indices = test_indices if original_indices is None else original_indices[test_indices]
    return LinearForecastResult(
        y_true=np.asarray(y_true, dtype=np.float64),
        y_predicted=np.asarray(y_predicted, dtype=np.float64),
        end_indices=np.asarray(end_indices, dtype=np.int64),
        metrics=calculate_metrics(y_true, y_predicted),
    )


def _validated_pair(first: ArrayLike, second: ArrayLike) -> tuple[FloatArray, FloatArray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size < 3
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("input series must be equal-length finite vectors")
    return left, right
