"""Ridge regression on signature features and delayed target increments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MetricName = Literal["rmse", "mae", "mape"]


class FeatureMatrix(Protocol):
    """The attributes the ridge functions need from a feature object."""

    values: FloatArray
    end_indices: IntArray


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Contiguous train, validation, and test partitions of usable rows."""

    train_stop: int
    validation_stop: int
    sample_count: int

    def __post_init__(self) -> None:
        if not 1 <= self.train_stop < self.validation_stop < self.sample_count:
            raise ValueError(
                "split must satisfy 1 <= train_stop < validation_stop < sample_count; "
                f"got {self.train_stop}, {self.validation_stop}, {self.sample_count}"
            )

    @classmethod
    def from_reference_counts(
        cls,
        *,
        train_count: int,
        validation_count: int,
        sample_count: int,
    ) -> TemporalSplit:
        """Copy the row counting used by the authors' notebook.

        The notebook counts every row in 2012-2013 and then takes that many
        signature rows. Signature rows only start at ``window``, so each
        boundary ends up one window later than the calendar date.
        """
        return cls(
            train_stop=train_count,
            validation_stop=train_count + validation_count,
            sample_count=sample_count,
        )

    @classmethod
    def from_calendar_boundaries(
        cls,
        *,
        dates: ArrayLike,
        end_indices: ArrayLike,
        train_end: str = "2014-01-01T00:00:00",
        validation_end: str = "2015-01-01T00:00:00",
    ) -> TemporalSplit:
        """Split rows by the dates on which their windows end."""
        all_dates = np.asarray(dates, dtype="datetime64[ns]")
        indices = np.asarray(end_indices, dtype=np.int64)
        if all_dates.ndim != 1 or indices.ndim != 1:
            raise ValueError("dates and end_indices must be one-dimensional")
        if indices.size < 3 or indices.min() < 0 or indices.max() >= all_dates.size:
            raise ValueError("end_indices are outside the supplied dates")
        end_dates = all_dates[indices]
        if np.any(end_dates[1:] < end_dates[:-1]):
            raise ValueError("window end dates must be sorted")

        train_stop = int(np.searchsorted(end_dates, np.datetime64(train_end), side="left"))
        validation_stop = int(
            np.searchsorted(end_dates, np.datetime64(validation_end), side="left")
        )
        return cls(
            train_stop=train_stop,
            validation_stop=validation_stop,
            sample_count=indices.size,
        )

    @property
    def train(self) -> slice:
        """Training rows."""
        return slice(0, self.train_stop)

    @property
    def validation(self) -> slice:
        """Validation rows."""
        return slice(self.train_stop, self.validation_stop)

    @property
    def train_and_validation(self) -> slice:
        """Rows used for the final refit."""
        return slice(0, self.validation_stop)

    @property
    def test(self) -> slice:
        """Held-out test rows."""
        return slice(self.validation_stop, self.sample_count)


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Forecast errors in the same units used by the paper."""

    rmse: float
    mae: float
    mape_percent: float


@dataclass(frozen=True, slots=True)
class RidgeForecastResult:
    """Fitted model details and held-out forecasts."""

    best_penalty: float
    validation_score: float
    selection_metric: MetricName
    coefficients: FloatArray
    intercept: float
    y_true: FloatArray
    y_predicted: FloatArray
    end_indices: IntArray
    metrics: ForecastMetrics


@dataclass(frozen=True, slots=True)
class RidgeTuningResult:
    """A ridge penalty chosen on validation rows, with no test evaluation."""

    best_penalty: float
    validation_score: float
    selection_metric: MetricName


@dataclass(frozen=True, slots=True)
class _PreparedForecastData:
    """Checked arrays that both tuning and the final refit need."""

    target: FloatArray
    end_indices: IntArray
    past_indices: IntArray
    increments: FloatArray


def fit_ridge_forecaster(
    features: FeatureMatrix,
    target: ArrayLike,
    *,
    delay: int,
    split: TemporalSplit,
    penalties: ArrayLike,
    selection_metric: MetricName = "rmse",
    standardize_features: bool = False,
) -> RidgeForecastResult:
    """Pick a ridge penalty, refit, and forecast the held-out rows.

    The model predicts the change ``target[t] - target[t - delay]``. Adding the
    known ``target[t - delay]`` back turns a predicted change into a predicted
    level, as in Section 4.1 and Algorithm 2 of the paper.
    """
    prepared = _prepare_forecast_data(features, target, delay=delay, split=split)
    tuning = _tune_prepared_ridge(
        features,
        prepared,
        split=split,
        penalties=penalties,
        selection_metric=selection_metric,
        standardize_features=standardize_features,
    )

    final_scaler: StandardScaler | None = None
    final_training_features = features.values[split.train_and_validation]
    if standardize_features:
        final_scaler = StandardScaler()
        final_training_features = final_scaler.fit_transform(final_training_features)
    final_model = Ridge(alpha=tuning.best_penalty)
    final_model.fit(
        final_training_features,
        prepared.increments[split.train_and_validation],
    )

    test_end_indices = prepared.end_indices[split.test]
    test_past_indices = prepared.past_indices[split.test]
    y_true = prepared.target[test_end_indices]
    test_features = features.values[split.test]
    if final_scaler is not None:
        test_features = final_scaler.transform(test_features)
    y_predicted = final_model.predict(test_features) + prepared.target[test_past_indices]
    metrics = calculate_metrics(y_true, y_predicted)

    coefficients = np.asarray(final_model.coef_, dtype=np.float64)
    intercept = float(final_model.intercept_)
    if final_scaler is not None:
        coefficients = coefficients / final_scaler.scale_
        intercept -= float(np.dot(coefficients, final_scaler.mean_))

    return RidgeForecastResult(
        best_penalty=tuning.best_penalty,
        validation_score=tuning.validation_score,
        selection_metric=selection_metric,
        coefficients=coefficients,
        intercept=intercept,
        y_true=np.asarray(y_true, dtype=np.float64),
        y_predicted=np.asarray(y_predicted, dtype=np.float64),
        end_indices=np.asarray(test_end_indices, dtype=np.int64),
        metrics=metrics,
    )


def tune_ridge_penalty(
    features: FeatureMatrix,
    target: ArrayLike,
    *,
    delay: int,
    split: TemporalSplit,
    penalties: ArrayLike,
    selection_metric: MetricName = "rmse",
    standardize_features: bool = False,
) -> RidgeTuningResult:
    """Select a ridge penalty from the training and validation rows only.

    It does not refit on train-plus-validation and does not read the test
    target, so it can be called inside a loop that is choosing the window size
    or the memory horizon.
    """
    prepared = _prepare_forecast_data(features, target, delay=delay, split=split)
    return _tune_prepared_ridge(
        features,
        prepared,
        split=split,
        penalties=penalties,
        selection_metric=selection_metric,
        standardize_features=standardize_features,
    )


def _prepare_forecast_data(
    features: FeatureMatrix,
    target: ArrayLike,
    *,
    delay: int,
    split: TemporalSplit,
) -> _PreparedForecastData:
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("target must be a finite one-dimensional array")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError("delay must be a positive integer")
    if features.values.ndim != 2 or features.end_indices.ndim != 1:
        raise ValueError("features must contain a matrix and one-dimensional end indices")
    if features.values.shape[0] != split.sample_count:
        raise ValueError("split sample_count does not match the signature matrix")
    if features.end_indices.size != features.values.shape[0]:
        raise ValueError("signature rows and end indices must have equal length")
    if features.end_indices[-1] >= y.size:
        raise ValueError("target is shorter than the signature end indices")
    if np.any(features.end_indices - delay < 0):
        raise ValueError("delay reaches before the start of the target series")

    end_indices = np.asarray(features.end_indices, dtype=np.int64)
    past_indices = end_indices - delay
    increments = y[end_indices] - y[past_indices]
    return _PreparedForecastData(
        target=y,
        end_indices=end_indices,
        past_indices=past_indices,
        increments=np.asarray(increments, dtype=np.float64),
    )


def _tune_prepared_ridge(
    features: FeatureMatrix,
    prepared: _PreparedForecastData,
    *,
    split: TemporalSplit,
    penalties: ArrayLike,
    selection_metric: MetricName,
    standardize_features: bool,
) -> RidgeTuningResult:
    if selection_metric not in {"rmse", "mae", "mape"}:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    candidate_penalties = np.asarray(penalties, dtype=np.float64)
    if (
        candidate_penalties.ndim != 1
        or candidate_penalties.size < 1
        or not np.isfinite(candidate_penalties).all()
        or np.any(candidate_penalties <= 0)
    ):
        raise ValueError("penalties must be a non-empty finite sequence of positive values")

    x_train = features.values[split.train]
    delta_train = prepared.increments[split.train]
    x_validation = features.values[split.validation]
    if standardize_features:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
    y_validation = prepared.target[prepared.end_indices[split.validation]]
    y_past_validation = prepared.target[prepared.past_indices[split.validation]]

    best_penalty = float(candidate_penalties[0])
    best_score = np.inf
    for penalty in candidate_penalties:
        candidate = Ridge(alpha=float(penalty))
        candidate.fit(x_train, delta_train)
        prediction = candidate.predict(x_validation) + y_past_validation
        score = metric_value(selection_metric, y_validation, prediction)
        if score < best_score:
            best_score = score
            best_penalty = float(penalty)
    return RidgeTuningResult(
        best_penalty=best_penalty,
        validation_score=float(best_score),
        selection_metric=selection_metric,
    )


def calculate_metrics(y_true: ArrayLike, y_predicted: ArrayLike) -> ForecastMetrics:
    """Calculate RMSE, MAE, and percentage MAPE."""
    actual, forecast = _validated_metric_arrays(y_true, y_predicted)
    error = forecast - actual
    return ForecastMetrics(
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mae=float(np.mean(np.abs(error))),
        mape_percent=float(100.0 * np.mean(np.abs(error / actual))),
    )


def metric_value(name: MetricName, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
    """Return the single metric used to compare candidate models."""
    metrics = calculate_metrics(y_true, y_predicted)
    if name == "rmse":
        return metrics.rmse
    if name == "mae":
        return metrics.mae
    return metrics.mape_percent


def _validated_metric_arrays(
    y_true: ArrayLike,
    y_predicted: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    actual = np.asarray(y_true, dtype=np.float64)
    forecast = np.asarray(y_predicted, dtype=np.float64)
    if actual.ndim != 1 or forecast.ndim != 1 or actual.shape != forecast.shape:
        raise ValueError("metric arrays must be one-dimensional and have equal shape")
    if actual.size < 1 or not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("metric arrays must be non-empty and finite")
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when a target value is zero")
    return actual, forecast
