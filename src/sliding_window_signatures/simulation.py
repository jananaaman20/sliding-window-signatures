"""Builds the synthetic demand series from Appendix D of the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticDemand:
    """Synthetic target and the fitted temperature relationship behind it."""

    values: FloatArray
    smoothed_temperature: FloatArray
    coefficients: FloatArray
    intercept: float
    noise_standard_deviation: float
    seed: int


def exponential_smoothing(series: ArrayLike, alpha: float) -> FloatArray:
    """Apply the paper's recursive exponential smoothing.

    A small ``alpha`` gives older observations more influence. ``alpha=1``
    returns the original series.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("series must be a non-empty finite one-dimensional array")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def simulate_synthetic_demand(
    temperature: ArrayLike,
    observed_demand: ArrayLike,
    *,
    alpha: float,
    noise_standard_deviation: float,
    seed: int = 141,
) -> SyntheticDemand:
    """Generate the quadratic, smoothed-temperature target from Appendix D.

    The notebook fits ``LinearRegression`` with its default intercept, even
    though Equations (3)-(4) only show two coefficients. This copies the
    notebook, not the printed formula.
    """
    temperature_values = np.asarray(temperature, dtype=np.float64)
    demand_values = np.asarray(observed_demand, dtype=np.float64)
    if (
        temperature_values.ndim != 1
        or demand_values.ndim != 1
        or temperature_values.shape != demand_values.shape
        or temperature_values.size < 2
        or not np.isfinite(temperature_values).all()
        or not np.isfinite(demand_values).all()
    ):
        raise ValueError("temperature and observed_demand must be equal-length finite vectors")
    if not np.isfinite(noise_standard_deviation) or noise_standard_deviation < 0:
        raise ValueError("noise_standard_deviation must be finite and non-negative")

    smoothed = exponential_smoothing(temperature_values, alpha)
    design = np.column_stack((smoothed, np.square(smoothed)))
    relationship = LinearRegression().fit(design, demand_values)
    fitted = relationship.predict(design)

    # RandomState matches the old np.random.seed()/np.random.normal stream the
    # notebook uses. The newer default_rng would give different numbers.
    random_state = np.random.RandomState(seed)
    noise = random_state.normal(0.0, noise_standard_deviation, fitted.size)

    return SyntheticDemand(
        values=np.asarray(fitted + noise, dtype=np.float64),
        smoothed_temperature=smoothed,
        coefficients=np.asarray(relationship.coef_, dtype=np.float64),
        intercept=float(relationship.intercept_),
        noise_standard_deviation=float(noise_standard_deviation),
        seed=seed,
    )
