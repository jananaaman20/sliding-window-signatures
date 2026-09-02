"""Exponentially fading memory (EFM) signatures.

Instead of a hard window, old data fades away smoothly. This is Abi Jaber and
Sotnikov's EFM signature with one decay rate for every path coordinate. Their
definition allows a different rate per coordinate, but sharing one rate is what
makes the exact update for straight segments cheap (Lemma 4.2 and equation 4.4
of arXiv:2507.03700).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import iisignature
import numpy as np
from numpy.typing import ArrayLike, NDArray

from sliding_window_signatures.signatures import (
    TensorSeries,
    chen_product,
    linear_segment_signature,
    signature_feature_count,
    time_only_feature_indices,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FadingMemoryFeatures:
    """A matrix of EFM-signature features and their source-series indices.

    Attributes:
        values: Feature matrix. Row ``i`` describes the path history ending at
            ``end_indices[i]``.
        end_indices: Indices in the original series at which states are read.
        half_life_intervals: Number of observation intervals after which a
            level-one contribution has half its original weight.
        order: Highest retained signature level.
        path_dimension: Covariate dimension plus the added time coordinate.
        time_scale_intervals: Number of observations represented by one unit
            of the augmented time coordinate.
        time_only_dropped: Whether one pure-time coefficient per level was
            removed, matching the rolling-signature experiments.
    """

    values: FloatArray
    end_indices: NDArray[np.int64]
    half_life_intervals: float
    order: int
    path_dimension: int
    time_scale_intervals: float
    time_only_dropped: bool

    def through_order(self, order: int) -> FadingMemoryFeatures:
        """Return the prefix containing EFM-signature levels up to ``order``."""
        if not 1 <= order <= self.order:
            raise ValueError(f"order must be between 1 and {self.order}; got {order}")
        width = signature_feature_count(
            self.path_dimension,
            order,
            drop_time_only=self.time_only_dropped,
        )
        return FadingMemoryFeatures(
            values=self.values[:, :width],
            end_indices=self.end_indices,
            half_life_intervals=self.half_life_intervals,
            order=order,
            path_dimension=self.path_dimension,
            time_scale_intervals=self.time_scale_intervals,
            time_only_dropped=self.time_only_dropped,
        )


def sequential_fading_memory_signatures(
    covariates: ArrayLike,
    *,
    half_life_intervals: float,
    order: int = 4,
    time_scale_intervals: float = 1.0,
    start_index: int = 1,
    drop_time_only: bool = True,
) -> FadingMemoryFeatures:
    """Compute equal-rate EFM signatures with the paper's exact update.

    With ``lambda = log(2) / half_life_intervals``, one observation interval of
    equation (4.4) reduces to

    ``state <- D_1(state) tensor exp_tensor(phi * delta_x)``,

    where level ``k`` of ``D_1(state)`` is scaled by ``exp(-k * lambda)`` and
    ``phi = (1 - exp(-lambda)) / lambda``. The tensor exponential and the Chen
    product run inside ``iisignature``, and the update is exact.

    Every path coordinate uses the same decay rate. This is the equal-rate
    case of Lemma 4.2. The time-augmented increment is
    ``(1 / time_scale_intervals, delta_covariates)``.

    Args:
        covariates: One- or two-dimensional covariate series, ordered in time.
        half_life_intervals: Positive level-one memory half-life, measured in
            observation intervals.
        order: Signature truncation order ``N``.
        time_scale_intervals: Positive scaling for the augmented time channel.
        start_index: First original-series index returned. Earlier
            observations still update the state, so they act as warm-up.
        drop_time_only: Remove one pure-time coefficient per retained level.

    Returns:
        EFM states ordered by their original-series end index.
    """
    matrix = _as_covariate_matrix(covariates)
    half_life = _positive_float(half_life_intervals, "half_life_intervals")
    time_scale = _positive_float(time_scale_intervals, "time_scale_intervals")
    _validate_order(order)
    _validate_start_index(start_index, matrix.shape[0])

    path_dimension = matrix.shape[1] + 1
    signature_width = int(iisignature.siglength(path_dimension, order))
    state = np.zeros(signature_width, dtype=np.float64)

    decay_rate = log(2.0) / half_life
    level_one_decay = np.exp(-decay_rate)
    segment_scale = -np.expm1(-decay_rate) / decay_rate
    decay_by_coefficient = np.concatenate(
        [
            np.full(path_dimension**level, level_one_decay**level, dtype=np.float64)
            for level in range(1, order + 1)
        ]
    )

    end_indices = np.arange(start_index, matrix.shape[0], dtype=np.int64)
    values = np.empty((end_indices.size, signature_width), dtype=np.float64)
    zero = np.zeros(path_dimension, dtype=np.float64)
    time_increment = 1.0 / time_scale

    output_row = 0
    for end_index in range(1, matrix.shape[0]):
        increment = np.concatenate(
            (
                np.asarray([time_increment], dtype=np.float64),
                matrix[end_index] - matrix[end_index - 1],
            )
        )
        effective_increment = segment_scale * increment
        segment = np.asarray(
            iisignature.sig(np.vstack((zero, effective_increment)), order),
            dtype=np.float64,
        )
        state = np.asarray(
            iisignature.sigcombine(
                state * decay_by_coefficient,
                segment,
                path_dimension,
                order,
            ),
            dtype=np.float64,
        )
        if end_index >= start_index:
            values[output_row] = state
            output_row += 1

    if drop_time_only:
        values = np.delete(values, time_only_feature_indices(path_dimension, order), axis=1)

    return FadingMemoryFeatures(
        values=values,
        end_indices=end_indices,
        half_life_intervals=half_life,
        order=order,
        path_dimension=path_dimension,
        time_scale_intervals=time_scale,
        time_only_dropped=drop_time_only,
    )


def reference_fading_memory_signature(
    path: ArrayLike,
    *,
    half_life_intervals: float,
    order: int,
) -> FloatArray:
    """Compute one EFM state with plain tensor algebra.

    ``path`` is already time-augmented and sampled one interval apart. Slow,
    but simple enough to check :func:`sequential_fading_memory_signatures` by
    hand.
    """
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("path must have shape (n_points, dimension) with n_points >= 2")
    if not np.isfinite(points).all():
        raise ValueError("path must contain only finite values")
    half_life = _positive_float(half_life_intervals, "half_life_intervals")
    _validate_order(order)

    dimension = points.shape[1]
    decay_rate = log(2.0) / half_life
    level_one_decay = np.exp(-decay_rate)
    segment_scale = -np.expm1(-decay_rate) / decay_rate

    identity: list[FloatArray] = [np.ones(1, dtype=np.float64)]
    identity.extend(np.zeros(dimension**level, dtype=np.float64) for level in range(1, order + 1))
    state: TensorSeries = tuple(identity)

    for increment in np.diff(points, axis=0):
        discounted = tuple(
            level_values * level_one_decay**level for level, level_values in enumerate(state)
        )
        segment = linear_segment_signature(segment_scale * increment, order)
        state = chen_product(discounted, segment)
    return np.concatenate(state[1:])


def _as_covariate_matrix(covariates: ArrayLike) -> FloatArray:
    matrix = np.asarray(covariates, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(
            "covariates must have shape (n_samples,) or (n_samples, n_covariates), "
            "with at least two samples"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("covariates must contain only finite values")
    return matrix


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 1:
        raise ValueError("order must be at least 1")


def _validate_start_index(start_index: int, sample_count: int) -> None:
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")
    if not 1 <= start_index < sample_count:
        raise ValueError(f"start_index must satisfy 1 <= start_index < {sample_count}")
