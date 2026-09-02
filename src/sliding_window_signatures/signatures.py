"""Signature features and the paper's sliding-window update.

The experiments call :func:`sequential_sliding_signatures`, which does its
arithmetic in ``iisignature``. :func:`reference_path_signature` and the small
tensor functions around it do the same thing in NumPy. They are much slower but
easier to read, and useful for checking the fast one.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import iisignature
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TensorSeries = tuple[FloatArray, ...]


@dataclass(frozen=True, slots=True)
class SignatureFeatures:
    """A matrix of signatures and the target-series index of every row.

    Attributes:
        values: Feature matrix. Row ``i`` describes the path ending at
            ``end_indices[i]``.
        end_indices: Indices in the original time series where windows end.
        window: Window length in observation intervals.
        order: Highest retained signature level.
        path_dimension: Covariate dimension plus the added time coordinate.
        time_only_dropped: Whether the pure-time coefficient of each level was
            removed, as in Section 4.1 of the paper.
    """

    values: FloatArray
    end_indices: NDArray[np.int64]
    window: int
    order: int
    path_dimension: int
    time_only_dropped: bool

    def through_order(self, order: int) -> SignatureFeatures:
        """Return the prefix containing signature levels up to ``order``."""
        if not 1 <= order <= self.order:
            raise ValueError(f"order must be between 1 and {self.order}; got {order}")
        width = signature_feature_count(
            self.path_dimension,
            order,
            drop_time_only=self.time_only_dropped,
        )
        return SignatureFeatures(
            values=self.values[:, :width],
            end_indices=self.end_indices,
            window=self.window,
            order=order,
            path_dimension=self.path_dimension,
            time_only_dropped=self.time_only_dropped,
        )


def signature_feature_count(
    path_dimension: int,
    order: int,
    *,
    drop_time_only: bool = False,
) -> int:
    """Return the flattened feature count, excluding the level-zero constant."""
    _validate_dimension_and_order(path_dimension, order)
    count = int(iisignature.siglength(path_dimension, order))
    return count - order if drop_time_only else count


def time_only_feature_indices(path_dimension: int, order: int) -> NDArray[np.int64]:
    """Return flat indices of coefficients involving only the time coordinate.

    Inside each level, ``iisignature`` lists the words in index order. Time is
    coordinate zero here, so the all-time word comes first in every level.
    """
    _validate_dimension_and_order(path_dimension, order)
    offsets: list[int] = []
    offset = 0
    for level in range(1, order + 1):
        offsets.append(offset)
        offset += path_dimension**level
    return np.asarray(offsets, dtype=np.int64)


def linear_segment_signature(increment: ArrayLike, order: int) -> TensorSeries:
    """Compute a straight segment's tensor series with ``dx^k / k!``.

    Example 1 in the paper. Level zero is just 1.
    """
    delta = np.asarray(increment, dtype=np.float64)
    if delta.ndim != 1 or delta.size < 1:
        raise ValueError("increment must be a non-empty one-dimensional array")
    if order < 1:
        raise ValueError("order must be at least 1")

    levels: list[FloatArray] = [np.ones(1, dtype=np.float64)]
    tensor_power = np.ones(1, dtype=np.float64)
    for level in range(1, order + 1):
        tensor_power = np.kron(tensor_power, delta)
        levels.append(np.asarray(tensor_power / factorial(level), dtype=np.float64))
    return tuple(levels)


def chen_product(left: TensorSeries, right: TensorSeries) -> TensorSeries:
    """Concatenate two truncated tensor series with Chen's identity."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("tensor series must have the same positive order")
    dimension = left[1].size
    order = len(left) - 1
    expected_sizes = [dimension**level for level in range(order + 1)]
    for name, series in (("left", left), ("right", right)):
        sizes = [level.size for level in series]
        if sizes != expected_sizes:
            raise ValueError(f"{name} tensor levels have sizes {sizes}, expected {expected_sizes}")

    result: list[FloatArray] = []
    for level in range(order + 1):
        combined = np.zeros(dimension**level, dtype=np.float64)
        for split in range(level + 1):
            combined += np.kron(left[split], right[level - split])
        result.append(combined)
    return tuple(result)


def reference_path_signature(path: ArrayLike, order: int) -> FloatArray:
    """Compute a piecewise-linear signature straight from the paper's formulas.

    Slow. The experiments use :func:`sequential_sliding_signatures` instead.
    """
    points = _as_path(path)
    identity: list[FloatArray] = [np.ones(1, dtype=np.float64)]
    identity.extend(
        np.zeros(points.shape[1] ** level, dtype=np.float64) for level in range(1, order + 1)
    )
    signature = tuple(identity)
    for increment in np.diff(points, axis=0):
        signature = chen_product(signature, linear_segment_signature(increment, order))
    return np.concatenate(signature[1:])


def sequential_sliding_signatures(
    covariates: ArrayLike,
    window: int,
    order: int = 4,
    *,
    drop_time_only: bool = True,
    rescale_time: bool = True,
) -> SignatureFeatures:
    """Compute signatures with the paper's two-step sliding algorithm.

    After the first window, each row is built from the previous one: prepend
    the reverse of the oldest segment to cancel it, then append the newest
    segment. That is two signature combinations per row, instead of one per
    segment if you recompute the whole window.

    Args:
        covariates: One- or two-dimensional covariate series. The first axis is
            time.
        window: Number of observation intervals in each window. A window uses
            ``window + 1`` observations.
        order: Signature truncation order ``N``.
        drop_time_only: Remove one redundant pure-time coefficient per level.
        rescale_time: Use the experiments' path ``(t / window, X_t)``. Set to
            false for the plain path ``(t, X_t)``.

    Returns:
        Signatures ordered by their window end index.
    """
    matrix = _as_covariate_matrix(covariates)
    _validate_window(matrix.shape[0], window)
    path = _time_augmented_path(matrix, window, rescale_time=rescale_time)
    path_dimension = path.shape[1]
    _validate_dimension_and_order(path_dimension, order)

    signature_width = signature_feature_count(path_dimension, order)
    n_windows = matrix.shape[0] - window
    values = np.empty((n_windows, signature_width), dtype=np.float64)

    current = np.asarray(iisignature.sig(path[: window + 1], order), dtype=np.float64)
    values[0] = current

    for row, end_index in enumerate(range(window + 1, matrix.shape[0]), start=1):
        oldest_reversed = path[end_index - window - 1 : end_index - window + 1][::-1]
        oldest_inverse = np.asarray(
            iisignature.sig(oldest_reversed, order),
            dtype=np.float64,
        )
        current = np.asarray(
            iisignature.sigcombine(oldest_inverse, current, path_dimension, order),
            dtype=np.float64,
        )

        newest_segment = path[end_index - 1 : end_index + 1]
        newest = np.asarray(iisignature.sig(newest_segment, order), dtype=np.float64)
        current = np.asarray(
            iisignature.sigcombine(current, newest, path_dimension, order),
            dtype=np.float64,
        )
        values[row] = current

    if drop_time_only:
        values = np.delete(values, time_only_feature_indices(path_dimension, order), axis=1)

    return SignatureFeatures(
        values=values,
        end_indices=np.arange(window, matrix.shape[0], dtype=np.int64),
        window=window,
        order=order,
        path_dimension=path_dimension,
        time_only_dropped=drop_time_only,
    )


def direct_sliding_signatures(
    covariates: ArrayLike,
    window: int,
    order: int = 4,
    *,
    drop_time_only: bool = True,
    rescale_time: bool = True,
) -> SignatureFeatures:
    """Recompute every window from scratch, as a check on the sliding update."""
    matrix = _as_covariate_matrix(covariates)
    _validate_window(matrix.shape[0], window)
    path = _time_augmented_path(matrix, window, rescale_time=rescale_time)
    path_dimension = path.shape[1]
    _validate_dimension_and_order(path_dimension, order)

    rows = [
        np.asarray(iisignature.sig(path[start : start + window + 1], order), dtype=np.float64)
        for start in range(matrix.shape[0] - window)
    ]
    values = np.vstack(rows)
    if drop_time_only:
        values = np.delete(values, time_only_feature_indices(path_dimension, order), axis=1)

    return SignatureFeatures(
        values=values,
        end_indices=np.arange(window, matrix.shape[0], dtype=np.int64),
        window=window,
        order=order,
        path_dimension=path_dimension,
        time_only_dropped=drop_time_only,
    )


def _time_augmented_path(
    covariates: FloatArray,
    window: int,
    *,
    rescale_time: bool,
) -> FloatArray:
    divisor = float(window) if rescale_time else 1.0
    time = np.arange(covariates.shape[0], dtype=np.float64) / divisor
    return np.column_stack((time, covariates))


def _as_covariate_matrix(covariates: ArrayLike) -> FloatArray:
    matrix = np.asarray(covariates, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("covariates must have shape (n_samples,) or (n_samples, n_covariates)")
    if not np.isfinite(matrix).all():
        raise ValueError("covariates must contain only finite values")
    return matrix


def _as_path(path: ArrayLike) -> FloatArray:
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or min(points.shape) < 2:
        raise ValueError("path must contain at least two points in at least two dimensions")
    if not np.isfinite(points).all():
        raise ValueError("path must contain only finite values")
    return points


def _validate_window(n_samples: int, window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer")
    if not 1 <= window < n_samples:
        raise ValueError(f"window must satisfy 1 <= window < {n_samples}; got {window}")


def _validate_dimension_and_order(path_dimension: int, order: int) -> None:
    if path_dimension < 1:
        raise ValueError("path_dimension must be positive")
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 1:
        raise ValueError("order must be at least 1")
