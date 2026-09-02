"""Matplotlib versions of Figures 2-5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sliding_window_signatures.baselines import LinearForecastResult
from sliding_window_signatures.data import SAMPLES_PER_DAY, PaperDataset
from sliding_window_signatures.model import RidgeForecastResult
from sliding_window_signatures.simulation import SyntheticDemand

PAPER_BLUE = "#1f77b4"
SIGNATURE_GREEN = "#35b779"
OBSERVED_BLACK = "#262626"


def save_window_sweep_figure(results: pd.DataFrame, destination: Path) -> Path:
    """Save Figure 2: RMSE by window size for two memory strengths."""
    required = {"alpha", "window_days", "validation_rmse_mw", "test_rmse_mw", "mode"}
    _require_columns(results, required)
    destination = _prepare_destination(destination)
    metric_column, metric_label = _sweep_metric(results)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
        for axis, alpha in zip(axes, (0.005, 0.05), strict=True):
            subset = results[np.isclose(results["alpha"], alpha)].sort_values("window_days")
            axis.plot(
                subset["window_days"],
                subset[metric_column],
                color=PAPER_BLUE,
                marker="o",
                markersize=3,
                linewidth=1.5,
            )
            best = subset.loc[subset[metric_column].idxmin()]
            axis.scatter(
                [best["window_days"]],
                [best[metric_column]],
                color=SIGNATURE_GREEN,
                zorder=3,
                label=f"minimum: {int(best['window_days'])} days",
            )
            axis.set_title(f"Smoothing parameter alpha = {alpha}")
            axis.set_ylabel(f"{metric_label} RMSE (MW)")
            axis.legend(frameon=True)
        axes[-1].set_xlabel("Window size (days)")
        axes[-1].set_xticks(sorted(results["window_days"].unique())[::2])
        figure.suptitle("Memory strength and the optimal sliding window", fontsize=13)
        figure.tight_layout()
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_order_sweep_figure(results: pd.DataFrame, destination: Path) -> Path:
    """Save Figure 3: RMSE by window and signature truncation order."""
    required = {
        "window_days",
        "order",
        "fitted_feature_count",
        "validation_rmse_mw",
        "test_rmse_mw",
        "mode",
    }
    _require_columns(results, required)
    destination = _prepare_destination(destination)
    metric_column, metric_label = _sweep_metric(results)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=(8.3, 5.1))
        colors = plt.colormaps["viridis"](np.linspace(0.08, 0.92, len(results["order"].unique())))
        for color, order in zip(colors, sorted(results["order"].unique()), strict=True):
            subset = results[results["order"] == order].sort_values("window_days")
            fitted_count = int(subset["fitted_feature_count"].iloc[0])
            raw_count = int(sum(2**level for level in range(1, int(order) + 1)))
            axis.plot(
                subset["window_days"],
                subset[metric_column],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=f"N={order} ({fitted_count} fitted; {raw_count} raw)",
            )
        axis.set_xlabel("Window size (days)")
        axis.set_ylabel(f"{metric_label} RMSE (MW)")
        axis.set_xticks(sorted(results["window_days"].unique())[::2])
        axis.set_title("Window size and signature truncation order")
        axis.legend(ncol=2, fontsize=8.5, frameon=True)
        figure.tight_layout()
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_real_forecast_figure(
    dataset: PaperDataset,
    ridge: RidgeForecastResult,
    baseline: LinearForecastResult,
    destination: Path,
) -> Path:
    """Save Figure 4 with timestamp-aligned winter and summer forecasts."""
    if not np.array_equal(ridge.end_indices, baseline.end_indices):
        raise ValueError("Figure 4 forecasts must share identical timestamps")
    destination = _prepare_destination(destination)
    dates = dataset.dates[ridge.end_indices]
    periods = (
        (np.datetime64("2015-02-02T00:00"), np.datetime64("2015-02-08T23:30"), "Winter"),
        (np.datetime64("2015-06-15T00:00"), np.datetime64("2015-06-21T23:30"), "Summer"),
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
        for axis, (start, end, season) in zip(axes, periods, strict=True):
            mask = (dates >= start) & (dates <= end)
            if np.count_nonzero(mask) != 7 * SAMPLES_PER_DAY:
                raise ValueError(f"missing forecast rows for the {season.lower()} week")
            x = np.arange(np.count_nonzero(mask)) / SAMPLES_PER_DAY
            axis.plot(x, ridge.y_true[mask], color=OBSERVED_BLACK, label="Observed", linewidth=1.4)
            axis.plot(
                x,
                ridge.y_predicted[mask],
                color=SIGNATURE_GREEN,
                label="RidgeSig (w=9 days, N=6)",
                linewidth=1.4,
            )
            axis.plot(
                x,
                baseline.y_predicted[mask],
                color=PAPER_BLUE,
                label="Temperature features + weekly lag",
                linewidth=1.1,
            )
            axis.set_title(
                f"{season}: {np.datetime_as_string(start, unit='D')} to "
                f"{np.datetime_as_string(end, unit='D')}"
            )
            axis.set_ylabel("Demand (MW)")
            axis.legend(fontsize=8.5, frameon=True)
        axes[-1].set_xlabel("Days")
        axes[-1].set_xticks(np.arange(0, 8))
        figure.tight_layout()
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_synthetic_fidelity_figure(
    dataset: PaperDataset,
    synthetic: SyntheticDemand,
    destination: Path,
) -> Path:
    """Save Figure 5 comparing observed and synthetic demand."""
    if synthetic.values.shape != dataset.demand.shape:
        raise ValueError("synthetic and observed series must have equal lengths")
    destination = _prepare_destination(destination)
    detail_start = np.datetime64("2013-02-11T00:00")
    detail_end = np.datetime64("2013-02-17T23:30")
    detail_mask = (dataset.dates >= detail_start) & (dataset.dates <= detail_end)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        axes[0].plot(
            dataset.dates, dataset.demand, color=PAPER_BLUE, label="Observed", linewidth=0.5
        )
        axes[0].plot(
            dataset.dates,
            synthetic.values,
            color="#00a6a6",
            label="Simulated",
            linewidth=0.5,
        )
        axes[0].set_title("Full four-year dataset")
        axes[0].set_ylabel("Demand (MW)")
        axes[0].legend(frameon=True)

        x = np.arange(np.count_nonzero(detail_mask)) / SAMPLES_PER_DAY
        axes[1].plot(x, dataset.demand[detail_mask], color=PAPER_BLUE, label="Observed")
        axes[1].plot(x, synthetic.values[detail_mask], color="#00a6a6", label="Simulated")
        axes[1].set_title("Week of 11-17 February 2013")
        axes[1].set_xlabel("Days")
        axes[1].legend(frameon=True)
        figure.tight_layout()
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def _prepare_destination(destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"results are missing required columns: {sorted(missing)}")


def _sweep_metric(results: pd.DataFrame) -> tuple[str, str]:
    modes = set(results["mode"].unique())
    if modes == {"corrected"}:
        return "validation_rmse_mw", "Validation"
    if modes == {"faithful"}:
        return "test_rmse_mw", "Test"
    raise ValueError(f"sweep results must contain one mode, got {sorted(modes)}")
