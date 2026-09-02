"""Figures that explain the rolling-versus-fading comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sliding_window_signatures.comparison import SignatureComparison

ROLLING_BLUE = "#2864A8"
FADING_ORANGE = "#D97706"
OBSERVED_BLACK = "#222222"
GRID_GREY = "#D7DCE2"


def save_comparison_figures(
    comparison: SignatureComparison,
    output_directory: Path,
) -> list[Path]:
    """Save the six figures that go with the comparison write-up."""
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        save_memory_kernel_figure(
            comparison.metrics,
            output / "comparison_01_memory_kernels.png",
        ),
        save_validation_sweep_figure(
            comparison.sweep,
            output / "comparison_02_validation_sweep.png",
        ),
        save_test_metrics_figure(
            comparison.metrics,
            output / "comparison_03_test_metrics.png",
        ),
        save_seasonal_rmse_figure(
            comparison.seasonal_metrics,
            output / "comparison_04_seasonal_rmse.png",
        ),
        save_forecast_examples_figure(
            comparison.forecasts,
            output / "comparison_05_forecast_examples.png",
        ),
        save_computation_figure(
            comparison.runtime,
            output / "comparison_06_computation.png",
        ),
    ]
    return paths


def save_memory_kernel_figure(metrics: pd.DataFrame, destination: Path) -> Path:
    """Put the hard rolling cutoff next to EFM's smooth decay."""
    _require_columns(
        metrics,
        {"method", "selected_parameter_days", "level_one_half_life_days"},
    )
    window_days = _selected_parameter(metrics, "Rolling Window Signature")
    fading_horizon_days = _selected_parameter(metrics, "Fading Memory Signature")
    half_life_days = _selected_value(
        metrics,
        "Fading Memory Signature",
        "level_one_half_life_days",
    )
    maximum_age = max(window_days * 1.45, fading_horizon_days * 1.25)
    age = np.linspace(0.0, maximum_age, 600)
    rolling_weight = (age <= window_days).astype(float)
    fading_level_one = np.power(2.0, -age / half_life_days)
    fading_level_three = np.power(2.0, -3.0 * age / half_life_days)

    destination = _prepare_destination(destination)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        axis.step(
            age,
            rolling_weight,
            where="post",
            color=ROLLING_BLUE,
            linewidth=2.5,
            label=f"Rolling level 1: keep {window_days:g} days, then drop",
        )
        axis.plot(
            age,
            fading_level_one,
            color=FADING_ORANGE,
            linewidth=2.5,
            label=f"Fading level 1: 1% remains after {fading_horizon_days:g} days",
        )
        axis.plot(
            age,
            fading_level_three,
            color=FADING_ORANGE,
            linestyle="--",
            linewidth=1.8,
            label="Fading level 3: three times the decay rate",
        )
        axis.axvline(window_days, color=ROLLING_BLUE, linestyle=":", alpha=0.8)
        axis.axvline(fading_horizon_days, color=FADING_ORANGE, linestyle=":", alpha=0.8)
        axis.scatter([half_life_days], [0.5], color=FADING_ORANGE, zorder=4)
        axis.annotate(
            f"half remains after {half_life_days:.2f} days",
            xy=(half_life_days, 0.5),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=9,
        )
        axis.set(
            xlabel="Age of a past path increment (days)",
            ylabel="Relative contribution",
            ylim=(-0.04, 1.08),
        )
        axis.set_title("Two ways to remember the past", pad=34)
        axis.text(
            0.0,
            1.015,
            "Rolling memory has a hard boundary; EFM forgets smoothly and faster at higher levels.",
            transform=axis.transAxes,
            fontsize=10,
            color="#4B5563",
        )
        axis.legend(loc="upper right", frameon=True, fontsize=9)
        axis.grid(color=GRID_GREY, linewidth=0.7)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_validation_sweep_figure(results: pd.DataFrame, destination: Path) -> Path:
    """Plot the validation-only selection curve for each memory parameter."""
    _require_columns(results, {"method", "parameter_days", "validation_rmse_mw"})
    destination = _prepare_destination(destination)
    specifications = (
        ("Rolling Window Signature", ROLLING_BLUE, "o", "Window length"),
        ("Fading Memory Signature", FADING_ORANGE, "s", "Level-1 1% horizon"),
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
        for method, color, marker, label in specifications:
            subset = results[results["method"] == method].sort_values("parameter_days")
            axis.plot(
                subset["parameter_days"],
                subset["validation_rmse_mw"],
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=5,
                label=f"{method} ({label})",
            )
            best = subset.loc[subset["validation_rmse_mw"].idxmin()]
            axis.scatter(
                [best["parameter_days"]],
                [best["validation_rmse_mw"]],
                s=85,
                facecolors="white",
                edgecolors=color,
                linewidths=2.2,
                zorder=4,
            )
            axis.annotate(
                f"selected: {best['parameter_days']:g} days",
                xy=(best["parameter_days"], best["validation_rmse_mw"]),
                xytext=(7, -18 if method.startswith("Rolling") else 12),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        axis.set(
            xlabel="Memory parameter (days)",
            ylabel="Validation RMSE (MW)",
        )
        axis.set_title("Both memory parameters are selected on 2014 only", pad=34)
        axis.text(
            0.0,
            1.015,
            "2013 trains each candidate; the shared 2015 evaluation year "
            "is not used in this sweep.",
            transform=axis.transAxes,
            fontsize=10,
            color="#4B5563",
        )
        axis.legend(frameon=True, fontsize=9)
        axis.grid(color=GRID_GREY, linewidth=0.7)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_test_metrics_figure(metrics: pd.DataFrame, destination: Path) -> Path:
    """Compare the shared 2015 errors, one panel per unit."""
    _require_columns(
        metrics,
        {"method", "test_rmse_mw", "test_mae_mw", "test_mape_percent"},
    )
    destination = _prepare_destination(destination)
    methods = ["Rolling Window Signature", "Fading Memory Signature"]
    colors = [ROLLING_BLUE, FADING_ORANGE]
    panels = (
        ("test_rmse_mw", "RMSE", "MW", ",.0f"),
        ("test_mae_mw", "MAE", "MW", ",.0f"),
        ("test_mape_percent", "MAPE", "%", ".2f"),
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.3), constrained_layout=True)
        for axis, (column, title, unit, number_format) in zip(axes, panels, strict=True):
            values = np.asarray(
                [metrics.loc[metrics["method"] == method, column].iloc[0] for method in methods]
            )
            bars = axis.bar([0, 1], values, color=colors, width=0.62)
            axis.set_xticks([0, 1], ["Rolling", "Fading"])
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.set_ylim(0.0, values.max() * 1.20)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:{number_format}}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            axis.grid(axis="x", visible=False)
            axis.grid(axis="y", color=GRID_GREY, linewidth=0.7)
        figure.suptitle("Shared 2015 forecast errors", fontsize=14)
        figure.supxlabel(
            "Lower is better. Every bar uses the same 17,520 half-hourly timestamps.",
            fontsize=9.5,
            color="#4B5563",
        )
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_seasonal_rmse_figure(results: pd.DataFrame, destination: Path) -> Path:
    """Check whether the overall result holds season by season."""
    _require_columns(results, {"season", "method", "rmse_mw"})
    destination = _prepare_destination(destination)
    seasons = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
    x = np.arange(len(seasons), dtype=np.float64)
    width = 0.36

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
        for offset, method, color, label in (
            (-width / 2, "Rolling Window Signature", ROLLING_BLUE, "Rolling"),
            (width / 2, "Fading Memory Signature", FADING_ORANGE, "Fading"),
        ):
            values = [
                results.loc[
                    (results["season"] == season) & (results["method"] == method),
                    "rmse_mw",
                ].iloc[0]
                for season in seasons
            ]
            axis.bar(x + offset, values, width=width, color=color, label=label)
        axis.set(
            title="Forecast accuracy by season in the shared evaluation year",
            ylabel="RMSE (MW)",
        )
        axis.set_xticks(x, [season.split()[0] for season in seasons])
        axis.legend(frameon=True)
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", color=GRID_GREY, linewidth=0.7)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_forecast_examples_figure(forecasts: pd.DataFrame, destination: Path) -> Path:
    """Plot one winter and one summer week on the common test horizon."""
    _require_columns(
        forecasts,
        {
            "datetime",
            "observed_mw",
            "rolling_window_prediction_mw",
            "fading_memory_prediction_mw",
        },
    )
    destination = _prepare_destination(destination)
    frame = forecasts.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    periods = (
        ("Winter week", pd.Timestamp("2015-02-02"), pd.Timestamp("2015-02-09")),
        ("Summer week", pd.Timestamp("2015-07-06"), pd.Timestamp("2015-07-13")),
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 1, figsize=(10.6, 6.4), sharey=True, constrained_layout=True)
        for axis, (title, start, stop) in zip(axes, periods, strict=True):
            subset = frame[(frame["datetime"] >= start) & (frame["datetime"] < stop)]
            axis.plot(
                subset["datetime"],
                subset["observed_mw"],
                color=OBSERVED_BLACK,
                linewidth=1.8,
                label="Observed",
            )
            axis.plot(
                subset["datetime"],
                subset["rolling_window_prediction_mw"],
                color=ROLLING_BLUE,
                linewidth=1.3,
                label="Rolling",
            )
            axis.plot(
                subset["datetime"],
                subset["fading_memory_prediction_mw"],
                color=FADING_ORANGE,
                linewidth=1.3,
                label="Fading",
            )
            axis.set_title(title, loc="left")
            axis.set_ylabel("Demand (MW)")
            axis.xaxis.set_major_locator(mdates.DayLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
            axis.grid(color=GRID_GREY, linewidth=0.7)
        axes[0].legend(ncol=3, frameon=True, loc="upper right")
        axes[-1].set_xlabel("2015")
        figure.suptitle("What the two forecasts look like on ordinary weeks", fontsize=14)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def save_computation_figure(runtime: pd.DataFrame, destination: Path) -> Path:
    """Compare the measured feature-building and fitting times."""
    _require_columns(
        runtime,
        {"method", "feature_construction_seconds", "ridge_tune_and_refit_seconds"},
    )
    destination = _prepare_destination(destination)
    methods = ["Rolling Window Signature", "Fading Memory Signature"]
    colors = [ROLLING_BLUE, FADING_ORANGE]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), constrained_layout=True)
        for axis, column, title in (
            (axes[0], "feature_construction_seconds", "Feature construction"),
            (axes[1], "ridge_tune_and_refit_seconds", "Ridge tuning + refit"),
        ):
            values = [
                runtime.loc[runtime["method"] == method, column].iloc[0] for method in methods
            ]
            bars = axis.bar([0, 1], values, color=colors, width=0.62)
            axis.set_xticks([0, 1], ["Rolling", "Fading"])
            axis.set_title(title)
            axis.set_ylabel("Wall-clock seconds")
            axis.set_ylim(0.0, max(values) * 1.22)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            axis.grid(axis="x", visible=False)
            axis.grid(axis="y", color=GRID_GREY, linewidth=0.7)
        figure.suptitle("Measured computation for the selected models", fontsize=14)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return destination


def _selected_parameter(metrics: pd.DataFrame, method: str) -> float:
    return _selected_value(metrics, method, "selected_parameter_days")


def _selected_value(metrics: pd.DataFrame, method: str, column: str) -> float:
    subset = metrics.loc[metrics["method"] == method, column]
    if subset.size != 1:
        raise ValueError(f"expected exactly one selected row for {method}")
    value = float(subset.iloc[0])
    if value <= 0.0:
        raise ValueError("selected memory parameters must be positive")
    return value


def _prepare_destination(destination: Path) -> Path:
    resolved = destination.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"results are missing required columns: {sorted(missing)}")
