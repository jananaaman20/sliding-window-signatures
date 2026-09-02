"""Figures that explain how the method works.

:mod:`sliding_window_signatures.plotting` redraws the paper's results. These
figures are for explaining the method instead. The signature values in them
are computed, not hard-coded.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LinearRegression, Ridge

from sliding_window_signatures.data import SAMPLES_PER_DAY, PaperDataset
from sliding_window_signatures.signatures import reference_path_signature

BLUE = "#2878B5"
GREEN = "#2A9D6F"
ORANGE = "#E1812C"
PURPLE = "#6F4E9C"
RED = "#C44E52"
CHARCOAL = "#303030"
LIGHT_GRAY = "#D9D9D9"


def save_beginner_explainers(dataset: PaperDataset, destination: Path) -> tuple[Path, ...]:
    """Create the whole set of explanation figures.

    Args:
        dataset: Validated half-hourly demand and temperature observations.
        destination: Directory to write the PNG files into.

    Returns:
        Absolute paths, in reading order.
    """
    directory = destination.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return (
        save_pipeline_explainer(dataset, directory / "beginner_01_pipeline.png"),
        save_signature_levels_explainer(directory / "beginner_02_signature_levels.png"),
        save_sliding_update_explainer(directory / "beginner_03_sliding_update.png"),
        save_parameter_explainer(directory / "beginner_04_parameters.png"),
        save_ridge_explainer(directory / "beginner_05_squared_loss_and_ridge.png"),
    )


def save_pipeline_explainer(dataset: PaperDataset, destination: Path) -> Path:
    """Show how one recent temperature window becomes one demand forecast."""
    window = 9 * SAMPLES_PER_DAY
    delay = 7 * SAMPLES_PER_DAY
    if len(dataset) <= window:
        raise ValueError(f"the pipeline explainer needs more than {window} observations")

    end = _preferred_end_index(dataset)
    if end < max(window, delay):
        end = len(dataset) - 1
    temperature = dataset.temperature[end - window : end + 1]
    demand = dataset.demand[end - delay : end + 1]
    window_days = np.linspace(-9.0, 0.0, temperature.size)
    delay_days = np.linspace(-7.0, 0.0, demand.size)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 5, figsize=(15.5, 4.6))
        figure.suptitle("From recent temperature to one electricity-demand forecast", fontsize=16)

        axes[0].plot(window_days, temperature, color=BLUE, linewidth=1.8)
        axes[0].fill_between(window_days, temperature, temperature.min(), color=BLUE, alpha=0.12)
        axes[0].set_title("1. Keep the recent path")
        axes[0].set_xlabel("Days before $t$")
        axes[0].set_ylabel("Temperature")
        axes[0].text(
            0.03,
            0.96,
            "$w=9$ days of remembered history",
            transform=axes[0].transAxes,
            va="top",
            fontsize=9,
        )

        normalized_time = np.linspace(0.0, 1.0, temperature.size)
        axes[1].plot(normalized_time, temperature, color=ORANGE, linewidth=1.8)
        axes[1].scatter(normalized_time[::48], temperature[::48], color=ORANGE, s=12, zorder=3)
        axes[1].set_title("2. Add normalized time")
        axes[1].set_xlabel("$t/w$ inside the window")
        axes[1].set_ylabel("Temperature")
        axes[1].text(
            0.03,
            0.96,
            "Linearly interpolated path $(t/w, T_t)$",
            transform=axes[1].transAxes,
            va="top",
            fontsize=9,
        )

        level_counts = np.array([2**level - 1 for level in range(1, 7)])
        axes[2].barh(np.arange(1, 7), level_counts, color=PURPLE, alpha=0.82)
        axes[2].set_title("3. Calculate the signature")
        axes[2].set_xlabel("Fitted coefficients at each level")
        axes[2].set_ylabel("Signature level")
        axes[2].set_yticks(np.arange(1, 7))
        axes[2].text(
            0.03,
            0.96,
            "$N=6$: 120 features in total",
            transform=axes[2].transAxes,
            va="top",
            fontsize=9,
        )

        axes[3].axis("off")
        box = FancyBboxPatch(
            (0.08, 0.21),
            0.84,
            0.58,
            boxstyle="round,pad=0.04",
            transform=axes[3].transAxes,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.14,
        )
        axes[3].add_patch(box)
        axes[3].text(0.5, 0.84, "4. Ridge predicts a change", ha="center", fontsize=11)
        axes[3].text(
            0.5,
            0.61,
            r"$\widehat{\Delta_DY_t}=b+\hat\theta^\top z_t$",
            ha="center",
            fontsize=13,
        )
        axes[3].text(
            0.5,
            0.42,
            "Squared errors measure fit.\n"
            "The ridge penalty keeps correlated\n"
            "signature weights stable.",
            ha="center",
            va="center",
            fontsize=9.5,
        )

        axes[4].plot(delay_days, demand, color=CHARCOAL, linewidth=1.6)
        axes[4].scatter([-7.0, 0.0], [demand[0], demand[-1]], color=[BLUE, GREEN], s=45, zorder=4)
        axes[4].plot([-7.0, 0.0], [demand[0], demand[-1]], color=GREEN, linestyle="--")
        axes[4].annotate(
            r"add predicted $\Delta_DY_t$",
            xy=(0.0, demand[-1]),
            xytext=(-6.7, demand.max()),
            arrowprops={"arrowstyle": "->", "color": GREEN},
            fontsize=9,
        )
        axes[4].set_title("5. Add the observed anchor")
        axes[4].set_xlabel("Days before $t$")
        axes[4].set_ylabel("Demand (MW)")
        axes[4].text(-7.0, demand[0], "$Y_{t-D}$", ha="right", va="bottom", color=BLUE)
        axes[4].text(0.0, demand[-1], r"$\widehat Y_t$", ha="right", va="bottom", color=GREEN)

        figure.text(
            0.5,
            0.015,
            "$w$ controls remembered history, $N$ controls signature detail, and "
            "$D$ chooses the already observed demand anchor.",
            ha="center",
            fontsize=10,
        )
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.92), w_pad=2.0)
        _add_flow_arrows(figure, axes)
        return _save_and_close(figure, destination)


def save_signature_levels_explainer(destination: Path) -> Path:
    """Show how mixed level-two terms tell early warming from late warming."""
    early = np.array([[0.0, 0.0], [0.25, 0.80], [0.60, 0.95], [1.0, 1.0]])
    late = np.array([[0.0, 0.0], [0.40, 0.05], [0.75, 0.20], [1.0, 1.0]])
    early_signature = reference_path_signature(early, order=2)
    late_signature = reference_path_signature(late, order=2)

    labels = [r"$S^t$", r"$S^T$", r"$S^{t,T}$", r"$S^{T,t}$"]
    indices = [0, 1, 3, 4]
    early_values = early_signature[indices]
    late_values = late_signature[indices]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure = plt.figure(figsize=(12.5, 4.8))
        grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.55))
        axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
        figure.suptitle("What signature coefficients remember", fontsize=16)

        for axis, path, title, color in (
            (axes[0], early, "Temperature rises early", BLUE),
            (axes[1], late, "Temperature rises late", ORANGE),
        ):
            axis.plot(path[:, 0], path[:, 1], color=color, marker="o", linewidth=2.2)
            for start, stop in pairwise(path):
                midpoint = 0.5 * (start + stop)
                axis.annotate(
                    "",
                    xy=stop,
                    xytext=midpoint,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
                )
            axis.set_xlim(-0.05, 1.05)
            axis.set_ylim(-0.05, 1.08)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Normalized time $t/w$")
            axis.set_ylabel("Scaled temperature")
            axis.set_title(title)

        positions = np.arange(len(labels))
        width = 0.36
        axes[2].bar(positions - width / 2, early_values, width, color=BLUE, label="rises early")
        axes[2].bar(positions + width / 2, late_values, width, color=ORANGE, label="rises late")
        axes[2].set_xticks(positions, labels)
        axes[2].set_ylabel("Signature coefficient")
        axes[2].set_title("Same totals; different order-sensitive terms")
        axes[2].legend(frameon=True)
        axes[2].axvline(1.5, color=LIGHT_GRAY, linewidth=1.2)

        figure.text(
            0.5,
            0.02,
            "Level 1 records total change. Mixed level-2 coefficients change when the "
            "same movement happens in a different order.",
            ha="center",
            fontsize=10,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.92), w_pad=2.0)
        return _save_and_close(figure, destination)


def save_sliding_update_explainer(destination: Path) -> Path:
    """Show how two neighboring windows share almost all of their work."""
    x = np.arange(7, dtype=float)
    y = np.array([0.25, 0.78, 0.47, 0.92, 0.58, 0.72, 0.38])

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharex=True, sharey=True)
        figure.suptitle("Move the window without recomputing the whole signature", fontsize=16)

        axes[0].plot(x[:6], y[:6], color=BLUE, marker="o", linewidth=2)
        axes[0].plot(x[:2], y[:2], color=RED, marker="o", linewidth=4)
        axes[0].set_title("1. Current window")
        axes[0].annotate(
            "oldest segment",
            xy=(0.5, 0.5 * (y[0] + y[1])),
            xytext=(1.7, 0.18),
            arrowprops={"arrowstyle": "->", "color": RED},
            color=RED,
        )

        axes[1].plot(x[:6], y[:6], color=LIGHT_GRAY, marker="o", linewidth=2)
        axes[1].annotate(
            "",
            xy=(x[0], y[0]),
            xytext=(x[1], y[1]),
            arrowprops={"arrowstyle": "->", "color": RED, "lw": 4},
        )
        axes[1].set_title("2. Traverse it backward")
        axes[1].text(0.5, 0.92, "its signature cancels", color=RED, ha="center")
        axes[1].text(3.2, 0.27, "overlap is reused", color=CHARCOAL, ha="center")

        axes[2].plot(x[1:6], y[1:6], color=BLUE, marker="o", linewidth=2)
        axes[2].plot(x[5:], y[5:], color=GREEN, marker="o", linewidth=4)
        axes[2].set_title("3. Append the new segment")
        axes[2].annotate(
            "new segment",
            xy=(5.5, 0.5 * (y[5] + y[6])),
            xytext=(4.0, 0.18),
            arrowprops={"arrowstyle": "->", "color": GREEN},
            color=GREEN,
        )

        for axis in axes:
            axis.set_xlim(-0.25, 6.25)
            axis.set_ylim(0.05, 1.05)
            axis.set_xlabel("Observation index")
            axis.set_yticks([])
        axes[0].set_ylabel("Temperature path")

        figure.text(
            0.5,
            0.075,
            r"$S_{new}=S(\overleftarrow{x}_{old})\otimes S_{current}\otimes S(x_{new})$",
            ha="center",
            fontsize=13,
        )
        figure.text(
            0.5,
            0.02,
            "After the first window, each step needs two signature combinations instead of "
            "recombining all $w$ segments.",
            ha="center",
            fontsize=10,
        )
        figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.91), w_pad=2.0)
        return _save_and_close(figure, destination)


def save_parameter_explainer(destination: Path) -> Path:
    """Explain how alpha, window, order, delay, and ridge penalty interact."""
    days = np.linspace(0.0, 14.0, 14 * SAMPLES_PER_DAY + 1)
    orders = np.arange(2, 8)
    fitted_counts = np.array([4, 11, 26, 57, 120, 247])

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
        figure.suptitle("What each paper parameter controls", fontsize=16)

        for alpha, color, label in (
            (0.005, BLUE, r"$\alpha=0.005$: slow forgetting"),
            (0.05, ORANGE, r"$\alpha=0.05$: fast forgetting"),
        ):
            weights = (1.0 - alpha) ** (days * SAMPLES_PER_DAY)
            axes[0, 0].plot(days, weights, color=color, linewidth=2, label=label)
        axes[0, 0].axvline(9, color=BLUE, linestyle="--", alpha=0.8)
        axes[0, 0].axvline(3, color=ORANGE, linestyle="--", alpha=0.8)
        axes[0, 0].annotate("best $w=9$", (9, 0.12), xytext=(9.4, 0.36), color=BLUE)
        axes[0, 0].annotate("best $w=3$", (3, 0.02), xytext=(4.0, 0.18), color=ORANGE)
        axes[0, 0].set_xlabel("Age of a temperature observation (days)")
        axes[0, 0].set_ylabel("Relative memory weight")
        axes[0, 0].set_title(r"$\alpha$ controls forgetting; $w$ keeps recent history")
        axes[0, 0].legend(frameon=True, fontsize=9)

        axes[0, 1].bar(orders, fitted_counts, color=PURPLE, alpha=0.85)
        for order, count in zip(orders, fitted_counts, strict=True):
            axes[0, 1].text(order, count + 5, str(count), ha="center", fontsize=9)
        axes[0, 1].set_xticks(orders)
        axes[0, 1].set_xlabel("Truncation order $N$")
        axes[0, 1].set_ylabel("Fitted signature features")
        axes[0, 1].set_title("$N$ adds nonlinear, ordered detail—and cost")

        axes[1, 0].hlines([1, 0], xmin=[-2, -7], xmax=[0, 0], color=[ORANGE, BLUE], linewidth=4)
        axes[1, 0].scatter([-2, 0, -7, 0], [1, 1, 0, 0], color=[ORANGE, ORANGE, BLUE, BLUE], s=50)
        axes[1, 0].text(-2, 1.12, "$Y_{t-D}$", ha="center", color=ORANGE)
        axes[1, 0].text(-7, 0.12, "$Y_{t-D}$", ha="center", color=BLUE)
        axes[1, 0].text(0, 1.12, "$Y_t$", ha="center")
        axes[1, 0].text(0, 0.12, "$Y_t$", ha="center")
        axes[1, 0].set_yticks([0, 1], ["Real demand: 7 days", "Synthetic: 2 days"])
        axes[1, 0].set_xticks(
            np.arange(-7, 1), [f"{value}d" if value else "$t$" for value in range(-7, 1)]
        )
        axes[1, 0].set_xlim(-7.7, 0.7)
        axes[1, 0].set_ylim(-0.35, 1.35)
        axes[1, 0].set_title("$D$ chooses an available seasonal anchor")
        axes[1, 0].grid(axis="x")

        synthetic_grid = np.logspace(-3, 3, 50)
        real_grid = np.logspace(-1, 2, 20)
        axes[1, 1].scatter(synthetic_grid, np.ones_like(synthetic_grid), color=PURPLE, s=14)
        axes[1, 1].scatter(real_grid, np.zeros_like(real_grid), color=GREEN, s=18)
        axes[1, 1].scatter([79.0604], [1], color=ORANGE, marker="*", s=180, zorder=4)
        axes[1, 1].scatter([0.1], [0], color=ORANGE, marker="*", s=180, zorder=4)
        axes[1, 1].annotate("Table 1 RMSE choice: 79.1", (79.0604, 1), xytext=(2.0, 1.23))
        axes[1, 1].annotate("Real-data choice: 0.1", (0.1, 0), xytext=(0.25, 0.25))
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yticks([0, 1], ["Real grid", "Synthetic grid"])
        axes[1, 1].set_ylim(-0.45, 1.45)
        axes[1, 1].set_xlabel(r"Candidate ridge penalty $\lambda$ (log scale)")
        axes[1, 1].set_title(r"Validation selects $\lambda$ from a logarithmic grid")

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=2.3, w_pad=2.0)
        return _save_and_close(figure, destination)


def save_ridge_explainer(destination: Path) -> Path:
    """Explain squared error and ridge stability on a small fixed example."""
    residuals = np.linspace(-4.0, 4.0, 300)
    rng = np.random.default_rng(2025)
    latent = rng.normal(size=90)
    feature_one = latent + rng.normal(scale=0.025, size=latent.size)
    feature_two = latent + rng.normal(scale=0.025, size=latent.size)

    ordinary_coefficients: list[np.ndarray] = []
    ridge_coefficients: list[np.ndarray] = []
    for _ in range(90):
        design = np.column_stack(
            (
                latent + rng.normal(scale=0.005, size=latent.size),
                latent + rng.normal(scale=0.005, size=latent.size),
            )
        )
        target = 4.0 * latent + rng.normal(scale=0.40, size=latent.size)
        ordinary_coefficients.append(LinearRegression().fit(design, target).coef_)
        ridge_coefficients.append(Ridge(alpha=1.0).fit(design, target).coef_)
    ordinary = np.vstack(ordinary_coefficients)
    regularized = np.vstack(ridge_coefficients)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        figure.suptitle(
            "Squared loss measures fit; ridge stabilizes correlated features", fontsize=16
        )

        axes[0].plot(residuals, residuals**2, color=RED, linewidth=2.2)
        axes[0].scatter([1, 3], [1, 9], color=[BLUE, RED], s=45, zorder=3)
        axes[0].annotate(
            "error 1 → loss 1", (1, 1), xytext=(-3.5, 4.0), arrowprops={"arrowstyle": "->"}
        )
        axes[0].annotate(
            "error 3 → loss 9", (3, 9), xytext=(-0.5, 13.0), arrowprops={"arrowstyle": "->"}
        )
        axes[0].set_xlabel("Prediction error")
        axes[0].set_ylabel("Squared error")
        axes[0].set_title("1. Large misses matter more")

        axes[1].scatter(feature_one, feature_two, color=PURPLE, alpha=0.65, s=22)
        correlation = np.corrcoef(feature_one, feature_two)[0, 1]
        axes[1].text(
            0.04,
            0.95,
            f"correlation = {correlation:.3f}",
            transform=axes[1].transAxes,
            va="top",
        )
        axes[1].set_xlabel("Signature feature 1")
        axes[1].set_ylabel("Signature feature 2")
        axes[1].set_title("2. Overlapping windows create similar columns")

        axes[2].scatter(
            ordinary[:, 0],
            ordinary[:, 1],
            color=RED,
            alpha=0.45,
            label="ordinary least squares",
            marker="x",
        )
        axes[2].scatter(
            regularized[:, 0],
            regularized[:, 1],
            color=GREEN,
            alpha=0.75,
            label=r"ridge ($\lambda=1$)",
            s=22,
        )
        axes[2].axline((0, 4), (4, 0), color=LIGHT_GRAY, linewidth=1.2)
        axes[2].set_xlabel("Weight on feature 1")
        axes[2].set_ylabel("Weight on feature 2")
        axes[2].set_title("3. Ridge shares and shrinks the weights")
        axes[2].legend(frameon=True, fontsize=8.5)

        figure.text(
            0.5,
            0.02,
            "The red fits trade large positive and negative weights between nearly duplicate "
            "features. Ridge keeps both weights small and stable.",
            ha="center",
            fontsize=10,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.92), w_pad=2.2)
        return _save_and_close(figure, destination)


def _preferred_end_index(dataset: PaperDataset) -> int:
    preferred = np.datetime64("2015-02-08T23:30")
    matches = np.flatnonzero(dataset.dates == preferred)
    return int(matches[0]) if matches.size else len(dataset) - 1


def _add_flow_arrows(figure: Figure, axes: np.ndarray) -> None:
    figure.canvas.draw()
    for left, right in pairwise(axes):
        left_box = left.get_position()
        right_box = right.get_position()
        arrow = FancyArrowPatch(
            (left_box.x1 + 0.004, 0.5 * (left_box.y0 + left_box.y1)),
            (right_box.x0 - 0.004, 0.5 * (right_box.y0 + right_box.y1)),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            color=CHARCOAL,
            linewidth=1.1,
        )
        figure.add_artist(arrow)


def _save_and_close(figure: Figure, destination: Path) -> Path:
    path = destination.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path
