"""Collects the comparison write-up, its numbers, and its tables into JSON."""

# ruff: noqa: E501

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sliding_window_signatures.comparison import SignatureComparison

ROLLING_METHOD = "Rolling Window Signature"
FADING_METHOD = "Fading Memory Signature"


def save_comparison_report_artifact(
    comparison: SignatureComparison,
    destination: Path,
    *,
    generated_at: str | None = None,
) -> Path:
    """Write the JSON that a report viewer turns into a readable page."""
    resolved = destination.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_comparison_report_artifact(comparison, generated_at=generated_at)
    resolved.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return resolved


def build_comparison_report_artifact(
    comparison: SignatureComparison,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the write-up: text, key numbers, tables, and the queries behind them."""
    timestamp = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    rolling = _method_row(comparison.metrics, ROLLING_METHOD)
    fading = _method_row(comparison.metrics, FADING_METHOD)
    bootstrap = comparison.bootstrap

    rolling_rmse = float(rolling["test_rmse_mw"])
    fading_rmse = float(fading["test_rmse_mw"])
    rmse_difference = fading_rmse - rolling_rmse
    relative_difference = rmse_difference / rolling_rmse
    lower = float(bootstrap["ci_95_lower_mw"])
    upper = float(bootstrap["ci_95_upper_mw"])
    resample_share = float(bootstrap["bootstrap_resample_share_fading_lower_rmse"])
    evidence_sentence = _uncertainty_sentence(rmse_difference, lower, upper)

    rolling_horizon = float(rolling["selected_parameter_days"])
    fading_horizon = float(fading["selected_parameter_days"])
    fading_half_life = float(fading["level_one_half_life_days"])
    rolling_penalty = float(rolling["selected_ridge_penalty"])
    fading_penalty = float(fading["selected_ridge_penalty"])

    headline = [
        {
            "rolling_rmse_mw": rolling_rmse,
            "fading_rmse_mw": fading_rmse,
            "rmse_difference_mw": rmse_difference,
            "relative_rmse_difference": relative_difference,
            "bootstrap_resample_share_fading_better": resample_share,
            "rolling_horizon_days": rolling_horizon,
            "fading_horizon_days": fading_horizon,
        }
    ]
    memory_kernel = _memory_kernel_rows(rolling_horizon, fading_horizon)
    sweep = comparison.sweep.copy()
    sweep["series"] = sweep["method"].map(
        {
            ROLLING_METHOD: "Rolling window",
            FADING_METHOD: "Fading 1% horizon",
        }
    )
    seasonal = comparison.seasonal_metrics.copy()
    seasonal["series"] = seasonal["method"].map(
        {ROLLING_METHOD: "Rolling window", FADING_METHOD: "Fading memory"}
    )
    forecasts = _forecast_example_rows(comparison.forecasts)

    metrics_table = comparison.metrics.copy()
    metrics_table["method_short"] = metrics_table["method"].map(
        {ROLLING_METHOD: "Rolling", FADING_METHOD: "Fading"}
    )
    runtime = comparison.runtime.copy()
    runtime["method_short"] = runtime["method"].map(
        {ROLLING_METHOD: "Rolling", FADING_METHOD: "Fading"}
    )
    tradeoffs = pd.DataFrame(
        [
            {
                "property": "Memory boundary",
                "rolling": "Hard cutoff at H",
                "fading": "Smooth exponential tail",
            },
            {
                "property": "Remote past",
                "rolling": "Exactly removed",
                "fading": "Never exactly zero",
            },
            {
                "property": "Online update",
                "rolling": "Remove oldest; append newest",
                "fading": "Discount state; append newest",
            },
            {
                "property": "Stored history",
                "rolling": "Signature state plus window buffer",
                "fading": "Signature state plus latest point",
            },
            {
                "property": "Initialization",
                "rolling": "Exact after first full window",
                "fading": "Needs a prior state or burn-in",
            },
            {
                "property": "Regime change",
                "rolling": "Old regime disappears after H",
                "fading": "Old regime decays gradually",
            },
            {
                "property": "Equal-rate state complexity",
                "rolling": "O(d^N)",
                "fading": "O(d^N)",
            },
            {
                "property": "State dynamics",
                "rolling": "Moving lower boundary",
                "fading": "Mean-reverting Markov state",
            },
        ]
    )

    seasonal["season_order"] = seasonal["season"].map(
        {
            "Winter (DJF)": 1,
            "Spring (MAM)": 2,
            "Summer (JJA)": 3,
            "Autumn (SON)": 4,
        }
    )
    dataset_frames = {
        "headline": pd.DataFrame(headline),
        "memory_kernel": pd.DataFrame(memory_kernel),
        "validation_sweep": sweep[
            ["series", "parameter_days", "validation_rmse_mw", "selected_ridge_penalty"]
        ],
        "test_metrics": metrics_table[
            [
                "method_short",
                "selected_parameter_days",
                "selected_ridge_penalty",
                "test_rmse_mw",
                "test_mae_mw",
                "test_mape_percent",
            ]
        ],
        "seasonal_metrics": seasonal[
            ["season", "season_order", "series", "rmse_mw", "mae_mw", "mape_percent"]
        ],
        "forecast_examples": pd.DataFrame(forecasts),
        "runtime": runtime[
            [
                "method_short",
                "feature_construction_seconds",
                "ridge_tune_and_refit_seconds",
                "feature_matrix_mib",
                "online_signature_state_kib",
            ]
        ],
        "tradeoffs": tradeoffs,
    }
    sql_queries = {
        "headline": "SELECT * FROM headline",
        "memory_kernel": "SELECT * FROM memory_kernel ORDER BY age_days, series",
        "validation_sweep": ("SELECT * FROM validation_sweep ORDER BY parameter_days, series"),
        "test_metrics": "SELECT * FROM test_metrics ORDER BY test_rmse_mw",
        "seasonal_metrics": ("SELECT * FROM seasonal_metrics ORDER BY season_order, series"),
        "forecast_examples": ("SELECT * FROM forecast_examples ORDER BY period, datetime, series"),
        "runtime": "SELECT * FROM runtime ORDER BY method_short",
        "tradeoffs": "SELECT * FROM tradeoffs ORDER BY property",
    }
    datasets, sql_sources = _materialize_sql_datasets(
        dataset_frames,
        sql_queries,
        generated_at=timestamp,
    )
    source_definitions = [*_source_definitions(), *sql_sources]
    manifest_sources = [
        {"id": source["id"], "label": source["label"], **_source_link(source)}
        for source in source_definitions
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Rolling Window vs Fading Memory Signatures",
        "description": "A controlled electricity-demand comparison, written out step by step.",
        "generatedAt": timestamp,
        "cards": [
            {
                "id": "rolling_rmse",
                "description": (
                    f"Shared 2015 RMSE after selecting a {rolling_horizon:g}-day window on 2014."
                ),
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Rolling test RMSE (MW)", "field": "rolling_rmse_mw"},
                    {"label": "Selected horizon (days)", "field": "rolling_horizon_days"},
                ],
            },
            {
                "id": "fading_rmse",
                "description": ("Shared 2015 RMSE for the equal-rate EFM state, selected on 2014."),
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Fading test RMSE (MW)", "field": "fading_rmse_mw"},
                    {"label": "Selected 1% horizon (days)", "field": "fading_horizon_days"},
                ],
            },
            {
                "id": "rmse_difference",
                "description": (
                    "Fading minus rolling RMSE. Negative favors fading; uncertainty uses weekly blocks."
                ),
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {
                        "label": "RMSE difference (MW)",
                        "field": "rmse_difference_mw",
                        "signed": True,
                    },
                    {
                        "label": "Relative difference",
                        "field": "relative_rmse_difference",
                        "format": "percent",
                        "signed": True,
                    },
                    {
                        "label": "Bootstrap resamples favoring fading",
                        "field": "bootstrap_resample_share_fading_better",
                        "format": "percent",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "memory_kernel",
                "title": "Memory weight by age",
                "subtitle": "Level-one comparison plus the faster level-three EFM decay.",
                "type": "line",
                "dataset": "memory_kernel",
                "sourceId": "memory_kernel_sql",
                "encodings": {
                    "x": {
                        "field": "age_days",
                        "type": "quantitative",
                        "label": "Age (days)",
                    },
                    "y": {
                        "field": "relative_weight",
                        "type": "quantitative",
                        "label": "Relative contribution",
                    },
                    "color": {"field": "series", "type": "nominal", "label": "Memory rule"},
                },
                "layout": "full",
            },
            {
                "id": "validation_sweep",
                "title": "Validation RMSE across memory horizons",
                "subtitle": "Both methods use the same candidate horizons and the 2014 validation year.",
                "type": "line",
                "dataset": "validation_sweep",
                "sourceId": "validation_sweep_sql",
                "encodings": {
                    "x": {
                        "field": "parameter_days",
                        "type": "quantitative",
                        "label": "Memory horizon (days)",
                    },
                    "y": {
                        "field": "validation_rmse_mw",
                        "type": "quantitative",
                        "label": "Validation RMSE (MW)",
                    },
                    "color": {"field": "series", "type": "nominal", "label": "Method"},
                    "tooltip": [
                        {"field": "selected_ridge_penalty", "label": "Ridge penalty"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "seasonal_rmse",
                "title": "Shared 2015 RMSE by season",
                "subtitle": "Every seasonal slice uses the same timestamps for both models.",
                "type": "bar",
                "dataset": "seasonal_metrics",
                "sourceId": "seasonal_metrics_sql",
                "encodings": {
                    "x": {"field": "season", "type": "ordinal", "label": "Season"},
                    "y": {
                        "field": "rmse_mw",
                        "type": "quantitative",
                        "label": "RMSE (MW)",
                    },
                    "color": {"field": "series", "type": "nominal", "label": "Method"},
                },
                "layout": "full",
            },
            {
                "id": "forecast_examples",
                "title": "Observed and predicted demand in two example weeks",
                "subtitle": "Two-hour sampling is used only to keep the report snapshot compact.",
                "type": "line",
                "dataset": "forecast_examples",
                "sourceId": "forecast_examples_sql",
                "encodings": {
                    "x": {"field": "datetime", "type": "temporal", "label": "Time"},
                    "y": {
                        "field": "demand_mw",
                        "type": "quantitative",
                        "label": "Demand (MW)",
                    },
                    "color": {"field": "series", "type": "nominal", "label": "Series"},
                    "facet": {"field": "period", "type": "nominal", "label": "Week"},
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "test_metrics",
                "title": "Shared 2015 accuracy",
                "subtitle": "One validation-selected model per memory representation.",
                "dataset": "test_metrics",
                "sourceId": "test_metrics_sql",
                "defaultSort": {"field": "test_rmse_mw", "direction": "asc"},
                "columns": [
                    {"field": "method_short", "label": "Method", "type": "text"},
                    {"field": "selected_parameter_days", "label": "Selected H (days)"},
                    {"field": "selected_ridge_penalty", "label": "Ridge penalty"},
                    {"field": "test_rmse_mw", "label": "RMSE (MW)"},
                    {"field": "test_mae_mw", "label": "MAE (MW)"},
                    {"field": "test_mape_percent", "label": "MAPE (%)"},
                ],
                "layout": "full",
            },
            {
                "id": "runtime",
                "title": "Measured computation for selected models",
                "subtitle": "Single-run wall-clock measurements in the pinned project environment.",
                "dataset": "runtime",
                "sourceId": "runtime_sql",
                "defaultSort": {"field": "method_short", "direction": "asc"},
                "columns": [
                    {"field": "method_short", "label": "Method", "type": "text"},
                    {"field": "feature_construction_seconds", "label": "Features (s)"},
                    {"field": "ridge_tune_and_refit_seconds", "label": "Ridge (s)"},
                    {"field": "feature_matrix_mib", "label": "Matrix (MiB)"},
                    {"field": "online_signature_state_kib", "label": "State (KiB)"},
                ],
                "layout": "full",
            },
            {
                "id": "tradeoffs",
                "title": "General method comparison",
                "subtitle": "Structural differences independent of this dataset.",
                "dataset": "tradeoffs",
                "sourceId": "tradeoffs_sql",
                "defaultSort": {"field": "property", "direction": "asc"},
                "columns": [
                    {"field": "property", "label": "Property", "type": "text"},
                    {"field": "rolling", "label": "Rolling window", "type": "text"},
                    {"field": "fading", "label": "Fading memory", "type": "text"},
                ],
                "layout": "full",
            },
        ],
        "sources": manifest_sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Rolling Window vs Fading Memory Signatures",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Technical summary\n\n"
                    f"The controlled test gives **{rolling_rmse:,.0f} MW RMSE** for rolling memory "
                    f"and **{fading_rmse:,.0f} MW** for fading memory. The fading-minus-rolling "
                    f"difference is **{rmse_difference:+,.0f} MW** ({relative_difference:+.2%}). "
                    f"{evidence_sentence}\n\n"
                    "This does not show that one signature is better in general. The two methods "
                    "differ in how old information disappears."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["rolling_rmse", "fading_rmse", "rmse_difference"],
            },
            {
                "id": "memory_concept",
                "type": "markdown",
                "body": (
                    "## How each method forgets the past\n\n"
                    "**Rolling Window Signature (RWS).** Keep the last `H` days with full weight and "
                    "remove anything older. When the window moves, reverse the oldest segment, combine "
                    "it with the current signature so it cancels, then append the newest segment.\n\n"
                    "**Fading Memory Signature (EFM).** Keep one state representing the whole observed "
                    "past, but continuously discount older path increments. Nothing is dropped outright. "
                    "A level-`k` coefficient fades `k` times as fast when every channel uses one common rate."
                ),
            },
            {"id": "memory_kernel_chart", "type": "chart", "chartId": "memory_kernel"},
            {
                "id": "math_update",
                "type": "markdown",
                "body": (
                    "## The exact EFM update, step by step\n\n"
                    "For one new half-hour segment, the implementation performs two exact operations:\n\n"
                    "1. **Fade the old state.** Multiply every level-`k` block by "
                    "`exp(-k * lambda * dt)`.\n"
                    "2. **Append the new segment.** Replace its increment `delta_x` by "
                    "`phi * delta_x`, where phi = (1 - exp(-lambda * dt)) / (lambda * dt)`, "
                    "then apply the ordinary tensor exponential and Chen product.\n\n"
                    "The code evaluates `phi` with `expm1`, which stays accurate when the decay rate is "
                    "small. This is Lemma 4.2 and equation (4.4) of the "
                    "[EFM paper](https://arxiv.org/abs/2507.03700v2), not an EMA applied after computing "
                    "ordinary signature rows."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Controlled experimental design\n\n"
                    "Only the memory representation changes. Both models use the same time-temperature "
                    "path, order `N=6`, **120 fitted coefficients**, seven-day target increment, train-only "
                    "feature standardization, squared loss, Ridge regression, and timestamps.\n\n"
                    "- **2012:** common context and EFM burn-in only\n"
                    "- **2013:** model training\n"
                    "- **2014:** memory-horizon and Ridge-penalty selection\n"
                    "- **2015:** shared evaluation, 17,520 half-hour observations; it does not "
                    "select `H` or Ridge `alpha` in this run\n\n"
                    "A common parameter `H` makes the comparison readable. Rolling memory uses a window "
                    "of `H` days. EFM sets `lambda = log(100) / H`, so a level-one contribution that is "
                    "`H` days old has 1% of its initial weight."
                ),
            },
            {
                "id": "ridge_reasoning",
                "type": "markdown",
                "body": (
                    "## Why squared loss and why Ridge for both methods\n\n"
                    "Squared loss matches the primary RMSE metric and gives unusually large demand errors "
                    "extra weight. Signature coordinates are strongly related to one another, especially "
                    "across neighboring levels. Ordinary least squares can therefore use large, unstable "
                    "coefficients. Ridge adds the squared coefficient penalty `alpha * ||beta||^2`, trading "
                    "a little bias for lower variance and a more stable forecast.\n\n"
                    "The EFM paper uses Elastic Net in one stationary-SDE illustration. This comparison "
                    "keeps Ridge for EFM because changing both the memory representation and estimator "
                    "would not isolate the question the experiment is meant to answer."
                ),
            },
            {
                "id": "validation_read",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Memory is selected before the test year\n\n"
                    f"Validation selects a **{rolling_horizon:g}-day rolling window** and a "
                    f"**{fading_horizon:g}-day EFM 1% horizon**. The latter corresponds to a "
                    f"{fading_half_life:.2f}-day level-one half-life. Ridge penalties are "
                    f"{rolling_penalty:g} and {fading_penalty:g}, respectively. The curve shows whether "
                    "each optimum is broad or fragile; 2015 is absent from this decision."
                ),
            },
            {"id": "validation_chart", "type": "chart", "chartId": "validation_sweep"},
            {
                "id": "test_result",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Shared 2015 performance is a paired comparison\n\n"
                    f"Both predictions are scored at the same 2015 timestamps. {evidence_sentence} "
                    "The interval comes from 2,000 circular moving-block bootstrap samples with "
                    "seven-day blocks, preserving short-range dependence better than resampling isolated "
                    "half-hours. The interval and resample share are conditional on the two fixed fitted "
                    "models and this block choice. MAE and MAPE are supporting measures, not separately "
                    "selected models."
                ),
            },
            {"id": "test_metrics_table", "type": "table", "tableId": "test_metrics"},
            {
                "id": "seasonal_read",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Checking the result season by season\n\n"
                    "The seasonal bars separate winter heating conditions, spring transitions, summer, "
                    "and autumn. A method that wins overall but loses in one season may be learning a "
                    "particular regime rather than a generally superior memory rule."
                ),
            },
            {"id": "seasonal_chart", "type": "chart", "chartId": "seasonal_rmse"},
            {
                "id": "forecast_read",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## What the forecast traces show\n\n"
                    "The winter and summer weeks show whether a method misses the level, smooths peaks, "
                    "or reacts too slowly. These weeks are descriptive examples from the common test "
                    "horizon; they were not chosen to select either model."
                ),
            },
            {"id": "forecast_chart", "type": "chart", "chartId": "forecast_examples"},
            {
                "id": "efficiency_read",
                "type": "markdown",
                "sourceId": "comparison_results",
                "body": (
                    "## Both methods have a fixed-size online signature state\n\n"
                    "At equal dimension and order, both store 126 raw nonconstant signature coordinates "
                    "before six pure-time words are dropped for regression. EFM needs only the current "
                    "state and latest point. The optimized rolling update also keeps a fixed-size signature "
                    "state, but it additionally needs the active path window so the oldest segment can be "
                    "removed. Wall-clock figures are single-run diagnostics, not hardware-independent "
                    "complexity claims."
                ),
            },
            {"id": "runtime_table", "type": "table", "tableId": "runtime"},
            {
                "id": "general_comparison",
                "type": "markdown",
                "body": (
                    "## General advantages and disadvantages\n\n"
                    "Rolling memory is easier to explain as a literal recent-history window and forgets an "
                    "obsolete regime completely after `H` days. Its hard boundary can also create abrupt "
                    "changes when one influential segment exits. EFM has smoother, time-invariant, "
                    "mean-reverting dynamics and no mathematically required window buffer, but its remote "
                    "past never becomes exactly zero and finite data require initialization or burn-in."
                ),
            },
            {"id": "tradeoffs_table", "type": "table", "tableId": "tradeoffs"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations\n\n"
                    "- This is one electricity dataset and one time-temperature path. It does not show "
                    "that either method forecasts better in general.\n"
                    "- The main EFM run uses one equal decay rate for both path channels. Separate rates "
                    "per channel are more flexible but harder to compute stably.\n"
                    "- The stationary EFM idealization uses the infinite past, although the signature itself "
                    "is also defined on finite intervals. This run initializes the state at zero in 2012. "
                    "One full year of common context makes the slowest candidate's level-one initialization "
                    "weight below 2e-20 of its initial value, but does not recreate infinite history.\n"
                    "- Order `N=6` was fixed from the rolling-window paper before this comparison. That "
                    "paper explored order and window on the same 2015 data, so 2015 is independent of `H` "
                    "and Ridge `alpha` selection here, but not an untouched holdout for every choice.\n"
                    "- Features use the temperature path all the way through the target time. A genuine "
                    "seven-day-ahead forecast therefore needs a weather forecast for that whole future "
                    "temperature path. Using the observed path, as here and in the rolling-window paper, "
                    "makes this a conditional or oracle-temperature evaluation rather than a fully "
                    "operational forecast.\n"
                    "- An [author-associated repository](https://github.com/DimitriSotnikov/signature) "
                    "contains later electricity notebooks, but it is not linked by the paper, has no license, "
                    "and its saved rolling and EFM transforms use different path dimensions. Those numbers "
                    "are not used here."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Possible next steps\n\n"
                    "1. Repeat the controlled grid over orders 3-6 and report accuracy against update "
                    "time.\n"
                    "2. Add the integrated-temperature path as a separate, capacity-matched experiment.\n"
                    "3. Run the synthetic generator across many noise seeds; EFM should be tested where the "
                    "true memory is exponential.\n"
                    "4. Measure sensitivity to shorter burn-in periods and abrupt regime shifts.\n"
                    "5. Only then test channel-specific EFM rates with a stable divided-difference or matrix-"
                    "exponential segment calculation."
                ),
            },
            {
                "id": "sources_note",
                "type": "markdown",
                "body": (
                    "## Source scope\n\n"
                    "The [rolling-window paper](https://arxiv.org/abs/2510.12337) supplies the electricity "
                    "task and RidgeSig design. The [EFM paper v2](https://arxiv.org/abs/2507.03700v2) supplies "
                    "the fading definition, modified Chen identity, exact linear-segment formula, and "
                    "mean-reverting interpretation. The EFM paper itself has no electricity experiment; all "
                    "electricity results in this report were generated by the independent code in this project."
                ),
            },
        ],
    }

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": source_definitions,
        "package_info": {
            "root": "results/comparison",
            "manifestPath": "results/comparison/report-artifact.json",
            "snapshotPath": "results/comparison/report-artifact.json",
            "originUrl": "artifact://rolling-vs-fading-memory-signatures",
        },
    }


def _memory_kernel_rows(rolling_horizon: float, fading_horizon: float) -> list[dict[str, Any]]:
    maximum_age = max(rolling_horizon * 1.4, fading_horizon * 1.2)
    ages = np.linspace(0.0, maximum_age, 121)
    rows: list[dict[str, Any]] = []
    for age in ages:
        rows.extend(
            [
                {
                    "age_days": float(age),
                    "series": "Rolling level 1",
                    "relative_weight": float(age <= rolling_horizon),
                },
                {
                    "age_days": float(age),
                    "series": "Fading level 1",
                    "relative_weight": float(100.0 ** (-age / fading_horizon)),
                },
                {
                    "age_days": float(age),
                    "series": "Fading level 3",
                    "relative_weight": float(100.0 ** (-3.0 * age / fading_horizon)),
                },
            ]
        )
    return rows


def _forecast_example_rows(forecasts: pd.DataFrame) -> list[dict[str, Any]]:
    frame = forecasts.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    periods = (
        ("Winter week", pd.Timestamp("2015-02-02"), pd.Timestamp("2015-02-09")),
        ("Summer week", pd.Timestamp("2015-07-06"), pd.Timestamp("2015-07-13")),
    )
    rows: list[dict[str, Any]] = []
    series = (
        ("Observed", "observed_mw"),
        ("Rolling", "rolling_window_prediction_mw"),
        ("Fading", "fading_memory_prediction_mw"),
    )
    for label, start, stop in periods:
        subset = frame[(frame["datetime"] >= start) & (frame["datetime"] < stop)].iloc[::4]
        for _, row in subset.iterrows():
            for series_label, column in series:
                rows.append(
                    {
                        "datetime": row["datetime"].isoformat(),
                        "period": label,
                        "series": series_label,
                        "demand_mw": float(row[column]),
                    }
                )
    return rows


def _uncertainty_sentence(difference: float, lower: float, upper: float) -> str:
    if upper < 0.0:
        return (
            f"The weekly-block 95% interval is [{lower:,.0f}, {upper:,.0f}] MW, "
            "supporting lower RMSE for fading memory on this test year."
        )
    if lower > 0.0:
        return (
            f"The weekly-block 95% interval is [{lower:,.0f}, {upper:,.0f}] MW, "
            "supporting lower RMSE for rolling memory on this test year."
        )
    point_winner = "fading" if difference < 0.0 else "rolling"
    return (
        f"The point estimate favors {point_winner}, but the weekly-block 95% interval "
        f"[{lower:,.0f}, {upper:,.0f}] MW includes zero, so this test year does not resolve "
        "a reliable winner."
    )


def _method_row(frame: pd.DataFrame, method: str) -> pd.Series:
    rows = frame[frame["method"] == method]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one metrics row for {method}")
    return rows.iloc[0]


def _materialize_sql_datasets(
    frames: dict[str, pd.DataFrame],
    queries: dict[str, str],
    *,
    generated_at: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Run each table's SQL query and keep the query next to its result."""
    if set(frames) != set(queries):
        raise ValueError("every report dataset must have exactly one materializing SQL query")

    descriptions = {
        "headline": "Selects the three headline 2015 comparison metrics.",
        "memory_kernel": "Selects the rolling and fading memory-weight curves.",
        "validation_sweep": "Selects validation RMSE and Ridge penalty by memory horizon.",
        "test_metrics": "Selects exact 2015 accuracy metrics for both methods.",
        "seasonal_metrics": "Selects 2015 error metrics by season and method.",
        "forecast_examples": "Selects the compact winter and summer forecast traces.",
        "runtime": "Selects measured feature, fit, matrix, and state costs.",
        "tradeoffs": "Selects the paper-derived structural method comparison.",
    }
    datasets: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    with sqlite3.connect(":memory:") as connection:
        for dataset, frame in frames.items():
            frame.to_sql(dataset, connection, if_exists="replace", index=False)
        for dataset, query in queries.items():
            materialized = pd.read_sql_query(query, connection)
            datasets[dataset] = _json_records(materialized)
            sources.append(
                {
                    "id": f"{dataset}_sql",
                    "label": f"{dataset.replace('_', ' ').title()} query",
                    "query": {
                        "engine": "sqlite",
                        "sql": query,
                        "description": descriptions[dataset],
                        "executed_at": generated_at,
                        "tables_used": [dataset],
                    },
                }
            )
    return datasets, sources


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _source_link(source: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    if "path" in source:
        result["path"] = source["path"]
    if "href" in source:
        result["href"] = source["href"]
    return result


def _source_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "comparison_results",
            "label": "Controlled Python comparison",
            "path": "src/sliding_window_signatures/comparison.py",
        },
        {
            "id": "rolling_paper",
            "label": "Sliding-Window Signatures paper",
            "href": "https://arxiv.org/abs/2510.12337",
        },
        {
            "id": "fading_paper",
            "label": "Exponentially Fading Memory Signature paper v2",
            "href": "https://arxiv.org/abs/2507.03700v2",
        },
        {
            "id": "paper_comparison",
            "label": "Structural comparison derived from both signature papers",
            "href": "https://arxiv.org/abs/2507.03700v2",
        },
        {
            "id": "author_repository",
            "label": "Author-associated exploratory repository",
            "href": "https://github.com/DimitriSotnikov/signature",
        },
    ]
