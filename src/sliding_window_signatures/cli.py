"""Command-line entry point for every reproduction command."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from sliding_window_signatures.comparison import run_signature_memory_comparison
from sliding_window_signatures.data import (
    PaperDataset,
    download_paper_data,
    load_paper_dataset,
)
from sliding_window_signatures.experiments import (
    ExperimentMode,
    make_figure5_synthetic_data,
    run_order_sweep,
    run_real_forecasts,
    run_real_table,
    run_smoke_experiment,
    run_synthetic_table,
    run_window_sweep,
)

DEFAULT_DATA_PATH = Path("data/data.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("results")


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse the arguments and run the requested command."""
    parser = _build_parser()
    options = parser.parse_args(arguments)

    if options.command == "download-data":
        path = download_paper_data(options.data, overwrite=options.overwrite)
        print(f"Verified paper dataset: {path}")
        return

    dataset = _load_or_download(options.data)
    output_directory = options.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    if options.command == "smoke":
        result = run_smoke_experiment(dataset, observations=options.observations)
        destination = output_directory / "smoke_results.json"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Saved {destination}")
        return

    if options.command == "figure5":
        _run_figure5(dataset, output_directory)
        return

    if options.command == "explain":
        _run_explainers(dataset, output_directory)
        return

    if options.command == "compare-memory":
        _run_memory_comparison(dataset, output_directory)
        return

    mode: ExperimentMode = options.mode
    if options.command == "table1":
        _run_table1(dataset, mode, output_directory)
    elif options.command == "table2":
        _run_table2(dataset, mode, output_directory)
    elif options.command == "figure2":
        _run_figure2(dataset, mode, output_directory)
    elif options.command == "figure3":
        _run_figure3(dataset, mode, output_directory)
    elif options.command == "figure4":
        _run_figure4(dataset, mode, output_directory)
    elif options.command == "all":
        _run_explainers(dataset, output_directory)
        _run_figure5(dataset, output_directory)
        _run_table1(dataset, mode, output_directory)
        _run_figure2(dataset, mode, output_directory)
        _run_figure3(dataset, mode, output_directory)
        _run_table2(dataset, mode, output_directory)
        _run_figure4(dataset, mode, output_directory)
    else:  # pragma: no cover - argparse enforces the choices.
        parser.error(f"unknown command: {options.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slidesig-reproduce",
        description="Reproduce the experiments in arXiv:2510.12337.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download and verify the paper CSV.")
    download.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    download.add_argument("--overwrite", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run a small end-to-end correctness check.")
    _add_data_and_output_arguments(smoke)
    smoke.add_argument("--observations", type=int, default=6_000)

    for command, help_text in (
        ("table1", "Reproduce the synthetic comparison table."),
        ("table2", "Reproduce the real-demand comparison table."),
        ("figure2", "Reproduce the memory/window sweep."),
        ("figure3", "Reproduce the truncation-order sweep."),
        ("figure4", "Reproduce the winter/summer forecast plot."),
        ("all", "Run every table and figure; this is compute intensive."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        _add_data_and_output_arguments(subparser)
        _add_mode_argument(subparser)

    figure5 = subparsers.add_parser(
        "figure5",
        help="Reproduce the observed-versus-synthetic data plot.",
    )
    _add_data_and_output_arguments(figure5)

    explain = subparsers.add_parser(
        "explain",
        help="Generate the annotated explanation figures.",
    )
    _add_data_and_output_arguments(explain)

    compare = subparsers.add_parser(
        "compare-memory",
        help="Compare rolling-window and fading-memory signatures fairly.",
    )
    _add_data_and_output_arguments(compare)
    return parser


def _add_data_and_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("faithful", "corrected"),
        default="faithful",
        help=(
            "faithful matches the released notebook's indexing; corrected uses strict calendar "
            "boundaries and a common test horizon"
        ),
    )


def _load_or_download(path: Path) -> PaperDataset:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        print(f"Dataset not found at {resolved}; downloading the pinned public CSV.")
        download_paper_data(resolved)
    return load_paper_dataset(resolved)


def _run_table1(dataset: PaperDataset, mode: ExperimentMode, output: Path) -> None:
    table = run_synthetic_table(dataset, mode=mode, progress=_progress)
    _save_table(table, output / f"table1_synthetic_{mode}.csv")


def _run_table2(dataset: PaperDataset, mode: ExperimentMode, output: Path) -> None:
    table = run_real_table(dataset, mode=mode, progress=_progress)
    _save_table(table, output / f"table2_real_{mode}.csv")


def _run_figure2(dataset: PaperDataset, mode: ExperimentMode, output: Path) -> None:
    from sliding_window_signatures.plotting import save_window_sweep_figure

    results = run_window_sweep(dataset, mode=mode, progress=_progress)
    csv_path = output / f"figure2_window_sweep_{mode}.csv"
    _save_table(results, csv_path)
    figure_path = save_window_sweep_figure(
        results,
        output / f"figure2_window_sweep_{mode}.png",
    )
    print(f"Saved {figure_path}")


def _run_figure3(dataset: PaperDataset, mode: ExperimentMode, output: Path) -> None:
    from sliding_window_signatures.plotting import save_order_sweep_figure

    results = run_order_sweep(dataset, mode=mode, progress=_progress)
    csv_path = output / f"figure3_order_sweep_{mode}.csv"
    _save_table(results, csv_path)
    figure_path = save_order_sweep_figure(
        results,
        output / f"figure3_order_sweep_{mode}.png",
    )
    print(f"Saved {figure_path}")


def _run_figure4(dataset: PaperDataset, mode: ExperimentMode, output: Path) -> None:
    from sliding_window_signatures.plotting import save_real_forecast_figure

    ridge, baseline = run_real_forecasts(dataset, mode=mode, progress=_progress)
    metrics = pd.DataFrame(
        [
            {
                "Model": "RidgeSig",
                "RMSE (MW)": ridge.metrics.rmse,
                "MAE (MW)": ridge.metrics.mae,
                "MAPE (%)": ridge.metrics.mape_percent,
            },
            {
                "Model": "Temperature features + weekly lag",
                "RMSE (MW)": baseline.metrics.rmse,
                "MAE (MW)": baseline.metrics.mae,
                "MAPE (%)": baseline.metrics.mape_percent,
            },
        ]
    )
    _save_table(metrics, output / f"figure4_forecast_metrics_{mode}.csv")
    figure_path = save_real_forecast_figure(
        dataset,
        ridge,
        baseline,
        output / f"figure4_real_forecasts_{mode}.png",
    )
    print(f"Saved {figure_path}")


def _run_figure5(dataset: PaperDataset, output: Path) -> None:
    from sliding_window_signatures.plotting import save_synthetic_fidelity_figure

    synthetic = make_figure5_synthetic_data(dataset)
    generator = {
        "alpha": 0.005,
        "noise_standard_deviation": synthetic.noise_standard_deviation,
        "seed": synthetic.seed,
        "intercept": synthetic.intercept,
        "linear_coefficient": float(synthetic.coefficients[0]),
        "quadratic_coefficient": float(synthetic.coefficients[1]),
    }
    metadata_path = output / "figure5_synthetic_generator.json"
    metadata_path.write_text(json.dumps(generator, indent=2) + "\n", encoding="utf-8")
    figure_path = save_synthetic_fidelity_figure(
        dataset,
        synthetic,
        output / "figure5_synthetic_fidelity.png",
    )
    print(json.dumps(generator, indent=2))
    print(f"Saved {metadata_path}")
    print(f"Saved {figure_path}")


def _run_explainers(dataset: PaperDataset, output: Path) -> None:
    from sliding_window_signatures.explainers import save_beginner_explainers

    paths = save_beginner_explainers(dataset, output / "explainers")
    for path in paths:
        print(f"Saved {path}")


def _run_memory_comparison(dataset: PaperDataset, output: Path) -> None:
    from sliding_window_signatures.comparison_plotting import save_comparison_figures
    from sliding_window_signatures.comparison_report import save_comparison_report_artifact

    result = run_signature_memory_comparison(dataset, progress=_progress)
    comparison_output = output / "comparison"
    _save_table(result.sweep, comparison_output / "memory_horizon_sweep.csv")
    _save_table(result.metrics, comparison_output / "test_metrics.csv")
    _save_table(result.seasonal_metrics, comparison_output / "seasonal_metrics.csv")
    _save_table(result.runtime, comparison_output / "runtime.csv")
    _save_table(result.forecasts, comparison_output / "test_forecasts.csv")

    bootstrap_path = comparison_output / "paired_block_bootstrap.json"
    bootstrap_path.write_text(json.dumps(result.bootstrap, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {bootstrap_path}")

    for path in save_comparison_figures(result, comparison_output / "figures"):
        print(f"Saved {path}")
    artifact_path = save_comparison_report_artifact(
        result,
        comparison_output / "report-artifact.json",
    )
    print(f"Saved {artifact_path}")


def _save_table(table: pd.DataFrame, destination: Path) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    json_path = destination.with_suffix(".json")
    json_path.write_text(table.to_json(orient="records", indent=2) + "\n", encoding="utf-8")
    display = table if len(table) <= 20 else pd.concat((table.head(5), table.tail(5)))
    print(display.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if len(display) < len(table):
        print(f"... showing 10 of {len(table):,} rows")
    print(f"Saved {destination}")
    print(f"Saved {json_path}")


def _progress(message: str) -> None:
    print(f"[slidesig] {message}", flush=True)
