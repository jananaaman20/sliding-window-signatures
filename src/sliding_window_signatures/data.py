"""Download and validate the public electricity-demand dataset."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PAPER_REPOSITORY_COMMIT = "4fc8e1a112f79673aae9322dc5da77c2cde58fe4"
PAPER_DATA_URL = (
    f"https://raw.githubusercontent.com/ninadrobac/slidesig/{PAPER_REPOSITORY_COMMIT}/data/data.csv"
)
PAPER_DATA_SHA256 = "548d6eee8dac1e21a1d8e20bcc134b4b45ccffb2d1aa7f6f9258ffb202a4259e"
EXPECTED_ROWS = 70_128
EXPECTED_START = np.datetime64("2012-01-01T00:00:00")
EXPECTED_END = np.datetime64("2015-12-31T23:30:00")
SAMPLES_PER_DAY = 48


@dataclass(frozen=True, slots=True)
class PaperDataset:
    """The four columns the experiments use."""

    dates: NDArray[np.datetime64]
    half_hour: NDArray[np.int64]
    demand: NDArray[np.float64]
    temperature: NDArray[np.float64]

    def __len__(self) -> int:
        return self.demand.size


def download_paper_data(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the authors' CSV from a pinned commit and verify its SHA-256."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        if file_sha256(destination) == PAPER_DATA_SHA256:
            return destination
        if not overwrite:
            raise FileExistsError(
                f"{destination} exists but does not match the paper dataset; "
                "pass overwrite=True to replace it"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            PAPER_DATA_URL,
            headers={"User-Agent": "sliding-window-signatures-reproduction/0.1"},
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)

        actual_hash = file_sha256(temporary_path)
        if actual_hash != PAPER_DATA_SHA256:
            raise ValueError(
                "downloaded dataset failed its checksum: "
                f"expected {PAPER_DATA_SHA256}, got {actual_hash}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_paper_dataset(path: Path, *, verify_checksum: bool = True) -> PaperDataset:
    """Load the paper CSV, refusing anything with unexpected shape or dates."""
    path = path.expanduser().resolve()
    if verify_checksum:
        actual_hash = file_sha256(path)
        if actual_hash != PAPER_DATA_SHA256:
            raise ValueError(
                f"dataset checksum mismatch: expected {PAPER_DATA_SHA256}, got {actual_hash}"
            )

    frame = pd.read_csv(path, parse_dates=["DateTime"])
    expected_columns = ["DateTime", "HalfHour", "Consumption", "Temperature"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"unexpected columns: {frame.columns.tolist()}; expected {expected_columns}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"unexpected row count: {len(frame)}; expected {EXPECTED_ROWS}")

    dates = frame["DateTime"].to_numpy(dtype="datetime64[ns]")
    half_hour = frame["HalfHour"].to_numpy(dtype=np.int64)
    demand = frame["Consumption"].to_numpy(dtype=np.float64)
    temperature = frame["Temperature"].to_numpy(dtype=np.float64)

    if dates[0] != EXPECTED_START or dates[-1] != EXPECTED_END:
        raise ValueError(f"unexpected date range: {dates[0]} through {dates[-1]}")
    if not np.all(np.diff(dates) == np.timedelta64(30, "m")):
        raise ValueError("DateTime must be a complete, strictly half-hourly sequence")
    if not np.array_equal(half_hour, np.arange(EXPECTED_ROWS) % SAMPLES_PER_DAY):
        raise ValueError("HalfHour must repeat the labels 0 through 47")
    if not np.isfinite(demand).all() or not np.isfinite(temperature).all():
        raise ValueError("demand and temperature must be finite")

    return PaperDataset(
        dates=dates,
        half_hour=half_hour,
        demand=demand,
        temperature=temperature,
    )


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
