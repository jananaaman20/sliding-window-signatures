# Sliding-window signatures

A reproduction of [Drobac et al., *Sliding-Window Signatures for Time Series:
Application to Electricity Demand Forecasting*](https://arxiv.org/abs/2510.12337).

The method takes a sliding window of recent temperature, turns it into
signature features, and fits a ridge regression that predicts the change in
electricity demand.

## Setup

Needs Python 3.11-3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run slidesig-reproduce download-data
uv run slidesig-reproduce smoke
```

`download-data` fetches the authors' CSV from a pinned commit and checks its
SHA-256. The other commands download it themselves if `data/data.csv` is
missing.

## Commands

```bash
uv run slidesig-reproduce table1          # synthetic-data comparison
uv run slidesig-reproduce table2          # real-demand comparison
uv run slidesig-reproduce figure2         # window-size sweep
uv run slidesig-reproduce figure3         # truncation-order sweep
uv run slidesig-reproduce figure4         # winter and summer forecasts
uv run slidesig-reproduce figure5         # observed versus synthetic demand
uv run slidesig-reproduce explain         # five annotated explanation figures
uv run slidesig-reproduce compare-memory  # rolling window versus fading memory
uv run slidesig-reproduce all             # everything except compare-memory
```

`figure3` is the slow one: 31 window sizes times six orders over 70,128 rows.
Level seven is computed once per window and the lower orders reuse it, but the
full run still takes a while.

## Faithful and corrected modes

`--mode faithful` is the default. It copies the released notebook's row
counting and its test-set sweeps, so these are the numbers to compare against
the paper.

`--mode corrected` uses exact calendar boundaries and one shared test period
for RidgeSig and all of its baselines. Its sweep plots use validation RMSE, so
the window and order are not picked using the test data.

```bash
uv run slidesig-reproduce table1 --mode corrected
```

The two modes differ because the public notebook shifts RidgeSig's train,
validation, and test periods forward by one window, while several baselines use
other periods.

## Expected faithful results

Rounded Table 1 targets, with RidgeSig at `w=9 days` and `N=4`. The model
names are the ones `table1` prints, so the rows line up:

| Model | RMSE (MW) | MAPE (%) |
|---|---:|---:|
| LR(T) | 5,435 | 8.3 |
| LR(T, T^2) | 4,729 | 6.4 |
| LR(smoothed T) | 3,079 | 4.9 |
| RidgeSig | 1,637 | 2.5 |
| Oracle LR(smoothed T, smoothed T^2) | 995 | 1.5 |

Rounded Table 2 targets, with RidgeSig at `w=9 days` and `N=6`:

| Model | RMSE (MW) | MAPE (%) |
|---|---:|---:|
| LR(smoothed T, smoothed T^2) | 6,518 | 10.7 |
| LR(T, T^2, smoothed T, smoothed T^2) | 6,172 | 9.9 |
| LR(Y[t-7 days]) | 4,149 | 5.3 |
| LR(T features, Y[t-7 days]) | 3,714 | 5.3 |
| RidgeSig | 3,150 | 4.4 |

The current public notebook prints 3,733 MW for the combined real-data
baseline because it no longer uses the nine-day alignment from the paper.
Faithful mode keeps that alignment and reproduces the published 3,714 MW.

## Rolling window versus fading memory

`compare-memory` is an extra experiment, not part of the paper. It replaces the
hard window with the equal-rate Exponentially Fading Memory (EFM) signature of
[Abi Jaber and Sotnikov](https://arxiv.org/abs/2507.03700v2). There the weight
on older data decays smoothly instead of stopping at a fixed cut-off. That
paper is theoretical and has no electricity-demand experiment, so this is a new
experiment rather than a reproduction.

Only the memory changes. Both methods use the same time-and-temperature path,
order six, 120 fitted coefficients, seven-day demand increment, ridge grid, and
standardization fitted on training rows only. 2012 is history, 2013 trains,
2014 chooses the memory horizon and ridge penalty, and 2015 is the shared test
year.

Both methods take one memory setting `H`. The rolling window keeps exactly `H`
days. Fading memory uses `lambda = log(100) / H`, so at level one something `H`
days old still has one percent of its weight, and level `k` fades `k` times
faster. The EFM update is exact for every half-hour segment.

Two things to keep in mind:

- Each feature uses temperature right up to its target timestamp, so the
  accuracy assumes you already know the weather. A real seven-day-ahead
  forecast would need a weather forecast instead.
- Order six comes from the rolling-window paper, which did look at 2015 when
  choosing it. So 2015 was not used to pick anything in this experiment, but it
  was used for that one setting.

The command writes to `results/comparison/`: the validation sweep, 2015
metrics, a paired weekly block bootstrap, seasonal errors, the full forecasts,
runtimes, six figures, and `report-artifact.json` with the write-up text and
numbers.

## Layout

```text
src/sliding_window_signatures/
├── signatures.py           # signature algebra and the sliding update
├── fading_memory.py        # exact equal-rate EFM recurrence
├── model.py                # ridge model on demand increments
├── baselines.py            # linear-regression comparisons
├── simulation.py           # Appendix D synthetic target
├── experiments.py          # Tables 1-2 and Figures 2-5
├── comparison.py           # rolling window versus fading memory
├── comparison_plotting.py  # comparison figures
├── comparison_report.py    # comparison write-up data
├── plotting.py             # paper figures
├── explainers.py           # annotated explanation figures
├── data.py                 # pinned download and validation
└── cli.py                  # command-line entry point
scripts/reproduce.py        # CLI wrapper
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
```

## Data

`data/data.csv` is not committed. It comes from Nina Drobac's
[`slidesig`](https://github.com/ninadrobac/slidesig) repository.