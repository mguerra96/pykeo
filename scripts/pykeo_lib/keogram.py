"""
Keogram (latitude × time dTEC plot) creation and grid export.
"""

import datetime as dt
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

from .constants import (
    DEFAULT_LON_SPAN,
    KEO_DPI,
    KEO_DTEC_CLIM,
    KEO_FIGSIZE,
    KEO_GAUSS_SIGMA,
    KEO_LAT_RANGE,
    KEO_LAT_STEP,
    KEO_TIME_STEP,
    MIN_ELEVATION,
)

logger = logging.getLogger(__name__)


def _keo_fill_2d(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells by nearest-neighbour interpolation from surrounding valid points."""
    valid = np.isfinite(grid)
    if valid.all() or not valid.any():
        return grid
    rows, cols = np.indices(grid.shape)
    interp  = NearestNDInterpolator(
        np.column_stack([rows[valid], cols[valid]]), grid[valid]
    )
    filled  = grid.copy()
    filled[~valid] = interp(rows[~valid], cols[~valid])
    return filled


def _keo_smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing on filled cells only — does not introduce new NaN cells."""
    valid    = np.isfinite(grid)
    smoothed = gaussian_filter(np.where(valid, grid, 0.0), sigma=sigma)
    weight   = gaussian_filter(valid.astype(float), sigma=sigma)
    out      = grid.copy()
    covered  = valid & (weight > 1e-6)
    out[covered] = (smoothed / np.where(weight > 1e-6, weight, 1.0))[covered]
    return out


def plot_keogram(
    df: pl.DataFrame,
    out_path: Path,
    input_date: date | None = None,
    lon_span: tuple[float, float] = DEFAULT_LON_SPAN,
    lat_range: tuple[float, float] = KEO_LAT_RANGE,
    min_ele: float = MIN_ELEVATION,
    dtec_clim: float = KEO_DTEC_CLIM,
    gauss_sigma: float = KEO_GAUSS_SIGMA,
    lat_step: float = KEO_LAT_STEP,
    time_step: int = KEO_TIME_STEP,
    grid_out_path: Path | None = None,
) -> None:
    """
    Create a latitude–time keogram of dTEC and optionally save the 2-D grid as parquet.

    Snaps IPP positions to a regular lat/time grid, takes the mean dTEC per cell,
    fills NaN cells by 2-D linear interpolation, then applies mild Gaussian smoothing.
    """
    mask = (
        (df["ele"] >= min_ele)
        & (df["lon_ipp"] >= lon_span[0]) & (df["lon_ipp"] < lon_span[1])
        & (df["lat_ipp"] >= lat_range[0]) & (df["lat_ipp"] <= lat_range[1])
    )
    df = df.filter(mask)
    if df.is_empty():
        logger.warning("Keogram: no data after spatial filter.")
        return

    t0 = (
        dt.datetime(input_date.year, input_date.month, input_date.day, tzinfo=dt.timezone.utc)
        if input_date is not None
        else df["epoch"].min()
    )

    t0_us    = int(t0.timestamp() * 1e6)
    sod_raw  = (df["epoch"].cast(pl.Int64) - t0_us).to_numpy() / 1e6
    sod_grid = np.arange(0, 86400, time_step)
    lat_grid = np.arange(lat_range[0], lat_range[1] + lat_step, lat_step)

    sod_snap = sod_grid[np.searchsorted(sod_grid, sod_raw).clip(0, len(sod_grid) - 1)]
    lat_snap = lat_grid[np.searchsorted(lat_grid, df["lat_ipp"].to_numpy()).clip(0, len(lat_grid) - 1)]

    agg = (
        pl.DataFrame({"sod": sod_snap, "lat": lat_snap, "dtec": df["vtec_detrended"].to_numpy()})
        .group_by(["sod", "lat"])
        .agg(pl.col("dtec").mean())
    )

    sod_idx = {v: i for i, v in enumerate(sod_grid)}
    lat_idx = {v: i for i, v in enumerate(lat_grid)}
    grid = np.full((len(lat_grid), len(sod_grid)), np.nan)
    for s, l, v in zip(agg["sod"].to_numpy(), agg["lat"].to_numpy(), agg["dtec"].to_numpy()):
        si, li = sod_idx.get(s), lat_idx.get(l)
        if si is not None and li is not None:
            grid[li, si] = v

    grid = _keo_fill_2d(grid)

    if gauss_sigma > 0:
        grid = _keo_smooth(grid, gauss_sigma)

    if grid_out_path is not None:
        pl.DataFrame(
            {"lat": lat_grid, **{str(int(s)): grid[:, i] for i, s in enumerate(sod_grid)}}
        ).write_parquet(grid_out_path, compression="zstd")
        logger.info(f"Saved keogram grid: {grid_out_path}")

    t_nums   = mdates.date2num([t0 + dt.timedelta(seconds=float(s)) for s in sod_grid])
    date_str = t0.strftime("%d/%m/%Y")

    fig, ax = plt.subplots(figsize=KEO_FIGSIZE)
    img = ax.pcolormesh(
        t_nums, lat_grid, grid,
        cmap="jet", vmin=-dtec_clim, vmax=dtec_clim, shading="nearest",
    )
    cbar = plt.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("dTEC [TECu]", fontsize=13, rotation=270, labelpad=15)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlabel("Time UT", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.set_ylim(lat_range)
    ax.set_title(
        f"Keogram  {date_str}  |  lon [{lon_span[0]}°, {lon_span[1]}°]",
        fontsize=13,
    )
    ax.tick_params(labelsize=13)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=KEO_DPI)
    plt.close(fig)
    logger.info(f"Saved keogram: {out_path}")
