"""
Standalone keogram plotter — reads a TEC parquet and produces a keogram PNG.

The grid is built by snapping IPP positions to a regular lat/time grid and
taking the median dTEC per cell, then filling short time-axis gaps and
applying a mild Gaussian smoothing.

Usage
-----
# date-only (resolves parquet from results/tec_data/ automatically)
python plot_keogram.py --date 2015-06-21

# explicit parquet path
python plot_keogram.py tec_2015-06-21.parquet
python plot_keogram.py tec_2015-06-21.parquet --date 2015-06-21
python plot_keogram.py tec_2015-06-21.parquet --lon-span 10 25 --lat-range 37.5 57.5
python plot_keogram.py tec_2015-06-21.parquet --clim 0.3 --out my_keogram.png
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LON_SPAN   = (-180.0, 180.0)
DEFAULT_LAT_RANGE  = (37.5, 57.5)
DEFAULT_MIN_ELE    = 20.0
DEFAULT_DTEC_CLIM  = 0.15
KEO_TIME_STEP      = 30          # seconds
KEO_LAT_STEP       = 0.05        # degrees
KEO_GAUSS_SIGMA    = 0         # pixels
KEO_FIGSIZE        = (10, 6)
KEO_DPI            = 150


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _keo_fill_2d(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells by 2-D linear interpolation from surrounding valid points."""
    valid = np.isfinite(grid)
    if valid.all() or not valid.any():
        return grid
    rows, cols = np.indices(grid.shape)
    points = np.column_stack([rows[valid], cols[valid]])
    values = grid[valid]
    missing = np.column_stack([rows[~valid], cols[~valid]])
    filled = grid.copy()
    filled[~valid] = griddata(points, values, missing, method="linear")
    return filled


def _keo_smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing on filled cells only — does not fill new NaN cells."""
    valid    = np.isfinite(grid)
    smoothed = gaussian_filter(np.where(valid, grid, 0.0), sigma=sigma)
    weight   = gaussian_filter(valid.astype(float), sigma=sigma)
    out = grid.copy()
    covered = valid & (weight > 1e-6)
    out[covered] = (smoothed / np.where(weight > 1e-6, weight, 1.0))[covered]
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_keogram(
    parquet_path: Path,
    out_path: Path,
    input_date: dt.date | None,
    lon_span: tuple[float, float],
    lat_range: tuple[float, float],
    min_ele: float,
    dtec_clim: float,
    grid_out_path: Path | None,
) -> None:
    print(f"Reading {parquet_path} …")
    df = pl.read_parquet(parquet_path)

    if input_date is None:
        stem = parquet_path.stem
        for part in stem.split("_"):
            try:
                input_date = dt.date.fromisoformat(part)
                break
            except ValueError:
                continue

    mask = (
        (df["ele"] >= min_ele)
        & (df["lon_ipp"] >= lon_span[0]) & (df["lon_ipp"] < lon_span[1])
        & (df["lat_ipp"] >= lat_range[0]) & (df["lat_ipp"] <= lat_range[1])
    )
    df = df.filter(mask)
    if df.is_empty():
        raise ValueError("No data remaining after spatial / elevation filter.")

    t0 = (
        dt.datetime(input_date.year, input_date.month, input_date.day, tzinfo=dt.timezone.utc)
        if input_date is not None
        else df["epoch"].min()
    )

    # Build a regular time (seconds-of-day) × latitude grid
    sod_raw  = np.array([(e - t0).total_seconds() for e in df["epoch"].to_list()])
    sod_grid = np.arange(0, 86400, KEO_TIME_STEP)
    lat_grid = np.arange(lat_range[0], lat_range[1] + KEO_LAT_STEP, KEO_LAT_STEP)

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

    # Step 1 — fill NaN cells by 2-D nearest-neighbour interpolation
    grid = _keo_fill_2d(grid)

    # Step 2 — mild Gaussian smoothing (visual polish only, no new NaN filling)
    if KEO_GAUSS_SIGMA > 0:
        grid = _keo_smooth(grid, KEO_GAUSS_SIGMA)

    if grid_out_path is not None:
        pl.DataFrame(
            {"lat": lat_grid, **{str(int(s)): grid[:, i] for i, s in enumerate(sod_grid)}}
        ).write_parquet(grid_out_path, compression="zstd")
        print(f"Saved grid: {grid_out_path}")

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
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_TEC_DIR      = _PROJECT_ROOT / "results" / "tec_data"
_KEO_DIR      = _PROJECT_ROOT / "results" / "keo_image"
_MAT_DIR      = _PROJECT_ROOT / "results" / "keo_mat"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Keogram plotter (median-snap gridding).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parquet",      metavar="PARQUET", nargs="?", default=None,
                                        help="Path to the TEC parquet file (omit to use --date)")
    parser.add_argument("--date",       default=None,      help="Date YYYY-MM-DD — resolves parquet from results/tec_data/ automatically")
    parser.add_argument("--out",        default=None,      help="Output PNG path (default: results/keo_image/keogram_<date>.png)")
    parser.add_argument("--lon-span",   nargs=2, type=float, default=list(DEFAULT_LON_SPAN),  metavar=("LON_MIN", "LON_MAX"))
    parser.add_argument("--lat-range",  nargs=2, type=float, default=list(DEFAULT_LAT_RANGE), metavar=("LAT_MIN", "LAT_MAX"))
    parser.add_argument("--min-ele",    type=float, default=DEFAULT_MIN_ELE,    help="Minimum satellite elevation (deg)")
    parser.add_argument("--clim",       type=float, default=DEFAULT_DTEC_CLIM,  help="Colour scale ±limit [TECu]")
    parser.add_argument("--save-grid",  action="store_true", help="Also save the 2-D grid as a parquet file")
    args = parser.parse_args()

    if args.parquet:
        parquet_path = Path(args.parquet)
        input_date   = dt.date.fromisoformat(args.date) if args.date else None
    elif args.date:
        input_date   = dt.date.fromisoformat(args.date)
        parquet_path = _TEC_DIR / f"tec_{input_date}.parquet"
    else:
        print("ERROR: provide a parquet path or --date YYYY-MM-DD.")
        sys.exit(1)

    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.")
        sys.exit(1)

    date_tag = str(input_date) if input_date else parquet_path.stem.replace("tec_", "")
    if args.out:
        out_path = Path(args.out)
    else:
        _KEO_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _KEO_DIR / f"keogram_{date_tag}.png"

    grid_out_path = None
    if args.save_grid:
        _MAT_DIR.mkdir(parents=True, exist_ok=True)
        grid_out_path = _MAT_DIR / f"keogram_grid_{date_tag}.parquet"

    plot_keogram(
        parquet_path  = parquet_path,
        out_path      = out_path,
        input_date    = input_date,
        lon_span      = tuple(args.lon_span),
        lat_range     = tuple(args.lat_range),
        min_ele       = args.min_ele,
        dtec_clim     = args.clim,
        grid_out_path = grid_out_path,
    )
