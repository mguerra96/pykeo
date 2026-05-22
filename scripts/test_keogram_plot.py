"""
Interactive keogram test plotter.

Reads a TEC parquet and calls the pipeline's plot_keogram function, with CLI
flags to override every tunable constant in constants.py.  Use this script to
experiment with gridding and display parameters without touching the pipeline.

Usage
-----
# by date (resolves parquet from results/tec_data/ automatically)
python test_keogram_plot.py --date 2015-06-21

# explicit parquet path
python test_keogram_plot.py tec_2015-06-21.parquet

# override parameters
python test_keogram_plot.py --date 2015-06-21 --clim 0.3 --gauss-sigma 1.5
python test_keogram_plot.py --date 2015-06-21 --lon-span 10 25 --lat-range 37.5 57.5
python test_keogram_plot.py --date 2015-06-21 --lat-step 0.1 --time-step 60
python test_keogram_plot.py --date 2015-06-21 --exclude-sv R09 R12
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import polars as pl

# Allow running from the scripts/ directory or the project root
sys.path.insert(0, str(Path(__file__).parent))

from pykeo_lib.constants import (
    DEFAULT_LON_SPAN,
    KEO_DTEC_CLIM,
    KEO_GAUSS_SIGMA,
    KEO_LAT_RANGE,
    KEO_LAT_STEP,
    KEO_TIME_STEP,
    MIN_ELEVATION,
)
from pykeo_lib.keogram import plot_keogram

_PROJECT_ROOT = Path(__file__).parent.parent
_TEC_DIR      = _PROJECT_ROOT / "results" / "tec_data"
_KEO_DIR      = _PROJECT_ROOT / "results" / "_keo_test"
_MAT_DIR      = _PROJECT_ROOT / "results" / "keo_mat"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Test keogram plotter — wraps the pipeline's plot_keogram with overridable constants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parquet", metavar="PARQUET", nargs="?", default=None,
                        help="Path to the TEC parquet (omit to use --date)")
    parser.add_argument("--date",        default=None,
                        help="Date YYYY-MM-DD — resolves parquet from results/tec_data/ automatically")
    parser.add_argument("--out",         default=None,
                        help="Output PNG path (default: results/keo_image/keogram_<date>.png)")
    parser.add_argument("--lon-span",    nargs=2, type=float, default=list(DEFAULT_LON_SPAN),
                        metavar=("LON_MIN", "LON_MAX"),
                        help="Longitude range to include")
    parser.add_argument("--lat-range",   nargs=2, type=float, default=list(KEO_LAT_RANGE),
                        metavar=("LAT_MIN", "LAT_MAX"),
                        help="Latitude range for the keogram")
    parser.add_argument("--min-ele",     type=float, default=MIN_ELEVATION,
                        help="Minimum satellite elevation cutoff [deg]")
    parser.add_argument("--clim",        type=float, default=KEO_DTEC_CLIM,
                        help="Colour scale ±limit [TECu]")
    parser.add_argument("--gauss-sigma", type=float, default=KEO_GAUSS_SIGMA,
                        help="Gaussian smoothing sigma [pixels]; 0 = no smoothing")
    parser.add_argument("--lat-step",    type=float, default=KEO_LAT_STEP,
                        help="Latitude grid step [deg]")
    parser.add_argument("--time-step",   type=int,   default=KEO_TIME_STEP,
                        help="Time grid step [seconds]")
    parser.add_argument("--exclude-sv",  nargs="+",  default=None, metavar="SV",
                        help="Drop these SV codes before plotting (e.g. R09)")
    parser.add_argument("--save-grid",   action="store_true",
                        help="Also save the 2-D grid as a parquet file")
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
    out_path = Path(args.out) if args.out else _KEO_DIR / f"keogram_{date_tag}.png"
    _KEO_DIR.mkdir(parents=True, exist_ok=True)

    grid_out_path = None
    if args.save_grid:
        _MAT_DIR.mkdir(parents=True, exist_ok=True)
        grid_out_path = _MAT_DIR / f"keogram_grid_{date_tag}.parquet"

    print(f"Reading {parquet_path} ...")
    df = pl.read_parquet(parquet_path)

    if args.exclude_sv:
        n_before = len(df)
        df = df.filter(~pl.col("sv").is_in(args.exclude_sv))
        print(f"Excluded SV {args.exclude_sv}: dropped {n_before - len(df):,} rows")

    plot_keogram(
        df            = df,
        out_path      = out_path,
        input_date    = input_date,
        lon_span      = tuple(args.lon_span),
        lat_range     = tuple(args.lat_range),
        min_ele       = args.min_ele,
        dtec_clim     = args.clim,
        gauss_sigma   = args.gauss_sigma,
        lat_step      = args.lat_step,
        time_step     = args.time_step,
        grid_out_path = grid_out_path,
    )
    print(f"Saved: {out_path}")
