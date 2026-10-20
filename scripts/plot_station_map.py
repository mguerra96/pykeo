"""
Plot a map of all stations found in a directory of decompressed RINEX obs files.
Works with any data source (e.g. the folder you pass to --source local).

Usage
-----
python plot_station_map.py --obs-dir path/to/my_rinex
python plot_station_map.py --obs-dir path/to/my_rinex --out map.png --workers 16
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from tqdm import tqdm

from pykeo_lib.stations import farthest_point_sample, read_rinex_coords

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORKERS  = 8
MAP_FIGSIZE      = (16, 10)   # inches
MAP_DPI          = 150
MARKER_SIZE      = 2
MARKER_COLOR     = "crimson"

# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def read_all_coords(obs_dir: Path, workers: int) -> dict[str, tuple[float, float]]:
    files = [
        f for f in obs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".rnx", ".o")
           or (f.is_file() and len(f.suffix) == 4 and f.suffix.lower().endswith("o"))
    ]

    coords: dict[str, tuple[float, float]] = {}
    seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(read_rinex_coords, f): f for f in files}
        with tqdm(total=len(futs), desc="Reading headers", unit="file", dynamic_ncols=True, file=sys.stderr) as pbar:
            for fut in as_completed(futs):
                f = futs[fut]
                sta = f.name[:4].upper()
                pbar.update()
                if sta in seen:
                    continue
                result = fut.result()
                if result is not None:
                    seen.add(sta)
                    coords[sta] = result

    return coords


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_map(coords: dict[str, tuple[float, float]], out_path: Path) -> None:
    lats = [v[0] for v in coords.values()]
    lons = [v[1] for v in coords.values()]

    # Auto-extent with a small margin
    lat_min, lat_max = min(lats) - 5, max(lats) + 5
    lon_min, lon_max = min(lons) - 5, max(lons) + 5

    fig, ax = plt.subplots(
        figsize=MAP_FIGSIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,       facecolor="whitesmoke")
    ax.add_feature(cfeature.OCEAN,      facecolor="lightcyan")
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    ax.scatter(
        lons, lats,
        s=MARKER_SIZE ** 2,
        color=MARKER_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=f"{len(coords)} stations",
    )

    ax.set_title(f"gnssgiving stations  —  {len(coords)} total", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(coords)} stations)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Plot a station map from decompressed RINEX obs files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obs-dir",    default="gnssgiving_obs", metavar="DIR",  help="Directory with decompressed RINEX files")
    parser.add_argument("--out",        default="gnssgiving_map.png",              help="Output PNG path")
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS,         help="Parallel header-reading threads")
    parser.add_argument("--n-stations", type=int, default=None,                    help="Select N stations by farthest-point sampling")
    args = parser.parse_args()

    obs_dir = Path(args.obs_dir)
    if not obs_dir.exists():
        print(f"ERROR: {obs_dir} does not exist.")
        sys.exit(1)

    coords = read_all_coords(obs_dir, args.workers)
    if not coords:
        print("No station coordinates found.")
        sys.exit(1)

    print(f"Found {len(coords)} stations.")

    if args.n_stations is not None:
        coords = farthest_point_sample(coords, args.n_stations)
        print(f"Selected {len(coords)} stations by farthest-point sampling.")

    plot_map(coords, Path(args.out))
