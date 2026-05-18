"""
Plot a map of all stations found in a directory of decompressed RINEX obs files.

Usage
-----
python plot_station_map.py --obs-dir gnssgiving_obs
python plot_station_map.py --obs-dir gnssgiving_obs --out map.png --workers 16
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pymap3d
from tqdm import tqdm

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

def _read_coords(f: Path) -> tuple[float, float] | None:
    """Parse APPROX POSITION XYZ from RINEX header and return (lat, lon), or None."""
    try:
        with f.open("r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                if "END OF HEADER" in line:
                    break
                if "APPROX POSITION XYZ" in line:
                    x, y, z = float(line[0:14]), float(line[14:28]), float(line[28:42])
                    if x == 0.0 and y == 0.0 and z == 0.0:
                        return None
                    lat, lon, _ = pymap3d.ecef2geodetic(x, y, z)
                    lat, lon = float(lat), float(lon)
                    if not (35.0 <= lat <= 60.0 and 0.0 <= lon <= 20.0):
                        return None
                    return lat, lon
    except Exception:
        pass
    return None


def read_all_coords(obs_dir: Path, workers: int) -> dict[str, tuple[float, float]]:
    files = [
        f for f in obs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".rnx", ".o")
           or (f.is_file() and len(f.suffix) == 4 and f.suffix.lower().endswith("o"))
    ]

    coords: dict[str, tuple[float, float]] = {}
    seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_read_coords, f): f for f in files}
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
# Spatial sub-sampling
# ---------------------------------------------------------------------------

def farthest_point_sample(
    coords: dict[str, tuple[float, float]],
    n: int,
) -> dict[str, tuple[float, float]]:
    if n <= 0 or n >= len(coords):
        return coords
    names = list(coords.keys())
    pts   = np.array([coords[s] for s in names])
    selected  = [0]
    min_dists = np.full(len(pts), np.inf)
    for _ in range(n - 1):
        last      = pts[selected[-1]]
        dists     = np.sqrt(((pts - last) ** 2).sum(axis=1))
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))
    return {names[i]: coords[names[i]] for i in selected}


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
