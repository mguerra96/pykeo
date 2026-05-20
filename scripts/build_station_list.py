"""
build_station_list.py — build the GNSS station network for the pykeo pipeline.

Downloads one day of RINEX obs from ALL gnssgiving networks, reads receiver
coordinates from the RINEX headers, filters by bounding box, sub-samples
stations for spatial coverage, and saves a station list + network map ready
for tec_pipeline / orchestrator.

Also writes a station→network JSON so the orchestrator only connects to the
networks that actually contain the selected stations (avoids iterating all
99 networks per day).

Networks
--------
  All networks under gnssgiving.int.ingv.it /CONTINUOUS/30s/

Decompression chain applied to every downloaded file:
  .Z   → unlzw3 (LZW) or gzip if the file is mislabelled
  .gz  → gzip (or Unix-compress if magic bytes are \x1f\x9d)
  .crx / .??d → crx2rnx.exe (Hatanaka RINEX decompressor)

Usage
-----
python build_station_list.py --date 2015-06-21
python build_station_list.py --date 2015-06-21 --workers 16
python build_station_list.py --date 2015-06-21 --n-stations 300
python build_station_list.py --date 2015-06-21 --map-only

Outputs
-------
  network/station_rnx/                        downloaded (decompressed) RINEX files
  network/station_map/stations_<year>.png     spatial coverage map
  network/station_lists/station_networks_<year>.json  station→network mapping

Note: crx2rnx.exe must be present in the project root directory.
"""

import argparse
import ftplib
import gzip
import io
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pymap3d
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "network" / "station_rnx"
CRX2RNX    = Path(__file__).parent.parent / "crx2rnx.exe"

GNSSGIVING_HOSTS = ("mga.int.ingv.it", "gnssgiving.int.ingv.it")
GNSSGIVING_BASE  = "/CONTINUOUS/30s"

# Bounding box for station filtering
LAT_MIN, LAT_MAX = 35.0, 60.0
LON_MIN, LON_MAX =  0.0, 20.0

# Default spatial sub-sample
DEFAULT_N_STATIONS = 300

FTP_TIMEOUT_LIST     = 120
FTP_TIMEOUT_DOWNLOAD = 120

MAP_PANEL_HEIGHT = 9
MAP_DPI          = 150
DEFAULT_WORKERS  = 8


# ---------------------------------------------------------------------------
# Decompression helpers
# ---------------------------------------------------------------------------

def _unlzw_safe(data: bytes) -> bytes | None:
    """Run unlzw3 in a subprocess so a C-level crash doesn't kill the main process."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from unlzw3 import unlzw; sys.stdout.buffer.write(unlzw(sys.stdin.buffer.read()))"],
            input=data, capture_output=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return None
    except Exception:
        return None


def _is_hatanaka_rnx2(p: Path) -> bool:
    return len(p.suffix) == 4 and p.suffix.upper().endswith("D")


def _is_hatanaka_rnx3(p: Path) -> bool:
    return p.suffix.lower() == ".crx"


def _run_crx2rnx(p: Path) -> Path | None:
    if not CRX2RNX.exists():
        log.warning("crx2rnx.exe not found — skipping Hatanaka decompression")
        return p
    try:
        subprocess.run([str(CRX2RNX), str(p)], capture_output=True, check=False)
        out = (
            p.with_suffix(p.suffix[:-1] + "O")
            if _is_hatanaka_rnx2(p)
            else p.with_suffix(".rnx")
        )
        if out.exists():
            p.unlink(missing_ok=True)
            return out
        log.warning(f"crx2rnx produced no output for {p.name}")
    except Exception as e:
        log.warning(f"crx2rnx error on {p.name}: {e}")
    return None


def decompress_and_dehatanaka(local_path: Path) -> Path | None:
    p = local_path

    if p.suffix == ".Z":
        out = p.with_suffix("")
        raw = p.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            try:
                out.write_bytes(gzip.open(io.BytesIO(raw)).read())
                p.unlink(missing_ok=True)
                p = out
            except Exception as e:
                log.warning(f"gunzip(.Z) failed for {local_path.name}: {e}")
                return None
        else:
            decompressed = _unlzw_safe(raw)
            if decompressed is None:
                log.warning(f"unlzw failed for {local_path.name}")
                return None
            out.write_bytes(decompressed)
            p.unlink(missing_ok=True)
            p = out

    if p.suffix == ".gz":
        out = p.with_suffix("")
        raw = p.read_bytes()
        try:
            if raw[:2] == b"\x1f\x9d":
                decompressed = _unlzw_safe(raw)
                if decompressed is None:
                    log.warning(f"unlzw(.gz) failed for {local_path.name}")
                    return None
                out.write_bytes(decompressed)
            else:
                out.write_bytes(gzip.open(io.BytesIO(raw)).read())
            p.unlink(missing_ok=True)
            p = out
        except Exception as e:
            log.warning(f"decompress(.gz) failed for {local_path.name}: {e}")
            return None

    if _is_hatanaka_rnx2(p) or _is_hatanaka_rnx3(p):
        result = _run_crx2rnx(p)
        if result is None:
            return None
        p = result

    return p


# ---------------------------------------------------------------------------
# FTP helpers
# ---------------------------------------------------------------------------

def _list_ftp_dir(host: str, remote_dir: str) -> list[str]:
    try:
        with ftplib.FTP(host, timeout=FTP_TIMEOUT_LIST) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            return [Path(f).name for f in ftp.nlst()]
    except Exception as e:
        log.debug(f"FTP listing failed {host}{remote_dir}: {e}")
        return []


def _list_ftp_dir_failover(remote_dir: str) -> tuple[list[str], str | None]:
    """Try GNSSGIVING_HOSTS in order; return (files, host_that_worked) or ([], None)."""
    for host in GNSSGIVING_HOSTS:
        files = _list_ftp_dir(host, remote_dir)
        if files:
            return files, host
        log.warning(f"[failover] {host}{remote_dir} empty/unreachable — trying next host")
    return [], None


def _list_networks() -> list[str]:
    files, _ = _list_ftp_dir_failover(GNSSGIVING_BASE)
    return files


def _is_obs_file(fname: str) -> bool:
    stem = fname
    for ext in (".gz", ".Z"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
    inner_ext = Path(stem).suffix.lower()
    if inner_ext in (".crx", ".rnx"):
        return True
    if len(inner_ext) == 4 and inner_ext[-1] in "od":
        return True
    return False


def _expected_final_path(fname: str, local_dir: Path) -> Path:
    stem = fname
    for ext in (".Z", ".gz"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
    stem_path = local_dir / stem
    if _is_hatanaka_rnx2(stem_path):
        return stem_path.with_suffix(stem_path.suffix[:-1] + "O")
    if _is_hatanaka_rnx3(stem_path):
        return stem_path.with_suffix(".rnx")
    return stem_path


def _download_one(fname: str, remote_dir: str, local_dir: Path, host: str) -> Path | None:
    final_path = _expected_final_path(fname, local_dir)
    if final_path.exists():
        return final_path
    raw_path = local_dir / fname
    try:
        with ftplib.FTP(host, timeout=FTP_TIMEOUT_DOWNLOAD) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            with open(raw_path, "wb") as f:
                ftp.retrbinary(f"RETR {fname}", f.write)
    except Exception as e:
        log.warning(f"[fail] {fname}: {e}")
        raw_path.unlink(missing_ok=True)
        return None
    return decompress_and_dehatanaka(raw_path)


# ---------------------------------------------------------------------------
# Per-network download
# ---------------------------------------------------------------------------

def download_network(network: str, remote_dir: str, local_dir: Path, workers: int) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    files, host = _list_ftp_dir_failover(remote_dir)
    obs_files = [f for f in files if _is_obs_file(f)]
    if not obs_files:
        return []

    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, f, remote_dir, local_dir, host): f for f in obs_files}
        with tqdm(total=len(futs), desc=f"{network:12s}", unit="file", leave=True) as pbar:
            for fut in as_completed(futs):
                result = fut.result()
                if result is not None:
                    downloaded.append(result)
                pbar.update()
    return downloaded


# ---------------------------------------------------------------------------
# Coordinate extraction (header-only, no pytecgg)
# ---------------------------------------------------------------------------

def _read_coords(f: Path) -> tuple[float, float] | None:
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
                    return float(lat), float(lon)
    except Exception:
        pass
    return None


def collect_station_coords(obs_dir: Path) -> dict[str, tuple[float, float]]:
    all_files = [
        p for p in obs_dir.iterdir()
        if p.is_file() and (
            p.suffix.lower() == ".rnx"
            or (len(p.suffix) == 4 and p.suffix[1:3].isdigit() and p.suffix[-1].lower() == "o")
        )
    ]

    seen:   set[str]                       = set()
    coords: dict[str, tuple[float, float]] = {}

    print(f"Reading RINEX headers from {len(all_files)} files...")
    with tqdm(total=len(all_files), unit="file") as pbar:
        for f in sorted(all_files):
            sta = f.name[:4].upper()
            pbar.update()
            if sta in seen:
                continue
            result = _read_coords(f)
            if result is None:
                continue
            seen.add(sta)
            coords[sta] = result

    return coords


# ---------------------------------------------------------------------------
# Station filtering and spatial sub-sampling
# ---------------------------------------------------------------------------

def filter_stations_by_bbox(
    coords: dict[str, tuple[float, float]],
    obs_dir: Path,
) -> dict[str, tuple[float, float]]:
    inside: dict[str, tuple[float, float]] = {}
    for sta, (lat, lon) in coords.items():
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            inside[sta] = (lat, lon)
        else:
            for f in (
                list(obs_dir.glob(f"{sta.lower()}*"))
                + list(obs_dir.glob(f"{sta.upper()}*"))
            ):
                f.unlink(missing_ok=True)

    removed = len(coords) - len(inside)
    if removed:
        print(
            f"Removed {removed} stations outside bbox "
            f"(lat {LAT_MIN}–{LAT_MAX}, lon {LON_MIN}–{LON_MAX}) "
            f"→ {len(inside)} remaining"
        )
    return inside


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
# Station → network mapping
# ---------------------------------------------------------------------------

def build_station_network_map(obs_dir: Path, networks_downloaded: dict[str, list[Path]]) -> dict[str, str]:
    """Return {station_id: network_name} for all downloaded files."""
    sta_to_net: dict[str, str] = {}
    for network, paths in networks_downloaded.items():
        for p in paths:
            sta = p.name[:4].upper()
            sta_to_net.setdefault(sta, network)
    return sta_to_net


# ---------------------------------------------------------------------------
# Station map
# ---------------------------------------------------------------------------

def _add_map_features(ax) -> None:
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d8eaf5", zorder=0)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")


def plot_station_map(
    coords_all: dict[str, tuple[float, float]],
    coords_sel: dict[str, tuple[float, float]] | None,
    out_path: Path,
) -> None:
    if not coords_all:
        print("No station coordinates found — map not generated.")
        return

    all_lats = np.array([v[0] for v in coords_all.values()])
    all_lons = np.array([v[1] for v in coords_all.values()])
    extent   = [
        all_lons.min() - 1, all_lons.max() + 1,
        all_lats.min() - 1, all_lats.max() + 1,
    ]

    lon_range = extent[1] - extent[0]
    lat_range = extent[3] - extent[2]
    panel_w   = MAP_PANEL_HEIGHT * (lon_range / lat_range)
    n_panels  = 2 if coords_sel is not None else 1

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels, MAP_PANEL_HEIGHT),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.05)

    ax = axes[0, 0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    _add_map_features(ax)
    ax.scatter(all_lons, all_lats, s=4, color="black",
               transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(f"{len(coords_all)} stations (all in bbox)", fontsize=13, fontweight="bold")

    if coords_sel is not None:
        sel_lats = np.array([v[0] for v in coords_sel.values()])
        sel_lons = np.array([v[1] for v in coords_sel.values()])
        dropped  = set(coords_all) - set(coords_sel)

        ax2 = axes[0, 1]
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        _add_map_features(ax2)

        if dropped:
            drop_lats = np.array([coords_all[s][0] for s in dropped])
            drop_lons = np.array([coords_all[s][1] for s in dropped])
            ax2.scatter(drop_lons, drop_lats, s=4, color="#bbbbbb",
                        transform=ccrs.PlateCarree(), zorder=2, label="dropped")

        ax2.scatter(sel_lons, sel_lats, s=4, color="crimson",
                    transform=ccrs.PlateCarree(), zorder=3, label="selected")
        ax2.legend(loc="lower left", fontsize=8, framealpha=0.7)
        ax2.set_title(f"{len(coords_sel)} stations (selected)", fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Map saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all gnssgiving obs and build a station list for tec_pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date",        required=True,                        help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS,  help="Parallel download threads per network")
    parser.add_argument("--n-stations",  type=int,   default=DEFAULT_N_STATIONS, help="Keep N stations by farthest-point sampling")
    parser.add_argument("--map-only",    action="store_true",                  help="Skip download, re-plot existing files")
    args = parser.parse_args()

    target:   date = datetime.strptime(args.date, "%Y-%m-%d").date()
    year, doy      = target.year, target.timetuple().tm_yday
    map_dir        = _PROJECT_ROOT / "network" / "station_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_path       = map_dir / f"stations_{year}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    networks_downloaded: dict[str, list[Path]] = {}

    if not args.map_only:
        print(f"Date: {target}  DOY={doy:03d}  ->  {OUTPUT_DIR.resolve()}\n")

        # Clean download directory
        existing = [f for f in OUTPUT_DIR.iterdir() if f.is_file()]
        for f in existing:
            f.unlink(missing_ok=True)
        if existing:
            print(f"Cleaned {len(existing)} existing file(s) from {OUTPUT_DIR}\n")

        print(f"Listing gnssgiving networks...")
        networks = _list_networks()
        if not networks:
            print("ERROR: Could not list networks.")
            return
        print(f"Found {len(networks)} networks.\n")

        for net in networks:
            remote_dir = f"{GNSSGIVING_BASE}/{net}/{year}/{doy:03d}"
            paths = download_network(net, remote_dir, OUTPUT_DIR, args.workers)
            if paths:
                networks_downloaded[net] = paths
                tqdm.write(f"  {net}: {len(paths)} files")

        total = sum(len(v) for v in networks_downloaded.values())
        print(f"\nTotal: {total} files from {len(networks_downloaded)} networks.\n")
    else:
        # In map-only mode we can't know which network each file came from
        networks_downloaded = {}

    # Read coordinates, filter bbox, sub-sample
    coords     = collect_station_coords(OUTPUT_DIR)
    coords     = filter_stations_by_bbox(coords, OUTPUT_DIR)
    coords_sel = farthest_point_sample(coords, args.n_stations)
    print(f"Selected {len(coords_sel)} stations by farthest-point sampling.")

    plot_station_map(coords, coords_sel, map_path)

    # Write station→network mapping (used by orchestrator to skip unused networks)
    list_dir = _PROJECT_ROOT / "network" / "station_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    if networks_downloaded:
        sta_to_net  = build_station_network_map(OUTPUT_DIR, networks_downloaded)
        sel_mapping = {sta: sta_to_net[sta] for sta in coords_sel if sta in sta_to_net}
        net_map_path = list_dir / f"station_networks_{year}.json"
        net_map_path.write_text(json.dumps(sel_mapping, indent=2), encoding="utf-8")
        used_nets = sorted(set(sel_mapping.values()))
        print(f"Network map saved: {net_map_path}  ({len(used_nets)} networks: {', '.join(used_nets)})")


if __name__ == "__main__":
    main()
