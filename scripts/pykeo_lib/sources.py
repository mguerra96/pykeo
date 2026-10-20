"""
Observation sources: where the RINEX obs files for one day come from.

The method (pipeline.run) only needs decompressed RINEX obs files on disk.
A source is any function with this signature:

    get_obs(day: date, stations: list[str] | None, obs_dir: Path) -> list[Path]

It puts the obs files for `day` into `obs_dir` and returns their paths,
decompressed (.rnx / .yyO), with standard RINEX 2 or 3 file names: the first
4 characters are the station ID, and the name holds the date ("2024172" in
RINEX 3, "1720.24" in RINEX 2). `stations` limits the download to those IDs;
None means every station the source has. Raise an exception when the source
cannot be reached, so the day fails and is retried instead of being saved
with missing stations; return [] when the source simply has no data.

obs_dir is a scratch cache: the orchestrator deletes files there once a day
no longer needs them, so a source must never point it at your own archive.

Three examples:
  obs_euref  public EUREF EPN archive, works for anyone
  obs_local  files you already have in another folder (copied, never modified)
  obs_ingv   INGV gnssgiving networks, reachable only from the INGV network
Copy one of them to add your own data provider.
"""

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from .constants import FTP_HOST, GNSSGIVING_HOSTS
from .decompress import can_decompress, decompress, expected_final_path
from .download import (
    _date_to_year_doy,
    _download_parallel,
    _filter_obs_filenames,
    _list_ftp_dir,
    _list_ftp_dir_failover,
)
from .stations import load_station_networks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EUREF EPN (public)
# ---------------------------------------------------------------------------

def obs_euref(
    day: date,
    stations: list[str] | None,
    obs_dir: Path,
    max_workers: int = 12,
) -> list[Path]:
    """Download one day of 30 s obs from the public EUREF EPN archive (~400 European stations)."""
    year, doy = _date_to_year_doy(day)
    obs_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"/pub/obs/{year}/{doy:03d}"

    all_files = _list_ftp_dir(FTP_HOST, remote_dir)
    if all_files is None:
        raise ConnectionError(f"EUREF unreachable: {FTP_HOST}{remote_dir}")

    stations_upper = {s.upper() for s in stations} if stations else None
    # _GO files are GPS-only extras next to some stations' mixed (_MO) file;
    # keeping both would load the same station twice.
    targets = [f for f in _filter_obs_filenames(all_files, stations_upper) if "_GO." not in f]
    if not targets:
        logger.warning(f"No EUREF obs for {day} ({remote_dir}).")
        return []
    return _download_parallel(targets, remote_dir, obs_dir, FTP_HOST, max_workers)


# ---------------------------------------------------------------------------
# Local folder
# ---------------------------------------------------------------------------

def obs_local(
    day: date,
    stations: list[str] | None,
    obs_dir: Path,
    src_dir: Path = Path("my_obs"),
) -> list[Path]:
    """
    Use obs files you already have in `src_dir` (one flat folder, any mix of
    RINEX 2/3, compressed .gz/.Z/Hatanaka or not). Matching files are copied
    into obs_dir and decompressed there; `src_dir` is never modified.
    """
    year, doy = _date_to_year_doy(day)
    obs_dir.mkdir(parents=True, exist_ok=True)
    tags = (f"{year}{doy:03d}", f"{doy:03d}0.{year % 100:02d}")
    stations_upper = {s.upper() for s in stations} if stations else None

    found: list[Path] = []
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or not any(t in f.name for t in tags):
            continue
        if stations_upper is not None and f.name[:4].upper() not in stations_upper:
            continue
        if not can_decompress(f.name):
            continue
        final = expected_final_path(f.name, obs_dir)
        if not final.exists():
            final = decompress(Path(shutil.copy2(f, obs_dir / f.name)))
        if final is not None:
            found.append(final)
    if not found:
        logger.warning(f"No local obs for {day} in {src_dir}.")
    return found


# ---------------------------------------------------------------------------
# INGV gnssgiving (INGV network only)
# ---------------------------------------------------------------------------

# Max obs networks to list/download concurrently. Listings are latency-bound,
# so several in parallel hides per-network round-trips; the per-network download
# fan-out is scaled down so total simultaneous FTP connections stay bounded.
MAX_CONCURRENT_NETWORKS = 6


def obs_ingv(
    day: date,
    stations: list[str] | None,
    obs_dir: Path,
    max_workers: int = 12,
    work_dir: Path = Path("."),
) -> list[Path]:
    """
    Download observation files for one day from INGV gnssgiving networks.

    Uses network/station_lists/station_networks_{year}.json (built by
    build_station_list.py) to map station IDs to network paths. On Dec 31 /
    Jan 1 it falls back to the adjacent year's JSON, since D-1 or D+1 may fall
    in a year that has none. Raises FileNotFoundError if no JSON exists.
    """
    year, doy = _date_to_year_doy(day)
    obs_dir.mkdir(parents=True, exist_ok=True)
    stations_upper = {s.upper() for s in stations} if stations else None

    fallback_year = {(12, 31): year + 1, (1, 1): year - 1}.get((day.month, day.day))
    sta_net_map = load_station_networks(year, work_dir, fallback_year=fallback_year)
    if sta_net_map is None:
        json_path = work_dir / "network" / "station_lists" / f"station_networks_{year}.json"
        raise FileNotFoundError(
            f"station_networks_{year}.json not found at {json_path}. "
            f"Run: python build_station_list.py --date {year}-06-21"
        )

    if stations_upper is not None:
        needed_networks = {net for sta, net in sta_net_map.items() if sta in stations_upper}
    else:
        needed_networks = set(sta_net_map.values())

    if not needed_networks:
        logger.debug("[gnssgiving] no matching networks for the requested stations — skipping")
        return []

    networks_to_query = {net: f"/CONTINUOUS/30s/{net}" for net in needed_networks}
    logger.debug(
        f"[gnssgiving] station_networks_{year}.json: "
        f"{len(networks_to_query)} network(s) → {sorted(networks_to_query)}"
    )

    def _fetch_network(network: str, base_path: str, per_net_workers: int) -> list[Path]:
        remote_dir = f"{base_path}/{year}/{doy:03d}"
        logger.debug(f"[{network}] listing {remote_dir} (hosts: {GNSSGIVING_HOSTS})")
        all_files, host = _list_ftp_dir_failover(GNSSGIVING_HOSTS, remote_dir)
        if all_files is None:
            raise ConnectionError(f"no host reachable for {remote_dir}")
        if not all_files:
            logger.warning(f"No data for {network} for DOY {doy:03d} ({day}).")
            return []

        targets = _filter_obs_filenames(all_files, stations_upper)
        if not targets:
            logger.debug(f"[{network}] No matching files for {day}.")
            return []

        logger.debug(f"[{network}] {len(targets)} files to download from {host}")
        return _download_parallel(targets, remote_dir, obs_dir, host, per_net_workers)

    # Process networks concurrently: each does a serial FTP listing then a
    # parallel per-file download, so overlapping networks hides their listing
    # latency and keeps more transfers in flight. FTP listings are cheap and
    # latency-bound, so we allow several networks to run at once, while keeping
    # the per-network download fan-out so the total simultaneous connections
    # (n_net_workers * per_net_workers) stays near `max_workers` and the server
    # doesn't throttle or refuse us.
    downloaded: list[Path] = []
    failed: list[str] = []
    n_net_workers   = min(len(networks_to_query), MAX_CONCURRENT_NETWORKS)
    per_net_workers = max(1, max_workers // n_net_workers)
    with ThreadPoolExecutor(max_workers=n_net_workers) as pool:
        futures = {
            pool.submit(_fetch_network, net, base, per_net_workers): net
            for net, base in networks_to_query.items()
        }
        for fut in as_completed(futures):
            net = futures[fut]
            try:
                downloaded.extend(fut.result())
            except Exception as e:
                logger.warning(f"[{net}] network download failed: {e}")
                failed.append(net)

    # Fail the day rather than save it with missing networks: a failed day has no
    # keogram, so the next run retries it. Files already downloaded stay cached.
    if failed:
        raise ConnectionError(f"{day}: network(s) unreachable: {sorted(failed)}")
    return downloaded
