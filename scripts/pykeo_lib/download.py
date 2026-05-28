"""
FTP download helpers for RINEX observation and navigation files.

Obs: gnssgiving FTP (CONTINUOUS/30s networks).
Nav: CDDIS multi-GNSS BRDC (IGS combined / DLR merged), with EUREF and BKG
     fallback.
"""

import ftplib
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import date
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .constants import (
    CDDIS_HOST,
    CDDIS_NAV_PRODUCTS,
    CDDIS_TIMEOUT,
    FTP_HOST,
    FTP_TIMEOUT_DOWNLOAD,
    FTP_TIMEOUT_LIST,
    FTP_TIMEOUT_TRANSFER,
    GNSSGIVING_HOSTS,
)
from .decompress import decompress, expected_final_path
from .stations import load_station_networks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level FTP helpers
# ---------------------------------------------------------------------------

def _date_to_year_doy(d: date) -> tuple[int, int]:
    """Return (year, day-of-year) for a given date."""
    return d.year, d.timetuple().tm_yday


def _list_ftp_dir(host: str, remote_dir: str) -> list[str]:
    """List filenames in an FTP directory; returns empty list on any error."""
    try:
        with ftplib.FTP(host, timeout=FTP_TIMEOUT_LIST) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            return [Path(f).name for f in ftp.nlst()]
    except ftplib.error_perm as e:
        logger.debug(f"FTP listing failed {host}{remote_dir}: {e}")
        return []
    except Exception as e:
        logger.debug(f"FTP listing failed {host}{remote_dir}: {e}")
        return []


def _list_ftp_dir_failover(hosts: tuple[str, ...], remote_dir: str) -> tuple[list[str], str | None]:
    """Try each host in order; return (files, host_that_worked) or ([], None)."""
    for host in hosts:
        files = _list_ftp_dir(host, remote_dir)
        if files:
            return files, host
        logger.debug(f"[failover] {host}{remote_dir} empty/unreachable — trying next host")
    return [], None


def _download_one(fname: str, remote_dir: str, local_dir: Path, host: str = FTP_HOST) -> Path | None:
    final_path = expected_final_path(fname, local_dir)
    if final_path.exists() and final_path.stat().st_size > 0:
        logger.debug(f"[skip] {fname} (already exists)")
        return final_path
    final_path.unlink(missing_ok=True)  # remove zero-byte leftover before re-downloading

    raw_path = local_dir / fname
    try:
        with ftplib.FTP(host, timeout=FTP_TIMEOUT_DOWNLOAD) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            with open(raw_path, "wb") as f:
                ftp.retrbinary(f"RETR {fname}", f.write)
    except Exception as e:
        logger.warning(f"[fail download] {fname}: {e}")
        raw_path.unlink(missing_ok=True)
        return None

    result = decompress(raw_path)
    if result:
        logger.debug(f"[ok] {fname} -> {result.name}")
    return result


def _filter_obs_filenames(
    all_files: list[str],
    stations_upper: set[str] | None,
) -> list[str]:
    """Return compressed obs files (.gz/.Z) optionally filtered to a station subset."""
    return [
        f for f in all_files
        if (f.endswith(".gz") or f.endswith(".Z"))
        and (stations_upper is None or f[:4].upper() in stations_upper)
    ]


def _download_parallel(
    targets: list[str],
    remote_dir: str,
    local_dir: Path,
    host: str,
    max_workers: int,
) -> list[Path]:
    """Download a list of files from an FTP host in parallel; returns successfully downloaded paths."""
    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {
            pool.submit(_download_one, fname, remote_dir, local_dir, host): fname
            for fname in targets
        }
        with tqdm(total=len(futures), unit="file", leave=False) as bar:
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    result = fut.result(timeout=FTP_TIMEOUT_TRANSFER)
                except TimeoutError:
                    logger.warning(f"[timeout] {fname} — download exceeded {FTP_TIMEOUT_TRANSFER}s, skipping")
                    fut.cancel()
                    result = None
                except Exception as e:
                    logger.warning(f"[error] {fname}: {e}")
                    result = None
                if result is not None:
                    downloaded.append(result)
                bar.update(1)
    return downloaded


# ---------------------------------------------------------------------------
# Obs downloader (public)
# ---------------------------------------------------------------------------

def download_obs_gnssgiving(
    input_date: date,
    stations: list[str] | None,
    local_dir: Path,
    max_workers: int = 12,
    work_dir: Path = Path("."),
    fallback_year: int | None = None,
) -> list[Path]:
    """
    Download observation files for one day from gnssgiving networks.

    Uses station_networks_{year}.json to map station IDs to network paths.
    Falls back to fallback_year JSON when input_date is a cross-year boundary
    day (D-1 of Jan 1 or D+1 of Dec 31) and the primary year JSON is absent.
    Raises FileNotFoundError if neither JSON exists.
    """
    year, doy = _date_to_year_doy(input_date)
    local_dir.mkdir(parents=True, exist_ok=True)
    stations_upper = {s.upper() for s in stations} if stations else None

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

    downloaded: list[Path] = []
    for network, base_path in networks_to_query.items():
        remote_dir = f"{base_path}/{year}/{doy:03d}"
        logger.debug(f"[{network}] listing {remote_dir} (hosts: {GNSSGIVING_HOSTS})")
        all_files, host = _list_ftp_dir_failover(GNSSGIVING_HOSTS, remote_dir)
        if not all_files:
            logger.warning(f"No data for {network} for DOY {doy:03d} ({input_date}).")
            continue

        targets = _filter_obs_filenames(all_files, stations_upper)
        if not targets:
            logger.debug(f"[{network}] No matching files for {input_date}.")
            continue

        logger.debug(f"[{network}] {len(targets)} files to download from {host}")
        downloaded.extend(
            _download_parallel(targets, remote_dir, local_dir, host, max_workers)
        )

    return downloaded


# ---------------------------------------------------------------------------
# Nav downloaders (public)
# ---------------------------------------------------------------------------

def _nav_priority(path: Path) -> int:
    """Source preference for a NAV file: lower sorts first.

    Downstream (pipeline.run) reads only nav_files[0] per day, so the preferred
    multi-GNSS product must sort ahead of the EUREF/GOP one. CDDIS products are
    ordered by their position in CDDIS_NAV_PRODUCTS; everything else trails.
    """
    for i, prod in enumerate(CDDIS_NAV_PRODUCTS):
        if prod in path.name:
            return i
    return len(CDDIS_NAV_PRODUCTS)


def _get_local_nav_files(nav_dir: Path, year: int, doy: int) -> list[Path]:
    """Return already-downloaded NAV files for a given year/DOY (RINEX 2 and 3 patterns).

    Sorted by source preference so the best multi-GNSS product is first.
    """
    yy = str(year)[-2:]
    files = list(nav_dir.glob(f"*BRDC*{year}*{doy:03d}*"))
    files.extend(nav_dir.glob(f"*BRDM*{year}*{doy:03d}*"))
    files.extend(nav_dir.glob(f"brdc{doy:03d}0.{yy}*"))
    return sorted(set(files), key=_nav_priority)


def _download_nav_cddis(year: int, doy: int, nav_dir: Path) -> list[Path]:
    """Download the preferred multi-GNSS BRDC product from CDDIS (anonymous FTPS).

    Tries each product in CDDIS_NAV_PRODUCTS order and stops at the first that
    downloads successfully. CDDIS allows anonymous FTPS login (no Earthdata
    credentials needed for this archive path).
    """
    yy = str(year)[-2:]
    remote_dir = f"/gnss/data/daily/{year}/{doy:03d}/{yy}p/"
    nav_dir.mkdir(parents=True, exist_ok=True)

    try:
        ftp = ftplib.FTP_TLS(CDDIS_HOST, timeout=CDDIS_TIMEOUT)
        ftp.login()
        ftp.prot_p()
        ftp.cwd(remote_dir)
    except Exception as e:
        logger.warning(f"NAV CDDIS connect/cwd error {year}/{doy:03d}: {e}")
        return []

    downloaded: list[Path] = []
    try:
        for prod in CDDIS_NAV_PRODUCTS:
            fname = f"{prod}_{year}{doy:03d}0000_01D_MN.rnx.gz"
            local_path = nav_dir / fname
            if local_path.exists():
                downloaded.append(local_path)
                break
            try:
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
                logger.debug(f"[nav cddis] {fname}")
                downloaded.append(local_path)
                break
            except Exception as e:
                logger.debug(f"[nav cddis miss] {fname}: {e}")
                local_path.unlink(missing_ok=True)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    if not downloaded:
        logger.warning(f"No CDDIS NAV product available for {year} DOY {doy:03d}.")
    return downloaded


def _download_nav_euref(year: int, doy: int, nav_dir: Path) -> list[Path]:
    """Download BRDC NAV file for a given year/DOY from the EUREF EPN FTP."""
    remote_dir = f"/pub/obs/BRDC/{year}"
    nav_dir.mkdir(parents=True, exist_ok=True)
    try:
        with ftplib.FTP(FTP_HOST, timeout=FTP_TIMEOUT_LIST) as ftp:
            ftp.login()
            ftp.cwd(remote_dir)
            all_files = [Path(f).name for f in ftp.nlst()]
    except ftplib.error_perm as e:
        logger.warning(f"NAV EUREF FTP error {year}/BRDC: {e}")
        return []

    targets = [f for f in all_files if f"{year}{doy:03d}" in f and f.endswith(".gz")]
    if not targets:
        logger.warning(f"No NAV file on EUREF for {year} DOY {doy:03d}.")
        return []

    downloaded = []
    for fname in targets:
        local_path = nav_dir / fname
        if local_path.exists():
            downloaded.append(local_path)
            continue
        try:
            with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
                ftp.login()
                ftp.cwd(remote_dir)
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
            logger.debug(f"[nav] {fname}")
            downloaded.append(local_path)
        except Exception as e:
            logger.warning(f"[nav fail] {fname}: {e}")
    return downloaded


def ensure_nav(year: int, doy: int, nav_dir: Path) -> list[Path]:
    """
    Return local NAV files for a given year/DOY, downloading if needed.

    Source preference: CDDIS multi-GNSS (IGS combined / DLR merged) first, then
    EUREF EPN, then BKG. CDDIS products avoid the cross-PRN BeiDou-3
    mis-attribution present in the EUREF/GOP product. pipeline.run reads only
    the first returned file, and _get_local_nav_files orders by source
    preference, so a cached CDDIS file always wins over a cached GOP one.
    """
    nav_files = _get_local_nav_files(nav_dir, year, doy)
    # Attempt CDDIS unless a CDDIS product is already cached. A cached GOP file
    # alone must not short-circuit the preferred source.
    have_cddis = bool(nav_files) and any(
        prod in nav_files[0].name for prod in CDDIS_NAV_PRODUCTS
    )
    if not have_cddis:
        logger.debug(f"Downloading preferred NAV from CDDIS for {year}/DOY {doy}...")
        _download_nav_cddis(year, doy, nav_dir)
        nav_files = _get_local_nav_files(nav_dir, year, doy)
    if not nav_files:
        logger.debug(f"CDDIS NAV unavailable — trying EUREF for {year}/DOY {doy}...")
        _download_nav_euref(year, doy, nav_dir)
        nav_files = _get_local_nav_files(nav_dir, year, doy)
    if not nav_files:
        logger.debug(f"EUREF NAV unavailable — trying BKG fallback for {year}/DOY {doy}...")
        from download_nav_bkg import download_nav_bkg  # project-level helper
        download_nav_bkg(year=year, doys=[doy], output_path=nav_dir)
        nav_files = _get_local_nav_files(nav_dir, year, doy)
    if not nav_files:
        logger.error(f"Could not obtain NAV for {year} DOY {doy}.")
    return nav_files


def _sanitize_glonass(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop corrupt GLONASS broadcast records.

    The 3rd header field (parsed as `clock_drift_rate`) is the message frame
    time. Legitimate values are at most a few × 10^5 seconds; uninitialized
    records carry the ~2^32 sentinel (≈ 4.29e9). The empirical gap between
    valid and sentinel is ~4 × 10^9, so 10^8 is a safe threshold.
    """
    if df.is_empty() or "clock_drift_rate" not in df.columns:
        return df
    SENTINEL_THRESHOLD = 1e8
    n_in = len(df)
    df = df.filter(
        (pl.col("clock_drift_rate").abs() < SENTINEL_THRESHOLD)
        | pl.col("clock_drift_rate").is_null()
    )
    n_dropped = n_in - len(df)
    if n_dropped:
        logger.warning(f"GLONASS NAV sanitizer dropped {n_dropped} corrupt record(s)")
    return df


def merge_nav_dicts(nav_dicts: list[dict]) -> dict:
    """Merge multiple navigation dictionaries keyed by GNSS constellation."""
    merged: dict[str, pl.DataFrame] = {}
    for nd in nav_dicts:
        for constellation, df in nd.items():
            if constellation in merged:
                merged[constellation] = pl.concat(
                    [merged[constellation], df], how="diagonal"
                ).unique()
            else:
                merged[constellation] = df
    if "GLONASS" in merged:
        merged["GLONASS"] = _sanitize_glonass(merged["GLONASS"])
    return merged
