"""
FTP download helpers for RINEX observation and navigation files.

Obs: generic FTP helpers used by the obs sources in sources.py.
Nav: CDDIS multi-GNSS BRDC (IGS combined / DLR merged), with EUREF
     fallback. Both are public, so NAV works for every obs source.
"""

import ftplib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)
from .decompress import can_decompress, decompress, expected_final_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level FTP helpers
# ---------------------------------------------------------------------------

def _date_to_year_doy(d: date) -> tuple[int, int]:
    """Return (year, day-of-year) for a given date."""
    return d.year, d.timetuple().tm_yday


def _list_ftp_dir(host: str, remote_dir: str) -> list[str] | None:
    """List filenames in an FTP directory.

    Returns [] when the server says the directory is missing (550: no data),
    None when the server could not be asked (timeout, refused, 4xx throttling).
    """
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
        return None


def _list_ftp_dir_failover(hosts: tuple[str, ...], remote_dir: str) -> tuple[list[str] | None, str | None]:
    """Try each host in order; return (files, host_that_worked).

    files is [] if at least one host answered "no data", None if no host was reachable.
    """
    result = None
    for host in hosts:
        files = _list_ftp_dir(host, remote_dir)
        if files:
            return files, host
        if files is not None:
            result = []
        logger.debug(f"[failover] {host}{remote_dir} empty/unreachable — trying next host")
    return result, None


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
        and can_decompress(f)
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
                    result = fut.result()
                except Exception as e:
                    logger.warning(f"[error] {fname}: {e}")
                    result = None
                if result is not None:
                    downloaded.append(result)
                bar.update(1)
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
    # .part = unfinished download (see _download_nav_*), never read it
    return sorted({f for f in files if f.suffix != ".part"}, key=_nav_priority)


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
            part_path = local_path.with_name(local_path.name + ".part")
            try:
                with open(part_path, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
                part_path.replace(local_path)
                logger.debug(f"[nav cddis] {fname}")
                downloaded.append(local_path)
                break
            except Exception as e:
                logger.debug(f"[nav cddis miss] {fname}: {e}")
                part_path.unlink(missing_ok=True)
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
    except Exception as e:
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
        part_path = local_path.with_name(local_path.name + ".part")
        try:
            with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
                ftp.login()
                ftp.cwd(remote_dir)
                with open(part_path, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
            part_path.replace(local_path)
            logger.debug(f"[nav] {fname}")
            downloaded.append(local_path)
        except Exception as e:
            logger.warning(f"[nav fail] {fname}: {e}")
            part_path.unlink(missing_ok=True)
    return downloaded


def ensure_nav(year: int, doy: int, nav_dir: Path) -> list[Path]:
    """
    Return local NAV files for a given year/DOY, downloading if needed.

    Source preference: CDDIS multi-GNSS (IGS combined / DLR merged) first, then
    EUREF EPN. CDDIS products avoid the cross-PRN BeiDou-3
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
