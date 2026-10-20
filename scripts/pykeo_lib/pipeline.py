"""
Per-station TEC calibration pipeline: obs grouping, SG detrending, run().
"""

import contextlib
import datetime as dt
import gzip
import logging
import os
import re
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, TimeoutError as FuturesTimeout, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from tqdm import tqdm

from pytecgg import GNSSContext
from pytecgg.linear_combinations import calculate_linear_combinations
from pytecgg.parsing import read_rinex_nav, read_rinex_obs
from pytecgg.satellites import calculate_ipp, prepare_ephemeris, satellite_coordinates
from pytecgg.tec_calibration import calculate_tec, extract_arcs

from .constants import (
    ARC_MAX_GAP,
    ARC_MIN_LENGTH,
    ARC_THRESHOLD_ABS,
    ARC_THRESHOLD_JUMP,
    ARC_THRESHOLD_STD,
    COLS_TO_KEEP,
    H_IPP,
    MIN_ELEVATION,
    MM_WINDOW,
    SG_ORDER,
    SG_WINDOW,
    SYSTEMS,
    VTEC_JUMP_THRESHOLD,
)
from .download import (
    _date_to_year_doy,
    ensure_nav,
    merge_nav_dicts,
)

logger = logging.getLogger(__name__)

# Max wall-clock to wait for the *next* worker to finish during calibration.
# A healthy station takes a few seconds; this only trips if a worker has wedged
# (e.g. a native-parser hang), so the run can abandon it and merge partials
# instead of freezing the whole year. Override via PYKEO_STATION_TIMEOUT.
STATION_TIMEOUT_S = float(os.environ.get("PYKEO_STATION_TIMEOUT", 600))

# How many times to recreate a broken ProcessPool and retry the stations that
# didn't finish. A worker crash (native-parser segfault / OOM) breaks the whole
# pool and fails every pending future; retrying the survivors recovers them. The
# last attempt runs single-worker so one poison-pill station can't take others
# down with it. Override via PYKEO_POOL_RETRIES.
MAX_POOL_RETRIES = int(os.environ.get("PYKEO_POOL_RETRIES", 3))


def _level_vtec_jumps(df: pl.DataFrame, threshold: float = VTEC_JUMP_THRESHOLD) -> pl.DataFrame:
    """Per-arc: when |vtec[t+1] - vtec[t]| > threshold, shift the post-jump segment by -diff so it aligns with the pre-jump level."""
    if df.is_empty() or "vtec" not in df.columns or "id_arc_valid" not in df.columns:
        return df
    df = df.sort(["id_arc_valid", "epoch"])
    diff = pl.col("vtec").diff().over("id_arc_valid")
    jump = pl.when(diff.abs() > threshold).then(diff).otherwise(0.0)
    correction = jump.fill_null(0.0).cum_sum().over("id_arc_valid")
    return df.with_columns((pl.col("vtec") - correction).alias("vtec"))


def _savitzky_golay_detrend(
    df: pl.DataFrame,
    sg_window: int = SG_WINDOW,
    sg_order: int = SG_ORDER,
    mm_window: int = MM_WINDOW,
) -> pl.DataFrame:
    """Apply Savitzky-Golay detrending + moving-mean smoothing to each arc; adds vtec_detrended column."""
    results = []
    for (arc_id,), group in df.group_by(["id_arc_valid"]):
        if len(group) < sg_window:
            continue
        group = group.sort("epoch")
        vtec_vals = group["vtec"].to_numpy()
        if not np.isfinite(vtec_vals).all():
            continue
        trend    = savgol_filter(vtec_vals, sg_window, sg_order)
        residual = vtec_vals - trend
        kernel   = np.ones(mm_window) / mm_window
        smoothed = np.convolve(residual, kernel, mode="same")
        results.append(group.with_columns(pl.Series("vtec_detrended", smoothed)))

    if not results:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("vtec_detrended"))
    return pl.concat(results)


@contextlib.contextmanager
def _suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def _strip_leap_seconds(p: Path) -> Path:
    raw = p.read_bytes()
    is_gz = raw[:2] == b"\x1f\x8b"
    text = gzip.decompress(raw).decode("latin-1") if is_gz else raw.decode("latin-1")
    cleaned = "\n".join(
        line for line in text.splitlines()
        if "LEAP SECONDS" not in line
    ) + "\n"
    data = gzip.compress(cleaned.encode("latin-1")) if is_gz else cleaned.encode("latin-1")
    suffix = ".rnx.gz" if is_gz else ".rnx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _read_rinex_obs_safe(p: Path):
    """Call read_rinex_obs, retrying once after stripping a malformed LEAP SECONDS header."""
    try:
        with _suppress_stderr():
            return read_rinex_obs(str(p))
    except BaseException as exc:
        if type(exc).__name__ != "PanicException":
            raise

    tmp_path = _strip_leap_seconds(p)
    try:
        with _suppress_stderr():
            result = read_rinex_obs(str(tmp_path))
        logger.warning(f"{p.name}: malformed LEAP SECONDS header stripped and re-parsed successfully")
        return result
    except BaseException as exc2:
        if type(exc2).__name__ != "PanicException":
            raise
        raise RuntimeError(
            f"{p.name}: RINEX obs parser failed even after stripping LEAP SECONDS header"
        ) from exc2
    finally:
        tmp_path.unlink(missing_ok=True)


# Matches the epoch date field of a RINEX-3 NAV record:
#   "G01 2017 01 00 06 00 00 ..."  ->  groups (sv-prefix, YYYY, MM, DD)
# Some IGS BRDC files write the midnight-boundary day as "DD = 00" (e.g.
# "2017 01 00" meaning the instant rolling out of 2016-12-31), which is not a
# valid Gregorian date and panics the Rust parser. We roll such records back to
# the real previous day, preserving the fixed-width "YYYY MM DD" layout.
_NAV_DAY_ZERO_RE = re.compile(r"^([A-Z]\d{2} )(\d{4}) (\d{2}) 00( )")


def _fix_nav_day_zero(p: Path) -> Path:
    """Rewrite NAV epoch records with an invalid 'day 00' to the previous day."""
    raw = p.read_bytes()
    is_gz = raw[:2] == b"\x1f\x8b"
    text = gzip.decompress(raw).decode("latin-1") if is_gz else raw.decode("latin-1")

    def _repl(m: "re.Match") -> str:
        prefix, year, month, tail = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        real = dt.date(year, month, 1) - dt.timedelta(days=1)  # day-0 of (y,m) → last day of prev month
        return f"{prefix}{real.year:04d} {real.month:02d} {real.day:02d}{tail}"

    cleaned = "\n".join(_NAV_DAY_ZERO_RE.sub(_repl, line) for line in text.splitlines()) + "\n"
    data = gzip.compress(cleaned.encode("latin-1")) if is_gz else cleaned.encode("latin-1")
    suffix = ".rnx.gz" if is_gz else ".rnx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _read_rinex_nav_safe(p: Path):
    """
    Call read_rinex_nav, retrying once after fixing 'day 00' epoch dates that
    panic the Rust parser ("invalid Gregorian date"). Returns None if the file
    still cannot be parsed; one bad NAV file out of D-1/D/D+1 is non-fatal, the
    caller proceeds as long as at least one parsed.
    """
    try:
        with _suppress_stderr():
            return read_rinex_nav(str(p))
    except BaseException as exc:
        if type(exc).__name__ != "PanicException":
            raise

    tmp_path = _fix_nav_day_zero(p)
    try:
        with _suppress_stderr():
            result = read_rinex_nav(str(tmp_path))
        logger.warning(f"{p.name}: fixed invalid 'day 00' NAV epochs and re-parsed successfully")
        return result
    except BaseException as exc2:
        if type(exc2).__name__ != "PanicException":
            raise
        logger.warning(
            f"{p.name}: RINEX nav parser still panicked after day-00 fix ({exc2}); skipping this NAV file"
        )
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def process_station(
    obs_paths: list[Path],
    glonass_channels: dict,
    sat_coords_path: Path,
    epoch_filter: tuple[dt.datetime, dt.datetime],
    epoch_clip: tuple[dt.datetime, dt.datetime],
    out_dir: Path | None = None,
) -> Path | tuple | None:
    """
    Calibrate TEC for one station using three daily obs files (D-1, D, D+1).

    Writes result to a temp parquet in out_dir and returns the Path.
    Returns None if the station should be skipped (no data, short arcs, etc.).
    Returns ("error", obs_paths) on unrecoverable error so the caller can log it.
    """
    label = obs_paths[0].name[:4].upper()
    logger.debug(f"Processing {label} ({len(obs_paths)} file(s))")
    warnings.filterwarnings("error", message="No valid arcs")
    try:
        frames, rec_pos, rinex_version = [], None, None
        for p in sorted(obs_paths):
            if not p.exists():
                logger.warning(f"[{label}] obs file missing on disk, skipping: {p.name}")
                continue
            df_i, rp, rv = _read_rinex_obs_safe(p)
            frames.append(df_i)
            rec_pos, rinex_version = rp, rv
        if not frames:
            return None
        df_obs = pl.concat(frames).sort("epoch")

        df_obs = df_obs.filter(
            (pl.col("epoch") >= epoch_filter[0]) & (pl.col("epoch") <= epoch_filter[1])
        )
        if df_obs.is_empty():
            return None

        ctx = GNSSContext(
            receiver_pos=rec_pos,
            receiver_name=obs_paths[0].name[:4].lower(),
            rinex_version=rinex_version,
            h_ipp=H_IPP,
            systems=SYSTEMS,
        )
        ctx.glonass_channels.update(glonass_channels)

        # Drop GLONASS SVs with no valid channel (absent or null in nav data) —
        # they would survive .replace() as strings and fail the strict Float32 cast.
        known_glonass = {sv for sv, k in ctx.glonass_channels.items() if k is not None}
        df_obs = df_obs.filter(
            ~pl.col("sv").str.starts_with("R") | pl.col("sv").is_in(known_glonass)
        )

        df_lc = calculate_linear_combinations(df_obs, ctx=ctx)

        # Only this station's observed (sv, epoch) pairs are needed from the
        # shared sat-coords table. Scan lazily and inner-join so Polars prunes
        # columns/row-groups instead of every worker materializing all ~Nk rows.
        # Keys mirror df_lc exactly (no dedup) to preserve the original join's
        # row multiplicity.
        df_coords = (
            pl.scan_parquet(sat_coords_path)
            .join(df_lc.select(["sv", "epoch"]).lazy(), on=["sv", "epoch"], how="inner")
            .collect()
        )
        df_geom   = df_lc.join(df_coords, on=["sv", "epoch"], how="left")
        df_ipp    = calculate_ipp(df_geom, ctx=ctx, min_elevation=MIN_ELEVATION)

        df_arcs = extract_arcs(
            df=df_ipp,
            ctx=ctx,
            threshold_abs=ARC_THRESHOLD_ABS,
            threshold_std=ARC_THRESHOLD_STD,
            threshold_jump=ARC_THRESHOLD_JUMP,
            min_arc_length=ARC_MIN_LENGTH,
            max_gap=ARC_MAX_GAP,
        )

        df_tec = calculate_tec(df_arcs, ctx=ctx)

        df_tec = _level_vtec_jumps(df_tec)

        df_out = _savitzky_golay_detrend(df_tec)

        df_out = df_out.filter(
            (pl.col("epoch") >= epoch_clip[0]) & (pl.col("epoch") <= epoch_clip[1])
        )

        available = [c for c in COLS_TO_KEEP if c in df_out.columns]
        df_out = (
            df_out.select(available)
            .with_columns(pl.col(pl.Float32, pl.Float64).round(4))
            .sort(["epoch", "sv"])
        )

        logger.debug(f"  -> {len(df_out):,} rows for {label}")
        # Date in the name: an abandoned worker finishing late must not overwrite the next day's file.
        tmp = (out_dir or Path(".")) / f"_tec_{epoch_clip[0]:%Y-%m-%d}_{label}.parquet"
        df_out.write_parquet(tmp)
        return tmp

    except BaseException as e:
        if "No valid arcs" in str(e):  # normal outcome, raised early by the filter above
            logger.debug(f"{label}: no valid arcs — skipped")
            return None
        logger.error(f"Error on {label}: {e}")
        return ("error", obs_paths)


def _drop_jumpy_svs(df_sat_coords: pl.DataFrame, max_step_m: float = 500_000.0) -> pl.DataFrame:
    """
    Drop every SV whose interpolated orbit has a jump > max_step_m between
    consecutive 30 s epochs. Catches corrupt broadcast records that survive
    upstream sanitization (e.g. R13 on 2017-01-03, where a garbled 20:45 nav
    message produced ~20 000 km position discontinuities).

    Sat coords are in metres; nominal GLONASS step at 30 s sampling is ~90 km,
    so 500 km is ~5× nominal — well below any plausible discontinuity.
    """
    if df_sat_coords.is_empty():
        return df_sat_coords
    df = df_sat_coords.sort(["sv", "epoch"])
    step_sq = sum(
        (pl.col(c) - pl.col(c).shift(1).over("sv")) ** 2
        for c in ("sat_x", "sat_y", "sat_z")
    )
    flagged = (
        df.with_columns(step_sq.sqrt().alias("_step"))
        .filter(pl.col("_step").is_finite() & (pl.col("_step") > max_step_m))
        .get_column("sv")
        .unique()
        .to_list()
    )
    if not flagged:
        return df_sat_coords
    logger.warning(
        f"Dropping {len(flagged)} SV(s) with orbit jumps > {max_step_m / 1000:g} km: "
        f"{sorted(flagged)}"
    )
    return df_sat_coords.filter(~pl.col("sv").is_in(flagged))


def _group_obs_by_station(obs_files: list[Path]) -> dict[str, list[Path]]:
    """Group obs files by 4-char station code; prefer RINEX 3 over RINEX 2 when both exist."""
    raw: dict[str, list[Path]] = {}
    for f in sorted(obs_files):
        raw.setdefault(f.name[:4].upper(), []).append(f)

    result: dict[str, list[Path]] = {}
    for sta, files in raw.items():
        rinex3 = [f for f in files if "_" in f.name]
        result[sta] = rinex3 if rinex3 else files
    return result


def run(
    input_date: date,
    obs_files: list[Path],
    work_dir: Path = Path("."),
    n_stations_target: int | None = None,
) -> pl.DataFrame:
    """
    Run the TEC calibration method for one day.

    obs_files: decompressed RINEX obs files for D-1, D and D+1 (see sources.py
    for how to get them). Downloads NAV files from CDDIS/EUREF, calibrates all
    stations in parallel, and returns a merged DataFrame clipped to day D.
    Returns an empty DataFrame if no data could be obtained.
    """
    nav_dir = work_dir / "tec_data" / "nav"
    nav_dir.mkdir(parents=True, exist_ok=True)

    date_prev = input_date - dt.timedelta(days=1)
    date_next = input_date + dt.timedelta(days=1)
    utc       = dt.timezone.utc

    epoch_filter = (
        dt.datetime(date_prev.year,  date_prev.month,  date_prev.day,  20, 0, 0,  tzinfo=utc),
        dt.datetime(date_next.year,  date_next.month,  date_next.day,   4, 0, 0,  tzinfo=utc),
    )
    epoch_clip = (
        dt.datetime(input_date.year, input_date.month, input_date.day,  0, 0, 0,   tzinfo=utc),
        dt.datetime(input_date.year, input_date.month, input_date.day, 23, 59, 30,  tzinfo=utc),
    )

    t_process = time.perf_counter()
    year, doy = _date_to_year_doy(input_date)
    target_ydoy_long  = f"{year}{doy:03d}"           # RINEX 3: STAT...20160010000...
    target_ydoy_short = f"{doy:03d}0.{year % 100:02d}"  # RINEX 2: xxxx0010.16o
    obs_today = [
        f for f in obs_files
        if target_ydoy_long in f.name or target_ydoy_short in f.name
    ]
    if not obs_today:
        logger.error("No obs files found for the target date. Aborting.")
        return pl.DataFrame()

    station_files = _group_obs_by_station(obs_files)
    n_avail = len(station_files)
    if n_stations_target:
        logger.warning(
            f"{n_avail} stations available out of {n_stations_target} for DOY {doy:03d} ({input_date})"
        )
    else:
        logger.warning(f"{n_avail} stations available for DOY {doy:03d} ({input_date})")

    logger.debug("Downloading NAV for D-1, D, D+1...")
    nav_dicts = []
    for d in (date_prev, input_date, date_next):
        nav_files = ensure_nav(*_date_to_year_doy(d), nav_dir)
        nav_dict = _read_rinex_nav_safe(nav_files[0]) if nav_files else None
        if nav_dict is not None:
            nav_dicts.append(nav_dict)
        elif d == input_date:
            # Neighbour-day NAV is out of validity for most of D: abort so the day is retried.
            logger.error(f"No usable NAV for the target day {d}. Aborting.")
            return pl.DataFrame()
    if not nav_dicts:
        logger.error("Could not obtain NAV files. Aborting.")
        return pl.DataFrame()

    nav_dict = merge_nav_dicts(nav_dicts)
    logger.debug(f"NAV merged: constellations = {list(nav_dict.keys())}")

    logger.debug("Preparing shared ephemeris for all stations...")
    shared_ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, 6_371_000.0),
        receiver_name="shared",
        rinex_version="3",
        systems=SYSTEMS,
    )
    ephem_dict       = prepare_ephemeris(nav_dict, ctx=shared_ctx)
    glonass_channels = dict(shared_ctx.glonass_channels)
    logger.info(f"Ephemeris ready: {len(ephem_dict)} SVs")

    logger.info("Precomputing satellite positions for all SVs × all epochs...")
    epochs_grid = [
        epoch_filter[0] + dt.timedelta(seconds=30 * i)
        for i in range(int((epoch_filter[1] - epoch_filter[0]).total_seconds() / 30) + 1)
    ]
    svs_all = list(ephem_dict.keys())
    _grid = (
        pl.DataFrame({"sv": svs_all})
        .join(pl.DataFrame({"epoch": pl.Series("epoch", epochs_grid)}), how="cross")
        .sort(["sv", "epoch"])
    )
    df_sat_coords = satellite_coordinates(_grid["sv"], _grid["epoch"], ephem_dict)
    logger.info(
        f"Sat coords computed: {len(df_sat_coords):,} rows "
        f"({len(svs_all)} SVs × {len(epochs_grid)} epochs)"
    )
    df_sat_coords = _drop_jumpy_svs(df_sat_coords)

    station_groups = list(station_files.values())
    _max_workers   = int(os.environ.get("PYKEO_WORKERS", 8))
    n_workers = min(_max_workers, len(station_groups))

    errors_dir     = work_dir / "tec_data" / "errors"
    error_log      = errors_dir / f"errors_{input_date}.txt"
    tmp_dir        = work_dir / "tec_data" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sat_coords_path = tmp_dir / f"_sat_coords_{input_date}.parquet"
    df_sat_coords.write_parquet(sat_coords_path)

    logger.info(f"Calibrating {len(station_groups)} stations with {n_workers} workers...")
    tmp_paths: list[Path] = []

    def _log_error_result(obs_paths: list[Path]) -> None:
        errors_dir.mkdir(parents=True, exist_ok=True)
        with error_log.open("a") as fh:
            for p in obs_paths:
                p.unlink(missing_ok=True)
                fh.write(p.name + "\n")
                logger.debug(f"[deleted, logged] {p.name}")

    def _handle_result(paths: list[Path], result) -> None:
        """Record one station's outcome: collect its parquet or log an error."""
        if isinstance(result, tuple) and result[0] == "error":
            _log_error_result(result[1])
        elif result is not None:
            tmp_paths.append(result)

    # Calibrate the station set, recovering from a broken pool. When a worker
    # crashes (native-parser segfault or OOM), ProcessPoolExecutor marks the
    # WHOLE pool broken and fails every still-pending future at once — which used
    # to silently discard ~all remaining stations. We handle this in two phases:
    #   1. Multi-worker passes: fast path; retry the survivors of a broken pool a
    #      few times (recovers transient/OOM-under-load crashes that don't repeat).
    #   2. Single-worker, one-station-at-a-time drain: a deterministic poison-pill
    #      station crashes only its own task, so every other station still runs.
    pending_groups = list(station_groups)
    total = len(pending_groups)

    with tqdm(total=total, desc="Calibrating", unit="station") as pbar:
        # ---- Phase 1: multi-worker passes -------------------------------------
        for attempt in range(1, MAX_POOL_RETRIES + 1):
            if not pending_groups:
                break
            if attempt > 1:
                logger.warning(
                    f"Pool broke — multi-worker retry {attempt}/{MAX_POOL_RETRIES} "
                    f"for {len(pending_groups)} unprocessed station(s)."
                )
            done_groups: list = []
            pool = ProcessPoolExecutor(max_workers=n_workers)
            try:
                futures = {
                    pool.submit(
                        process_station,
                        paths, glonass_channels,
                        sat_coords_path, epoch_filter, epoch_clip, tmp_dir,
                    ): paths
                    for paths in pending_groups
                }
                not_done = set(futures)
                try:
                    # Timeout restarts each time any station finishes (as_completed's
                    # timeout would cap the whole day instead).
                    while not_done:
                        done, not_done = wait(
                            not_done, timeout=STATION_TIMEOUT_S, return_when=FIRST_COMPLETED
                        )
                        if not done:
                            raise FuturesTimeout
                        for fut in done:
                            paths = futures[fut]
                            try:
                                result = fut.result()
                            except BrokenProcessPool:
                                raise  # abort pass; survivors stay pending for retry
                            except Exception as e:
                                logger.warning(f"Worker error (skipping station): {e}")
                                done_groups.append(paths)
                                pbar.update()
                                continue
                            done_groups.append(paths)
                            _handle_result(paths, result)
                            pbar.update()
                except FuturesTimeout:
                    n_pending = sum(1 for f in futures if not f.done())
                    logger.error(
                        f"Calibration stalled: no worker finished within "
                        f"{STATION_TIMEOUT_S:.0f}s; abandoning {n_pending} pending "
                        f"station(s) and proceeding with {len(tmp_paths)} done."
                    )
                    done_groups = list(pending_groups)  # a stall won't recover
            except Exception as e:
                logger.error(f"ProcessPoolExecutor broke (likely worker crash/OOM): {e}")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            done_ids = {id(g) for g in done_groups}
            pending_groups = [g for g in pending_groups if id(g) not in done_ids]

        # ---- Phase 2: single-worker, one station at a time --------------------
        # Each submit runs in its own fresh pool, so a crash isolates exactly that
        # station and the rest still get processed.
        if pending_groups:
            logger.warning(
                f"Draining {len(pending_groups)} remaining station(s) one at a "
                f"time to isolate the crash."
            )
        for paths in pending_groups:
            pool = ProcessPoolExecutor(max_workers=1)
            try:
                fut = pool.submit(
                    process_station,
                    paths, glonass_channels,
                    sat_coords_path, epoch_filter, epoch_clip, tmp_dir,
                )
                _handle_result(paths, fut.result(timeout=STATION_TIMEOUT_S))
            except (BrokenProcessPool, FuturesTimeout):
                logger.error(
                    f"Station {paths[0].name[:4].upper()} crashed/stalled its "
                    f"worker — skipping it."
                )
            except Exception as e:
                logger.warning(f"Worker error (skipping station): {e}")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
                pbar.update()

    if not tmp_paths:
        logger.error("No data produced after calibration (pool failure or stall).")
        sat_coords_path.unlink(missing_ok=True)
        return pl.DataFrame()

    frames = [pl.read_parquet(p) for p in tmp_paths]
    for p in tmp_paths:
        p.unlink(missing_ok=True)
    sat_coords_path.unlink(missing_ok=True)

    merged = pl.concat(frames).sort(["epoch", "sv"])
    processing_elapsed = time.perf_counter() - t_process
    logger.info(f"Pipeline done: {len(merged):,} rows from {len(frames)} stations")
    logger.info(f"Calibration completed in {processing_elapsed:.0f}s for DOY {doy:03d} ({input_date})")
    return merged
