"""
Per-station TEC calibration pipeline: obs grouping, SG detrending, run().
"""

import datetime as dt
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
)
from .download import (
    _date_to_year_doy,
    download_obs_gnssgiving,
    ensure_nav,
    merge_nav_dicts,
)

logger = logging.getLogger(__name__)


def _savitzky_golay_detrend(
    df: pl.DataFrame,
    sg_window: int = SG_WINDOW,
    sg_order: int = SG_ORDER,
    mm_window: int = MM_WINDOW,
) -> pl.DataFrame:
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


def process_station(
    obs_paths: list[Path],
    nav_dict: dict,
    glonass_channels: dict,
    df_sat_coords: pl.DataFrame,
    epoch_filter: tuple[dt.datetime, dt.datetime],
    epoch_clip: tuple[dt.datetime, dt.datetime],
    pipeline_kwargs: dict,
) -> pl.DataFrame | None:
    """
    Calibrate TEC for one station using three daily obs files (D-1, D, D+1).

    Returns a DataFrame with COLS_TO_KEEP columns, clipped to the target day.
    Returns None if the station should be skipped (no data, short arcs, etc.).
    Returns ("error", obs_paths) on unrecoverable error so the caller can log it.
    """
    label = obs_paths[0].name[:4].upper()
    logger.debug(f"Processing {label} ({len(obs_paths)} file(s))")
    warnings.filterwarnings("error", message="No valid arcs")
    try:
        frames, rec_pos, rinex_version = [], None, None
        for p in sorted(obs_paths):
            df_i, rp, rv = read_rinex_obs(str(p))
            frames.append(df_i)
            rec_pos, rinex_version = rp, rv
        df_obs = pl.concat(frames).sort("epoch")

        df_obs = df_obs.filter(
            (pl.col("epoch") >= epoch_filter[0]) & (pl.col("epoch") <= epoch_filter[1])
        )
        if df_obs.is_empty():
            return None

        rec_name = obs_paths[0].name[:4].lower()
        systems  = pipeline_kwargs.get("systems", SYSTEMS)
        ctx = GNSSContext(
            receiver_pos=rec_pos,
            receiver_name=rec_name,
            rinex_version=rinex_version,
            h_ipp=pipeline_kwargs.get("h_ipp", H_IPP),
            systems=systems,
        )
        ctx.glonass_channels.update(glonass_channels)

        df_lc   = calculate_linear_combinations(df_obs, ctx=ctx)
        df_arcs = extract_arcs(
            df=df_lc,
            ctx=ctx,
            threshold_abs=pipeline_kwargs.get("arc_threshold_abs", ARC_THRESHOLD_ABS),
            threshold_std=pipeline_kwargs.get("arc_threshold_std", ARC_THRESHOLD_STD),
            threshold_jump=pipeline_kwargs.get("arc_threshold_jump", ARC_THRESHOLD_JUMP),
            min_arc_length=pipeline_kwargs.get("arc_min_length", ARC_MIN_LENGTH),
            max_gap=pipeline_kwargs.get("arc_max_gap", ARC_MAX_GAP),
        )

        df_coords = df_sat_coords.join(df_arcs.select(["sv", "epoch"]), on=["sv", "epoch"], how="inner")
        df_geom   = df_arcs.join(df_coords, on=["sv", "epoch"], how="left")
        df_final  = calculate_ipp(
            df_geom, ctx=ctx,
            min_elevation=pipeline_kwargs.get("min_elevation", MIN_ELEVATION),
        )
        df_tec = calculate_tec(df_final, ctx=ctx)

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
        return df_out

    except BaseException as e:
        logger.error(f"Error on {label}: {e}")
        return ("error", obs_paths)


def _group_obs_by_station(obs_files: list[Path]) -> dict[str, list[Path]]:
    """
    Group obs files by 4-char station code.
    When both RINEX 2 and RINEX 3 are present for the same station, keep only RINEX 3.
    """
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
    stations: list[str] | None = None,
    work_dir: Path = Path("."),
    pipeline_kwargs: dict | None = None,
    fallback_year: int | None = None,
) -> pl.DataFrame:
    """
    Run the full TEC calibration pipeline for one day.

    Downloads obs (D-1, D, D+1) from gnssgiving, downloads NAV files from
    EUREF/BKG, calibrates all stations in parallel, and returns a merged
    DataFrame.  Returns an empty DataFrame if no data could be obtained.

    fallback_year is forwarded to download_obs_gnssgiving for cross-year
    boundary days (D-1 of Jan 1 or D+1 of Dec 31).
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    obs_dir = work_dir / "tec_data" / "obs"
    nav_dir = work_dir / "tec_data" / "nav"
    obs_dir.mkdir(parents=True, exist_ok=True)
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

    logger.info(f"{'='*60}")
    logger.info(f"Downloading obs for {date_prev} / {input_date} / {date_next}")
    logger.info(f"{'='*60}")

    # Determine per-day fallback years for cross-year boundary days
    def _fallback_for(d: date) -> int | None:
        if d.month == 12 and d.day == 31:
            return input_date.year  # D+1 of Dec 31 → use primary year JSON
        if d.month == 1 and d.day == 1:
            return input_date.year  # D-1 of Jan 1 → use primary year JSON
        return None

    all_obs: list[Path] = []
    for d in (date_prev, input_date, date_next):
        all_obs += download_obs_gnssgiving(
            d, stations, obs_dir,
            work_dir=work_dir,
            fallback_year=_fallback_for(d),
        )

    year, doy = _date_to_year_doy(input_date)
    target_ydoy_long  = f"{year}{doy:03d}"           # RINEX 3: STAT...20160010000...
    target_ydoy_short = f"{doy:03d}0.{year % 100:02d}"  # RINEX 2: xxxx0010.16o
    obs_today = [
        f for f in all_obs
        if target_ydoy_long in f.name or target_ydoy_short in f.name
    ]
    if not obs_today:
        logger.error("No obs files found for the target date. Aborting.")
        return pl.DataFrame()

    station_files = _group_obs_by_station(all_obs)
    logger.info(f"Stations with obs: {len(station_files)}")

    logger.debug("Downloading NAV for D-1, D, D+1...")
    nav_dicts = []
    for d in (date_prev, input_date, date_next):
        y, doy = _date_to_year_doy(d)
        nav_files = ensure_nav(y, doy, nav_dir)
        if nav_files:
            nav_dicts.append(read_rinex_nav(str(nav_files[0])))
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
        systems=pipeline_kwargs.get("systems", SYSTEMS),
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

    station_groups = list(station_files.values())
    n_workers      = min(12, len(station_groups))
    errors_dir     = work_dir / "tec_data" / "errors"
    error_log      = errors_dir / f"errors_{input_date}.txt"

    logger.info(f"Calibrating {len(station_groups)} stations with {n_workers} workers...")
    frames: list[pl.DataFrame] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                process_station,
                paths, nav_dict, glonass_channels,
                df_sat_coords, epoch_filter, epoch_clip,
                dict(pipeline_kwargs),
            ): paths
            for paths in station_groups
        }
        with tqdm(total=len(futures), desc="Calibrating", unit="station") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if isinstance(result, tuple) and result[0] == "error":
                    errors_dir.mkdir(parents=True, exist_ok=True)
                    with error_log.open("a") as fh:
                        for p in result[1]:
                            p.unlink(missing_ok=True)
                            fh.write(p.name + "\n")
                            logger.info(f"[deleted, logged] {p.name}")
                elif result is not None and len(result) > 0:
                    frames.append(result)
                pbar.update()

    if not frames:
        logger.error("No data produced after calibration.")
        return pl.DataFrame()

    merged = pl.concat(frames).sort(["epoch", "sv"])
    logger.info(f"Pipeline done: {len(merged):,} rows from {len(frames)} stations")
    return merged
