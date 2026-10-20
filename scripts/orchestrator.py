"""
TEC pipeline orchestrator — yearly or single-day processing.

Usage
-----
# Process the full year 2025 (obs from the public EUREF archive, the default):
python orchestrator.py --year 2025

# Process a single day:
python orchestrator.py --date 2025-06-15

# Use your own RINEX files, or INGV networks (INGV network only):
python orchestrator.py --date 2025-06-15 --source local --local-dir path/to/my_rinex
python orchestrator.py --date 2025-06-15 --source ingv

# Resume a year run from a specific date:
python orchestrator.py --year 2025 --start-date 2025-03-01

# Limit to a station subset:
python orchestrator.py --year 2025 --stations-file network/station_lists/station_networks_2025.json

# Regenerate keograms from existing parquets (no downloads):
python orchestrator.py --year 2025 --keo-only
python orchestrator.py --date 2025-06-15 --keo-only

Outputs (relative to --work-dir)
-------
  results/tec_data/tec_<date>.parquet
  results/keo_mat/keogram_grid_<date>.parquet
  results/keo_image/keogram_<date>.png
  orchestrator_<year|date>.log
  pipeline_log_<year>.csv                         (--year mode only)
"""

import argparse
import csv
import datetime as dt
import json
import logging
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pykeo_lib.constants import CRX2RNX
from pykeo_lib.download import _date_to_year_doy
from pykeo_lib.pipeline import run
from pykeo_lib.sources import obs_euref, obs_ingv, obs_local
from pykeo_lib.stations import ensure_station_networks, station_ids_from_networks


def _obs_source(source: str, work_dir: Path, local_dir: Path | None):
    """Return the get_obs(day, stations, obs_dir) function for --source (see pykeo_lib/sources.py)."""
    if not CRX2RNX.exists():
        logging.getLogger("orchestrator").warning(
            "crx2rnx not found (PATH or project root): RINEX 2 Hatanaka (.yyD) files will be "
            "skipped. RINEX 3 works without it. Install RNXCMP to read RINEX 2 Hatanaka."
        )
    if source == "local":
        return partial(obs_local, src_dir=local_dir)
    if source == "ingv":
        return partial(obs_ingv, work_dir=work_dir)
    return obs_euref

# NOTE: plot_keogram is imported lazily inside process_day() to avoid loading
# scipy in every ProcessPoolExecutor worker process spawned by pipeline.run().


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    """Configure console + file logging for both the orchestrator and pykeo_lib loggers."""
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    # Attach handlers to the pykeo_lib package logger so download/pipeline
    # warnings propagate to the orchestrator log file.
    lib_logger = logging.getLogger("pykeo_lib")
    lib_logger.setLevel(logging.INFO)
    lib_logger.addHandler(console_handler)
    lib_logger.addHandler(file_handler)

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Obs-file helpers
# ---------------------------------------------------------------------------

def _delete_obs_day(obs_dir: Path, target_date: date, logger: logging.Logger) -> None:
    """Delete all obs files for *target_date*, identified by year+DOY in their filename."""
    year, doy = _date_to_year_doy(target_date)
    year2 = year % 100
    patterns = [f"*{year}{doy:03d}*", f"*{doy:03d}0.{year2:02d}[oOnN]"]
    deleted = sum(
        1
        for pat in patterns
        for f in obs_dir.glob(pat)
        if f.is_file() and (f.unlink(), True)[1]
    )
    if deleted:
        logger.info(f"  Deleted {deleted} obs file(s) for {target_date} (DOY {doy:03d})")


def _cleanup_obs_dir(obs_dir: Path, start_date: date, logger: logging.Logger) -> None:
    """Delete obs files outside the D-1/D/D+1 window of *start_date*."""
    keep_dates = {
        start_date - dt.timedelta(days=1),
        start_date,
        start_date + dt.timedelta(days=1),
    }
    keep_tags = set()
    for d in keep_dates:
        year, doy = _date_to_year_doy(d)
        year2 = year % 100
        keep_tags.add(f"{year}{doy:03d}")
        keep_tags.add(f"{doy:03d}0.{year2:02d}")

    deleted = 0
    for f in obs_dir.iterdir():
        if not f.is_file():
            continue
        if not any(tag in f.name for tag in keep_tags):
            f.unlink()
            deleted += 1

    if deleted:
        logger.info(
            f"Startup cleanup: deleted {deleted} obs file(s) outside "
            f"D-1/D/D+1 window of {start_date}"
        )


# ---------------------------------------------------------------------------
# CSV progress log
# ---------------------------------------------------------------------------

def _append_csv_row(csv_path: Path, row: dict) -> None:
    """Append one row to a CSV progress log, writing the header on first call."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single-day processing
# ---------------------------------------------------------------------------

def _output_dirs(work_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create and return (obs, tec parquet, keogram PNG, keogram grid) folders under work_dir."""
    dirs = (
        work_dir / "tec_data" / "obs",
        work_dir / "results" / "tec_data",
        work_dir / "results" / "keo_image",
        work_dir / "results" / "keo_mat",
    )
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _keogram_exists(keo_dir: Path, date_str: str) -> bool:
    """A day counts as done once its keogram PNG exists (it is written last)."""
    return (keo_dir / f"keogram_{date_str}.png").exists()


def process_day(
    target_date: date,
    work_dir: Path,
    tec_dir: Path,
    keo_dir: Path,
    mat_dir: Path,
    stations: list[str] | None,
    logger: logging.Logger,
    csv_path: Path | None = None,
    keo_only: bool = False,
    n_stations_target: int | None = None,
    exclude_sv: list[str] | None = None,
    get_obs=obs_euref,
) -> bool:
    """
    Run the full pipeline (or keogram-only) for one calendar day.

    Returns True on success, False if the pipeline raised an exception or
    produced an empty result.
    """
    date_str     = target_date.isoformat()
    out_parquet  = tec_dir / f"tec_{date_str}.parquet"
    out_keogram  = keo_dir / f"keogram_{date_str}.png"
    out_keo_grid = mat_dir / f"keogram_grid_{date_str}.parquet"

    if keo_only:
        if not out_parquet.exists():
            logger.warning(f"[KEO-ONLY] {date_str} — parquet missing, skipping.")
            return False
        logger.info(f"[KEO-ONLY] Regenerating keogram for {date_str}")
        import polars as pl
        from pykeo_lib.keogram import plot_keogram
        df = pl.read_parquet(out_parquet)
        if exclude_sv:
            n_before = len(df)
            df = df.filter(~pl.col("sv").is_in(exclude_sv))
            logger.info(f"  Excluded SV {exclude_sv}: dropped {n_before - len(df):,} rows")
        plot_keogram(df, out_keogram, input_date=target_date, grid_out_path=out_keo_grid)
        return True

    logger.info("=" * 60)
    logger.info(f"Processing {date_str}")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    try:
        obs_dir = work_dir / "tec_data" / "obs"
        window  = [target_date + dt.timedelta(days=k) for k in (-1, 0, 1)]
        obs_files = [p for d in window for p in get_obs(d, stations, obs_dir)]
        logger.info(f"  Obs ready in {time.perf_counter() - t_start:.0f}s: {len(obs_files)} file(s)")
        df = run(
            input_date=target_date,
            obs_files=obs_files,
            work_dir=work_dir,
            n_stations_target=n_stations_target,
        )
    except Exception as e:
        logger.error(f"  Pipeline failed for {date_str}: {e}")
        return False

    elapsed = time.perf_counter() - t_start

    if df.is_empty():
        logger.warning(f"  No data for {date_str} — skipping outputs.")
        return False

    n_stations     = df["id_arc_valid"].str.split("_").list.get(0).n_unique()
    n_observations = len(df)
    n_valid_arcs   = df["id_arc_valid"].drop_nulls().n_unique()
    logger.info(
        f"  Done in {elapsed:.0f}s — "
        f"{n_observations:,} rows, {n_stations} stations, {n_valid_arcs} valid arcs"
    )

    df.write_parquet(out_parquet, compression="zstd")
    logger.info(f"  Saved: {out_parquet.name}")

    from pykeo_lib.keogram import plot_keogram
    plot_keogram(df, out_keogram, input_date=target_date, grid_out_path=out_keo_grid)

    if csv_path is not None:
        _append_csv_row(csv_path, {
            "date":           date_str,
            "n_stations":     n_stations,
            "n_observations": n_observations,
            "n_valid_arcs":   n_valid_arcs,
            "elapsed_s":      round(elapsed, 1),
        })

    return True


# ---------------------------------------------------------------------------
# Year-mode orchestrator loop
# ---------------------------------------------------------------------------

def orchestrate_year(
    year: int,
    work_dir: Path,
    stations: list[str] | None,
    start_date: date | None = None,
    keo_only: bool = False,
    n_stations: int = 300,
    exclude_sv: list[str] | None = None,
    source: str = "euref",
    local_dir: Path | None = None,
) -> None:
    """Process all days in a year sequentially, with rolling obs cleanup."""
    log_path = work_dir / f"orchestrator_{year}.log"
    csv_path = work_dir / f"pipeline_log_{year}.csv"
    logger   = _setup_logging(log_path)

    obs_dir, tec_dir, keo_dir, mat_dir = _output_dirs(work_dir)

    # INGV source only: auto-build the station→network JSON if needed
    if not keo_only and source == "ingv":
        mapping  = ensure_station_networks(year, work_dir, n_stations=n_stations)
        stations = station_ids_from_networks(mapping) if stations is None else stations
    get_obs = _obs_source(source, work_dir, local_dir)

    first_day = start_date or date(year, 1, 1)
    last_day  = date(year, 12, 31)
    days = [
        first_day + dt.timedelta(days=i)
        for i in range((last_day - first_day).days + 1)
    ]

    if not keo_only:
        _cleanup_obs_dir(obs_dir, first_day, logger)

    logger.info(f"Orchestrator started — year={year}, {len(days)} days to process")

    success_count = 0
    fail_count    = 0

    for target_date in days:
        date_str  = target_date.isoformat()
        date_prev = target_date - dt.timedelta(days=1)

        if not keo_only and _keogram_exists(keo_dir, date_str):
            logger.debug(f"[SKIP] {date_str} — keogram exists.")
            _delete_obs_day(obs_dir, target_date - dt.timedelta(days=2), logger)
            success_count += 1
            continue

        ok = process_day(
            target_date=target_date,
            work_dir=work_dir,
            tec_dir=tec_dir,
            keo_dir=keo_dir,
            mat_dir=mat_dir,
            stations=stations,
            logger=logger,
            csv_path=csv_path,
            keo_only=keo_only,
            n_stations_target=n_stations if source == "ingv" else None,
            exclude_sv=exclude_sv,
            get_obs=get_obs,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

        if not keo_only:
            _delete_obs_day(obs_dir, date_prev, logger)

    logger.info("=" * 60)
    logger.info(f"Orchestrator finished — {success_count} ok, {fail_count} failed")


# ---------------------------------------------------------------------------
# Single-date mode
# ---------------------------------------------------------------------------

def orchestrate_date(
    target_date: date,
    work_dir: Path,
    stations: list[str] | None,
    keo_only: bool = False,
    n_stations: int = 300,
    exclude_sv: list[str] | None = None,
    source: str = "euref",
    local_dir: Path | None = None,
) -> None:
    """Process a single calendar day and clean up D-1/D+1 obs afterwards."""
    date_str = target_date.isoformat()
    log_path = work_dir / f"orchestrator_{date_str}.log"
    logger   = _setup_logging(log_path)

    obs_dir, tec_dir, keo_dir, mat_dir = _output_dirs(work_dir)

    if not keo_only:
        if source == "ingv":
            mapping  = ensure_station_networks(target_date.year, work_dir, n_stations=n_stations)
            stations = station_ids_from_networks(mapping) if stations is None else stations
        _cleanup_obs_dir(obs_dir, target_date, logger)

    logger.info(f"Processing single date: {date_str}")

    ok = process_day(
        target_date=target_date,
        work_dir=work_dir,
        tec_dir=tec_dir,
        keo_dir=keo_dir,
        mat_dir=mat_dir,
        stations=stations,
        logger=logger,
        csv_path=None,
        keo_only=keo_only,
        n_stations_target=n_stations if source == "ingv" else None,
        exclude_sv=exclude_sv,
        get_obs=_obs_source(source, work_dir, local_dir),
    )

    if not keo_only:
        # Clean up D-1 and D+1 obs after processing
        _delete_obs_day(obs_dir, target_date - dt.timedelta(days=1), logger)
        _delete_obs_day(obs_dir, target_date + dt.timedelta(days=1), logger)

    logger.info("Done." if ok else "Failed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="TEC pipeline orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--year", type=int, metavar="YYYY",
                            help="Process the full year (Jan 1 – Dec 31)")
    mode_group.add_argument("--date", metavar="YYYY-MM-DD",
                            help="Process a single day (D-1/D/D+1 window)")

    parser.add_argument("--source",        default="euref", choices=["euref", "local", "ingv"],
                        help="Where obs files come from: euref = public EUREF EPN archive, "
                             "local = your own files in --local-dir, "
                             "ingv = INGV gnssgiving (INGV network only)")
    parser.add_argument("--local-dir",     default=None, metavar="DIR",
                        help="Folder with your own RINEX obs files (--source local)")
    parser.add_argument("--work-dir",      default=".", metavar="DIR",
                        help="Root data folder")
    parser.add_argument("--start-date",    default=None, metavar="YYYY-MM-DD",
                        help="Resume year run from this date (--year only)")
    parser.add_argument("--stations-file", default=None, metavar="FILE",
                        help="JSON (station→network) or text file (one ID per line) "
                             "to restrict the station set; omit to use all available")
    parser.add_argument("--keo-only",      action="store_true",
                        help="Regenerate keogram PNG + grid parquet from existing "
                             "parquets — no FTP downloads")
    parser.add_argument("--n-stations",   type=int, default=300, metavar="N",
                        help="Number of stations to select when auto-building station network")
    parser.add_argument("--exclude-sv",    nargs="+", default=None, metavar="SV",
                        help="Drop these SV codes (e.g. R09) from the keogram only; "
                             "the saved TEC parquet is not modified")

    args = parser.parse_args()
    if args.source == "local" and not args.keo_only and not args.local_dir:
        parser.error("--source local needs --local-dir")
    local_dir = Path(args.local_dir) if args.local_dir else None

    work_dir = Path(args.work_dir)

    # Resolve optional station filter
    stations: list[str] | None = None
    if args.stations_file:
        p = Path(args.stations_file)
        if p.suffix.lower() == ".json":
            stations = list(json.loads(p.read_text(encoding="utf-8-sig")).keys())
        else:
            # utf-8-sig: Notepad and PowerShell may add a BOM that would corrupt the first ID
            stations = [s.strip() for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

    if args.year:
        start_date = (
            dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
            if args.start_date else None
        )
        orchestrate_year(
            year=args.year,
            work_dir=work_dir,
            stations=stations,
            start_date=start_date,
            keo_only=args.keo_only,
            n_stations=args.n_stations,
            exclude_sv=args.exclude_sv,
            source=args.source,
            local_dir=local_dir,
        )
    else:
        if args.start_date:
            parser.error("--start-date is only valid with --year")
        target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        orchestrate_date(
            target_date=target_date,
            work_dir=work_dir,
            stations=stations,
            keo_only=args.keo_only,
            n_stations=args.n_stations,
            exclude_sv=args.exclude_sv,
            source=args.source,
            local_dir=local_dir,
        )
