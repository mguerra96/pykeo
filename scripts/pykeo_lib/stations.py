"""
Station list management.

Loads the station→network JSON produced by build_station_list.py, with
optional cross-year fallback for boundary days (Dec 31 / Jan 1).
If the JSON is missing entirely, invokes build_station_list.py as a
subprocess to create it automatically.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _json_path(year: int, work_dir: Path) -> Path:
    return work_dir / "network" / "station_lists" / f"station_networks_{year}.json"


def _load_json(year: int, work_dir: Path) -> dict[str, str] | None:
    p = _json_path(year, work_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not read {p}: {e}")
        return None


def load_station_networks(
    year: int,
    work_dir: Path,
    fallback_year: int | None = None,
) -> dict[str, str] | None:
    """
    Return station→network mapping for *year*.

    Falls back to *fallback_year* if the primary JSON is missing and a
    fallback year is supplied (used for boundary days: D-1 of Jan 1 and
    D+1 of Dec 31).  Returns None only if neither file exists.
    """
    mapping = _load_json(year, work_dir)
    if mapping is not None:
        return mapping
    if fallback_year is not None and fallback_year != year:
        logger.info(
            f"station_networks_{year}.json missing — "
            f"using fallback from {fallback_year}"
        )
        return _load_json(fallback_year, work_dir)
    return None


def ensure_station_networks(
    year: int,
    work_dir: Path,
    n_stations: int = 300,
) -> dict[str, str]:
    """
    Return station→network mapping for *year*, auto-building it if missing.

    Invokes build_station_list.py --date <year>-06-21 --n-stations <n_stations>
    as a subprocess when the JSON does not exist.  Raises RuntimeError if the
    file is still missing after the subprocess completes.
    """
    mapping = _load_json(year, work_dir)
    if mapping is not None:
        return mapping

    json_p = _json_path(year, work_dir)
    build_script = Path(__file__).parent.parent / "build_station_list.py"
    cmd = [
        sys.executable, str(build_script),
        "--date", f"{year}-06-21",
        "--n-stations", str(n_stations),
        "--workers", "12",
    ]
    logger.warning(
        f"{json_p.name} not found — auto-building station network.\n"
        f"  Running: {' '.join(cmd)}\n"
        f"  This may take several minutes. "
        f"To run manually: python {build_script} --date {year}-06-21 --n-stations {n_stations}"
    )
    result = subprocess.run(cmd, cwd=str(work_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"build_station_list.py failed (exit {result.returncode}). "
            f"Please run it manually: python {build_script} --date {year}-06-21 --n-stations {n_stations}"
        )

    mapping = _load_json(year, work_dir)
    if mapping is None:
        raise RuntimeError(
            f"build_station_list.py completed but {json_p} was not created."
        )
    logger.info(f"Station networks loaded: {len(mapping)} stations")
    return mapping


def station_ids_from_networks(mapping: dict[str, str]) -> list[str]:
    """Return sorted station IDs from a station→network mapping."""
    return sorted(mapping.keys())
