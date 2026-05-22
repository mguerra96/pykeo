# pykeo — GNSS TEC Pipeline

GNSS-based Total Electron Content (TEC) calibration and keogram generation for European networks (EUREF, RING, NOA, SWEPOS, ASG-EUPOS).

---

## Directory structure

```
pykeo/                          ← project root
│
├── scripts/                    ← all Python scripts
│   ├── orchestrator.py         ← sole CLI entry point
│   ├── build_station_list.py   ← one-time station network builder
│   └── pykeo_lib/              ← internal library package
│       ├── constants.py
│       ├── decompress.py
│       ├── stations.py
│       ├── download.py
│       ├── pipeline.py
│       └── keogram.py
│
├── tec_data/                   ← raw GNSS data (created at runtime)
│   ├── obs/                    ← RINEX obs files (rolling 3-day window)
│   ├── nav/                    ← NAV/BRDC files (kept for the whole run)
│   ├── errors/                 ← obs files that caused processing errors
│   └── _tmp/                   ← per-station temp parquets (created and deleted each day)
│
├── results/                    ← pipeline outputs (created at runtime)
│   ├── tec_data/               ← tec_<date>.parquet  (full calibrated output)
│   ├── keo_image/              ← keogram_<date>.png
│   └── keo_mat/                ← keogram_grid_<date>.parquet  (2-D grid for re-plotting)
│
└── network/                    ← station metadata
    ├── station_rnx/            ← RINEX obs downloaded by build_station_list.py
    ├── station_map/            ← stations_<year>.png  (spatial coverage map)
    └── station_lists/          ← station_networks_<year>.json  (station→network mapping)
```

---

## Scripts

### 1. `build_station_list.py` — build the station network

Downloads one day of RINEX obs from all configured networks, reads receiver
coordinates from the RINEX headers, optionally sub-samples stations for spatial
coverage using farthest-point sampling, and writes a station→network JSON ready
for `orchestrator.py`.

**Networks:** EUREF (`www.epncb.oma.be`), RING, NOA, SWEPOS, ASG-EUPOS (`mga.int.ingv.it`, falling back to `gnssgiving.int.ingv.it`)

**Decompression chain** applied to every downloaded file:
`.Z` → unlzw3 (or gzip if mislabelled) → `.gz` → gzip → Hatanaka (crx2rnx.exe)

RINEX 3 is always preferred over RINEX 2 when both exist for the same station.

```
python build_station_list.py --date 2015-06-21
python build_station_list.py --date 2015-06-21 --workers 16
python build_station_list.py --date 2015-06-21 --n-stations 80   # farthest-point sampling
python build_station_list.py --date 2015-06-21 --map-only        # skip download, re-plot existing
```

| Argument | Default | Description |
|---|---|---|
| `--date` | required | Reference date (`YYYY-MM-DD`) |
| `--workers` | 8 | Parallel download threads per network |
| `--n-stations` | all | Keep N stations with best spatial coverage |
| `--map-only` | false | Skip download, just re-generate the map |

**Outputs:**
- `network/station_rnx/` — downloaded and decompressed RINEX files
- `network/station_map/stations_<year>.png` — spatial coverage map
- `network/station_lists/station_networks_<year>.json` — station→network mapping

**Station filtering:** only stations within lat 30–65°, lon 10–25° are kept;
files for out-of-bbox stations are deleted automatically.

> Note: `crx2rnx.exe` must be present at the project root for Hatanaka decompression.

If `station_networks_<year>.json` is missing when `orchestrator.py` runs, it
invokes `build_station_list.py --date <year>-06-21` automatically.

---

### 2. `orchestrator.py` — TEC calibration pipeline (single day or full year)

The sole CLI entry point. Downloads obs from gnssgiving networks and NAV (BRDC)
from EUREF/BKG, calibrates TEC per station in parallel (`ProcessPoolExecutor`),
applies Savitzky-Golay detrending, computes IPP positions, and generates keograms.

```
# Full year:
python orchestrator.py --year 2025

# Resume from a specific date:
python orchestrator.py --year 2025 --start-date 2025-06-01

# Single day:
python orchestrator.py --date 2025-06-21

# Limit to a station subset:
python orchestrator.py --year 2025 --stations-file network/station_lists/station_networks_2025.json

# Regenerate keograms from existing parquets (no downloads):
python orchestrator.py --year 2025 --keo-only
python orchestrator.py --date 2025-06-21 --keo-only

# Regenerate keograms excluding specific SVs (parquets are not modified):
python orchestrator.py --year 2017 --keo-only --exclude-sv R09
python orchestrator.py --year 2017 --keo-only --exclude-sv R09 R12
```

| Argument | Default | Description |
|---|---|---|
| `--year YYYY` | — | Process full year (Jan 1 → Dec 31); mutually exclusive with `--date` |
| `--date YYYY-MM-DD` | — | Process a single day; mutually exclusive with `--year` |
| `--work-dir` | `.` (project root) | Root folder; `tec_data/` and `results/` are created inside it |
| `--stations-file` | — | JSON (station→network) or text (one ID per line) to restrict stations |
| `--start-date` | Jan 1 | Resume year run from this date (`--year` only) |
| `--keo-only` | false | Regenerate keogram PNG + grid parquet from existing parquets — no FTP downloads |
| `--exclude-sv` | — | Drop one or more SV codes (e.g. `R09`) from the keogram only; the saved TEC parquets are not modified |

**Processing steps:**
1. Download obs for D-1, D, D+1 from gnssgiving
2. Download NAV (BRDC) for D-1, D, D+1 from EUREF and BKG
3. Parse RINEX obs + nav, precompute satellite coordinates on a common epoch grid; write to `_tmp/` so workers read from disk rather than receiving the large DataFrame through IPC pipes
4. Per station in parallel: filter GLONASS SVs with no valid channel → extract arcs → compute STEC/VTEC → Savitzky-Golay detrend → compute IPP → write result to `_tmp/`
5. Main process reads per-station parquets, concatenates, deletes temp files, saves outputs

**Skip logic (year mode):** a day is skipped only when ALL three outputs already exist:
`tec_<date>.parquet`, `keogram_<date>.png`, and `keogram_grid_<date>.parquet`.

**Obs-file lifecycle (rolling 3-day window):**
- Each day D downloads D-1, D, and D+1 obs on demand
- After processing day D: delete D-1 obs to free disk space
- NAV files cover the whole year and are never deleted

**Cross-year boundary:** Dec 31 of year Y uses the station JSON from year Y for
Jan 1 of Y+1 (D+1 boundary), and vice versa for Jan 1.

**Outputs** (relative to `--work-dir`):
- `results/tec_data/tec_<date>.parquet`
- `results/keo_image/keogram_<date>.png`
- `results/keo_mat/keogram_grid_<date>.parquet`
- `orchestrator_<year|date>.log`
- `pipeline_log_<year>.csv` — per-day summary: `date`, `n_stations`, `n_observations`, `n_valid_arcs`, `elapsed_s` (year mode only)

**Key pipeline constants:**

| Constant | Value | Meaning |
|---|---|---|
| `H_IPP` | 300 km | Ionospheric pierce point height |
| `MIN_ELEVATION` | 20° | Satellite elevation cutoff |
| `ARC_THRESHOLD_ABS` | 1 TECU | Arc validity: max absolute VTEC |
| `ARC_THRESHOLD_STD` | 10 TECU | Arc validity: max VTEC standard deviation |
| `ARC_THRESHOLD_JUMP` | 1 TECU | Arc validity: max inter-epoch jump |
| `ARC_MIN_LENGTH` | 190 epochs | Arc validity: minimum arc length |
| `SG_WINDOW` | 181 epochs | Savitzky-Golay detrending window (~90 min at 30 s) |
| `SG_ORDER` | 3 | Savitzky-Golay polynomial order |
| `DEFAULT_LON_SPAN` | (-180, 180) | Longitude range for the keogram plot |

**Output columns in the parquet:**
`epoch`, `sv`, `id_arc_valid`, `lat_ipp`, `lon_ipp`, `azi`, `ele`, `stec`, `vtec`, `vtec_detrended`

> `id_arc_valid` is `null` for rejected arcs and a non-null string for valid ones.

---

## Typical workflow

```
# 1. Build station network (once per year / region)
python build_station_list.py --date 2015-06-21 --n-stations 80

# 2a. Single day
python orchestrator.py --date 2015-06-21 --stations-file network/station_lists/station_networks_2015.json

# 2b. Full year
python orchestrator.py --year 2015 --stations-file network/station_lists/station_networks_2015.json

# 3. Re-plot keograms without re-downloading (e.g. after changing colour scale)
python orchestrator.py --year 2015 --keo-only

# 4. Re-plot excluding a noisy SV (parquets untouched)
python orchestrator.py --year 2015 --keo-only --exclude-sv R09
```

---

## Dependencies

- `pytecgg` — core GNSS TEC library (installed in `.venv` from the `develop` branch; requires Python ≥ 3.11)
- `polars` — DataFrame processing
- `numpy`, `scipy` — numerical operations and Savitzky-Golay filtering
- `matplotlib`, `cartopy` — plotting and maps
- `pymap3d` — ECEF ↔ geodetic coordinate conversion (used in `build_station_list.py`)
- `unlzw3` — LZW decompression for `.Z` files
- `tqdm` — progress bars
- `crx2rnx.exe` — Hatanaka RINEX decompressor (must be at project root)
