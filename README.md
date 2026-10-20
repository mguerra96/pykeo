# pykeo — GNSS TEC calibration and keograms

pykeo turns daily RINEX observation files from a network of GNSS receivers into
calibrated vertical Total Electron Content (vTEC) and detrended TEC (dTEC) at the
ionospheric pierce points, and plots the dTEC as latitude–time keograms.

For each day D it:

1. reads the obs files of D-1, D and D+1 for every station (20:00 of D-1 to 04:00 of D+1,
   so arcs that cross midnight are complete);
2. computes satellite positions from broadcast ephemerides (GPS, Galileo, GLONASS, BeiDou)
   and drops satellites whose orbit jumps by more than 500 km between epochs;
3. computes ionospheric pierce points (IPP) at 300 km, keeping elevations ≥ 20°;
4. extracts continuous arcs and calibrates TEC per arc
   ([pytecgg](https://github.com/viventriglia/PyTECGg));
5. removes residual vTEC jumps inside each arc, then detrends each arc with a
   Savitzky-Golay filter and smooths the residual (dTEC);
6. keeps day D only, writes one parquet per day and draws the keogram.

The method does not depend on where the data comes from. Three example data
sources are included (public EUREF archive, your own files, INGV networks), and
adding your own is one function — see [Data sources](#data-sources).

---

## Install

Python ≥ 3.11.

```
pip install -r requirements.txt
```

**pytecgg.** The minimum version is the current release. Recent fixes are on its
`develop` branch until the next release; to use them (needs a Rust toolchain to build):

```
pip install git+https://github.com/viventriglia/PyTECGg.git@develop
```

**crx2rnx (optional).** Many archives distribute Hatanaka-compressed RINEX.
pytecgg reads RINEX 3 Hatanaka (`.crx`) directly, but **RINEX 2 Hatanaka (`.yyD`)
files need `crx2rnx`** from [RNXCMP](https://terras.gsi.go.jp/ja/crx2rnx.html).
pykeo looks for `crx2rnx` on your `PATH`, then for `crx2rnx.exe` at the project root
(a Windows build is included). Without it, RINEX 2 Hatanaka files are skipped with a
warning. This matters mostly for older years: in 2015, about 80% of EUREF files are RINEX 2.

## Quick start

Process one day with three EUREF stations (about 2 minutes, a few MB of downloads).
Create `stations.txt` with one station ID per line — `AQUI`, `AJAC`, `GENO` — then run:

```
python scripts/orchestrator.py --date 2024-06-20 --stations-file stations.txt
```

Outputs appear under `results/` (see [Outputs](#outputs)). Drop `--stations-file`
to use all ~400 EUREF stations.

---

## Data sources

Choose with `--source`:

| `--source` | Obs files from | Who can use it |
|---|---|---|
| `euref` (default) | public EUREF EPN archive (`www.epncb.oma.be`, ~400 European stations, 30 s) | anyone |
| `local` | your own folder of RINEX files, given with `--local-dir` | anyone |
| `ingv` | INGV gnssgiving networks (`*.int.ingv.it`) | INGV network only |

Navigation (broadcast ephemeris) files are always downloaded automatically from public
archives: CDDIS (IGS combined `BRDC00IGS_R`, or DLR merged `BRDM00DLR_S` for older
years), with EUREF as fallback. A day is skipped if no NAV file is available for D itself.

### Your own files (`--source local`)

```
python scripts/orchestrator.py --year 2024 --source local --local-dir path/to/my_rinex
```

- One flat folder; any mix of RINEX 2 and 3, compressed (`.gz`, `.Z`, Hatanaka) or not.
- Files need standard RINEX names: the first 4 characters are the station ID, and the
  name contains the date (`2024172` in RINEX 3 long names, `1720.24` in RINEX 2 names).
- The folder needs the days around the ones you process (D-1 and D+1).
- Your folder is never modified: matching files are copied into `tec_data/obs/` and
  decompressed there.

### Writing your own source

A source is one function in [scripts/pykeo_lib/sources.py](scripts/pykeo_lib/sources.py):

```python
def get_obs(day: date, stations: list[str] | None, obs_dir: Path) -> list[Path]:
    ...
```

It puts the obs files for `day` into `obs_dir` and returns their paths (decompressed,
standard RINEX names). `stations` limits the download to those IDs; `None` means all.
Raise an exception when the server cannot be reached, so the day fails and is retried
on the next run instead of being saved with missing stations; return `[]` when there is
simply no data. Copy `obs_euref` (a single FTP folder per day) or `obs_ingv` (one folder
per network) as a starting point, then add it to `_obs_source()` in
[scripts/orchestrator.py](scripts/orchestrator.py).

### INGV source

`--source ingv` maps stations to networks with
`network/station_lists/station_networks_<year>.json`. These files are included for
2015–2025. For another year, `orchestrator.py` builds one automatically by running
[scripts/build_station_list.py](scripts/build_station_list.py), which downloads one day
from every network, keeps stations inside lat 35–60°, lon 0–20°, and picks `--n-stations`
of them (default 300) spread out by farthest-point sampling.

---

## Running

```
python scripts/orchestrator.py --date 2024-06-20                       # one day
python scripts/orchestrator.py --year 2024                             # whole year
python scripts/orchestrator.py --year 2024 --start-date 2024-06-01     # resume
python scripts/orchestrator.py --year 2024 --keo-only                  # re-plot keograms only
python scripts/orchestrator.py --year 2024 --keo-only --exclude-sv R09 # re-plot without one satellite
```

| Option | Default | Meaning |
|---|---|---|
| `--year YYYY` / `--date YYYY-MM-DD` | — | process a whole year, or one day (one is required) |
| `--source` | `euref` | `euref`, `local` or `ingv` (see above) |
| `--local-dir DIR` | — | your RINEX folder, for `--source local` |
| `--work-dir DIR` | `.` | root folder for all data and outputs |
| `--stations-file FILE` | all | restrict stations: text file with one ID per line, or a JSON whose keys are IDs |
| `--start-date` | Jan 1 | resume a `--year` run from this day |
| `--keo-only` | off | redraw keograms from existing parquets, no downloads |
| `--exclude-sv SV ...` | — | drop these satellites (e.g. `R09`) from the keogram only; parquets are not changed |
| `--n-stations N` | 300 | stations to select when building an INGV station list |

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PYKEO_WORKERS` | 8 | parallel worker processes for station calibration |
| `PYKEO_STATION_TIMEOUT` | 600 | seconds without any station finishing before the day gives up on the stuck ones |
| `PYKEO_POOL_RETRIES` | 3 | retries after a worker crash, before falling back to one station at a time |

**Year runs can be stopped and restarted.** A day counts as done once its keogram PNG
exists, so a re-run skips finished days and retries failed ones (for example days where
a server could not be reached).

---

## Outputs

All paths are relative to `--work-dir`.

```
results/
├── tec_data/tec_<date>.parquet          calibrated TEC, one row per station × satellite × epoch
├── keo_image/keogram_<date>.png         keogram
├── keo_mat/keogram_grid_<date>.parquet  keogram grid (lat × seconds of day), for re-plotting
└── _keo_test/                           output of keogram_tuner.py
tec_data/
├── obs/       obs cache: only the D-1/D/D+1 window is kept, older files are deleted
├── nav/       NAV files, kept
├── errors/    errors_<date>.txt: obs files that failed to parse (they are re-downloaded next time)
└── _tmp/      per-station temporary files during a day
orchestrator_<year|date>.log
pipeline_log_<year>.csv                  per day: date, n_stations, n_observations, n_valid_arcs, elapsed_s
```

Columns of `tec_<date>.parquet`:

| Column | Meaning |
|---|---|
| `epoch` | UTC time (30 s) |
| `sv` | satellite, e.g. `G01`, `E11`, `R09`, `C30` |
| `id_arc_valid` | arc ID `<station>_<sv>_<arc start YYYYMMDD>_<n>`, e.g. `aqui_g01_20240620_003` |
| `lat_ipp`, `lon_ipp` | IPP position [deg] |
| `azi`, `ele` | satellite azimuth and elevation [deg] |
| `stec`, `vtec` | calibrated slant and vertical TEC [TECU] |
| `vtec_detrended` | dTEC [TECU] |

Only arcs that pass the validity checks are kept.

---

## Processing settings

All settings are in [scripts/pykeo_lib/constants.py](scripts/pykeo_lib/constants.py).

| Setting | Value | Meaning |
|---|---|---|
| `SYSTEMS` | G, E, R, C | constellations used |
| `H_IPP` | 300 km | ionospheric shell height |
| `MIN_ELEVATION` | 20° | elevation cutoff |
| `ARC_MIN_LENGTH` | 190 epochs | minimum arc length |
| `ARC_MAX_GAP` | 2 min | a longer data gap is a loss of lock and starts a new arc |
| `ARC_THRESHOLD_ABS`, `ARC_THRESHOLD_STD`, `ARC_THRESHOLD_JUMP` | 1, 5, 1 | cycle-slip detection in pytecgg `extract_arcs`: absolute threshold, standard-deviation multiplier, jump threshold |
| `VTEC_JUMP_THRESHOLD` | 0.1 TECU | vTEC steps larger than this inside an arc are levelled out |
| `SG_WINDOW`, `SG_ORDER` | 181 epochs (~90 min), 3 | Savitzky-Golay detrending |
| `MM_WINDOW` | 20 epochs | moving mean applied to the detrended residual |
| `KEO_LAT_RANGE` | 37.5–57.5° | keogram latitude range |
| `KEO_LAT_STEP`, `KEO_TIME_STEP` | 0.05°, 30 s | keogram cell size |
| `DEFAULT_LON_SPAN` | all | IPP longitudes included in the keogram |
| `KEO_DTEC_CLIM` | ±0.15 TECU | keogram colour range |
| `KEO_GAUSS_SIGMA` | 0 | optional Gaussian smoothing of the keogram, in cells |

Keogram cells hold the mean dTEC of the IPPs that fall in them; empty cells are filled
from the nearest non-empty cell.

---

## Helper tools

- [scripts/keogram_tuner.py](scripts/keogram_tuner.py) — redraw a keogram from an
  existing parquet with different settings, without touching the pipeline outputs:
  ```
  python scripts/keogram_tuner.py --date 2024-06-20 --clim 0.3 --gauss-sigma 1.5
  python scripts/keogram_tuner.py --date 2024-06-20 --lat-step 0.1 --time-step 60 --exclude-sv R09
  ```
  Run `--help` for all options. Output goes to `results/_keo_test/`.
- [scripts/plot_station_map.py](scripts/plot_station_map.py) — map the stations in any
  folder of RINEX files: `python scripts/plot_station_map.py --obs-dir path/to/my_rinex`.
- [scripts/plot_station_vtec.py](scripts/plot_station_vtec.py) — interactive HTML plot of
  elevation, sTEC, vTEC and dTEC per arc for chosen stations on one day:
  `python scripts/plot_station_vtec.py 2024-06-20`.

These read from `results/` in the project root, i.e. the default `--work-dir .`.

## Tests

Offline checks for file naming, NAV fixes, server failover, station grouping and the
local source (no network, a few seconds):

```
pip install pytest
pytest tests
```

The [Quick start](#quick-start) command is the end-to-end check.

---

## How to cite

Citation details will be added when the accompanying manuscript is published.

## Licence

MIT, © INGV. See [LICENSE](LICENSE).
