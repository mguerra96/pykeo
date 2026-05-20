import datetime as dt
from pathlib import Path

# EUREF EPN FTP
FTP_HOST             = "www.epncb.oma.be"
FTP_TIMEOUT_LIST     = 120
FTP_TIMEOUT_DOWNLOAD = 120

# gnssgiving FTP — primary + fallback hosts (same directory layout)
GNSSGIVING_HOSTS = ("mga.int.ingv.it", "gnssgiving.int.ingv.it")

# Ionospheric pierce point
H_IPP         = 300_000   # m (300 km)
MIN_ELEVATION = 20        # degrees

# Arc validity thresholds
ARC_THRESHOLD_ABS  = 1    # TECU
ARC_THRESHOLD_STD  = 5   # TECU
ARC_THRESHOLD_JUMP = 1    # TECU
ARC_MIN_LENGTH     = 190  # epochs
ARC_MAX_GAP        = dt.timedelta(minutes=2)

# GNSS constellations
SYSTEMS = ["G", "E", "R", "C"]

# vTEC consecutive-epoch jump leveling (applied per arc after calibration)
VTEC_JUMP_THRESHOLD = 0.5  # TECU

# Savitzky-Golay detrending
SG_WINDOW = 181
SG_ORDER  = 3
MM_WINDOW = 20

# Keogram grid and plot
KEO_TIME_STEP   = 30            # seconds
KEO_LAT_STEP    = 0.05          # degrees
KEO_LAT_RANGE   = (37.5, 57.5)
KEO_DTEC_CLIM   = 0.15          # TECU
KEO_GAUSS_SIGMA = 0
KEO_FIGSIZE     = (10, 6)
KEO_DPI         = 150

DEFAULT_LON_SPAN = (-180.0, 180.0)

# Columns written to the output parquet
COLS_TO_KEEP = [
    "epoch", "sv", "id_arc_valid",
    "lat_ipp", "lon_ipp", "azi", "ele",
    "stec", "vtec", "vtec_detrended",
]

# Hatanaka decompressor — sits at the project root (parent of scripts/)
CRX2RNX = Path(__file__).parent.parent.parent / "crx2rnx.exe"
