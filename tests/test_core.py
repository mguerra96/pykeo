"""
Offline checks for the risky small pieces: file naming, NAV fixes, failover,
station grouping, jump levelling. No network, no sample data.

    pip install pytest
    pytest
"""

import datetime as dt
import ftplib
import gzip
import io
import socket
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pykeo_lib import decompress as dc
from pykeo_lib import download as dl
from pykeo_lib.pipeline import _fix_nav_day_zero, _group_obs_by_station, _level_vtec_jumps
from pykeo_lib.sources import obs_local
from pykeo_lib.stations import is_obs_file, read_rinex_coords

R3 = "AQUI00ITA_R_20241720000_01D_30S_MO.crx.gz"   # RINEX 3 Hatanaka
R2 = "ACOR1720.15D.Z"                               # RINEX 2 Hatanaka
PLAIN = "abcd1720.24o.gz"                           # RINEX 2, gzip only


# ---------------------------------------------------------------------------
# NAV "day 00" fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gz", [False, True])
def test_fix_nav_day_zero_rolls_back_to_previous_day(tmp_path, gz):
    text = (
        "G01 2016 03 00 06 00 00 1.0\n"   # leap year: day 0 of March = Feb 29
        "R05 2017 01 00 00 15 00 2.0\n"   # year boundary: day 0 of Jan = Dec 31
        "G02 2016 03 01 00 00 00 3.0\n"   # valid date: unchanged
    )
    p = tmp_path / ("nav.rnx.gz" if gz else "nav.rnx")
    p.write_bytes(gzip.compress(text.encode()) if gz else text.encode())

    out = _fix_nav_day_zero(p)
    raw = out.read_bytes()
    out.unlink()
    lines = (gzip.decompress(raw) if gz else raw).decode().splitlines()

    assert lines == [
        "G01 2016 02 29 06 00 00 1.0",
        "R05 2016 12 31 00 15 00 2.0",
        "G02 2016 03 01 00 00 00 3.0",
    ]


# ---------------------------------------------------------------------------
# File naming: must match what decompress() produces, or every run re-downloads
# ---------------------------------------------------------------------------

def test_names_keep_gz_and_hatanaka(tmp_path):
    # pytecgg reads .gz and Hatanaka itself; only .Z is unpacked
    assert dc.expected_final_path(R3, tmp_path).name == R3
    assert dc.expected_final_path(R2, tmp_path).name == "ACOR1720.15D"
    assert dc.expected_final_path(PLAIN, tmp_path).name == PLAIN
    assert dl._filter_obs_filenames([R3, R2, PLAIN, "x.txt"], None) == [R3, R2, PLAIN]


def test_decompress_matches_expected_name(tmp_path):
    real_gz = tmp_path / R3
    real_gz.write_bytes(gzip.compress(b"crinex"))
    assert dc.decompress(real_gz) == dc.expected_final_path(R3, tmp_path)
    assert gzip.decompress(real_gz.read_bytes()) == b"crinex"   # left untouched

    z_holding_gzip = tmp_path / R2   # mislabelled .Z
    z_holding_gzip.write_bytes(gzip.compress(b"hatanaka"))
    out = dc.decompress(z_holding_gzip)
    assert out == dc.expected_final_path(R2, tmp_path) and out.read_bytes() == b"hatanaka"
    assert not z_holding_gzip.exists()

    gz_holding_zip = tmp_path / PLAIN   # mislabelled .gz: re-written as real gzip, same name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner", b"obs")
    gz_holding_zip.write_bytes(buf.getvalue())
    out = dc.decompress(gz_holding_zip)
    assert out == dc.expected_final_path(PLAIN, tmp_path)
    assert gzip.decompress(out.read_bytes()) == b"obs"


def test_read_rinex_coords_reads_gzip(tmp_path):
    xyz = f"{4642432.6:14.4f}{1028629.0:14.4f}{4246494.8:14.4f}"
    header = f"{xyz:<60}APPROX POSITION XYZ\n{'':60}END OF HEADER\n"
    f = tmp_path / R3
    f.write_bytes(gzip.compress(header.encode()))
    lat, lon = read_rinex_coords(f)
    assert lat == pytest.approx(42.0, abs=0.5) and lon == pytest.approx(12.5, abs=0.5)
    assert is_obs_file(R3) and is_obs_file(R2) and is_obs_file(PLAIN) and not is_obs_file("nav.json")


# ---------------------------------------------------------------------------
# Station grouping
# ---------------------------------------------------------------------------

def test_group_prefers_rinex3_per_station():
    r3 = Path("AQUI00ITA_R_20241720000_01D_30S_MO.rnx")
    groups = _group_obs_by_station([Path("aqui1720.24o"), r3, Path("geno1720.24o")])
    assert groups["AQUI"] == [r3]
    assert groups["GENO"] == [Path("geno1720.24o")]


# ---------------------------------------------------------------------------
# vTEC jump levelling
# ---------------------------------------------------------------------------

def test_level_vtec_jumps_per_arc():
    df = pl.DataFrame({
        "id_arc_valid": ["a"] * 5 + ["b"] * 3,
        "epoch": [0, 1, 2, 3, 4, 0, 1, 2],
        "vtec": [1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.05, 5.1],
    })
    out = _level_vtec_jumps(df)
    assert out.filter(pl.col("id_arc_valid") == "a")["vtec"].to_list() == [1.0] * 5
    # small steps below the threshold, and arc a's correction must not leak into arc b
    assert out.filter(pl.col("id_arc_valid") == "b")["vtec"].to_list() == pytest.approx([5.0, 5.05, 5.1])


# ---------------------------------------------------------------------------
# FTP failover: an outage must not look like "no data"
# ---------------------------------------------------------------------------

class _FakeFTP:
    behaviour: dict = {}

    def __init__(self, host, timeout=None):
        self.host = host

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def login(self):
        pass

    def cwd(self, path):
        b = self.behaviour[self.host]
        if b == "timeout":
            raise socket.timeout("timed out")
        if b == "421":
            raise ftplib.error_temp("421 too many connections")
        if b == "550":
            raise ftplib.error_perm("550 no such directory")

    def nlst(self):
        return ["/x/abcd1720.24d.Z"]


@pytest.mark.parametrize("a, b, expected", [
    ("timeout", "timeout", (None, None)),            # nobody reachable
    ("421", "timeout", (None, None)),                # throttled counts as unreachable
    ("timeout", "550", ([], None)),                  # one host says "no data"
    ("550", "550", ([], None)),
    ("timeout", "ok", (["abcd1720.24d.Z"], "b")),    # failover works
])
def test_list_ftp_dir_failover(monkeypatch, a, b, expected):
    monkeypatch.setattr(dl.ftplib, "FTP", _FakeFTP)
    monkeypatch.setattr(_FakeFTP, "behaviour", {"a": a, "b": b})
    assert dl._list_ftp_dir_failover(("a", "b"), "/x") == expected


# ---------------------------------------------------------------------------
# NAV file selection
# ---------------------------------------------------------------------------

def test_local_nav_ignores_partial_and_prefers_igs(tmp_path):
    (tmp_path / "BRDC00IGS_R_20240010000_01D_MN.rnx.gz.part").write_bytes(b"x")
    assert dl._get_local_nav_files(tmp_path, 2024, 1) == []

    (tmp_path / "BRDC00GOP_R_20240010000_01D_MN.rnx.gz").write_bytes(b"x")
    (tmp_path / "BRDC00IGS_R_20240010000_01D_MN.rnx.gz").write_bytes(b"x")
    names = [p.name for p in dl._get_local_nav_files(tmp_path, 2024, 1)]
    assert names == ["BRDC00IGS_R_20240010000_01D_MN.rnx.gz", "BRDC00GOP_R_20240010000_01D_MN.rnx.gz"]


# ---------------------------------------------------------------------------
# Local source: must never touch the user's own folder
# ---------------------------------------------------------------------------

def test_obs_local_copies_and_never_modifies_source(tmp_path):
    src, obs = tmp_path / "mine", tmp_path / "obs"
    src.mkdir()
    (src / "ABCD00ITA_R_20241720000_01D_30S_MO.rnx").write_text("x")   # day 172
    (src / "efgh1720.24o.gz").write_bytes(gzip.compress(b"y"))          # day 172, gzipped
    (src / "ABCD00ITA_R_20241730000_01D_30S_MO.rnx").write_text("z")   # other day
    before = sorted((f.name, f.read_bytes()) for f in src.iterdir())

    got = obs_local(dt.date(2024, 6, 20), None, obs, src_dir=src)

    assert sorted(p.name for p in got) == ["ABCD00ITA_R_20241720000_01D_30S_MO.rnx", "efgh1720.24o.gz"]
    assert all(p.parent == obs for p in got)
    assert sorted((f.name, f.read_bytes()) for f in src.iterdir()) == before
    assert [p.name for p in obs_local(dt.date(2024, 6, 20), ["EFGH"], obs, src_dir=src)] == ["efgh1720.24o.gz"]


# ---------------------------------------------------------------------------
# INGV source: a file listed in two networks is downloaded once, and a station
# whose own network is empty is still found in another
# ---------------------------------------------------------------------------

def test_obs_ingv_downloads_each_file_once(monkeypatch, tmp_path):
    import pykeo_lib.sources as src

    asir, geno = "ASIR00ITA_R_20260210000_01D_30S_MO.crx.gz", "GENO00ITA_R_20260210000_01D_30S_MO.crx.gz"
    listings = {"EUREF": [asir, geno], "NETB": [asir, geno], "EMPTY": []}
    monkeypatch.setattr(src, "load_station_networks",
                        lambda *a, **k: {"ASIR": "EUREF", "GENO": "NETB", "DUNG": "EMPTY"})
    monkeypatch.setattr(src, "_list_ftp_dir_failover", lambda hosts, d: (listings[d.split("/")[3]], "h"))
    got = []
    monkeypatch.setattr(src, "_download_parallel", lambda targets, *a: got.extend(targets) or [])

    src.obs_ingv(dt.date(2026, 1, 21), ["ASIR", "GENO", "DUNG"], tmp_path, work_dir=tmp_path)

    assert sorted(got) == [asir, geno]
