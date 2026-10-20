"""
Offline checks for the risky small pieces: file naming, NAV fixes, failover,
station grouping, jump levelling. No network, no sample data.

    pip install pytest
    pytest
"""

import datetime as dt
import ftplib
import gzip
import socket
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pykeo_lib import decompress as dc
from pykeo_lib import download as dl
from pykeo_lib.pipeline import _fix_nav_day_zero, _group_obs_by_station, _level_vtec_jumps
from pykeo_lib.sources import obs_local

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

@pytest.fixture
def with_crx2rnx(monkeypatch, tmp_path):
    fake = tmp_path / "crx2rnx.exe"
    fake.touch()
    monkeypatch.setattr(dc, "CRX2RNX", fake)


@pytest.fixture
def without_crx2rnx(monkeypatch):
    monkeypatch.setattr(dc, "CRX2RNX", Path("no_such_crx2rnx"))


def test_names_with_crx2rnx(with_crx2rnx, tmp_path):
    assert dc.expected_final_path(R3, tmp_path).name == "AQUI00ITA_R_20241720000_01D_30S_MO.rnx"
    assert dc.expected_final_path(R2, tmp_path).name == "ACOR1720.15O"
    assert dc.expected_final_path(PLAIN, tmp_path).name == "abcd1720.24o"
    assert all(dc.can_decompress(n) for n in (R3, R2, PLAIN))


def test_names_without_crx2rnx(without_crx2rnx, tmp_path):
    # RINEX 3 Hatanaka is kept as .crx (pytecgg reads it); RINEX 2 Hatanaka is skipped.
    assert dc.expected_final_path(R3, tmp_path).name == "AQUI00ITA_R_20241720000_01D_30S_MO.crx"
    assert dc.can_decompress(R3) and dc.can_decompress(PLAIN)
    assert not dc.can_decompress(R2)
    assert R2 not in dl._filter_obs_filenames([R3, R2, PLAIN], None)

    f = tmp_path / R3
    f.write_bytes(gzip.compress(b"crinex"))
    assert dc.decompress(f) == dc.expected_final_path(R3, tmp_path)


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

    assert sorted(p.name for p in got) == ["ABCD00ITA_R_20241720000_01D_30S_MO.rnx", "efgh1720.24o"]
    assert all(p.parent == obs for p in got)
    assert sorted((f.name, f.read_bytes()) for f in src.iterdir()) == before
    assert [p.name for p in obs_local(dt.date(2024, 6, 20), ["EFGH"], obs, src_dir=src)] == ["efgh1720.24o"]
