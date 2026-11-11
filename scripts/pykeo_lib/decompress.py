"""
Make a downloaded RINEX file readable by pytecgg.

pytecgg (>= 1.5) reads plain, gzip (.gz) and Hatanaka (.crx, RINEX 2 .yyD)
RINEX itself, so those are left as they are. Only Unix-compress (.Z) files and
mislabelled .gz files (LZW or ZIP inside) are unpacked here.
"""

import gzip
import logging
import zipfile
import io
from pathlib import Path

logger = logging.getLogger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"


def _unpack(raw: bytes) -> bytes:
    """Decompress gzip, ZIP (first member) or Unix-compress bytes, by magic number."""
    from unlzw3 import unlzw

    if raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw)
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                raise ValueError("empty ZIP")
            return zf.read(names[0])
    return unlzw(raw)


def decompress(raw_path: Path) -> Path | None:
    """
    Prepare a downloaded file for pytecgg and return its final path (see
    expected_final_path), or None on failure:
      .Z  → unpacked (LZW, or gzip/ZIP if mislabelled), .Z dropped from the name
      .gz → kept; re-written as gzip if it holds LZW or ZIP
      anything else (plain, .crx, .yyD) → kept as is
    """
    p = raw_path
    if p.suffix not in (".Z", ".gz"):
        return p
    raw = p.read_bytes()
    if p.suffix == ".gz" and raw[:2] == _GZIP_MAGIC:
        return p
    try:
        data = _unpack(raw)
    except Exception as e:
        logger.warning(f"decompress failed for {p.name}: {e}")
        return None
    if p.suffix == ".gz":
        # keep the .gz name so expected_final_path() still finds it on the next run
        p.write_bytes(gzip.compress(data, compresslevel=1))
        return p
    out = p.with_suffix("")
    out.write_bytes(data)
    p.unlink(missing_ok=True)
    return out


def expected_final_path(fname: str, local_dir: Path) -> Path:
    """Predict decompress()'s output path for a filename before downloading."""
    return local_dir / (fname[:-2] if fname.endswith(".Z") else fname)
