import gzip
import io
import logging
import subprocess
import zipfile
from pathlib import Path

from .constants import CRX2RNX

logger = logging.getLogger(__name__)


def _is_hatanaka_rinex2(p: Path) -> bool:
    return len(p.suffix) == 4 and p.suffix.upper().endswith("D")


def _is_hatanaka_rinex3(p: Path) -> bool:
    return p.suffix.lower() == ".crx"


def _run_crx2rnx(p: Path) -> Path | None:
    if not CRX2RNX.exists():
        logger.warning("crx2rnx.exe not found — skipping Hatanaka decompression")
        return p
    try:
        subprocess.run([str(CRX2RNX), "-f", "-d", str(p)], capture_output=True, check=False, timeout=5)
        out = p.with_suffix(p.suffix[:-1] + "O") if _is_hatanaka_rinex2(p) else p.with_suffix(".rnx")
        if out.exists():
            p.unlink(missing_ok=True)
            return out
        logger.warning(f"crx2rnx produced no output for {p.name}")
    except subprocess.TimeoutExpired:
        logger.warning(f"crx2rnx timed out on {p.name} — skipping")
        p.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"crx2rnx error on {p.name}: {e}")
    return None


def decompress(raw_path: Path) -> Path | None:
    """
    Decompress a downloaded file through all needed steps:
      .Z  → unlzw3 (or gzip if mislabelled)
      .gz → gzip (or unlzw3 if mislabelled)
      .??d / .crx (Hatanaka) → .??o / .rnx
    Returns the final RINEX path, or None on failure.
    """
    from unlzw3 import unlzw

    p = raw_path

    if p.suffix == ".Z":
        out = p.with_suffix("")
        raw = p.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            try:
                with gzip.open(io.BytesIO(raw), "rb") as fi:
                    out.write_bytes(fi.read())
                p.unlink(missing_ok=True)
                p = out
            except Exception as e:
                logger.warning(f"gunzip(.Z) failed for {raw_path.name}: {e}")
                return None
        else:
            try:
                out.write_bytes(unlzw(raw))
                p.unlink(missing_ok=True)
                p = out
            except Exception as e:
                logger.warning(f"unlzw failed for {raw_path.name}: {e}")
                return None

    if p.suffix == ".gz":
        out = p.with_suffix("")
        raw = p.read_bytes()
        try:
            if raw[:2] == b"\x1f\x9d":
                out.write_bytes(unlzw(raw))
            elif raw[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    names = zf.namelist()
                    if not names:
                        logger.warning(f"empty ZIP for {raw_path.name}")
                        return None
                    out.write_bytes(zf.read(names[0]))
            else:
                with gzip.open(io.BytesIO(raw), "rb") as fi:
                    out.write_bytes(fi.read())
            p.unlink(missing_ok=True)
            p = out
        except Exception as e:
            logger.warning(f"decompress failed for {raw_path.name}: {e}")
            return None

    if _is_hatanaka_rinex2(p) or _is_hatanaka_rinex3(p):
        result = _run_crx2rnx(p)
        if result is None:
            return None
        p = result

    return p


def expected_final_path(fname: str, local_dir: Path) -> Path:
    """Predict the final decompressed path for a filename before downloading."""
    stem = fname
    for ext in (".Z", ".gz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    stem_path = local_dir / stem
    if _is_hatanaka_rinex2(stem_path):
        return stem_path.with_suffix(stem_path.suffix[:-1] + "O")
    if _is_hatanaka_rinex3(stem_path):
        return stem_path.with_suffix(".rnx")
    return stem_path
