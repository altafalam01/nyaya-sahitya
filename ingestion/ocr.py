"""Tesseract OCR for pages that have no usable text layer.

Tesseract is a separate program, not a Python package, so nothing here assumes it is
present: `check_tesseract()` reports what is (and is not) installed, and the parser
skips OCR with a clear warning when it is missing instead of failing.
"""
import csv
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

WINDOWS_DEFAULT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

INSTALL_HELP = """\
OCR needs Tesseract (a separate program). To install it on Windows:
  1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
     (the 'tesseract-ocr-w64-setup-...exe' link) and run it.
  2. On the 'Choose Components' page expand 'Additional language data (download)'
     and tick 'Hindi' (needed for Devanagari scans); English is included by default.
  3. Add its folder to PATH (the installer has an option for this; the default folder is
     C:\\Program Files\\Tesseract-OCR), or set TESSERACT_CMD in .env to the full path of
     tesseract.exe. Open a NEW terminal afterwards.
To verify:
  tesseract --version        (should print 'tesseract v5...')
  tesseract --list-langs     (should list 'eng' and, for Hindi, 'hin')
  python -m ingestion.parse --check-ocr
Already installed but 'hin' is missing? Download hin.traineddata from
https://github.com/tesseract-ocr/tessdata_fast and copy it into the 'tessdata' folder next to
tesseract.exe (needs administrator rights), then re-run `tesseract --list-langs`.
"""


@dataclass(frozen=True)
class OcrStatus:
    available: bool
    path: Optional[str]
    version: Optional[str]
    languages: tuple[str, ...]
    message: str

    def language_arg(self, wanted: tuple[str, ...] = ("eng", "hin")) -> str:
        """Tesseract -l value using only the wanted languages that are actually installed."""
        have = [lang for lang in wanted if lang in self.languages]
        return "+".join(have) if have else "eng"


@dataclass(frozen=True)
class OcrWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def find_tesseract() -> Optional[str]:
    candidates = [os.getenv("TESSERACT_CMD"), shutil.which("tesseract"), *WINDOWS_DEFAULT_PATHS]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return None


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def check_tesseract(path: Optional[str] = None) -> OcrStatus:
    path = path or find_tesseract()
    if not path:
        return OcrStatus(False, None, None, (), "Tesseract was not found.\n" + INSTALL_HELP)
    try:
        ver = _run([path, "--version"])
        langs = _run([path, "--list-langs"])
    except (OSError, subprocess.SubprocessError) as e:
        return OcrStatus(False, path, None, (), f"Tesseract at {path} could not be run: {e}\n" + INSTALL_HELP)
    if ver.returncode != 0:
        return OcrStatus(False, path, None, (), f"'{path} --version' failed: {ver.stderr.strip()}\n" + INSTALL_HELP)
    version = (ver.stdout or ver.stderr).splitlines()[0].strip()
    names = tuple(line.strip() for line in langs.stdout.splitlines()[1:] if line.strip())
    msg = f"{version} at {path}; languages: {', '.join(names) or 'none'}"
    if "eng" not in names:
        return OcrStatus(False, path, version, names, msg + "\nEnglish data ('eng') is missing.\n" + INSTALL_HELP)
    if "hin" not in names:
        msg += "\nNote: 'hin' (Hindi) is not installed, so Devanagari scans cannot be OCR'd yet.\n" + INSTALL_HELP
    return OcrStatus(True, path, version, names, msg)


def ocr_png(png: bytes, status: OcrStatus, scale: float, timeout: int = 180) -> list[OcrWord]:
    """OCR one page image. `scale` converts image pixels to PDF points (72 / render_dpi)."""
    if not status.available or not status.path:
        raise RuntimeError("Tesseract is not available")
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        img.write_bytes(png)
        res = _run([status.path, str(img), "stdout", "-l", status.language_arg(), "tsv"], timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"tesseract failed: {res.stderr.strip()[:300]}")
    words: list[OcrWord] = []
    reader = csv.DictReader(io.StringIO(res.stdout), delimiter="\t", quoting=csv.QUOTE_NONE)
    for r in reader:
        text = (r.get("text") or "").strip()
        if r.get("level") != "5" or not text:
            continue
        try:
            left, top, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
        except (TypeError, ValueError):
            continue
        words.append(OcrWord(text, left * scale, top * scale, (left + w) * scale, (top + h) * scale))
    return words
