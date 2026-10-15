"""Page-level extraction: words with coordinates, from the PDF text layer or from OCR.

A page only goes to OCR when it has no usable text layer (empty, or mostly garbage) AND
contains an image to read; a truly blank page is left alone. If OCR is needed but Tesseract
is missing, the page is recorded as `ocr_skipped` so the caller can warn, never silently
dropped.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

from .ocr import OcrStatus, ocr_png

log = logging.getLogger(__name__)
pymupdf.TOOLS.mupdf_display_errors(False)

TEXT, OCR, BLANK, OCR_SKIPPED = "text", "ocr", "blank", "ocr_skipped"
MIN_TEXT_CHARS = 25  # below this a page counts as having no text layer
MIN_ALNUM_RATIO = 0.55  # below this the text layer is treated as garbage
RENDER_DPI = 300


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PageWords:
    number: int  # 1-based
    width: float
    height: float
    words: list[Word] = field(default_factory=list)
    source: str = TEXT
    weak: bool = False  # text layer present but poor, and OCR was not available to improve it


def text_layer_is_usable(words: list[Word]) -> bool:
    chars = "".join(w.text for w in words)
    if len(chars) < MIN_TEXT_CHARS:
        return False
    return sum(c.isalnum() for c in chars) / len(chars) >= MIN_ALNUM_RATIO


def extract_pages(pdf_path: Path, ocr_mode: str = "auto", ocr_status: Optional[OcrStatus] = None) -> list[PageWords]:
    """ocr_mode: 'auto' (only pages lacking text), 'force' (every page), 'off' (never)."""
    pages: list[PageWords] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pw = PageWords(i + 1, page.rect.width, page.rect.height)
            pw.words = [Word(w[4], w[0], w[1], w[2], w[3]) for w in page.get_text("words") if w[4].strip()]
            usable = text_layer_is_usable(pw.words)
            if ocr_mode == "force" or (not usable and ocr_mode != "off"):
                has_image = bool(page.get_images())
                if ocr_mode != "force" and not has_image and not pw.words:
                    pw.source = BLANK
                elif ocr_status and ocr_status.available:
                    pix = page.get_pixmap(dpi=RENDER_DPI)
                    ocr_words = ocr_png(pix.tobytes("png"), ocr_status, 72 / RENDER_DPI)
                    pw.words = [Word(w.text, w.x0, w.y0, w.x1, w.y1) for w in ocr_words]
                    pw.source = OCR
                elif pw.words:
                    pw.weak = True  # keep the poor text layer rather than drop real words
                else:
                    pw.source = OCR_SKIPPED
            elif not usable and not pw.words:
                pw.source = BLANK if not page.get_images() else OCR_SKIPPED
            pages.append(pw)
    return pages
