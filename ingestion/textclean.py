"""Turn page words into clean running text.

Pipeline per document:  words -> rows (one per visual line) -> drop noise (page numbers,
running headers/footers, margin markers, rules) -> paragraphs (line wraps repaired,
hyphenation resolved against the document's own vocabulary) -> Unicode normalisation.

Running headers are found from the document itself: a short line in the top/bottom band of
a page whose letters (digits ignored, so the changing page number does not matter) repeat at
the same height on many pages is a header/footer. Nothing is removed on a guess about one page.
"""
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .pdf_extract import PageWords, Word

TOP_BAND = 0.19  # fraction of page height searched for running headers
BOTTOM_BAND = 0.78  # fraction of page height below which footers are searched
HEADER_MIN_PAGES = 3
KEY_LETTERS = 12  # letters that identify a running head: long enough not to collide with body text
CLAUSE_OPENER_RE = re.compile(
    r"^[^\w(]*(?:Explanation|Illustrations?\b|Provided|\(?\d{1,3}[A-Z]{0,2}[.)]|\([0-9]{1,3}[A-Z]?\)|"
    r"\([A-Za-z]{1,2}\)|\([ivxlcdm]{1,6}\))", re.I)
MIN_SEQUENCE = 5  # shortest counting-up run of bare numbers treated as page numbers
HEADER_MIN_FRACTION = 0.08
HEADER_MAX_CHARS = 110
COLUMN_GAP = 0.08  # gap wider than this fraction of page width separates table columns
PARA_GAP = 1.45  # vertical pitch (in normal line pitches) that starts a new paragraph

PAGE_NUMBER_RE = re.compile(r"^[^\w]*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?[^\w.]*$", re.I)
JUDGE_HEADER_END_RE = re.compile(r"\b(?:JJ?|CJ|CJI|C\.J\.(?:I\.)?)[.\s\]\)]*$")
ROMAN_RE = re.compile(r"^[\s\[\]()\-]*(?=[ivxlcdm]{2,7}[\s\[\]()\-]*$)[ivxlcdm]+[\s\[\]()\-]*$", re.I)
_NAV = r"(?:index|home\s*page|chapters?\s+and\s+sections?|note:?\s*for\s+reference\s+only)"
NAV_LINK_RE = re.compile(rf"^[\W_]*{_NAV}(?:[\W_]+{_NAV})*[\W_]*$", re.I)  # website navigation baked into a PDF
RULE_RE = re.compile(r"^[\s_\-=.*~—–]{6,}$")
MARGIN_LETTER_RE = re.compile(r"^[A-Ha-h]$")
LONE_MARKER_RE = re.compile(r"^[A-Ha-h0-9Oo]$")  # a lone marker row; OCR reads margin letters as digits
MARKER_GAP = 7.0  # points of empty space that detach a margin letter from the text
STRAY_SYMBOL_RE = re.compile(r"^[·•▪◦∙⋅¨´`]+$")
SENTENCE_END_RE = re.compile(r"[.:;?!][\"”’')\]]*$")
MARKER_RE = re.compile(
    r"^(?:\(?\d{1,3}[A-Z]{0,2}[.)]|\([0-9]{1,3}[A-Z]?\)|\([A-Za-z]{1,2}\)|\([ivxlcdm]{1,6}\)|"
    r"[IVXL]{1,5}\.|Explanation|Illustration|Provided|CHAPTER|PART\b|SECTION|ARTICLE)(?=[\s.—-]|$)")
ZERO_WIDTH_RE = re.compile("[​⁠﻿]")  # NOT U+200C/U+200D: Devanagari needs them
PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff]")  # glyphs from symbol fonts: meaningless as text
CONTROL_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
SOFT_HYPHEN = "­"
REPLACEMENT = "�"


# ---------------------------------------------------------------- normalisation
def fix_replacement_chars(s: str) -> tuple[str, int]:
    """Repair U+FFFD (a glyph the PDF could not map) only where the context makes it obvious.

    Returns (text, number_of_unrecoverable_characters_removed).
    """
    if REPLACEMENT not in s:
        return s, 0
    s = re.sub(rf"(?<=[A-Za-z]){REPLACEMENT}(?=(?:s|t|re|ll|ve|d)\b)", "’", s)  # woman?s
    s = re.sub(rf"(?<=[\d.]){REPLACEMENT}+(?=[A-Za-z(])", "—", s)  # 2.?For -> 2.-For
    s = re.sub(rf"(?<![\w]){REPLACEMENT}+(?=[\w(])", "“", s)  # opening quote
    s = re.sub(rf"(?<=[\w.,;)]){REPLACEMENT}+(?![\w])", "”", s)  # closing quote
    left = s.count(REPLACEMENT)
    return s.replace(REPLACEMENT, ""), left


def normalize_text(s: str) -> str:
    """Unicode-normalise (NFKC, valid for Devanagari too) and strip invisible junk."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(SOFT_HYPHEN, "")
    s = PRIVATE_USE_RE.sub(lambda m: "\u2022" if m.group() == "\uf0b7" else "", s)  # Symbol-font bullet
    s = ZERO_WIDTH_RE.sub("", s)
    s = CONTROL_RE.sub(" ", s)
    return s


def tidy_spacing(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+([,;:.?!])(?=\s|$)", r"\1", s)
    return s.strip()


# ---------------------------------------------------------------- rows
@dataclass
class Row:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    h: float
    page: int = 0
    words: list[Word] = field(default_factory=list, repr=False)


def _join_words(words: list[Word], page_w: float, columns: bool) -> str:
    out: list[str] = []
    prev = None
    for w in words:
        t = w.text
        if SOFT_HYPHEN in t:
            if t.endswith(SOFT_HYPHEN):  # hyphenation point at end of a line
                before = t[:-1]
                t = before + ("-" if before and before[-1].isalpha() else "")
            t = t.replace(SOFT_HYPHEN, "")
        if prev is not None:
            out.append(" | " if columns and w.x0 - prev.x1 > COLUMN_GAP * page_w else " ")
        out.append(t)
        prev = w
    return "".join(out)


def build_rows(words: list[Word], page_w: float, page: int = 0, columns: bool = False) -> list[Row]:
    """Group words into visual lines by vertical position (robust to word-per-line text layers).

    columns=True marks wide gaps with ' | ' so table columns (statute indexes) stay distinguishable.
    """
    rows: list[Row] = []
    cur: list[Word] = []
    centre_sum = height_sum = 0.0  # running sums keep this linear in the number of words

    def flush() -> None:
        if not cur:
            return
        ws = sorted(cur, key=lambda w: w.x0)
        rows.append(Row(_join_words(ws, page_w, columns), min(w.x0 for w in ws), min(w.y0 for w in ws),
                        max(w.x1 for w in ws), max(w.y1 for w in ws),
                        statistics.median(w.y1 - w.y0 for w in ws), page, ws))

    for w in sorted(words, key=lambda w: (w.y0 + w.y1) / 2):
        c = (w.y0 + w.y1) / 2
        h = max(w.y1 - w.y0, 1.0)
        if cur and abs(c - centre_sum / len(cur)) <= 0.5 * (height_sum + h) / (len(cur) + 1):
            cur.append(w)
            centre_sum += c
            height_sum += h
        else:
            flush()
            cur, centre_sum, height_sum = [w], c, h
    flush()
    return rows


def _is_marker(ws: list[Word], i: int) -> bool:
    """Word i is a lone margin letter (A-H) detached from its neighbour by a wide gap."""
    if not MARGIN_LETTER_RE.match(ws[i].text) or len(ws) < 2:
        return False
    if i == 0:
        return ws[1].x0 - ws[0].x1 > MARKER_GAP
    return i == len(ws) - 1 and ws[i].x0 - ws[i - 1].x1 > MARKER_GAP


def _text_extent(r: Row) -> tuple[float, float]:
    """Row extent ignoring margin letters, which would otherwise widen the text column."""
    ws = r.words
    lo = ws[1].x0 if len(ws) > 1 and _is_marker(ws, 0) else r.x0
    hi = ws[-2].x1 if len(ws) > 1 and _is_marker(ws, len(ws) - 1) else r.x1
    return lo, hi


def _band(rows: list[Row]) -> tuple[float, float] | None:
    long_rows = [r for r in rows if len(r.text) >= 45]
    if len(long_rows) < 5:
        return None
    xs0 = sorted(_text_extent(r)[0] for r in long_rows)
    xs1 = sorted(_text_extent(r)[1] for r in long_rows)
    return xs0[int(0.15 * (len(xs0) - 1))], xs1[int(0.90 * (len(xs1) - 1))]


def strip_margin_marks(words: list[Word], page_w: float, page: int) -> tuple[list[Word], int]:
    """Judgment reports print paragraph letters (A-H) and small notes in the margins; drop them."""
    rows = build_rows(words, page_w, page)
    band = _band(rows)
    drop: set[int] = set()
    for r in rows:
        ws = r.words
        if len(ws) == 1 and LONE_MARKER_RE.match(ws[0].text):
            drop.add(id(ws[0]))  # a lone marker letter on its own line
            continue
        if _is_marker(ws, 0) and (band is None or ws[0].x1 <= band[0] + 6):
            drop.add(id(ws[0]))
        if len(ws) > 1 and _is_marker(ws, len(ws) - 1) and (band is None or ws[-1].x0 >= band[1] - 6):
            drop.add(id(ws[-1]))
        if band:
            for w in ws:
                if w.x1 < band[0] - 4 or w.x0 > band[1] + 4:
                    drop.add(id(w))  # note fully outside the text column
    kept = [w for w in words if id(w) not in drop]
    return kept, len(words) - len(kept)


# ---------------------------------------------------------------- noise detection
def _key(text: str) -> str:
    """Letters of the row (runs of 3+ only, so OCR specks and page numbers do not change it)."""
    return "".join(re.findall(r"[a-z]{3,}", text.lower()))[:KEY_LETTERS]


def looks_like_clause(text: str) -> bool:
    """Rows that open like body text ('Explanation 1.', '(a)', '12.') are never running headers."""
    return bool(CLAUSE_OPENER_RE.match(text.strip()))


def is_page_number(text: str) -> bool:
    t = text.strip()
    return bool(PAGE_NUMBER_RE.match(t) or ROMAN_RE.match(t))


def is_judge_header(text: str) -> bool:
    """Running head naming the judge, e.g. '[S. A. BOBDE, J.]' (judgments only, header zone only)."""
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    return (len(t) <= 90 and len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.9
            and bool(JUDGE_HEADER_END_RE.search(t)))


def is_junk_row(text: str) -> bool:
    t = text.strip()
    if not t or RULE_RE.match(t) or NAV_LINK_RE.match(t):
        return True
    alnum = sum(c.isalnum() for c in t)
    return alnum == 0 or (len(t) <= 12 and alnum / len(t) < 0.3)


def _zone_rows(rows: list[Row], height: float) -> list[Row]:
    """Rows inside the top or bottom band (continued footnotes can push a page number down)."""
    return [r for r in rows if r.y0 < TOP_BAND * height or r.y1 > BOTTOM_BAND * height]


def find_running_headers(pages_rows: list[tuple[PageWords, list[Row]]]) -> set[tuple[str, int]]:
    """(letters-key, height-bucket) pairs that repeat on enough pages to be headers/footers."""
    n = len(pages_rows)
    per_page: list[set[tuple[str, int]]] = []
    for pw, rows in pages_rows:
        found = set()
        for r in _zone_rows(rows, pw.height):
            k = _key(r.text)
            if len(k) >= 4 and len(r.text) <= HEADER_MAX_CHARS and not looks_like_clause(r.text):
                found.add((k, int(r.y0 / pw.height * 50)))
        per_page.append(found)
    counts: Counter = Counter()
    for found in per_page:
        counts.update(found)
    need = max(HEADER_MIN_PAGES, HEADER_MIN_FRACTION * n)
    headers = set()
    for (k, b), _ in counts.items():
        if counts[(k, b)] + counts.get((k, b - 1), 0) + counts.get((k, b + 1), 0) >= need:
            headers.add((k, b))
            headers.add((k, b - 1))
            headers.add((k, b + 1))
    return headers


def find_sequence_page_numbers(pages_rows: list[tuple[PageWords, list[Row]]]) -> set[int]:
    """ids of rows that are printed page numbers, found as a counting-up run of bare numbers.

    Some PDFs (the IPC) reproduce the original book's page numbers inline, at a different height on
    every page and not one per PDF page. A run of 5+ bare numbers that rise by 1 (one missed number
    allowed) in reading order cannot be anything else.
    """
    cands: list[tuple[int, Row]] = []
    for _pw, rows in pages_rows:
        for r in rows:
            t = r.text.strip()
            if PAGE_NUMBER_RE.match(t) and re.search(r"\d", t) and not re.search(r"[a-z]", t, re.I):
                cands.append((int(re.sub(r"\D", "", t)), r))
    length = [1] * len(cands)
    prev = [-1] * len(cands)
    for i, (v, _) in enumerate(cands):
        for j in range(i):
            if 1 <= v - cands[j][0] <= 2 and length[j] + 1 > length[i]:
                length[i], prev[i] = length[j] + 1, j
    members: set[int] = set()
    for i in sorted(range(len(cands)), key=lambda k: -length[k]):
        if length[i] < MIN_SEQUENCE or id(cands[i][1]) in members:
            continue
        k = i
        while k != -1:
            members.add(id(cands[k][1]))
            k = prev[k]
    return members


def _close(a: str, b: str, limit: int = 2) -> bool:
    """Edit distance <= limit (OCR spells the same running head slightly differently)."""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= limit


def is_header_row(r: Row, height: float, headers: set[tuple[str, int]]) -> bool:
    k = _key(r.text)
    if len(k) < 4 or len(r.text) > HEADER_MAX_CHARS or looks_like_clause(r.text):
        return False
    b = int(r.y0 / height * 50)
    if (k, b) in headers:
        return True
    if len(k) < KEY_LETTERS:
        return False
    # OCR spells one running head several ways; accept a one-letter miss at (almost) the same height
    return any(_close(k, hk, 1) for hk, hb in headers if abs(hb - b) <= 1 and len(hk) >= KEY_LETTERS)


@dataclass
class CleanStats:
    page_number_rows: int = 0  # page numbers and judge-name running heads
    header_footer_rows: int = 0
    margin_words: int = 0
    junk_rows: int = 0
    repeated_headings: int = 0
    unrecoverable_chars: int = 0
    header_samples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- paragraphs
def _vocab(rows_by_page: list[list[Row]]) -> Counter:
    c: Counter = Counter()
    for rows in rows_by_page:
        for r in rows:
            for tok in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", r.text.lower()):
                c[tok] += 1
    return c


def _join_lines(prev: str, nxt: str, vocab: Counter) -> str:
    m = re.search(r"([^\W\d_]+)-$", prev[-40:])  # only the tail: paragraphs grow long
    if m and nxt[:1].islower():
        first = re.match(r"[^\W\d_]+", nxt)
        stem = m.group(1).lower()
        if first:
            joined, hyph = stem + first.group(0).lower(), stem + "-" + first.group(0).lower()
            if vocab.get(hyph, 0) > 0 and vocab.get(joined, 0) == 0:
                return prev + nxt  # genuine compound, e.g. non-existent: keep the hyphen
            return prev[:-1] + nxt  # hyphenation point: rejoin the word
    return prev + " " + nxt


def _page_geometry(rows: list[Row], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """(left edge, right edge, line pitch) of the body text on one page."""
    band = _band(rows)
    pitches = [b.y0 - a.y0 for a, b in zip(rows, rows[1:]) if 0 < b.y0 - a.y0 < 3 * max(a.h, 1)]
    pitch = statistics.median(pitches) if len(pitches) >= 3 else fallback[2]
    return (band[0], band[1], pitch) if band else (fallback[0], fallback[1], pitch)


def assemble_paragraphs(rows_by_page: list[list[Row]]) -> str:
    vocab = _vocab(rows_by_page)
    all_rows = [r for rows in rows_by_page for r in rows]
    if not all_rows:
        return ""
    fb = (statistics.median(r.x0 for r in all_rows), max(r.x1 for r in all_rows),
          statistics.median(r.h for r in all_rows) * 1.25)
    paras: list[str] = []
    cur = ""
    prev: Row | None = None
    prev_geo = fb
    for rows in rows_by_page:
        geo = _page_geometry(rows, fb)
        left, right, pitch = geo
        for r in rows:
            if prev is None or not cur:
                cur = r.text
            else:
                p_left, p_right, _ = prev_geo
                width = max(p_right - p_left, 1.0)
                short = prev.x1 < p_right - 0.12 * width
                ended = bool(SENTENCE_END_RE.search(prev.text.rstrip()))
                allcaps = prev.text.isupper() and len(prev.text) < 90
                same_page = prev.page == r.page
                gap = same_page and (r.y0 - prev.y0) > PARA_GAP * pitch
                indent = r.x0 - prev.x0 > max(6.0, 0.9 * r.h) if same_page else r.x0 - left > max(6.0, 0.9 * r.h)
                boundary = (ended or short or allcaps) and (indent or bool(MARKER_RE.match(r.text)) or short or allcaps)
                if gap or boundary:
                    paras.append(cur)
                    cur = r.text
                else:
                    cur = _join_lines(cur, r.text, vocab)
            prev, prev_geo = r, geo
    if cur:
        paras.append(cur)
    return "\n\n".join(paras)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def drop_repeated_headings(paras: list[str]) -> tuple[list[str], int]:
    """Bare Acts from India Code print each section's number and marginal title, then repeat both
    at the start of the section text ('1.' / 'Title.' / '1. Title.--...'). Drop the first copy."""
    out: list[str] = []
    removed = 0
    i = 0
    while i < len(paras):
        if re.fullmatch(r"\d{1,3}[A-Z]{0,2}\.", paras[i].strip()):
            for k in (1, 2, 3):
                if i + k < len(paras):
                    head = _norm(" ".join(paras[i:i + k]))
                    nxt = _norm(re.sub(r"^\s*\d{0,3}\*+\s*\[?", "", paras[i + k]))
                    if len(head) >= 6 and nxt.startswith(head):
                        removed += k
                        i += k
                        break
            else:
                out.append(paras[i])
                i += 1
            continue
        out.append(paras[i])
        i += 1
    return out, removed


NOISE_PARA_RE = re.compile(
    r"(?:[-–—]\s*)?(?:page|pg\.?)?\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?(?:\s*[-–—])?|\[\d{1,4}\]", re.I)


def page_noise_report(text: str) -> dict:
    """Paragraphs that are nothing but a page number ('12', '- 12 -', 'Page 12 of 40', '[12]').

    Clause markers such as '(2)', section numbers such as '12.' and Roman-numeral headings are not noise.
    """
    bad = [p.strip() for p in text.split("\n\n") if NOISE_PARA_RE.fullmatch(p.strip())]
    return {"total": len(bad), "examples": bad[:10]}


# ---------------------------------------------------------------- driver
def clean_document(pages: list[PageWords], doc_type: str) -> tuple[str, CleanStats]:
    """Return (clean text, stats). doc_type 'judgment' also strips report margin marks."""
    stats = CleanStats()
    judgment = doc_type == "judgment"
    pages_rows: list[tuple[PageWords, list[Row]]] = []
    for pw in pages:
        words = [w for w in pw.words if not STRAY_SYMBOL_RE.match(w.text)]
        if judgment and words:
            words, dropped = strip_margin_marks(words, pw.width, pw.number)
            stats.margin_words += dropped
        pages_rows.append((pw, build_rows(words, pw.width, pw.number, columns=not judgment)))

    headers = find_running_headers(pages_rows)
    sequence = find_sequence_page_numbers(pages_rows)
    kept_by_page: list[list[Row]] = []
    for pw, rows in pages_rows:
        zone = {id(r) for r in _zone_rows(rows, pw.height)}
        kept: list[Row] = []
        for r in rows:
            in_zone = id(r) in zone
            if (in_zone and (is_page_number(r.text) or (judgment and is_judge_header(r.text)))) \
                    or id(r) in sequence:
                stats.page_number_rows += 1
            elif in_zone and is_header_row(r, pw.height, headers):
                stats.header_footer_rows += 1
                if len(stats.header_samples) < 8 and r.text not in stats.header_samples:
                    stats.header_samples.append(r.text)
            elif judgment and len(r.words) == 1 and LONE_MARKER_RE.match(r.text.strip()):
                stats.margin_words += 1  # a marker left alone once its neighbour in the margin was dropped
            elif is_junk_row(r.text):
                stats.junk_rows += 1
            else:
                kept.append(r)
        kept_by_page.append(kept)

    text = assemble_paragraphs(kept_by_page)
    text = normalize_text(text)
    text, stats.unrecoverable_chars = fix_replacement_chars(text)
    paras = [p for p in (tidy_spacing(p) for p in text.split("\n\n")) if p]
    if not judgment:
        paras, stats.repeated_headings = drop_repeated_headings(paras)
    return "\n\n".join(paras), stats
