"""Split a processed document into retrieval chunks. Pure functions: no file or network access.

Acts and the Constitution are split on their own numbering (sections, or articles for a
constitution), not on a layout specific to any one Act:

* Numbered paragraphs ("12.", "53A.") are candidate unit starts. The longest counting-up chains of
  candidates are the real numbering sequences; a footnote such as "1. Subs. by Act 3 of 1951"
  cannot extend one because its number does not follow on.
* Several chains usually exist (table of contents, body, correspondence tables). The chain
  covering the most text is the body; the others become `index` / `reference` chunks, kept but
  labelled, so the contents list never competes with the real section text.
* PART / CHAPTER headings give each unit its place in the hierarchy; a SCHEDULE heading ends
  numbered units (schedule entries have their own numbering) and is split by size.
* One chunk per section. A section longer than the token budget is cut only between its own
  clauses, and a proviso, explanation, illustration or sub-clause "(a)" always stays with the
  clause it qualifies; only a single clause longer than the budget is cut, at a sentence end.

Judgments are packed paragraph by paragraph into 300-500 token windows with a one-paragraph
overlap, starting a new window when the kind of passage changes. Each window is tagged
(facts / issues / arguments / ratio / order / headnote) only where cue phrases make that
detectable; otherwise the tag is empty.

Every chunk records the exact paragraph and character span it came from, keeps the document's
legal_status / usage_note (so a repealed text can never lose its warning), and its `text` is a
verbatim slice of the processed text.
"""
import difflib
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

CHUNKER_VERSION = "2.3"
MIN_FRAGMENT_TOKENS = 5  # an unnumbered scrap smaller than this (a stray heading or number) is not a chunk
# words and punctuation marks; Devanagari vowel signs belong to their word (plain \w would split them off)
_PIECE_RE = re.compile(r"[\w\u0900-\u0963\u0966-\u097f]+|[^\w\s\u0900-\u097f]|[\u0964\u0965]")
TOKENS_PER_PIECE = 1.5  # calibrated against multilingual-e5-small: see estimate_tokens


def estimate_tokens(text: str) -> int:
    """Token count without a tokenizer: words and punctuation marks, times 1.5.

    Calibrated on 2,500 real chunks against the embedding model's own tokenizer: it under-counts
    for 0.04% of chunks, and then by at most 4%. It over-counts the typical chunk by about 25%
    (noisy OCR text and numbers split into many tokens, so a cheaper rule is unsafe), which means a
    chunk estimated at <= 480 tokens is always below the model's 512-token input limit.
    """
    return math.ceil(len(_PIECE_RE.findall(text)) * TOKENS_PER_PIECE)


@dataclass(frozen=True)
class ChunkParams:
    max_tokens: int = 480  # estimated tokens; the embedding model's input limit is 512 real tokens
    min_tokens: int = 300  # judgments: a window is not closed on a change of passage type below this
    overlap_tokens: int = 60  # judgments: repeat the previous window's last paragraph if it is this small
    min_chain: int = 5  # shortest counting-up run of numbered paragraphs treated as a sequence
    max_gap: int = 60  # largest jump in numbering inside a chain (omitted/repealed provisions)

    def fingerprint(self) -> str:
        blob = json.dumps({**self.__dict__, "version": CHUNKER_VERSION}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# the word used for a numbered unit; documents of other types default to "Section"
UNIT_LABELS = {"constitution": "Article"}

# "12. text", "53A. text", "115.(1) text" and "8*[4. text" (a number preceded by an amendment footnote mark)
START_RE = re.compile(r"^(?:\d{0,3}\*+\s*\[\s*)?(\d{1,4})([A-Z]{0,3})\.(?=[\s(])\s*(.+)$", re.S)
# amendment notes printed under each page: "1. Subs. by Act 3 of 1951 ..." are not provisions
FOOTNOTE_RE = re.compile(
    r"^(?:Subs\.|Ins\.|Rep\.|Omitted|Added|Cl\.|Sub-s\.|Ibid|Vide\b|See\b|The words|The figures|Now\b|Renumbered)", re.I)
CONTAINER_RE = re.compile(
    r"^(?:(PART|CHAPTER|SCHEDULE|APPENDIX|ANNEXURE)\s+([IVXLCDM]+|\d+[A-Z]{0,2}|[A-Z])\b|THE\s+(?:[A-Z]+\s+)?SCHEDULES?\b)(.*)$")
INLINE_HEADING_RE = re.compile(r"^\d{1,4}[A-Z]{0,3}\.\s*(.{3,200}?[a-z0-9\)\]])\.\s*[-–—]{1,2}")
CLAUSE_OPENER_RE = re.compile(r"^(?:Explanation|Illustrations?\b|Exception|Provided|\(|\d)", re.I)
# a paragraph that qualifies the clause above it and must travel with it
GLUE_RE = re.compile(r"^(?:(?:Provided|Explanation|Illustrations?|Exception)\b|\([a-z]{1,2}\)|\([ivxl]{1,5}\))", re.I)
GLUE_SUBCLAUSE_RE = re.compile(r"^(?:\([a-z]{1,2}\)|\([ivxl]{1,5}\))")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:?!।॥])\s+")  # includes the Devanagari danda


@dataclass
class Seg:
    """One paragraph (or one sentence-sized piece of an over-long paragraph) of the processed text."""
    idx: int  # paragraph index in the document
    text: str
    label: str = ""  # judgments: facts / issues / arguments / ratio / order / headnote
    glue: bool = False  # statutes: must stay with the previous segment
    cues: tuple = ()


@dataclass
class Chunk:
    chunk_type: str  # section | article | schedule | judgment_text | front_matter | index | reference | text
    text: str
    unit_label: str = ""
    unit_number: str = ""
    heading: str = ""
    group_heading: str = ""
    hierarchy: dict = field(default_factory=dict)  # part / chapter / schedule
    part_index: int = 1
    part_count: int = 1
    paras: list[int] = field(default_factory=list)  # numbered paragraphs inside a judgment chunk
    pstart: int = 0  # first / last paragraph index the chunk covers (overlap paragraph included)
    pend: int = 0
    overlap: int = 0  # leading paragraphs repeated from the previous chunk
    section_type: str = ""
    cues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- small helpers
def paragraph_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Paragraphs of the processed text and the [start, end) character span of each in that text."""
    paras, spans, pos = [], [], 0
    for raw in text.split("\n\n"):
        p = raw.strip()
        if p:
            start = text.find(p, pos)
            paras.append(p)
            spans.append((start, start + len(p)))
            pos = start + len(p)
    return paras, spans


def _key(num: str, suffix: str) -> tuple[int, str]:
    return int(num), suffix


def _sentences(par: str, max_tokens: int) -> list[str]:
    """Split one over-long paragraph at sentence ends; a single endless sentence is cut at spaces."""
    out, cur = [], ""
    for s in SENTENCE_SPLIT_RE.split(par):
        if estimate_tokens(s) > max_tokens:  # one sentence over the budget: pack its words
            if cur:
                out.append(cur)
                cur = ""
            for w in s.split(" "):
                if cur and estimate_tokens(cur + " " + w) > max_tokens:
                    out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
        elif cur and estimate_tokens(cur + " " + s) > max_tokens:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _join(segs: list[Seg]) -> str:
    """Paragraphs are separated by a blank line; pieces of one split paragraph are rejoined with a space."""
    out = ""
    for k, s in enumerate(segs):
        out += s.text if k == 0 else ((" " if s.idx == segs[k - 1].idx else "\n\n") + s.text)
    return out


def _tokens(segs: list[Seg]) -> int:
    return sum(estimate_tokens(s.text) for s in segs)


def make_segs(idxs: list[int], paras: list[str], labels: Optional[list] = None, glue: bool = False) -> list[Seg]:
    out = []
    for i in idxs:
        lab, cues = labels[i] if labels else ("", ())
        out.append(Seg(i, paras[i], lab, glue and bool(GLUE_RE.match(paras[i])), cues))
    return out


def pack(segs: list[Seg], p: ChunkParams, overlap: bool = False, by_label: bool = False) -> list[tuple[list[Seg], int]]:
    """Group segments into windows of at most max_tokens; returns (segments, leading-overlap-count).

    A paragraph is never cut unless it alone exceeds the budget (then at sentence ends). Segments
    flagged `glue` stay with the one before them. With by_label, a window is closed early when the
    passage type changes, once it holds at least min_tokens.
    """
    flat: list[Seg] = []
    for s in segs:
        if estimate_tokens(s.text) > p.max_tokens:
            flat.extend(Seg(s.idx, piece, s.label, s.glue if not flat or flat[-1].idx != s.idx else False, s.cues)
                        for piece in _sentences(s.text, p.max_tokens))
        else:
            flat.append(s)
    groups: list[list[Seg]] = []
    for s in flat:
        if s.glue and groups and _tokens(groups[-1]) + estimate_tokens(s.text) <= p.max_tokens:
            groups[-1].append(s)
        else:
            groups.append([s])
    windows: list[tuple[list[Seg], int]] = []
    cur: list[Seg] = []
    kept = 0
    for g in groups:
        gt = _tokens(g)
        cur_tokens = _tokens(cur)
        label_change = (by_label and cur and cur_tokens >= p.min_tokens and g[0].label and cur[-1].label
                        and g[0].label != cur[-1].label)
        if cur and len(cur) > kept and (cur_tokens + gt > p.max_tokens or label_change):
            windows.append((cur, kept))
            keep = [cur[-1]] if overlap and estimate_tokens(cur[-1].text) <= p.overlap_tokens else []
            cur, kept = keep[:], len(keep)
        cur.extend(g)
    if len(cur) > kept:
        windows.append((cur, kept))
    if len(windows) > 1:  # fold a tiny tail into the previous window if it still fits comfortably
        tail, tail_kept = windows[-1]
        body = tail[tail_kept:]
        if _tokens(body) < p.min_tokens // 2 and _tokens(windows[-2][0]) + _tokens(body) <= int(p.max_tokens * 1.15):
            windows.pop()
            windows[-1][0].extend(body)
    return windows


# ---------------------------------------------------------------- numbering chains
def find_chains(paras: list[str], p: ChunkParams) -> list[list[tuple[int, tuple[int, str]]]]:
    """Longest counting-up runs of numbered paragraphs, as lists of (paragraph index, (number, suffix))."""
    cands = []
    for i, par in enumerate(paras):
        m = START_RE.match(par)
        if m and not FOOTNOTE_RE.match(m.group(3).strip()):
            cands.append((i, _key(m.group(1), m.group(2))))
    chains = []
    pool = cands
    while len(pool) >= p.min_chain and len(chains) < 8:
        n = len(pool)
        length, first, prev = [1] * n, [pool[i][0] for i in range(n)], [-1] * n
        for i in range(n):
            for j in range(i):
                kj, ki = pool[j][1], pool[i][1]
                if kj < ki and ki[0] - kj[0] <= p.max_gap:
                    cand = (length[j] + 1, first[j])
                    # longest chain; on a tie the one that starts later (a shorter stretch of the document)
                    if cand > (length[i], first[i]):
                        length[i], first[i], prev[i] = cand[0], cand[1], j
        end = max(range(n), key=lambda k: (length[k], -(pool[k][0] - first[k])))
        if length[end] < p.min_chain:
            break
        chain, k = [], end
        while k != -1:
            chain.append(pool[k])
            k = prev[k]
        chain.reverse()
        chains.append(chain)
        used = {c[0] for c in chain}
        pool = [c for c in pool if c[0] not in used]
    return chains


# ---------------------------------------------------------------- statutes / constitutions
def _container(par: str) -> Optional[tuple[str, str, str]]:
    """(kind, label, rest-of-heading) if the paragraph is a PART / CHAPTER / SCHEDULE heading."""
    if len(par) > 140:
        return None
    m = CONTAINER_RE.match(par.strip())
    if not m:
        return None
    if m.group(1):
        return m.group(1).lower(), f"{m.group(1)} {m.group(2)}", m.group(3).strip(" .-:–—")
    return "schedule", par.strip(), ""


def _title_lines(paras: list[str], i: int) -> tuple[str, int]:
    """Capitalised title paragraphs that directly follow a container heading ('OF PUNISHMENTS')."""
    parts, j = [], i
    while j < len(paras) and len(parts) < 2 and paras[j].isupper() and len(paras[j]) <= 120 \
            and not _container(paras[j]) and not START_RE.match(paras[j]):
        parts.append(paras[j])
        j += 1
    return " ".join(parts), j - i


def _is_lead_heading(par: str) -> bool:
    """A short unpunctuated line directly above a numbered unit (a marginal / cross heading)."""
    return (len(par) <= 160 and not START_RE.match(par) and not CLAUSE_OPENER_RE.match(par)
            and not _container(par) and par[:1].isupper() and not par.rstrip().endswith((".", ";", ":", ",", "-", "—")))


def _toc_titles(paras: list[str], chain) -> dict[tuple[int, str], str]:
    out = {}
    for i, key in chain:
        m = START_RE.match(paras[i])
        if m:
            out[key] = m.group(3).strip().rstrip(".").strip()
    return out


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _same_title(a: str, b: str) -> bool:
    """Headings differ in commas/case between the contents list and the body; compare the letters only."""
    x, y = _letters(a), _letters(b)
    if len(x) < 4 or len(y) < 4:
        return False
    return x == y or (min(len(x), len(y)) >= 12 and (x.startswith(y) or y.startswith(x))) \
        or difflib.SequenceMatcher(None, x, y).ratio() >= 0.88


def _walk_back(paras: list[str], k: int) -> int:
    """Move k up over the PART / CHAPTER headings (with their capitalised title lines) directly above it.

    Title lines are only absorbed when a heading sits above them, so a document title or an
    'ARRANGEMENT OF SECTIONS' line is never mistaken for part of the numbered text.
    """
    while True:
        j, titles = k, 0
        while j > 0 and titles < 2 and paras[j - 1].isupper() and len(paras[j - 1]) <= 120 \
                and not _container(paras[j - 1]) and not START_RE.match(paras[j - 1]):
            j -= 1
            titles += 1
        if j > 0 and _container(paras[j - 1]):
            k = j - 1
        else:
            return k


def _window_chunks(kind_: str, wins, base: Chunk, label: str = "") -> list[Chunk]:
    out = []
    for n, (segs, kept) in enumerate(wins, 1):
        out.append(Chunk(kind_, _join(segs), base.unit_label, base.unit_number, base.heading or label,
                         base.group_heading, dict(base.hierarchy), n, len(wins), pstart=segs[0].idx,
                         pend=segs[-1].idx, overlap=kept))
    return out


def chunk_numbered(paras: list[str], chains, doc: dict, p: ChunkParams) -> tuple[list[Chunk], dict]:
    unit = UNIT_LABELS.get(doc.get("type", ""), "Section")
    kind = unit.lower()
    members = sorted((i, key, c) for c, ch in enumerate(chains) for i, key in ch)
    nxt = {m[0]: members[k + 1][0] if k + 1 < len(members) else len(paras) for k, m in enumerate(members)}
    span_len = [0] * len(chains)
    for i, _key_, c in members:
        span_len[c] += sum(len(x) for x in paras[i:nxt[i]])
    body_c = max(range(len(chains)), key=lambda c: span_len[c])
    body = chains[body_c]
    elem = {i: key for i, key in body}
    first, last = body[0][0], body[-1][0]
    other_starts = sorted(i for i, _k, c in members if c != body_c)

    # leading-heading paragraph of each unit, and where the body region begins
    toc: dict[tuple[int, str], list[str]] = {}
    for c, ch in enumerate(chains):
        if c != body_c and ch[0][0] < first:
            for key_, title_ in _toc_titles(paras, ch).items():
                toc.setdefault(key_, []).append(title_)
    # a heading directly above a unit: it matches that unit's title in the contents list, or (when
    # there is no contents list) looks like one -- short, capitalised and unpunctuated
    lead = {i - 1 for i, key_ in elem.items() if i - 1 >= 0 and i - 1 not in elem and not _container(paras[i - 1])
            and len(paras[i - 1]) <= 160 and (any(_same_title(paras[i - 1], t) for t in toc.get(key_, ()))
                                              or _is_lead_heading(paras[i - 1]))}
    group_lead = {i - 1 for i in lead if i - 1 >= 0 and i - 1 not in elem and i - 1 not in lead
                  and not _container(paras[i - 1]) and _is_lead_heading(paras[i - 1])}
    body_start = _walk_back(paras, first - 1 if first - 1 in lead else first)
    # the body ends at the next other-numbering start or the first schedule heading after the last unit
    end = len(paras)
    for i in range(last + 1, len(paras)):
        c = _container(paras[i])
        if i in other_starts or (c and c[0] == "schedule"):
            end = i
            break
    chunks: list[Chunk] = []
    ctx = {"part": "", "chapter": ""}

    def emit_region(kind_: str, idxs: list[int], hierarchy: dict, label: str = "") -> None:
        wins = pack(make_segs(idxs, paras), p)
        chunks.extend(_window_chunks(kind_, wins, Chunk(kind_, "", hierarchy=hierarchy), label))

    # front matter and index (before the body)
    toc_paras = {i for c, ch in enumerate(chains) if c != body_c and ch[0][0] < first
                 for i in range(_walk_back(paras, ch[0][0]), ch[-1][0] + 1)}
    pre = list(range(0, body_start))
    front = [i for i in pre if i not in toc_paras]
    index = [i for i in pre if i in toc_paras]
    if front:
        emit_region("front_matter", front, {})
    if index:
        emit_region("index", index, {})

    # the numbered body
    cur: Optional[dict] = None
    pending: Optional[str] = None
    pending_group = ""

    def finish() -> None:
        nonlocal cur
        if not cur:
            return
        wins = pack(make_segs(cur["idxs"], paras, glue=True), p)
        base = Chunk(kind, "", unit, cur["number"], cur["heading"], cur["group"], dict(cur["hier"]))
        chunks.extend(_window_chunks(kind, wins, base))
        cur = None

    i = body_start
    while i < end:
        par = paras[i]
        cont = _container(par)
        if i in elem:
            finish()
            sm = START_RE.match(par)
            num = f"{sm.group(1)}{sm.group(2)}"  # as printed (e.g. Devanagari digits)
            m = INLINE_HEADING_RE.match(par)
            heading, group = "", pending_group
            if m:
                heading, group = m.group(1).strip(), " / ".join(x for x in (pending_group, pending) if x)
            elif pending:
                heading = pending
            else:
                heading = (toc.get(elem[i]) or [""])[0]
            pending = None
            pending_group = ""
            cur = {"number": num, "heading": heading, "group": group, "hier": {k_: v for k_, v in ctx.items() if v}, "idxs": [i]}
        elif i in lead and i + 1 in elem:
            finish()
            pending = par.strip().rstrip(". ")
        elif i in group_lead:
            finish()  # a cross-heading ("Of theft") above a unit's own heading
            pending_group = par.strip().rstrip(". ")
        elif cont and cont[0] in ("part", "chapter"):
            finish()
            title, used = _title_lines(paras, i + 1)
            ctx[cont[0]] = " - ".join(x for x in (cont[1], cont[2], title) if x)
            if cont[0] == "part":
                ctx["chapter"] = ""
            i += used
        elif cur:
            cur["idxs"].append(i)
        else:
            cur = {"number": "", "heading": "", "group": "", "hier": {k_: v for k_, v in ctx.items() if v}, "idxs": [i]}
        i += 1
    finish()
    # an unnumbered run (e.g. text under a heading before the first unit) is plain text, not a unit
    for c in chunks:
        if c.chunk_type == kind and not c.unit_number:
            c.chunk_type, c.unit_label, c.heading, c.group_heading = "text", "", "", ""

    # after the body: schedules, other numbering tables, anything else
    zone, label, hier = "back_matter", "", {}
    buf: list[int] = []

    def flush() -> None:
        if buf:
            emit_region(zone if zone != "back_matter" else "text", buf[:], hier, label)
            buf.clear()

    other_set = set(other_starts)
    for i in range(end, len(paras)):
        cont = _container(paras[i])
        if cont and cont[0] == "schedule":
            flush()
            zone, label, hier = "schedule", cont[1], {"schedule": cont[1]}
        elif i in other_set and zone != "reference" and zone != "schedule":
            flush()
            zone, label, hier = "reference", "", {}
        buf.append(i)
    flush()

    before = len(chunks)
    chunks = [c for c in chunks if not (c.chunk_type == "text" and estimate_tokens(c.text) < MIN_FRAGMENT_TOKENS)]
    nums = [k[0] for _, k in body]
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    info = {"structure": "numbered", "unit": unit, "units": len(body), "first": f"{nums[0]}", "last": f"{nums[-1]}",
            "missing_numbers": missing[:50], "missing_count": len(missing), "other_numbering_runs": len(chains) - 1,
            "dropped_fragments": before - len(chunks)}
    return chunks, info


# ---------------------------------------------------------------- judgments: passage types
# (label, pattern). Cue phrases of Indian law reports. A tag needs two different cues in a chunk,
# so one stray phrase quoted from another case does not tag it; the order is the exception (below).
_ORDER_RE = re.compile(
    r"\b(?:(?:appeals?|petitions?|writ petitions?|applications?|suits?|references?)(?:\s+(?:is|are|stands?|be|shall be|hereby))*"
    r"(?:\s+accordingly)?\s+(?:allowed|dismissed|disposed of|answered)|dispose[sd]?\s+of|disposed\s+(?:in|of)|"
    r"no order as to costs|in the result|order accordingly|"
    r"(?:conviction|sentence|order|judgment|decree)\s+(?:is|are|be|stands?)\s+(?:set aside|quashed|upheld|confirmed|restored))\b", re.I)
_PASSAGE_CUES = [
    ("order", _ORDER_RE),
    ("issues", re.compile(
        r"\b(?:the\s+following\s+(?:questions?|issues?|points)|(?:questions?|issues?|points)\s+(?:that|which)\s+"
        r"(?:arises?|fall|falls|need|requires?|arose)|points?\s+for\s+(?:determination|consideration)|issues?\s+(?:framed|raised)|"
        r"the\s+question\s+(?:that\s+)?(?:arises|falls)\s+for)\b", re.I)),
    ("arguments", re.compile(
        r"\b(?:submitted|contended|argued|urged|contentions?|submissions?|learned (?:senior )?counsel|"
        r"it was (?:also )?(?:argued|submitted|contended)|on behalf of the (?:appellant|respondent|petitioner)s?)\b", re.I)),
    ("facts", re.compile(
        r"\b(?:the facts|brief facts|facts of the case|factual (?:matrix|background)|background of the case|"
        r"filed a (?:writ )?petition|being aggrieved|was (?:detained|arrested|convicted|dismissed|married)|"
        r"in the year \d{4})\b", re.I)),
    ("ratio", re.compile(
        r"\b(?:we\s+(?:are of the (?:view|opinion)|hold that|are satisfied|are unable|find that)|in our (?:view|opinion|judgment)|"
        r"it follows that|in (?:the )?(?:light|view) of (?:the )?(?:above|foregoing)|the ratio|principle laid down)\b", re.I)),
]
_DELIVERED_RE = re.compile(r"(?:judgment|order|opinion) of the (?:court|bench) was delivered by|delivered the following (?:judgment|order)|"
                           r"^\s*(?:[A-Z][A-Za-z. ,']{2,60}),\s*(?:C\.?\s?J\.?(?:\s?I\.?)?|J\.?|JJ\.?)\s*[.—-]", re.I)
MIN_DISTINCT_CUES = 2  # different cue phrases needed in a chunk before it is tagged
END_MARK = "@end"  # an order cue near the end of the judgment is enough on its own


def label_paragraphs(paras: list[str]) -> list[tuple[str, tuple]]:
    """(label, cue phrases) per paragraph; ('', ()) where nothing is detectable."""
    n = len(paras)
    start = next((i for i, par in enumerate(paras[:max(1, int(n * 0.3))]) if _DELIVERED_RE.search(par[:200])), -1)
    tail = n - max(4, int(n * 0.03))  # the operative order sits in the last few paragraphs
    out: list[tuple[str, tuple]] = []
    for i, par in enumerate(paras):
        if i < start:
            out.append(("headnote", ()))  # head note, appearances and catch-words before the opinion begins
            continue
        best, hits = "", ()
        for label, rx in _PASSAGE_CUES:
            found = tuple(m.group(0).lower() for m in rx.finditer(par))
            if len(found) > len(hits):
                best, hits = label, found
        if i >= tail:
            ends = tuple(m.group(0).lower() for m in _ORDER_RE.finditer(par))
            if ends and len(par) < 600:
                best, hits = "order", ends + (END_MARK,)
        out.append((best, hits))
    return out


def tag_chunk(segs: list[Seg]) -> tuple[str, list[str]]:
    """Chunk-level passage type, if the evidence is enough; ('', []) when it is not detectable."""
    cues: dict[str, set] = {}
    for s in segs:
        if s.label == "headnote":
            cues.setdefault("headnote", set()).add("before the opinion")
        elif s.label:
            cues.setdefault(s.label, set()).update(s.cues)
    best, best_n = "", 0
    for label, found in cues.items():
        need = 1 if label == "headnote" or (label == "order" and END_MARK in found) else MIN_DISTINCT_CUES
        if len(found - {END_MARK}) >= need and len(found - {END_MARK}) > best_n:
            best, best_n = label, len(found - {END_MARK})
    return (best, sorted(cues[best] - {END_MARK})[:5]) if best else ("", [])


# ---------------------------------------------------------------- judgments and unstructured text
def chunk_windows(paras: list[str], p: ChunkParams, chunk_type: str) -> list[Chunk]:
    judgment = chunk_type == "judgment_text"
    labels = label_paragraphs(paras) if judgment else None
    wins = pack(make_segs(list(range(len(paras))), paras, labels), p, overlap=True, by_label=judgment)
    out = []
    for n, (segs, kept) in enumerate(wins, 1):
        nums = []
        for s in segs:
            m = re.match(r"^(\d{1,3})\.\s", s.text)
            if m and judgment:
                nums.append(int(m.group(1)))
        stype, cues = tag_chunk(segs[kept:] or segs) if judgment else ("", [])
        out.append(Chunk(chunk_type, _join(segs), part_index=n, part_count=len(wins), paras=nums,
                         pstart=segs[0].idx, pend=segs[-1].idx, overlap=kept, section_type=stype, cues=cues))
    return out


# ---------------------------------------------------------------- records
def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def locator_for(c: Chunk) -> str:
    if c.unit_number:
        base = f"{c.unit_label} {c.unit_number}"
    elif c.chunk_type == "schedule":
        base = c.heading or "Schedule"
    elif c.paras:
        base = f"para {c.paras[0]}" if len(set(c.paras)) == 1 else f"paras {c.paras[0]}-{c.paras[-1]}"
    else:
        base = ""
    return base


def build_records(doc: dict, chunks: list[Chunk], spans: list[tuple[int, int]]) -> list[dict]:
    title = doc.get("title") or doc.get("doc_id", "")
    is_judgment = doc.get("type") == "judgment"
    records = []
    for seq, c in enumerate(chunks, 1):
        loc = locator_for(c)
        cite_parts = [title]
        if is_judgment and doc.get("citation"):
            cite_parts.append(doc["citation"])
        if loc:
            cite_parts.append(loc)
        trail = [v for v in (c.hierarchy.get("part"), c.hierarchy.get("chapter"), c.hierarchy.get("schedule")) if v]
        label = f"{c.unit_label} {c.unit_number}".strip()
        if c.heading and c.chunk_type in ("section", "article"):
            label = f"{label}. {c.heading}"
        context = " > ".join([title, *trail, *([label] if label else [])])
        text_hash = _sha(c.text)
        records.append({
            "chunk_id": _sha(doc["doc_id"], c.chunk_type, locator_for(c), str(c.part_index), text_hash)[:20],
            "doc_id": doc["doc_id"], "seq": seq, "chunk_type": c.chunk_type,
            "unit_label": c.unit_label, "unit_number": c.unit_number, "heading": c.heading,
            "group_heading": c.group_heading, "hierarchy": c.hierarchy,
            "locator": loc, "cite": ", ".join(cite_parts), "context": context,
            "part_index": c.part_index, "part_count": c.part_count, "paragraph_numbers": c.paras,
            "section_type": c.section_type, "section_cues": c.cues,
            # exact source location: paragraph indexes and character offsets into the processed text
            "para_start": c.pstart, "para_end": c.pend, "overlap_paragraphs": c.overlap,
            "char_start": spans[c.pstart][0], "char_end": spans[c.pend][1],
            "doc_title": title, "doc_type": doc.get("type", ""), "act": doc.get("act", ""),
            "legal_status": doc.get("legal_status", ""), "usage_note": doc.get("usage_note", ""),
            "court": doc.get("court", ""), "date": doc.get("date", ""), "citation": doc.get("citation", ""),
            "source_url": doc.get("source_url", ""),
            "content_hash": text_hash, "char_count": len(c.text), "token_estimate": estimate_tokens(c.text),
            "chunker_version": CHUNKER_VERSION,
            "text": c.text,
        })
    return records


def chunk_document(doc: dict, p: ChunkParams = ChunkParams()) -> tuple[list[dict], dict]:
    """Return (chunk records, structure info) for one processed document."""
    paras, spans = paragraph_spans(doc.get("text", ""))
    if not paras:
        return [], {"structure": "empty"}
    if doc.get("type") == "judgment":
        chunks, info = chunk_windows(paras, p, "judgment_text"), {"structure": "paragraph_windows"}
    else:
        chains = find_chains(paras, p)
        if chains:
            chunks, info = chunk_numbered(paras, chains, doc, p)
        else:
            chunks, info = chunk_windows(paras, p, "text"), {"structure": "no_numbering_found"}
    return build_records(doc, chunks, spans), info
