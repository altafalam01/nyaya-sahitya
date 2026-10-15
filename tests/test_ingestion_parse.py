import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import parse as parse_mod
from ingestion.ocr import INSTALL_HELP, OcrStatus, check_tesseract
from ingestion.pdf_extract import BLANK, OCR, OCR_SKIPPED, TEXT, extract_pages
from ingestion.textclean import (clean_document, fix_replacement_chars, normalize_text, page_noise_report)

ROOT = Path(__file__).resolve().parent.parent
PAGE_W, PAGE_H = 595, 842
UNAVAILABLE = OcrStatus(False, None, None, (), "not installed")
TESSERACT = check_tesseract()

BODY = [
    "The petitioner was detained under the Preventive Deten-",
    "tion Act and applied to this Court for a writ of habeas",
    "corpus. The question is whether the detention was lawful.",
    "Counsel for the State argued that the law was non-",
    "existent in this form before the amendment was made.",
]


def make_pdf(path: Path, n_pages: int, *, header=True, footer=True, margin_letters=False, body=BODY) -> Path:
    doc = pymupdf.open()
    for p in range(1, n_pages + 1):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if header:  # running head whose number changes on every page, alternating sides like a law report
            text = f"{p + 100}   SUPREME COURT REPORTS   [1950]" if p % 2 else "[1950]   SUPREME COURT REPORTS   " + str(p + 100)
            page.insert_text((72, 60), text, fontsize=9)
        y = 200  # below the header band, otherwise identical text on every page would look like a running head
        for i, line in enumerate(body):
            x = 92 if i == 0 else 72  # first-line indent
            page.insert_text((x, y), line, fontsize=11)
            if margin_letters and i % 2 == 0:
                page.insert_text((40, y), "ABC"[i // 2], fontsize=9)
            y += 15
        page.insert_text((72, y + 20), f"Page {p} of body continues here in sentence {p}.", fontsize=11)
        page.insert_text((72, y + 40), "A non-existent remedy cannot be enforced by the Court.", fontsize=11)
        if footer:
            page.insert_text((290, 800), f"- {p + 100} -", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path) -> Path:
    """A page that is only a picture of text, with no text layer, like a scanned judgment."""
    src = pymupdf.open()
    sp = src.new_page(width=PAGE_W, height=PAGE_H)
    sp.insert_text((72, 200), "The Constitution of India guarantees", fontsize=26)
    sp.insert_text((72, 260), "equality before the law to everyone", fontsize=26)
    png = sp.get_pixmap(dpi=200).tobytes("png")
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=png)
    doc.new_page(width=PAGE_W, height=PAGE_H)  # a genuinely blank page
    doc.save(path)
    return path


def meta_for(pdf: Path, **extra) -> dict:
    data = pdf.read_bytes()
    meta = {"source_id": pdf.stem, "name": "Test Document", "doc_type": "judgment", "url": "https://example.org/page",
            "file_url": "https://example.org/file.pdf", "sha256": hashlib.sha256(data).hexdigest(),
            "date_fetched": "2026-01-01", "method": "automated"}
    meta.update(extra)
    return meta


# ---------------------------------------------------------------- cleaning
def test_page_numbers_and_running_headers_are_removed(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", 10)
    text, stats = clean_document(extract_pages(pdf, "off"), "judgment")
    assert "SUPREME COURT REPORTS" not in text
    assert "[1950]" not in text
    for n in range(101, 111):
        assert str(n) not in text  # neither the header numbers nor the "- N -" footers
    assert stats.header_footer_rows >= 10 and stats.page_number_rows >= 10
    assert page_noise_report(text)["total"] == 0
    assert "Page 3 of body continues here in sentence 3." in text  # real text that mentions 'Page' stays


def test_page_number_sequence_is_found_wherever_it_sits(tmp_path):
    """Statute PDFs print the number mid-page; a bare number that counts up with the pages is noise."""
    doc = pymupdf.open()
    for p in range(1, 9):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for i in range(4):
            page.insert_text((72, 60 + 15 * i), f"{i + 1}. Subs. by Act {p} of 18{p}9, s. {i}, for the original words.", fontsize=10)
        page.insert_text((72, 200 + 10 * p), str(p + 101), fontsize=10)  # lands at a different height each page
        page.insert_text((72, 400), f"Section text for page {p} that goes on and on for a while here.", fontsize=11)
    doc.save(tmp_path / "act.pdf")
    text, _ = clean_document(extract_pages(tmp_path / "act.pdf", "off"), "act")
    assert page_noise_report(text)["total"] == 0
    assert "Section text for page 3" in text


def test_clause_openers_repeated_near_page_tops_are_not_mistaken_for_headers(tmp_path):
    """Statutes open many paragraphs with 'Explanation 1.-' or 'Illustration'; that is text, not a header."""
    doc = pymupdf.open()
    for p in range(1, 13):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text((72, 90), f"Explanation {p}.-It is immaterial by what means the thing is done in case {p}.", fontsize=11)
        page.insert_text((72, 110), "Illustration", fontsize=11)
        page.insert_text((72, 130), f"(a) A person does the act number {p} knowing that it is an offence under this Act.", fontsize=11)
        page.insert_text((72, 400), f"Ordinary section text for page {p} of the Act, long enough to be body text.", fontsize=11)
    doc.save(tmp_path / "act.pdf")
    text, stats = clean_document(extract_pages(tmp_path / "act.pdf", "off"), "act")
    for p in range(1, 13):
        assert f"Explanation {p}.-It is immaterial" in text and f"act number {p} knowing" in text
    assert text.count("Illustration") == 12 and stats.header_footer_rows == 0


def test_line_wraps_are_repaired_and_hyphenation_resolved(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", 4, header=False, footer=False)
    text, _ = clean_document(extract_pages(pdf, "off"), "judgment")
    assert "Preventive Detention Act and applied to this Court" in text  # hyphenation point rejoined
    assert "Deten- tion" not in text and "Deten-\n" not in text
    assert "the law was non-existent in this form" in text  # a genuine compound keeps its hyphen
    assert "\n" not in text.split("\n\n")[0].strip()  # wrapped lines became one paragraph


def test_margin_letters_are_dropped_from_judgments(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", 4, header=False, footer=False, margin_letters=True)
    text, stats = clean_document(extract_pages(pdf, "off"), "judgment")
    assert stats.margin_words > 0
    assert not any(p.strip() in ("A", "B", "C") for p in text.split("\n\n"))
    assert " A The petitioner" not in text and text.startswith("The petitioner")


def test_unicode_normalisation_keeps_hindi_and_fixes_ligatures():
    hindi = "भारत का संविधान १२३ क्‍ष"  # Devanagari digits and a ZWJ must survive
    out = normalize_text("ﬁnal ­" + hindi + "​")
    assert out.startswith("final ") and "भारत का संविधान" in out and "१२३" in out
    assert "‍" in normalize_text("क्‍ष")  # ZWJ is meaningful in Devanagari
    assert "­" not in out and "​" not in out
    assert unicodedata.is_normalized("NFKC", out)
    assert normalize_text("क़") == unicodedata.normalize("NFKC", "क़")


def test_symbol_font_glyphs_are_normalised():
    assert normalize_text("\uf0b7 first point \ue123 second") == "\u2022 first point  second"


def test_replacement_characters_are_repaired_only_when_context_is_clear():
    fixed, left = fix_replacement_chars("woman�s consent. Explanation 2.�For this section �illicit intercourse� means")
    assert fixed == "woman’s consent. Explanation 2.—For this section “illicit intercourse” means"
    assert left == 0
    fixed, left = fix_replacement_chars("junk ��� here")
    assert "�" not in fixed and left == 3


def test_noise_report_flags_page_number_paragraphs():
    assert page_noise_report("Some text.\n\n123\n\nMore text.")["total"] == 1
    assert page_noise_report("Text.\n\nPage 4 of 30\n\nText.")["total"] == 1
    assert page_noise_report("Text.\n\n- 12 -\n\nText.")["total"] == 1
    assert page_noise_report("Text.\n\n(2) Clause two begins here.\n\n12. Section.\n\nII\n\nText.")["total"] == 0


# ---------------------------------------------------------------- OCR
def test_install_help_gives_windows_steps_and_a_verify_command():
    for needle in ("github.com/UB-Mannheim/tesseract", "Hindi", "PATH", "tesseract --version", "--list-langs",
                   "python -m ingestion.parse --check-ocr"):
        assert needle in INSTALL_HELP


def test_missing_tesseract_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "nope.exe"))
    monkeypatch.setattr("ingestion.ocr.shutil.which", lambda _: None)
    monkeypatch.setattr("ingestion.ocr.WINDOWS_DEFAULT_PATHS", ())
    st = check_tesseract()
    assert not st.available and "UB-Mannheim" in st.message


def test_scanned_page_without_ocr_is_flagged_and_blank_page_is_not(tmp_path):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf")
    pages = extract_pages(pdf, "auto", UNAVAILABLE)
    assert [p.source for p in pages] == [OCR_SKIPPED, BLANK]
    record, warnings = parse_mod.parse_document(pdf, meta_for(pdf), "auto", UNAVAILABLE)
    assert any("need OCR but Tesseract is unavailable" in w for w in warnings)
    assert record["extraction"]["pages_ocr_skipped"] == [1] and record["extraction"]["pages_blank"] == 1


@pytest.mark.skipif(not TESSERACT.available, reason="Tesseract not installed")
def test_scanned_page_is_read_by_ocr(tmp_path):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf")
    pages = extract_pages(pdf, "auto", TESSERACT)
    assert pages[0].source == OCR and pages[1].source == BLANK
    record, warnings = parse_mod.parse_document(pdf, meta_for(pdf), "auto", TESSERACT)
    text = record["text"].lower()
    assert "constitution of india" in text and "equality before the law" in text
    assert record["extraction"]["pages_ocr"] == 1 and not warnings


def test_pages_with_a_text_layer_are_never_sent_to_ocr(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", 2)
    pages = extract_pages(pdf, "auto", UNAVAILABLE)
    assert [p.source for p in pages] == [TEXT, TEXT]


# ---------------------------------------------------------------- records
def test_record_has_required_fields_and_leaves_unknown_ones_empty(tmp_path):
    pdf = make_pdf(tmp_path / "sc_test.pdf", 3)
    meta = meta_for(pdf, source_id="sc_test", name="Fallback Name", origin="third_party_mirror",
                    case_metadata={"title": "A v. B", "citation": "[1950] 1 S.C.R. 1", "decision_date": "19-05-1950",
                                   "court": "Supreme Court of India"})
    rec, _ = parse_mod.parse_document(pdf, meta, "auto", UNAVAILABLE)
    for key in ("doc_id", "title", "type", "court", "date", "bench", "citation", "source_url", "text"):
        assert key in rec
    assert (rec["doc_id"], rec["title"], rec["type"], rec["court"], rec["date"], rec["citation"]) == (
        "sc_test", "A v. B", "judgment", "Supreme Court of India", "1950-05-19", "[1950] 1 S.C.R. 1")
    assert rec["bench"] == "" and rec["source_url"] == "https://example.org/file.pdf"
    assert rec["source"]["origin"] == "third_party_mirror" and rec["source"]["sha256"] == meta["sha256"]


def test_act_keeps_court_and_date_empty_and_carries_legal_status(tmp_path):
    pdf = make_pdf(tmp_path / "ipc_test.pdf", 3)
    meta = meta_for(pdf, source_id="ipc_test", doc_type="act", name="The Indian Penal Code, 1860", act="IPC_1860",
                    legal_status="repealed", repealed_on="2024-07-01",
                    usage_note="REPEALED. Never present as current law.")
    rec, _ = parse_mod.parse_document(pdf, meta, "auto", UNAVAILABLE)
    assert rec["court"] == "" and rec["date"] == "" and rec["bench"] == "" and rec["citation"] == ""
    assert rec["legal_status"] == "repealed" and rec["act"] == "IPC_1860"
    assert rec["usage_note"] == "REPEALED. Never present as current law." and rec["repealed_on"] == "2024-07-01"
    assert rec["title"] == "The Indian Penal Code, 1860"


def test_bad_or_missing_date_is_left_empty_not_guessed():
    assert parse_mod.iso_date("24-04-1973") == "1973-04-24"
    assert parse_mod.iso_date("") == "" and parse_mod.iso_date("31-02-1973") == "" and parse_mod.iso_date("April 1973") == ""


def test_file_that_no_longer_matches_its_metadata_hash_is_refused(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", 2)
    meta = meta_for(pdf, sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        parse_mod.parse_document(pdf, meta, "auto", UNAVAILABLE)


def test_cli_writes_one_json_per_document_and_a_report(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    d = raw / "doc_a"
    d.mkdir(parents=True)
    pdf = make_pdf(d / "doc_a.pdf", 3)
    (d / "doc_a.pdf.meta.json").write_text(json.dumps(meta_for(pdf, source_id="doc_a")), encoding="utf-8")
    monkeypatch.setattr(parse_mod, "settings", type("S", (), {"raw_data_dir": raw, "processed_data_dir": out})())
    assert parse_mod.main(["--ocr", "off"]) == 0
    rec = json.loads((out / "doc_a.json").read_text(encoding="utf-8"))
    assert rec["doc_id"] == "doc_a" and rec["text"]
    report = json.loads((out / "parse_report.json").read_text(encoding="utf-8"))
    assert report["documents"][0]["status"] == "parsed"


# ---------------------------------------------------------------- the real corpus
PROCESSED = ROOT / "data" / "processed"
REAL = sorted(p for p in PROCESSED.glob("*.json") if p.name != "parse_report.json") if PROCESSED.exists() else []


@pytest.mark.skipif(len(REAL) < 10, reason="run `python -m ingestion.parse` first to create data/processed")
def test_processed_corpus_is_clean_and_complete():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in REAL]
    assert len(recs) >= 10
    assert sum(r["type"] == "judgment" for r in recs) >= 3
    # every statute that is present in data/raw was parsed
    raw_acts = {json.loads(m.read_text(encoding="utf-8"))["source_id"]
                for m in (ROOT / "data" / "raw").glob("*/*.meta.json")
                if json.loads(m.read_text(encoding="utf-8")).get("doc_type") in ("act", "constitution")}
    assert raw_acts and raw_acts <= {r["doc_id"] for r in recs}
    for r in recs:
        for key in ("doc_id", "title", "type", "court", "date", "bench", "citation", "source_url", "text",
                    "legal_status", "usage_note"):
            assert key in r, (r["doc_id"], key)
        assert len(r["text"]) > 5000, r["doc_id"]
        assert not r["extraction"]["pages_ocr_skipped"], r["doc_id"]
        noise = page_noise_report(r["text"])
        assert noise["total"] == 0, (r["doc_id"], noise["examples"])
        assert "�" not in r["text"] and "­" not in r["text"], r["doc_id"]
        assert unicodedata.is_normalized("NFKC", r["text"]), r["doc_id"]


@pytest.mark.skipif(not (PROCESSED / "ipc_1860.json").exists(), reason="IPC not parsed yet")
def test_repealed_ipc_stays_clearly_marked():
    ipc = json.loads((PROCESSED / "ipc_1860.json").read_text(encoding="utf-8"))
    assert ipc["legal_status"] == "repealed" and ipc["act"] == "IPC_1860"
    assert "never be presented as current law" in ipc["usage_note"]
