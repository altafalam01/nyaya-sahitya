import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.downloader import INVALID, ADOPTED, DOWNLOADED, MANUAL, PRESENT, Downloader, FetchResult
from ingestion.sources import SC_BUCKET_URL, SC_JUDGMENTS, SOURCES, get_source

PDF = b"%PDF-1.4 raw bytes \x00\xff unmodified" + b"x" * 30_000


class FakeSite:
    """Records calls; answers per-URL from a dict, 404 otherwise."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, user_agent):
        self.calls.append((url, user_agent))
        return self.responses.get(url, FetchResult(404))


def make(tmp_path, responses, delay=5.0):
    site = FakeSite(responses)
    naps = []
    now = [0.0]

    def sleep(s):
        naps.append(s)
        now[0] += s

    dl = Downloader(tmp_path, "TestBot/1.0", delay, fetcher=site, sleep=sleep, clock=lambda: now[0])
    return dl, site, naps


def ok_site(src):
    return {
        "https://" + src.file_url.split("/")[2] + "/robots.txt": FetchResult(200, b"User-agent: *\nAllow: /\n"),
        src.file_url: FetchResult(200, PDF, "application/pdf"),
    }


def test_registry_has_the_four_required_sources():
    ids = {s.id for s in SOURCES}
    assert {"constitution_of_india", "bns_2023", "bnss_2023", "bsa_2023"} <= ids
    for s in SOURCES:
        assert s.url.startswith("https://") and s.licence_note and s.name


def test_download_saves_unmodified_bytes_with_metadata(tmp_path):
    src = get_source("bns_2023")
    dl, _, _ = make(tmp_path, ok_site(src))
    out = dl.collect(src)
    assert out.status == DOWNLOADED
    assert out.path.read_bytes() == PDF
    meta = json.loads(out.path.with_name(out.path.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(PDF).hexdigest()
    assert meta["url"] == src.url and meta["file_url"] == src.file_url
    assert meta["licence_note"] and meta["date_fetched"] and meta["method"] == "automated"


def test_robots_disallow_is_respected(tmp_path):
    src = get_source("bns_2023")
    host = "https://" + src.file_url.split("/")[2]
    responses = ok_site(src)
    responses[host + "/robots.txt"] = FetchResult(200, b"User-agent: *\nDisallow: /bitstream/\n")
    dl, site, _ = make(tmp_path, responses)
    out = dl.collect(src)
    assert out.status == MANUAL and "robots.txt" in out.detail
    assert src.file_url not in [c[0] for c in site.calls]
    assert not (tmp_path / src.id / src.filename).exists()


def test_unreadable_robots_counts_as_blocked(tmp_path):
    src = get_source("bsa_2023")
    host = "https://" + src.file_url.split("/")[2]
    dl, site, _ = make(tmp_path, {host + "/robots.txt": FetchResult(403)})
    out = dl.collect(src)
    assert out.status == MANUAL
    assert src.file_url not in [c[0] for c in site.calls]


def test_403_on_file_gives_manual_instructions(tmp_path):
    src = get_source("bns_2023")
    responses = ok_site(src)
    responses[src.file_url] = FetchResult(403)
    dl, _, _ = make(tmp_path, responses)
    out = dl.collect(src)
    assert out.status == MANUAL
    assert str(tmp_path / src.id / src.filename) in out.detail
    assert src.url in out.detail


def test_source_without_verified_file_url_is_manual_and_makes_no_requests(tmp_path):
    src = get_source("constitution_of_india")
    dl, site, _ = make(tmp_path, {})
    assert dl.collect(src).status == MANUAL
    assert site.calls == []


def test_rate_limit_between_requests_to_same_host(tmp_path):
    a, b = get_source("bns_2023"), get_source("bsa_2023")
    responses = {**ok_site(a), **ok_site(b)}
    dl, _, naps = make(tmp_path, responses, delay=5.0)
    dl.collect(a)
    dl.collect(b)
    assert naps and all(n > 0 for n in naps)
    assert max(naps) <= 5.0


def test_manually_placed_file_is_adopted_with_metadata(tmp_path):
    src = get_source("constitution_of_india")
    p = tmp_path / src.id / src.filename
    p.parent.mkdir(parents=True)
    p.write_bytes(PDF)
    dl, site, _ = make(tmp_path, {})
    out = dl.collect(src)
    assert out.status == ADOPTED and site.calls == []
    meta = json.loads(p.with_name(p.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["method"] == "manual" and meta["sha256"] == hashlib.sha256(PDF).hexdigest()
    assert p.read_bytes() == PDF
    assert dl.collect(src).status == PRESENT  # idempotent


def test_cli_dry_run_makes_no_network_calls(capsys):
    from ingestion import collect
    assert collect.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    for s in SOURCES:
        assert s.id in out and s.url in out


def test_html_error_page_served_as_200_is_rejected(tmp_path):
    src = get_source("bns_2023")
    responses = ok_site(src)
    responses[src.file_url] = FetchResult(200, b"<html><body>Access Denied</body></html>" * 1000, "text/html")
    dl, _, _ = make(tmp_path, responses)
    out = dl.collect(src)
    assert out.status == MANUAL and "not a document" in out.detail
    assert not (tmp_path / src.id / src.filename).exists()


def test_tiny_or_non_pdf_manual_file_is_flagged_invalid_without_metadata(tmp_path):
    src = get_source("bsa_2023")
    p = tmp_path / src.id / src.filename
    p.parent.mkdir(parents=True)
    p.write_bytes(b"<html>403 Forbidden</html>")
    dl, _, _ = make(tmp_path, {})
    assert dl.collect(src).status == INVALID
    assert not p.with_name(p.name + ".meta.json").exists()
    p.write_bytes(b"%PDF-1.4 tiny")
    assert dl.collect(src).status == INVALID


# --- Supreme Court judgments (third-party S3 mirror) ---------------------------------

SC_PDF = b"%PDF-1.4 judgment " + b"x" * 30_000 + b"\n%%EOF\n"


def test_registry_lists_at_least_20_landmark_judgments_with_provenance():
    assert len(SC_JUDGMENTS) >= 20
    assert len({s.id for s in SC_JUDGMENTS}) == len(SC_JUDGMENTS)
    titles = " ".join(s.name for s in SC_JUDGMENTS).lower()
    for landmark in ("kesavananda", "maneka gandhi", "puttaswamy"):
        assert landmark in titles
    for s in SC_JUDGMENTS:
        assert s.origin == "third_party_mirror" and "eCourts" in s.mirror_of
        assert s.bucket_url == SC_BUCKET_URL and s.file_url.startswith(SC_BUCKET_URL + "data/pdf/year=")
        assert s.file_url.endswith("_EN.pdf") and "/english/" in s.file_url
        assert s.case_metadata["bucket_key"] == s.file_url[len(SC_BUCKET_URL):]
        assert "CC-BY-4.0" in s.licence_note and s.doc_type == "judgment"


def test_judgment_download_records_mirror_provenance(tmp_path):
    src = get_source("sc_1973_kesavananda_bharati")
    responses = {
        SC_BUCKET_URL + "robots.txt": FetchResult(404),
        src.file_url: FetchResult(200, SC_PDF, "application/pdf"),
    }
    dl, _, _ = make(tmp_path, responses)
    out = dl.collect(src)
    assert out.status == DOWNLOADED and out.path.read_bytes() == SC_PDF
    meta = json.loads(out.path.with_name(out.path.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["origin"] == "third_party_mirror" and "eCourts" in meta["mirror_of"]
    assert meta["bucket_url"] == SC_BUCKET_URL and meta["file_url"] == src.file_url
    assert meta["date_fetched"] and meta["case_metadata"]["citation"] == "[1973] Supp. 1 S.C.R. 1"


def test_official_source_metadata_has_no_mirror_fields(tmp_path):
    src = get_source("bns_2023")
    dl, _, _ = make(tmp_path, ok_site(src))
    out = dl.collect(src)
    meta = json.loads(out.path.with_name(out.path.name + ".meta.json").read_text(encoding="utf-8"))
    assert "origin" not in meta and "case_metadata" not in meta


def test_truncated_judgment_pdf_is_rejected(tmp_path):
    src = get_source("sc_1978_maneka_gandhi")
    truncated = b"%PDF-1.4 " + b"x" * 30_000  # valid header, no %%EOF
    responses = {SC_BUCKET_URL + "robots.txt": FetchResult(404), src.file_url: FetchResult(200, truncated, "application/pdf")}
    dl, _, _ = make(tmp_path, responses)
    out = dl.collect(src)
    assert out.status == MANUAL and "truncated" in out.detail
    assert not (tmp_path / src.id / src.filename).exists()


def test_judgment_served_as_html_is_rejected(tmp_path):
    src = get_source("sc_2017_puttaswamy")
    responses = {SC_BUCKET_URL + "robots.txt": FetchResult(404),
                 src.file_url: FetchResult(200, b"<html>AccessDenied</html>" * 2000, "text/html")}
    dl, _, _ = make(tmp_path, responses)
    assert dl.collect(src).status == MANUAL


def test_registry_json_lists_every_judgment_with_provenance(tmp_path):
    from ingestion.collect import write_registry
    src = SC_JUDGMENTS[0]
    responses = {SC_BUCKET_URL + "robots.txt": FetchResult(404), src.file_url: FetchResult(200, SC_PDF, "application/pdf")}
    dl, _, _ = make(tmp_path, responses)
    out = dl.collect(src)
    write_registry(tmp_path, SOURCES, [out])
    reg = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in reg["sources"]}
    assert all(s.id in rows for s in SC_JUDGMENTS) and "bns_2023" in rows
    first = rows[src.id]
    assert first["status"] == DOWNLOADED and first["date_fetched"]
    assert first["origin"] == "third_party_mirror" and first["bucket_url"] == SC_BUCKET_URL
    assert rows[SC_JUDGMENTS[1].id]["status"] == "not_run"
    # a later partial run still reports what is already on disk
    write_registry(tmp_path, SOURCES, [])
    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["sources"][0]["id"]
    again = {r["id"]: r for r in json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["sources"]}
    assert again[src.id]["status"] == "present"


def test_rate_limit_applies_across_judgment_downloads(tmp_path):
    a, b = SC_JUDGMENTS[0], SC_JUDGMENTS[1]
    responses = {SC_BUCKET_URL + "robots.txt": FetchResult(404),
                 a.file_url: FetchResult(200, SC_PDF), b.file_url: FetchResult(200, SC_PDF)}
    dl, site, naps = make(tmp_path, responses, delay=5.0)
    dl.collect(a)
    dl.collect(b)
    assert len(naps) >= 2 and all(0 < n <= 5.0 for n in naps)
    assert [c[0] for c in site.calls].count(SC_BUCKET_URL + "robots.txt") == 1


def test_cli_dry_run_doc_type_filter(capsys):
    from ingestion import collect
    assert collect.main(["--dry-run", "--doc-type", "judgment"]) == 0
    out = capsys.readouterr().out
    assert all(s.file_url in out for s in SC_JUDGMENTS) and "bns_2023" not in out
