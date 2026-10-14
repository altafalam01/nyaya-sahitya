import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.downloader import ADOPTED, DOWNLOADED, MANUAL, PRESENT, Downloader, FetchResult
from ingestion.sources import SOURCES, get_source

PDF = b"%PDF-1.4 raw bytes \x00\xff unmodified"


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
