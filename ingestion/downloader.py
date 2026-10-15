"""Polite downloader: honours robots.txt and rate limits, stores raw bytes untouched.

Layout per source:  <raw_dir>/<source.id>/<source.filename>
                    <raw_dir>/<source.id>/<source.filename>.meta.json
If a site refuses automated access we never work around it: the source is reported
as MANUAL with instructions, and a file the user places there is adopted later.
"""
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .sources import Source

log = logging.getLogger(__name__)

DOWNLOADED, ADOPTED, PRESENT, MANUAL = "downloaded", "adopted", "present", "manual_required"
INVALID = "invalid_file"
MIN_DOC_BYTES = 20_000  # real Act/Constitution PDFs are hundreds of KB; error pages are ~1 KB


def validate_document(content: bytes, strict: bool = False) -> tuple[bool, str]:
    """Return (ok, file_type_or_reason). Accepts PDFs only; rejects HTML/error pages and tiny files.

    strict additionally requires the %%EOF trailer, which a truncated download lacks.
    """
    head = content[:1024].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in head:
        return False, "HTML page, not a document"
    if not content.startswith(b"%PDF-"):
        return False, "does not start with %PDF- header"
    if len(content) < MIN_DOC_BYTES:
        return False, f"only {len(content)} bytes (minimum {MIN_DOC_BYTES})"
    if strict and b"%%EOF" not in content[-2048:]:
        return False, "no %%EOF trailer (file looks truncated)"
    return True, "application/pdf"


@dataclass(frozen=True)
class FetchResult:
    status: int  # HTTP status, 0 for network failure
    content: bytes = b""
    content_type: str = ""


class Fetcher(Protocol):
    def __call__(self, url: str, user_agent: str) -> FetchResult: ...


def http_fetch(url: str, user_agent: str, timeout: int = 60) -> FetchResult:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return FetchResult(r.status, r.read(), r.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as e:
        return FetchResult(e.code)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("network error for %s: %s", url, e)
        return FetchResult(0)


@dataclass(frozen=True)
class Outcome:
    source_id: str
    status: str
    detail: str
    path: Optional[Path] = None


def manual_instructions(source: Source, raw_dir: Path) -> str:
    target = raw_dir / source.id / source.filename
    return (
        f"Open {source.url} in a browser, download the official PDF, and save it exactly as:\n"
        f"    {target}\n"
        f"Then re-run `python -m ingestion.collect` to record its provenance."
    )


class Downloader:
    def __init__(
        self,
        raw_dir: Path,
        user_agent: str,
        delay_seconds: float = 5.0,
        fetcher: Fetcher = http_fetch,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.user_agent = user_agent
        self.delay = delay_seconds
        self._fetch = fetcher
        self._sleep = sleep
        self._clock = clock
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, Optional[RobotFileParser]] = {}

    # -- politeness -------------------------------------------------------
    def _throttle(self, host: str) -> None:
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.delay - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_hit[host] = self._clock()

    def _robots_for(self, url: str) -> Optional[RobotFileParser]:
        """Return a parser, or None meaning 'robots.txt could not be read: treat as blocked'."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._throttle(parts.netloc)
            res = self._fetch(f"{origin}/robots.txt", self.user_agent)
            rp: Optional[RobotFileParser] = RobotFileParser()
            if res.status == 200:
                rp.parse(res.content.decode("utf-8", errors="replace").splitlines())
            elif res.status == 404:
                rp.parse([])  # no robots.txt: everything allowed
            else:
                rp = None  # 401/403/5xx/network: cannot confirm permission
            self._robots[origin] = rp
        return self._robots[origin]

    def allowed(self, url: str) -> tuple[bool, str]:
        rp = self._robots_for(url)
        if rp is None:
            return False, "robots.txt could not be read (access refused or site unreachable)"
        if not rp.can_fetch(self.user_agent, url):
            return False, "disallowed by robots.txt"
        return True, "allowed"

    # -- storage ----------------------------------------------------------
    def _write_metadata(self, source: Source, path: Path, *, method: str, content: bytes,
                        http_status: Optional[int], content_type: Optional[str]) -> None:
        meta = {
            "source_id": source.id,
            "name": source.name,
            "doc_type": source.doc_type,
            "url": source.url,
            "file_url": source.file_url,
            "licence_note": source.licence_note,
            "url_note": source.url_note,
            "date_fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": method,  # automated | manual
            "filename": path.name,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "http_status": http_status,
            "content_type": content_type,
            "user_agent": self.user_agent if method == "automated" else None,
        }
        if source.origin != "official":
            meta.update({
                "origin": source.origin,  # third_party_mirror
                "mirror_of": source.mirror_of,
                "bucket_url": source.bucket_url,
                "case_metadata": source.case_metadata,
            })
        meta_path = path.with_name(path.name + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- main entry -------------------------------------------------------
    def collect(self, source: Source) -> Outcome:
        dest_dir = self.raw_dir / source.id
        path = dest_dir / source.filename
        meta_path = path.with_name(path.name + ".meta.json")

        if path.exists():
            if meta_path.exists():
                return Outcome(source.id, PRESENT, "file and metadata already exist", path)
            data = path.read_bytes()
            ok, why = validate_document(data, source.strict_pdf)
            if not ok:
                return Outcome(source.id, INVALID, f"{path} is not a valid document: {why}. "
                               f"Replace it with the real PDF.", path)
            self._write_metadata(source, path, method="manual", content=data,
                                 http_status=None, content_type=why)
            return Outcome(source.id, ADOPTED, "manually placed file; metadata written", path)

        manual = manual_instructions(source, self.raw_dir)
        if not source.file_url:
            return Outcome(source.id, MANUAL, f"no verified direct file link.\n{manual}")

        ok, why = self.allowed(source.file_url)
        if not ok:
            return Outcome(source.id, MANUAL, f"{why}.\n{manual}")

        self._throttle(urlsplit(source.file_url).netloc)
        res = self._fetch(source.file_url, self.user_agent)
        if res.status != 200 or not res.content:
            return Outcome(source.id, MANUAL, f"server answered HTTP {res.status}.\n{manual}")

        ok, why = validate_document(res.content, source.strict_pdf)
        if not ok:
            return Outcome(source.id, MANUAL, f"server sent something that is not a document ({why}).\n{manual}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(res.content)  # raw bytes, unmodified
        self._write_metadata(source, path, method="automated", content=res.content,
                             http_status=res.status, content_type=res.content_type)
        return Outcome(source.id, DOWNLOADED, f"saved {len(res.content)} bytes", path)
