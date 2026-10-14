"""Source registry: every document we collect, where it officially lives, and its terms.

URLs here were located via search results pointing at official government sites
(India Code, Legislative Department). None has been fetched successfully by this
project yet; `url_note` records how confident we are in each.
"""
from dataclasses import dataclass
from typing import Optional

INDIA_CODE_TERMS = (
    "Official Government of India publication hosted on India Code (indiacode.nic.in, "
    "Legislative Department / NIC). Central Acts are public records; check the site's "
    "terms and disclaimer before redistributing. Consolidated PDFs may carry an 'as on' date."
)
CONSTITUTION_TERMS = (
    "Official Government of India publication (Legislative Department, Ministry of Law "
    "and Justice). Government works of this kind are published for public access; check "
    "the site's terms before redistributing."
)


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    doc_type: str  # constitution | act
    url: str  # official page (or file) we found
    file_url: Optional[str]  # direct file link, only if seen in search results
    filename: str  # name to store/expect under data/raw/<id>/
    licence_note: str
    url_note: str


SOURCES: tuple[Source, ...] = (
    Source(
        id="constitution_of_india",
        name="The Constitution of India (English)",
        doc_type="constitution",
        url="https://lddashboard.legislative.gov.in/constitution-of-india",
        file_url=None,  # PDF link not seen yet; do not guess
        filename="constitution_of_india_en.pdf",
        licence_note=CONSTITUTION_TERMS,
        url_note="Page found via search; PDF link not yet verified, so download is manual.",
    ),
    Source(
        id="bns_2023",
        name="Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023)",
        doc_type="act",
        url="https://www.indiacode.nic.in/handle/123456789/20062",
        file_url="https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf",
        filename="bns_2023_a202345.pdf",
        licence_note=INDIA_CODE_TERMS,
        url_note="Handle and PDF both appeared in search results; site returned 403 to us.",
    ),
    Source(
        id="bnss_2023",
        name="Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023)",
        doc_type="act",
        url="https://indiacode.nic.in/bitstream/123456789/20099/3/a2023-46.pdf",
        file_url="https://indiacode.nic.in/bitstream/123456789/20099/3/a2023-46.pdf",
        filename="bnss_2023_a2023-46.pdf",
        licence_note=INDIA_CODE_TERMS,
        url_note="Only the PDF link was seen in search; the handle page was not, so it is not listed.",
    ),
    Source(
        id="bsa_2023",
        name="Bharatiya Sakshya Adhiniyam, 2023 (Act 47 of 2023)",
        doc_type="act",
        url="https://indiacode.nic.in/handle/123456789/20063",
        file_url="https://www.indiacode.nic.in/bitstream/123456789/20063/1/aa202347.pdf",
        filename="bsa_2023_aa202347.pdf",
        licence_note=INDIA_CODE_TERMS,
        url_note="Handle and PDF both appeared in search results; site returned 403 to us.",
    ),
)


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(source_id)
