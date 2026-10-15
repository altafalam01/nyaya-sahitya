"""Source registry: every document we collect, where it lives, and its terms.

URLs here were located via search results pointing at official government sites
(India Code, Legislative Department). None has been fetched successfully by this
project yet; `url_note` records how confident we are in each.
"""
from dataclasses import dataclass, field
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
NCRB_BNS_URL = "https://www.ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf"
NCRB_TERMS = (
    "Official Government of India publication hosted by the National Crime Records Bureau (NCRB), "
    "an attached office of the Ministry of Home Affairs (Sankalan portal). Check the site's terms "
    "before redistributing."
)
IPC_URL = "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf"


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    doc_type: str  # constitution | act | judgment
    url: str  # official page (or file) we found
    file_url: Optional[str]  # direct file link, only if seen in search results
    filename: str  # name to store/expect under data/raw/<id>/
    licence_note: str
    url_note: str
    # Provenance for non-official copies; defaults keep the official sources unchanged.
    origin: str = "official"  # official | third_party_mirror
    mirror_of: Optional[str] = None  # the official system the mirror copies from
    bucket_url: Optional[str] = None  # base URL of the mirror's bucket
    case_metadata: dict = field(default_factory=dict)  # fields taken from the mirror's metadata
    strict_pdf: bool = False  # also require the %%EOF trailer (catches truncated downloads)
    # Legal standing and hand-placed-file provenance (all optional).
    act: Optional[str] = None  # stable act key for old-to-new mapping, e.g. IPC_1860
    legal_status: Optional[str] = None  # in_force | repealed
    in_force_from: Optional[str] = None  # ISO date
    repealed_on: Optional[str] = None  # ISO date
    usage_note: Optional[str] = None  # how the text may / may not be used downstream
    content_note: Optional[str] = None  # what else the file contains
    retrieved_on: Optional[str] = None  # ISO date the file was fetched by the user (manual placement)
    expected_sha256: Optional[str] = None  # user-verified hash; a placed file must match it

    def legal_fields(self) -> dict:
        """Non-empty legal/provenance extras, shared by the registry row and the .meta.json."""
        keys = ("act", "legal_status", "in_force_from", "repealed_on", "usage_note",
                "content_note", "retrieved_on", "expected_sha256")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


OFFICIAL_SOURCES: tuple[Source, ...] = (
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
        url=NCRB_BNS_URL,
        file_url=NCRB_BNS_URL,
        filename="bns_2023_a202345.pdf",
        licence_note=NCRB_TERMS,
        url_note="Published by NCRB (an attached office of the Ministry of Home Affairs); India Code "
                 "was unreachable (403). File fetched by the user on 2026-09-21, not by this project.",
        act="BNS_2023",
        legal_status="in_force",
        in_force_from="2024-07-01",
        content_note="The PDF also contains an index and an IPC-to-BNS correspondence table.",
        retrieved_on="2026-09-21",
        expected_sha256="a83f12a93e32c9e0b39a85f75850a448dc4682b1b44b2d6bd680165bb9931549",
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
    Source(
        id="ipc_1860",
        name="The Indian Penal Code, 1860 (Act 45 of 1860)",
        doc_type="act",
        url=IPC_URL,
        file_url=IPC_URL,
        filename="ipc_1860_bare_act.pdf",
        licence_note=INDIA_CODE_TERMS,
        url_note="URL supplied by the user, who fetched the file on 2026-09-21; not re-fetched by this project.",
        act="IPC_1860",
        legal_status="repealed",
        repealed_on="2024-07-01",
        usage_note="REPEALED on 1 July 2024 (replaced by the Bharatiya Nyaya Sanhita, 2023). Kept only "
                   "for old-to-new section mapping; must never be presented as current law.",
        retrieved_on="2026-09-21",
        expected_sha256="ef8945c5d1b02904da67959e245b87bd5751ed5563d03ab0079758909f145309",
    ),
)


# --- Supreme Court judgments: third-party mirror of eCourts data -----------------------
# Layout confirmed from https://github.com/vanga/indian-supreme-court-judgments (README):
#   data/pdf/year=YYYY/english/<path>_EN.pdf ; <path> comes from metadata/parquet/year=YYYY.
# Each <path>, title, citation, decision date and case id below was read from that
# metadata (not guessed), and every PDF key was confirmed to exist with a HEAD request.
SC_BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com/"
SC_REPO_URL = "https://github.com/vanga/indian-supreme-court-judgments"
SC_MIRROR_OF = "eCourts Supreme Court judgments portal (https://scr.sci.gov.in)"
SC_TERMS = (
    "Third-party mirror, not an official source: vanga/indian-supreme-court-judgments republishes "
    "eCourts data and licenses the dataset CC-BY-4.0 (attribution required: 'Indian Supreme Court "
    "Judgments, vanga/indian-supreme-court-judgments'). The judgment text originates from the "
    "Supreme Court of India via eCourts; verify against the official text before relying on it."
)


def _sc(year: int, slug: str, path: str, title: str, citation: str, decided: str, case_id: str) -> Source:
    key = f"data/pdf/year={year}/english/{path}_EN.pdf"
    return Source(
        id=f"sc_{year}_{slug}",
        name=f"{title} ({citation})",
        doc_type="judgment",
        url=SC_REPO_URL,
        file_url=SC_BUCKET_URL + key,
        filename=f"{path}_EN.pdf",
        licence_note=SC_TERMS,
        url_note=f"Key found in metadata/parquet/year={year}/metadata.parquet (path={path}); "
                 "PDF exists in the public bucket.",
        origin="third_party_mirror",
        mirror_of=SC_MIRROR_OF,
        bucket_url=SC_BUCKET_URL,
        case_metadata={"title": title, "citation": citation, "decision_date": decided,
                       "case_id": case_id, "court": "Supreme Court of India", "language": "English",
                       "bucket_key": key, "bucket_path_id": path, "year": year},
        strict_pdf=True,
    )


SC_JUDGMENTS: tuple[Source, ...] = (
    _sc(1950, "gopalan", "1950_1_88_334", "A.K. Gopalan v. State of Madras", "[1950] 1 S.C.R. 88", "19-05-1950", "1950 INSC 13"),
    _sc(1967, "golak_nath", "1967_2_762_948", "I.C. Golak Nath & Ors. v. State of Punjab & Anrs.", "[1967] 2 S.C.R. 762", "27-02-1967", "1967 INSC 45"),
    _sc(1973, "kesavananda_bharati", "S_1973_1_1_1002", "His Holiness Kesavananda Bharati Sripadagalavaru v. State of Kerala", "[1973] Supp. 1 S.C.R. 1", "24-04-1973", "1973 INSC 91"),
    _sc(1976, "adm_jabalpur", "S_1976_1_172_477", "Additional District Magistrate, Jabalpur v. S.S. Shukla etc.", "[1976] Supp. 1 S.C.R. 172", "28-04-1976", "1976 INSC 129"),
    _sc(1978, "maneka_gandhi", "1978_2_621_776", "Maneka Gandhi v. Union of India", "[1978] 2 S.C.R. 621", "25-01-1978", "1978 INSC 16"),
    _sc(1980, "minerva_mills", "1981_1_206_342", "Minerva Mills Ltd. & Ors. v. Union of India & Ors.", "[1981] 1 S.C.R. 206", "31-07-1980", "1980 INSC 142"),
    _sc(1980, "bachan_singh", "1983_1_145_371", "Bachan Singh etc. etc. v. State of Punjab etc. etc.", "[1983] 1 S.C.R. 145", "09-05-1980", "1980 INSC 120"),
    _sc(1985, "shah_bano", "1985_3_844_868", "Mohd. Ahmed Khan v. Shah Bano Begum and Ors.", "[1985] 3 S.C.R. 844", "23-04-1985", "1985 INSC 97"),
    _sc(1985, "olga_tellis", "S_1985_2_51_99", "Olga Tellis & Ors. v. Bombay Municipal Corporation & Ors. etc.", "[1985] Supp. 2 S.C.R. 51", "10-07-1985", "1985 INSC 151"),
    _sc(1992, "indra_sawhney", "S_1992_2_454_1007", "Indra Sawhney and Ors. etc. etc. v. Union of India and Ors. etc. etc.", "[1992] Supp. 2 S.C.R. 454", "16-11-1992", ""),
    _sc(1994, "sr_bommai", "1994_2_644_1000", "S.R. Bommai v. Union of India and Ors.", "[1994] 2 S.C.R. 644", "11-03-1994", "1994 INSC 111"),
    _sc(1996, "dk_basu", "S_1996_10_284_320", "D.K. Basu v. State of West Bengal", "[1996] Supp. 10 S.C.R. 284", "18-12-1996", "1996 INSC 1508"),
    _sc(1997, "vishaka", "S_1997_3_404_418", "Vishaka and Ors. v. State of Rajasthan and Ors.", "[1997] Supp. 3 S.C.R. 404", "13-08-1997", "1997 INSC 604"),
    _sc(1997, "l_chandra_kumar", "1997_2_1186_1250", "L. Chandra Kumar etc. etc. v. The Union of India and Ors. etc. etc.", "[1997] 2 S.C.R. 1186", "18-03-1997", "1997 INSC 288"),
    _sc(2007, "ir_coelho", "2007_1_706_773", "I.R. Coelho (Dead) by LRs. v. State of Tamil Nadu", "[2007] 1 S.C.R. 706", "11-01-2007", "2007 INSC 28"),
    _sc(2013, "lalita_kumari", "2013_14_713_802", "Lalita Kumari v. Govt. of U.P. and Ors.", "[2013] 14 S.C.R. 713", "12-11-2013", "2013 INSC 748"),
    _sc(2015, "shreya_singhal", "2015_5_963_1074", "Shreya Singhal v. Union of India", "[2015] 5 S.C.R. 963", "24-03-2015", "2015 INSC 257"),
    _sc(2017, "puttaswamy", "2017_10_569_999", "Justice K.S. Puttaswamy (Retd.), and Anr. v. Union of India and Ors.", "[2017] 10 S.C.R. 569", "24-08-2017", "2017 INSC 801"),
    _sc(2017, "shayara_bano", "2017_9_797_1158", "Shayara Bano v. Union of India and Others", "[2017] 9 S.C.R. 797", "22-08-2017", "2017 INSC 785"),
    _sc(2018, "navtej_johar", "2018_7_379_746", "Navtej Singh Johar & Ors. v. Union of India thr. Secretary Ministry of Law and Justice", "[2018] 7 S.C.R. 379", "06-09-2018", "2018 INSC 790"),
)

SOURCES: tuple[Source, ...] = OFFICIAL_SOURCES + SC_JUDGMENTS


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(source_id)
