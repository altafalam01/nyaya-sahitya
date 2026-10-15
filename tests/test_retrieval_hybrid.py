"""Step 6 tests: BM25, RRF, query pre-processor, section map, hybrid pipeline, evaluation. No model downloads."""
import json
import zlib
from pathlib import Path

import pytest

from retrieval import mapping
from retrieval.bm25 import BM25Index, load_or_build, tokenize
from retrieval.embedder import Embedder
from retrieval.evaluate import check_gold, format_table, is_gold, recall_at_k, summarise
from retrieval.fusion import rrf
from retrieval.hybrid import accept_fn, exact_matches, hybrid_search
from retrieval.index import run as index_run
from retrieval.query import Ref, find_cases, find_refs, preprocess
from retrieval.rerank import Reranker
from retrieval.search import Hit
from retrieval.store import Store


# ------------------------------------------------------------------ helpers
def fake_vec(text):
    v = [0.0] * 32
    for w in text.lower().replace("passage:", " ").replace("query:", " ").split():
        v[zlib.crc32(w.encode()) % 32] += 1.0
    return v


def rec(cid, doc, text, **kw):
    base = {"chunk_id": cid, "doc_id": doc, "text": text, "context": doc, "chunk_type": "section", "doc_type": "act",
            "act": "BNS_2023", "legal_status": "in_force", "court": "", "date": "", "unit_label": "Section",
            "unit_number": "1", "locator": "Section 1", "heading": "", "cite": f"{doc}, Section 1", "doc_title": doc,
            "section_type": ""}
    base.update(kw)
    return base


CORPUS = {
    "bns_2023": [rec("b318", "bns_2023", "Cheating. Whoever by deceiving any person dishonestly induces delivery of property",
                     unit_number="318", locator="Section 318", heading="Cheating"),
                 rec("b103", "bns_2023", "Punishment for murder. Whoever commits murder shall be punished with death",
                     unit_number="103", locator="Section 103", heading="Punishment for murder"),
                 rec("bidx", "bns_2023", "cheating contents list table", chunk_type="index", locator="")],
    "ipc_1860": [rec("i302", "ipc_1860", "Punishment for murder. Whoever commits murder shall be punished with death",
                     act="IPC_1860", legal_status="repealed", unit_number="302", locator="Section 302",
                     heading="Punishment for murder"),
                 rec("i420", "ipc_1860", "Cheating and dishonestly inducing delivery of property",
                     act="IPC_1860", legal_status="repealed", unit_number="420", locator="Section 420",
                     heading="Cheating and dishonestly inducing delivery of property")],
    "sc_1978_maneka": [rec("j1", "sc_1978_maneka", "Maneka Gandhi v. Union of India passport impounded personal liberty",
                           chunk_type="judgment_text", doc_type="judgment", act="", court="Supreme Court of India",
                           date="1978-01-25", unit_label="", unit_number="", locator="para 1",
                           doc_title="Maneka Gandhi v. Union of India")],
}
SECTION_MAP = {"ipc_to_bns": {"302": ["103"], "420": ["318"]}, "bns_to_ipc": {"103": ["302"], "318": ["415", "420"]}}


@pytest.fixture
def env(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for doc, recs in CORPUS.items():
        (chunks / f"{doc}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    emb = Embedder(model_name="org/T", batch_size=4, encoder=lambda ts: [fake_vec(t) for t in ts])
    store = Store(emb.collection_name, tmp_path / "idx")
    index_run(chunks, store, emb)
    bm = load_or_build(chunks, tmp_path / "bm25.pkl")
    return dict(embedder=emb, store=store, bm25=bm, section_map=SECTION_MAP)


# ------------------------------------------------------------------ BM25
def test_tokenize_keeps_numbers_and_folds_devanagari_digits():
    assert tokenize("Section 302 of the IPC") == ["section", "302", "ipc"]
    assert tokenize("धारा ४२०") == ["धारा", "420"]
    assert "498a" in tokenize("Section 498A")


def test_bm25_ranks_exact_terms_and_respects_filters():
    ix = BM25Index.build([rec("a", "d1", "cheating by deceit"), rec("b", "d1", "murder and death"),
                          rec("c", "d2", "cheating cheating cheating", act="IPC_1860")])
    assert ix.search("cheating")[0][0] == "c"
    assert [c for c, _ in ix.search("cheating", accept=lambda m: m["act"] == "BNS_2023")] == ["a"]
    assert ix.search("zebra") == []


def test_bm25_indexes_the_heading_and_locator():
    ix = BM25Index.build([rec("a", "d", "body text only", locator="Section 318", heading="Cheating"),
                          rec("b", "d", "other body", locator="Section 5", heading="Other")])
    assert ix.search("318 cheating")[0][0] == "a"


def test_bm25_cache_is_reused_and_rebuilt_when_chunks_change(env, tmp_path):
    chunks = tmp_path / "chunks"
    cache = tmp_path / "bm25.pkl"
    first = cache.stat().st_mtime_ns
    load_or_build(chunks, cache)
    assert cache.stat().st_mtime_ns == first
    p = chunks / "bns_2023.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + json.dumps(rec("bnew", "bns_2023", "brand new text")) + "\n", encoding="utf-8")
    ix = load_or_build(chunks, cache)
    assert "bnew" in ix.ids


# ------------------------------------------------------------------ RRF
def test_rrf_scores_and_order():
    out = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in out]
    assert ids[:2] == ["a", "b"] and set(ids) == {"a", "b", "c", "d"}  # a and b tie; first appearance wins
    assert dict(out)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([]) == []


def test_rrf_weights():
    out = rrf([["a"], ["b"]], weights=[1.0, 3.0])
    assert out[0][0] == "b"


# ------------------------------------------------------------------ query pre-processor
@pytest.mark.parametrize("q,expected", [
    ("Section 302 IPC", Ref("section", "302", "IPC")),
    ("section 302 of the Indian Penal Code", Ref("section", "302", "IPC")),
    ("IPC 420", Ref("section", "420", "IPC")),
    ("s. 318 BNS", Ref("section", "318", "BNS")),
    ("BNS Section 103", Ref("section", "103", "BNS")),
    ("धारा ४२० आईपीसी", Ref("section", "420", "IPC")),
    ("Section 498A IPC", Ref("section", "498A", "IPC")),
    ("Article 21 of the Constitution", Ref("article", "21", "CONST")),
    ("अनुच्छेद 21", Ref("article", "21", "CONST")),
])
def test_references_are_found(q, expected):
    assert expected in find_refs(q)


def test_plain_text_gives_no_references():
    assert find_refs("section of the road was closed") == []
    assert find_refs("punishment for cheating") == []


def test_bare_section_has_no_act():
    assert find_refs("what is Section 420") == [Ref("section", "420", None)]


def test_case_names():
    assert "Maneka Gandhi v. Union of India" in find_cases("explain Maneka Gandhi v. Union of India please")
    assert find_cases("the right to privacy") == []


def test_ipc_reference_is_expanded_with_the_bns_counterpart():
    pq = preprocess("Section 302 IPC", SECTION_MAP, {"BNS": {"103": "Punishment for murder"}})
    assert pq.mapped == [Ref("section", "103", "BNS")]
    assert "BNS Section 103 Punishment for murder" in pq.search_text and pq.search_text.startswith("Section 302 IPC")
    assert "repealed" in pq.notes[0]


def test_bns_reference_expands_to_ipc_and_unmapped_or_bare_gets_nothing():
    assert [r.number for r in preprocess("BNS 318", SECTION_MAP).mapped] == ["415", "420"]
    assert preprocess("Section 999 IPC", SECTION_MAP).search_text == "Section 999 IPC"
    assert preprocess("Section 420", SECTION_MAP).mapped == []


# ------------------------------------------------------------------ section map parsing
def test_section_map_is_built_from_the_table_and_validated_against_headings(tmp_path):
    chunks = tmp_path
    table = ("CORRESPONDING SECTION TABLE OF BNS WITH REPEALED ACT\n\n"
             "103. Punishment for murder. | 302. Punishment for murder.\n\n"
             "318(4) Cheating | 420. Cheating and dishonestly inducing delivery of property.\n\n"
             "999. Invented section | 888. Nothing like it\n\n"
             "103. Punishment for murder. | 420. Something unrelated entirely here.\n")
    bns = [rec("t", "bns_2023", table, chunk_type="index")] + CORPUS["bns_2023"][:2]
    (chunks / "bns_2023.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bns), encoding="utf-8")
    (chunks / "ipc_1860.jsonl").write_text("".join(json.dumps(r) + "\n" for r in CORPUS["ipc_1860"]), encoding="utf-8")
    m = mapping.build_map(chunks)
    assert m["ipc_to_bns"]["302"] == ["103"]
    assert m["ipc_to_bns"]["420"] == ["318"]          # the bogus 103 | 420 pairing is rejected: the right side does not fit IPC 420
    assert "888" not in m["ipc_to_bns"] and m["bns_to_ipc"]["318"] == ["420"]


def test_mapping_of_the_real_table_if_chunks_exist():
    from config import settings
    if not (settings.chunks_data_dir / "bns_2023.jsonl").exists():
        pytest.skip("no chunks")
    m = mapping.build_map(settings.chunks_data_dir)
    assert m["ipc_to_bns"]["302"] == ["103"] and "318" in m["ipc_to_bns"]["420"] and "64" in m["ipc_to_bns"]["376"]
    assert m["stats"]["ipc_sections_mapped"] > 300


# ------------------------------------------------------------------ hybrid pipeline
def test_accept_fn_matches_filter_semantics():
    ok = accept_fn(act="IPC_1860", year=1860)
    assert ok({"act": "IPC_1860", "chunk_type": "section", "date": ""})
    assert not ok({"act": "BNS_2023", "chunk_type": "section", "date": ""})
    assert not accept_fn()({"chunk_type": "index"}) and accept_fn(include_index=True)({"chunk_type": "index"})
    assert accept_fn(year_from=1970, year_to=1980)({"chunk_type": "judgment_text", "date": "1978-01-01"})


def test_exact_matches_find_the_named_section(env):
    assert exact_matches(env["bm25"], Ref("section", "302", "IPC")) == ["i302"]
    assert exact_matches(env["bm25"], Ref("section", "302", "BNS")) == []


def test_reference_query_puts_named_section_first_then_its_counterpart(env):
    hits = hybrid_search("Section 302 IPC", 3, rerank=False, **env)
    assert [h.chunk_id for h in hits[:2]] == ["i302", "b103"]
    assert hits[0].legal_status == "repealed" and "reference" in hits[0].detail
    assert "repealed" in hits[0].detail["notes"][0]


def test_modes_and_filters(env):
    assert hybrid_search("cheating", 3, mode="bm25", **env)[0].chunk_id in ("b318", "i420")
    assert all(h.chunk_id != "bidx" for h in hybrid_search("cheating", 5, mode="hybrid", rerank=False, **env))
    only_ipc = hybrid_search("cheating", 5, mode="hybrid", rerank=False, act="IPC_1860", **env)
    assert only_ipc and all(h.doc_id == "ipc_1860" for h in only_ipc)
    assert hybrid_search("cheating", 5, mode="dense", court="No Such", **env) == []
    with pytest.raises(ValueError):
        hybrid_search("x", mode="nope", **env)


def test_reranker_reorders_and_falls_back(env):
    boom = Reranker(scorer=lambda pairs: [1.0 if "Cheating." in t else 0.0 for _, t in pairs])
    hits = hybrid_search("property deceiving", 3, rerank=True, reranker=boom, **env)
    assert hits[0].chunk_id == "b318" and "rerank" in hits[0].detail

    def broken(pairs):
        raise RuntimeError("no model")
    r = Reranker(scorer=broken)
    assert r.rerank("q", [("a", "x"), ("b", "y")]) == [("a", 0.0), ("b", 0.0)] and r.failed
    assert Reranker(scorer=lambda p: []).rerank("q", []) == []


# ------------------------------------------------------------------ evaluation
def hit(doc, locator=""):
    return Hit("id", 0.0, "", locator, doc, "", "", 0, "section", "")


def test_gold_matching_and_recall():
    assert is_gold(hit("bns_2023", "Section 318"), [{"doc": "bns_2023", "unit": "318"}])
    assert not is_gold(hit("bns_2023", "Section 319"), [{"doc": "bns_2023", "unit": "318"}])
    assert is_gold(hit("sc_x", "para 5"), [{"doc": "sc_x"}])
    hits = [hit("a"), hit("b"), hit("sc_x")]
    assert recall_at_k(hits, [{"doc": "sc_x"}], 5) == 1 and recall_at_k(hits, [{"doc": "sc_x"}], 2) == 0


def test_summary_table_and_gold_check(tmp_path):
    rows = [{"q": "a", "type": "meaning", "dense": 0, "hybrid": 1}, {"q": "b", "type": "reference", "dense": 1, "hybrid": 1}]
    s = summarise(rows, ["dense", "hybrid"])
    assert s["all"] == {"dense": 0.5, "hybrid": 1.0} and s["meaning"]["hybrid"] == 1.0
    assert "recall@5 all (2)" in format_table(rows, ["dense", "hybrid"])
    (tmp_path / "d.jsonl").write_text(json.dumps({"unit_number": "5"}) + "\n", encoding="utf-8")
    bad = check_gold([{"q": "x", "gold": [{"doc": "d", "unit": "5"}, {"doc": "d", "unit": "6"}, {"doc": "zz"}]}], tmp_path)
    assert len(bad) == 2


def test_real_gold_set_matches_the_corpus():
    from config import BASE_DIR, settings
    qpath = BASE_DIR / "eval" / "queries.json"
    qs = json.loads(qpath.read_text(encoding="utf-8"))
    assert len(qs) == 30 and {q["type"] for q in qs} == {"meaning", "reference"}
    if (settings.chunks_data_dir / "bns_2023.jsonl").exists():
        assert check_gold(qs, settings.chunks_data_dir) == []
