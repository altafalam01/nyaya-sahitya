"""Step 5 tests: fake encoder + temp Chroma dir, so no model is downloaded."""
import json
import math
import zlib

import pytest

from retrieval.embedder import PASSAGE_PREFIX, QUERY_PREFIX, Embedder, normalise, passage_text
from retrieval.index import run
from retrieval.search import search
from retrieval.store import Store, build_where, to_metadata, year_of

DIM = 32


def fake_vec(text: str) -> list[float]:
    """Bag-of-words hash vector: texts sharing words are close, so search is testable."""
    v = [0.0] * DIM
    for w in text.lower().replace("passage:", " ").replace("query:", " ").split():
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    return v


SEEN: list[str] = []


def fake_encoder(texts):
    SEEN.extend(texts)
    return [fake_vec(t) for t in texts]


def rec(chunk_id, doc_id, text, **kw):
    base = {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "context": kw.pop("context", doc_id),
            "chunk_type": "section", "doc_type": "act", "act": "BNS_2023", "legal_status": "in_force",
            "court": "", "date": "", "unit_label": "Section", "unit_number": "1", "locator": "Section 1",
            "cite": f"{doc_id}, Section 1", "doc_title": doc_id, "section_type": ""}
    base.update(kw)
    return base


def write(chunks_dir, doc_id, recs):
    (chunks_dir / f"{doc_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    emb = Embedder(model_name="org/Test-Model", batch_size=2, encoder=fake_encoder)
    store = Store(emb.collection_name, tmp_path / "index")
    SEEN.clear()
    return chunks, emb, store


def test_embedder_adds_prefixes_normalises_and_batches():
    emb = Embedder(batch_size=2, encoder=fake_encoder)
    SEEN.clear()
    vecs = emb.embed_passages(["a b", "c d", "e f"])
    assert len(vecs) == 3 and all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-9 for v in vecs)
    assert all(t.startswith(PASSAGE_PREFIX) for t in SEEN)
    SEEN.clear()
    emb.embed_query("hello")
    assert SEEN == [QUERY_PREFIX + "hello"]
    assert normalise([0, 0]) == [0.0, 0.0]


def test_collection_name_encodes_the_model():
    assert Embedder(model_name="intfloat/multilingual-e5-small").collection_name == "nyaya_multilingual_e5_small_v1"


def test_passage_text_uses_the_breadcrumb():
    assert passage_text({"context": "BNS > Section 318. Cheating", "text": "body"}) == "BNS > Section 318. Cheating\nbody"
    assert passage_text({"context": "", "text": "body"}) == "body"


def test_year_and_metadata():
    assert year_of({"date": "2017-08-24"}) == 2017
    assert year_of({"act": "IPC_1860"}) == 1860 and year_of({"act": "", "date": ""}) == 0
    m = to_metadata(rec("1", "d", "t", date="1973-04-24"))
    assert m["year"] == 1973 and m["act"] == "BNS_2023" and all(not isinstance(v, (list, dict)) for v in m.values())


def test_build_where():
    assert build_where(include_index=True) is None
    assert build_where(court="X", include_index=True) == {"court": "X"}
    w = build_where(year_from=1990, year_to=2000)
    assert {"year": {"$gte": 1990}} in w["$and"] and {"year": {"$lte": 2000}} in w["$and"]
    assert {"chunk_type": {"$nin": ["index", "front_matter"]}} in w["$and"]


def test_first_run_indexes_second_is_noop_edit_reembeds_only_that_chunk_and_removals_are_deleted(env):
    chunks, emb, store = env
    write(chunks, "d1", [rec("a", "d1", "cheating deceives person"), rec("b", "d1", "murder punished death")])
    r1 = run(chunks, store, emb)
    assert r1[0]["status"] == "indexed" and r1[0]["embedded"] == 2 and store.count() == 2
    assert run(chunks, store, emb)[0]["status"] == "up_to_date"
    write(chunks, "d1", [rec("a", "d1", "cheating deceives person"), rec("b", "d1", "murder changed text")])
    r3 = run(chunks, store, emb)[0]
    assert (r3["status"], r3["embedded"], r3["deleted"]) == ("updated", 1, 0)
    write(chunks, "d1", [rec("a", "d1", "cheating deceives person")])
    r4 = run(chunks, store, emb)[0]
    assert r4["deleted"] == 1 and store.count() == 1


def test_changed_breadcrumb_reembeds(env):
    chunks, emb, store = env
    write(chunks, "d1", [rec("a", "d1", "text", context="old")])
    run(chunks, store, emb)
    write(chunks, "d1", [rec("a", "d1", "text", context="new")])
    assert run(chunks, store, emb)[0]["embedded"] == 1


def test_only_rebuild_and_prune(env):
    chunks, emb, store = env
    write(chunks, "d1", [rec("a", "d1", "one")])
    write(chunks, "d2", [rec("b", "d2", "two")])
    assert [r["doc_id"] for r in run(chunks, store, emb, only=["d2"])] == ["d2"] and store.count() == 1
    run(chunks, store, emb)
    (chunks / "d2.jsonl").unlink()
    assert run(chunks, store, emb)[-1]["status"] == "orphan" and store.count() == 2
    assert run(chunks, store, emb, prune=True)[-1]["status"] == "pruned" and store.count() == 1
    r = run(chunks, store, emb, rebuild=True)
    assert r[0]["embedded"] == 1


def test_a_bad_document_does_not_stop_the_others(env):
    chunks, emb, store = env
    (chunks / "bad.jsonl").write_text("{not json\n", encoding="utf-8")
    write(chunks, "good", [rec("a", "good", "fine")])
    st = {r["doc_id"]: r["status"] for r in run(chunks, store, emb)}
    assert st == {"bad": "failed", "good": "indexed"}


def test_search_ranks_by_meaning_and_filters_work(env):
    chunks, emb, store = env
    write(chunks, "bns_2023", [rec("c318", "bns_2023", "Cheating deceiving person dishonestly induces delivery of property",
                                   locator="Section 318"),
                               rec("c101", "bns_2023", "Murder culpable homicide death punishment", locator="Section 101"),
                               rec("idx", "bns_2023", "cheating cheating cheating contents list", chunk_type="index")])
    write(chunks, "ipc_1860", [rec("i415", "ipc_1860", "Cheating deceiving person dishonestly", act="IPC_1860",
                                   legal_status="repealed")])
    write(chunks, "sc_2017_x", [rec("j1", "sc_2017_x", "privacy cheating fundamental right", chunk_type="judgment_text",
                                    doc_type="judgment", act="", court="Supreme Court of India", date="2017-08-24"),
                                rec("j2", "sc_1973_y", "privacy basic structure", chunk_type="judgment_text",
                                    doc_type="judgment", act="", court="Supreme Court of India", date="1973-04-24")])
    run(chunks, store, emb)
    hits = search("cheating deceiving", 5, embedder=emb, store=store)
    assert "c318" in [h.chunk_id for h in hits[:3]] and "idx" not in [h.chunk_id for h in hits]  # index chunks hidden
    assert "idx" in [h.chunk_id for h in search("cheating", 5, embedder=emb, store=store, include_index=True)]
    assert {h.chunk_id for h in search("cheating", 9, embedder=emb, store=store, act="IPC_1860")} == {"i415"}
    ipc = search("cheating", 3, embedder=emb, store=store, act="IPC_1860")[0]
    assert ipc.legal_status == "repealed" and ipc.year == 1860
    assert all(h.legal_status == "in_force" for h in search("cheating", 9, embedder=emb, store=store, legal_status="in_force"))
    y = search("privacy", 9, embedder=emb, store=store, court="Supreme Court of India", year=2017)
    assert [h.chunk_id for h in y] == ["j1"]
    assert {h.chunk_id for h in search("privacy", 9, embedder=emb, store=store, year_from=1970, year_to=1980,
                                       doc_type="judgment")} == {"j2"}
    assert search("cheating", 5, embedder=emb, store=store, court="No Such Court") == []


def test_search_on_an_empty_index_returns_nothing(env):
    _, emb, store = env
    assert search("anything", 5, embedder=emb, store=store) == []
