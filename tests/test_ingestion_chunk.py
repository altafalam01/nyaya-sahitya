import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import chunk as chunk_cli
from ingestion.chunk import fingerprint, run
from ingestion.chunker import ChunkParams, chunk_document, pack

ROOT = Path(__file__).resolve().parent.parent
P = ChunkParams()


def doc(doc_id, paras, type_="act", title=None, **extra):
    d = {"doc_id": doc_id, "title": title or doc_id, "type": type_, "court": "", "date": "", "bench": "",
         "citation": "", "source_url": "https://example.org/x.pdf", "act": "", "legal_status": "", "usage_note": "",
         "text": "\n\n".join(paras), "extraction": {"parsed_at": "2026-01-01T00:00:00+00:00"}}
    d.update(extra)
    return d


# A made-up Act: contents list, chapters, headings above the number (with and without a full stop),
# a cross-heading, a number glued to its bracket ("3.(1)"), a page footnote, inline headings, a schedule.
TEST_ACT = [
    "THE TEST ACT, 2024",
    "ARRANGEMENT OF SECTIONS",
    "CHAPTER I", "PRELIMINARY",
    "1. Short title.", "2. Definitions.",
    "CHAPTER II", "OF OFFENCES",
    "3. Theft.", "4. Cheating.", "5. Robbery.", "6. Fraud.", "7. Penalty.", "8. Repeal.",
    "CHAPTER I", "PRELIMINARY",
    "Short title",
    "1. (1) This Act may be called the Test Act, 2024.",
    "(2) It extends to the whole of India.",
    "Definitions.",
    "2. In this Act, unless the context otherwise requires,—",
    "(a) \"court\" means a court of law;",
    "CHAPTER II", "OF OFFENCES",
    "Of property",
    "Theft.",
    "3.(1) Whoever takes movable property out of possession without consent commits theft.",
    "1. Subs. by Act 3 of 1951, s. 3, for the words \"the States\".",
    "Cheating",
    "4. Whoever cheats any person shall be punished with imprisonment.",
    "5. Robbery.--In all robbery there is either theft or extortion.",
    "6. Fraud.--Whoever commits fraud shall be liable to fine.",
    "7. Penalty.--The penalty for any offence under this Act is as stated in each section.",
    "8. Repeal.--The Old Act is hereby repealed.",
    "THE FIRST SCHEDULE",
    "1. Form of notice to the accused person.",
    "2. Form of order of the court.",
]

TEST_CONSTITUTION = [
    "THE CONSTITUTION OF TESTLAND", "INDEX",
    "PART I", "THE UNION AND ITS TERRITORY",
    "1. Name and territory of the Union.", "2. Admission or establishment of new States.",
    "PART III", "FUNDAMENTAL RIGHTS",
    "14. Equality before law.", "15. Prohibition of discrimination.", "21. Protection of life and personal liberty.",
    "21A. Right to education.", "22. Protection against arrest and detention.",
    "PREAMBLE", "WE, THE PEOPLE OF TESTLAND, having solemnly resolved, give to ourselves this Constitution.",
    "PART I", "THE UNION AND ITS TERRITORY",
    "1. Name and territory of the Union.—(1) Testland, that is Test, shall be a Union of States.",
    "(2) The States and the territories thereof shall be as specified in the First Schedule.",
    "2. Admission or establishment of new States.—Parliament may by law admit new States into the Union.",
    "PART III", "FUNDAMENTAL RIGHTS",
    "Right to Equality",
    "14. Equality before law.—The State shall not deny to any person equality before the law.",
    "15. Prohibition of discrimination.—(1) The State shall not discriminate against any citizen.",
    "Right to Freedom",
    "21. Protection of life and personal liberty.—No person shall be deprived of his life except by law.",
    "21A. Right to education.—The State shall provide free and compulsory education to all children.",
    "22. Protection against arrest and detention.—(1) No person who is arrested shall be detained without cause.",
    "THE FIRST SCHEDULE", "I. The States", "1. Alpha", "2. Beta", "3. Gamma",
    "THE SECOND SCHEDULE", "Emoluments of the President", "1. There shall be paid to the President a salary.",
]

# A different Act in a different layout: every heading inline, no chapters, no contents list.
INLINE_ACT = [f"{n}. Heading number {n}.—Text of provision number {n} that is long enough to count." for n in range(1, 9)]


def units(records, kind):
    return [r for r in records if r["chunk_type"] == kind]


# ---------------------------------------------------------------- sections
def test_sections_are_found_from_the_numbering_with_headings_and_chapters():
    recs, info = chunk_document(doc("test_act", TEST_ACT, title="The Test Act, 2024"))
    secs = units(recs, "section")
    assert [r["unit_number"] for r in secs] == [str(n) for n in range(1, 9)]
    by = {r["unit_number"]: r for r in secs}
    assert info["units"] == 8 and info["missing_count"] == 0
    assert by["1"]["heading"] == "Short title" and by["1"]["hierarchy"]["chapter"] == "CHAPTER I - PRELIMINARY"
    assert by["2"]["heading"] == "Definitions"  # heading with a full stop, matched to the contents list
    assert by["3"]["heading"] == "Theft" and by["3"]["group_heading"] == "Of property"
    assert by["3"]["hierarchy"]["chapter"] == "CHAPTER II - OF OFFENCES"
    assert by["4"]["heading"] == "Cheating"
    assert by["5"]["heading"] == "Robbery" and by["6"]["heading"] == "Fraud"  # inline headings
    assert by["3"]["text"].startswith("3.(1) Whoever")  # glued "3.(1)" is still a section start
    assert by["3"]["locator"] == "Section 3" and by["3"]["cite"] == "The Test Act, 2024, Section 3"


def test_headings_above_a_number_do_not_leak_into_the_previous_section():
    secs = {r["unit_number"]: r for r in units(chunk_document(doc("a", TEST_ACT))[0], "section")}
    for n in ("1", "2", "3"):
        assert "Cheating" not in secs[n]["text"] and "Of property" not in secs["2"]["text"]
    assert "Definitions." not in secs["1"]["text"] and "Short title" not in secs["1"]["text"]


def test_contents_list_and_front_matter_are_kept_but_labelled_not_mixed_with_sections():
    recs, _ = chunk_document(doc("a", TEST_ACT))
    kinds = {r["chunk_type"] for r in recs}
    assert {"front_matter", "index", "section", "schedule"} <= kinds
    assert "1. Short title." in units(recs, "index")[0]["text"]
    assert all("1. Short title." not in r["text"] for r in units(recs, "section"))
    assert len(units(recs, "section")) == 8  # the contents list did not add duplicate sections


def test_page_footnotes_do_not_become_sections():
    recs, _ = chunk_document(doc("a", TEST_ACT))
    assert sorted(int(r["unit_number"]) for r in units(recs, "section")) == list(range(1, 9))
    assert all(not r["text"].startswith("1. Subs.") for r in units(recs, "section"))


def test_schedule_is_split_by_size_not_numbered_as_sections():
    recs, _ = chunk_document(doc("a", TEST_ACT))
    sched = units(recs, "schedule")
    assert sched and sched[0]["locator"] == "THE FIRST SCHEDULE"
    assert "Form of notice" in sched[0]["text"] and sched[0]["hierarchy"] == {"schedule": "THE FIRST SCHEDULE"}
    assert "Form of notice" not in "".join(r["text"] for r in units(recs, "section"))


def test_a_differently_laid_out_act_uses_the_same_chunker_unchanged():
    recs, info = chunk_document(doc("other_act", INLINE_ACT, title="The Other Act, 1999"))
    secs = units(recs, "section")
    assert [r["unit_number"] for r in secs] == [str(n) for n in range(1, 9)]
    assert secs[2]["heading"] == "Heading number 3" and secs[2]["cite"] == "The Other Act, 1999, Section 3"
    assert info["structure"] == "numbered" and info["unit"] == "Section"


def test_numbering_with_suffix_letters_and_gaps_is_read_in_order():
    paras = [f"{n}. Provision {n}. Text of provision {n} goes here in full." for n in (1, 2, 3, 3, 4)]
    paras = ["1. Alpha provision text.", "2. Beta provision text.", "2A. Inserted provision text.", "3. Gamma provision text.",
             "5. Epsilon provision text after a repealed number.", "6. Zeta provision text."]
    recs, info = chunk_document(doc("a", paras))
    assert [r["unit_number"] for r in units(recs, "section")] == ["1", "2", "2A", "3", "5", "6"]
    assert info["missing_numbers"] == [4]  # reported, not invented


# ---------------------------------------------------------------- constitution
def test_constitution_uses_articles_parts_and_schedules():
    recs, info = chunk_document(doc("constitution_of_india", TEST_CONSTITUTION, "constitution", title="The Constitution of Testland"))
    arts = units(recs, "section") + units(recs, "article")
    assert info["unit"] == "Article" and [r["chunk_type"] for r in arts] == ["article"] * 7
    by = {r["unit_number"]: r for r in arts}
    assert list(by) == ["1", "2", "14", "15", "21", "21A", "22"]
    assert by["21A"]["locator"] == "Article 21A" and by["21A"]["cite"] == "The Constitution of Testland, Article 21A"
    assert by["14"]["hierarchy"]["part"] == "PART III - FUNDAMENTAL RIGHTS"
    assert by["1"]["hierarchy"]["part"] == "PART I - THE UNION AND ITS TERRITORY"
    assert by["14"]["heading"] == "Equality before law" and by["14"]["group_heading"] == "Right to Equality"
    assert by["21"]["group_heading"] == "Right to Freedom"
    assert "Right to Freedom" not in by["15"]["text"]
    assert by["21A"]["context"].endswith("PART III - FUNDAMENTAL RIGHTS > Article 21A. Right to education")


def test_constitution_schedules_are_separate_and_not_read_as_articles():
    recs, _ = chunk_document(doc("constitution_of_india", TEST_CONSTITUTION, "constitution"))
    sched = units(recs, "schedule")
    assert [r["locator"] for r in sched] == ["THE FIRST SCHEDULE", "THE SECOND SCHEDULE"]
    assert "Alpha" in sched[0]["text"] and "salary" in sched[1]["text"]
    assert "Alpha" not in "".join(r["text"] for r in units(recs, "article"))
    pre = [r for r in recs if r["chunk_type"] == "front_matter"]
    assert pre and "WE, THE PEOPLE" in pre[0]["text"]  # the preamble is kept, as front matter


# ---------------------------------------------------------------- sizes, overlap, Hindi, metadata
def test_judgment_without_numbered_paragraphs_gets_no_invented_pinpoint():
    recs, _ = chunk_document(doc("sc_y", ["Plain paragraph. " * 40] * 8, "judgment", title="C v. D", citation="[1990] 2 S.C.R. 5"))
    assert all(r["locator"] == "" and r["cite"] == "C v. D, [1990] 2 S.C.R. 5" for r in recs)


def test_long_section_is_split_between_clauses_and_loses_nothing():
    clauses = [f"({i}) " + ("word " * 60).strip() + "." for i in range(1, 11)]  # about 90 tokens each
    paras = INLINE_ACT[:3] + ["4. Long provision.—" + clauses[0]] + clauses[1:] + INLINE_ACT[4:]
    recs, _ = chunk_document(doc("a", paras), ChunkParams(max_tokens=200, min_tokens=100))
    parts = [r for r in units(recs, "section") if r["unit_number"] == "4"]
    assert len(parts) >= 3 and [r["part_index"] for r in parts] == list(range(1, len(parts) + 1))
    assert all(r["part_count"] == len(parts) and r["token_estimate"] <= int(200 * 1.15) for r in parts)
    assert all(r["locator"] == "Section 4" and r["heading"] == "Long provision" for r in parts)
    for r in parts:  # every cut falls between whole clauses: each piece is a complete original paragraph
        assert all(piece in paras for piece in r["text"].split("\n\n"))
    assert "\n\n".join(r["text"] for r in parts) == "\n\n".join(paras[3:3 + 10])


def test_provisos_explanations_and_sub_clauses_stay_with_the_clause_they_qualify():
    body = ("word " * 55).strip() + "."  # about 85 tokens
    paras = INLINE_ACT[:3] + ["4. Heading four.—(1) " + body, "Provided that " + body, "Explanation.—" + body,
                              "(2) " + body, "(a) " + body, "(b) " + body] + INLINE_ACT[4:]
    recs, _ = chunk_document(doc("a", paras), ChunkParams(max_tokens=300, min_tokens=100))
    parts = [r for r in units(recs, "section") if r["unit_number"] == "4"]
    assert len(parts) == 2
    assert "Provided that" in parts[0]["text"] and "Explanation." in parts[0]["text"]
    assert parts[1]["text"].startswith("(2)") and "(a) " in parts[1]["text"] and "(b) " in parts[1]["text"]
    for r in parts:
        first = r["text"].split("\n\n")[0]
        assert not first.startswith(("Provided", "Explanation", "(a)", "(b)"))


def test_chunk_sizes_follow_the_token_budget():
    from ingestion.chunker import estimate_tokens
    paras = [f"{n}. " + ("The court held the matter carefully. " * 14).strip() for n in range(1, 41)]
    prm = ChunkParams(max_tokens=200, min_tokens=100, overlap_tokens=80)
    recs, _ = chunk_document(doc("sc_z", paras, "judgment"), prm)
    assert len(recs) > 10
    assert all(r["token_estimate"] == estimate_tokens(r["text"]) <= int(200 * 1.15) for r in recs)
    assert sum(r["token_estimate"] >= 100 for r in recs) >= len(recs) * 0.8


def test_hindi_words_are_not_over_counted_as_tokens():
    from ingestion.chunker import estimate_tokens
    assert estimate_tokens("भारत का संविधान") == 5  # three words, not one piece per vowel sign
    assert estimate_tokens("Whoever commits murder shall be punished with death.") < 20


def test_hindi_text_is_split_at_the_danda_and_numbering_in_devanagari_digits_is_read():
    from ingestion.chunker import estimate_tokens, make_segs
    sentence = "यह एक परीक्षण वाक्य है जो बहुत लंबा है और पूर्ण विराम पर समाप्त होता है। "
    text = sentence * 60
    windows = pack(make_segs([0], [text]), ChunkParams(max_tokens=120, min_tokens=60))
    assert len(windows) > 3
    for segs, _kept in windows:
        joined = " ".join(s.text for s in segs)
        assert estimate_tokens(joined) <= 140 and joined.rstrip().endswith("।")
    paras = [f"{d}. धारा {d} का पाठ यहाँ पर लिखा गया है।" for d in "१२३४५६"]
    recs, _ = chunk_document(doc("hindi_act", paras, title="परीक्षण अधिनियम"))
    assert [r["unit_number"] for r in units(recs, "section")] == list("१२३४५६")


def test_judgment_is_windowed_with_overlap_and_numbered_paragraph_pinpoints():
    paras = ["The Judgment of the Court was delivered by SOME J.— This petition raises a question."]
    paras += [f"{n}. " + ("This is paragraph text about the right to equality. " * 4).strip() for n in range(1, 21)]
    recs, info = chunk_document(doc("sc_x", paras, "judgment", title="A v. B", citation="[1950] 1 S.C.R. 1"),
                                ChunkParams(max_tokens=200, min_tokens=100, overlap_tokens=80))
    assert info["structure"] == "paragraph_windows" and len(recs) > 3
    assert all(r["chunk_type"] == "judgment_text" and r["token_estimate"] <= int(200 * 1.15) for r in recs)
    assert recs[1]["paragraph_numbers"] and recs[1]["locator"].startswith("para")
    assert recs[1]["cite"].startswith("A v. B, [1950] 1 S.C.R. 1, para")
    # the last paragraph of one window opens the next (overlap), and the record says so
    assert recs[0]["text"].split("\n\n")[-1] == recs[1]["text"].split("\n\n")[0] and recs[1]["overlap_paragraphs"] == 1


def test_every_chunk_points_back_to_its_exact_place_in_the_source_text():
    from ingestion.chunker import paragraph_spans
    long_one = ("Sentence number one is here. " * 80).strip()  # a single paragraph over the budget
    jud = ["Headnote text."] + [f"{n}. Paragraph {n} text of the judgment goes here." for n in range(1, 30)] + [long_one]
    docs = [(doc("act", TEST_ACT), ChunkParams(max_tokens=60, min_tokens=30)),
            (doc("con", TEST_CONSTITUTION, "constitution"), ChunkParams(max_tokens=60, min_tokens=30)),
            (doc("sc_t", jud, "judgment"), ChunkParams(max_tokens=100, min_tokens=50, overlap_tokens=40))]
    for d, prm in docs:
        paras, spans = paragraph_spans(d["text"])
        recs, _ = chunk_document(d, prm)
        assert recs
        for r in recs:
            a, b = r["para_start"], r["para_end"]
            assert (r["char_start"], r["char_end"]) == (spans[a][0], spans[b][1])
            span = d["text"][r["char_start"]:r["char_end"]]
            assert span.startswith(paras[a]) and span.endswith(paras[b])
            assert all(piece in span for piece in r["text"].split("\n\n")), (d["doc_id"], r["seq"])
        assert recs[0]["para_start"] >= 0 and recs[-1]["para_end"] == len(paras) - 1


# ---------------------------------------------------------------- judgment passage types
FILL = "The matter was considered with care and the record was examined in detail by the Bench. "


def _facts_para(n):
    return f"The facts of the case are brief. The petitioner filed a writ petition in the year 1975 about matter {n}. " + FILL


def _arg_para(n):
    return f"Learned counsel for the petitioner submitted that point {n} must succeed, and it was contended on the other side. " + FILL


def _ratio_para(n):
    return f"In our view the law is plain and we hold that the impugned provision is valid on ground {n}. " + FILL


def test_judgment_chunks_are_tagged_only_where_cue_phrases_make_it_detectable():
    paras = ["Constitution of India - Art.21 - catch words of the head note."] + ["HELD: " + FILL * 3] * 5
    paras += ["The Judgment of the Court was delivered by SOME, J.— This writ petition is before us."]
    paras += [_facts_para(n) for n in range(4)] + [FILL * 3] * 3 + [_arg_para(n) for n in range(4)]
    paras += [_ratio_para(n) for n in range(4)] + [FILL * 2] * 3
    paras += ["The appeal is dismissed. There will be no order as to costs."]
    recs, _ = chunk_document(doc("sc_tag", paras, "judgment"), ChunkParams(max_tokens=200, min_tokens=100))
    tags = [r["section_type"] for r in recs]
    assert {"headnote", "facts", "arguments", "ratio", "order"} <= set(tags)
    assert "" in tags  # filler-only passages stay untagged: nothing is guessed
    by = {t: next(r for r in recs if r["section_type"] == t) for t in ("facts", "arguments", "ratio", "order")}
    assert "The facts of the case" in by["facts"]["text"] and "the facts" in by["facts"]["section_cues"]
    assert "Learned counsel" in by["arguments"]["text"] and "we hold that" in by["ratio"]["section_cues"]
    assert "dismissed" in " ".join(by["order"]["section_cues"]) and recs[-1]["section_type"] == "order"
    assert all(r["section_type"] == "" for r in recs if r["doc_type"] != "judgment")


def test_one_stray_cue_phrase_is_not_enough_to_tag_a_chunk():
    paras = ["The Judgment of the Court was delivered by SOME, J.— Here we begin."]
    paras += [FILL * 3] * 6 + ["Counsel submitted one thing. " + FILL * 2] + [FILL * 3] * 6
    recs, _ = chunk_document(doc("sc_one", paras, "judgment"), ChunkParams(max_tokens=200, min_tokens=100))
    assert all(r["section_type"] in ("", "headnote") for r in recs)


def test_a_new_window_starts_when_the_kind_of_passage_changes():
    facts = [_facts_para(n) for n in range(2)]  # about 130 tokens: already past the minimum
    args = [_arg_para(n) for n in range(2)]
    paras = ["The Judgment of the Court was delivered by SOME, J.— Begin."] + facts + args
    recs, _ = chunk_document(doc("sc_split", paras, "judgment"), ChunkParams(max_tokens=480, min_tokens=100, overlap_tokens=0))
    body = [r for r in recs if "facts of the case" in r["text"] or "Learned counsel" in r["text"]]
    assert len(body) == 2 and "Learned counsel" not in body[0]["text"] and "facts of the case" not in body[1]["text"]
    assert [r["section_type"] for r in body] == ["facts", "arguments"]


def test_statute_chunks_carry_no_passage_type():
    recs, _ = chunk_document(doc("a", TEST_ACT))
    assert all(r["section_type"] == "" and r["section_cues"] == [] for r in recs)


def test_repealed_status_and_usage_note_stay_on_every_chunk():
    d = doc("ipc_test", TEST_ACT, act="IPC_1860", legal_status="repealed", usage_note="REPEALED. Never present as current law.")
    recs, _ = chunk_document(d)
    assert recs and all(r["legal_status"] == "repealed" and "Never present as current law" in r["usage_note"]
                        and r["act"] == "IPC_1860" for r in recs)


def test_chunk_ids_are_stable_unique_and_text_is_verbatim():
    d = doc("a", TEST_ACT)
    a, _ = chunk_document(d)
    b, _ = chunk_document(d)
    assert [r["chunk_id"] for r in a] == [r["chunk_id"] for r in b]
    assert len({r["chunk_id"] for r in a}) == len(a)
    for r in a:
        assert r["text"] in d["text"] or all(p in d["text"] for p in r["text"].split("\n\n"))


def test_document_with_no_numbering_falls_back_to_windows_and_empty_text_gives_nothing():
    recs, info = chunk_document(doc("plain", ["Some prose paragraph. " * 20] * 6))
    assert info["structure"] == "no_numbering_found" and recs and all(r["chunk_type"] == "text" for r in recs)
    assert chunk_document(doc("empty", []))[0] == []


# ---------------------------------------------------------------- re-runnable behaviour
@pytest.fixture
def dirs(tmp_path):
    processed, chunks = tmp_path / "processed", tmp_path / "chunks"
    processed.mkdir()
    return processed, chunks


def put(processed, d):
    (processed / f"{d['doc_id']}.json").write_text(json.dumps(d), encoding="utf-8")


def statuses(result):
    return {r["doc_id"]: r["status"] for r in result["documents"]}


def test_second_run_skips_everything_that_is_up_to_date(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    put(processed, doc("sc_a", ["1. Para text. " * 30] * 8, "judgment", citation="[1] 1 S.C.R. 1"))
    first = run(processed, chunks, P)
    assert statuses(first) == {"act_a": "chunked", "sc_a": "chunked"}
    before = {p.name: p.read_bytes() for p in chunks.glob("*.jsonl")}
    mtimes = {p.name: p.stat().st_mtime_ns for p in chunks.glob("*.jsonl")}
    second = run(processed, chunks, P)
    assert statuses(second) == {"act_a": "up_to_date", "sc_a": "up_to_date"}
    assert {p.name: p.stat().st_mtime_ns for p in chunks.glob("*.jsonl")} == mtimes
    assert {p.name: p.read_bytes() for p in chunks.glob("*.jsonl")} == before
    assert not list(chunks.glob("*.tmp"))


def test_a_new_document_is_picked_up_and_the_others_left_alone(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    run(processed, chunks, P)
    put(processed, doc("constitution_of_india", TEST_CONSTITUTION, "constitution"))  # parsed later
    result = run(processed, chunks, P)
    assert statuses(result) == {"act_a": "up_to_date", "constitution_of_india": "chunked"}
    assert (chunks / "constitution_of_india.jsonl").exists()


def test_a_changed_document_is_rebuilt_and_only_changed_chunks_are_new(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    put(processed, doc("act_b", TEST_ACT))
    run(processed, chunks, P)
    edited = [p.replace("without consent commits theft", "without consent commits the offence of theft") for p in TEST_ACT]
    put(processed, doc("act_a", edited))
    result = run(processed, chunks, P)
    assert statuses(result) == {"act_a": "rebuilt", "act_b": "up_to_date"}
    r = next(x for x in result["documents"] if x["doc_id"] == "act_a")
    assert r["added"] == 1 and r["removed"] == 1 and r["unchanged"] == r["chunks"] - 1


def test_reparsing_with_a_new_timestamp_but_the_same_text_is_not_a_change(dirs):
    processed, chunks = dirs
    d = doc("act_a", TEST_ACT)
    put(processed, d)
    run(processed, chunks, P)
    d["extraction"]["parsed_at"] = "2027-12-31T00:00:00+00:00"
    put(processed, d)
    assert statuses(run(processed, chunks, P)) == {"act_a": "up_to_date"}
    assert fingerprint(d) == fingerprint(doc("act_a", TEST_ACT))


def test_changed_settings_force_a_rebuild_and_force_flag_works(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    run(processed, chunks, P)
    assert statuses(run(processed, chunks, replace(P, max_tokens=150))) == {"act_a": "rebuilt"}
    assert statuses(run(processed, chunks, replace(P, max_tokens=150))) == {"act_a": "up_to_date"}
    assert statuses(run(processed, chunks, replace(P, max_tokens=150), force=True)) == {"act_a": "rebuilt"}


def test_missing_output_or_damaged_manifest_is_rebuilt_not_trusted(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    run(processed, chunks, P)
    (chunks / "act_a.jsonl").unlink()
    assert statuses(run(processed, chunks, P)) == {"act_a": "chunked"}
    (chunks / "manifest.json").write_text("{ not json", encoding="utf-8")
    assert statuses(run(processed, chunks, P)) == {"act_a": "rebuilt"}


def test_orphans_are_reported_and_only_deleted_with_prune(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    put(processed, doc("act_gone", TEST_ACT))
    run(processed, chunks, P)
    (processed / "act_gone.json").unlink()
    assert statuses(run(processed, chunks, P))["act_gone"] == "orphan" and (chunks / "act_gone.jsonl").exists()
    assert statuses(run(processed, chunks, P, prune=True))["act_gone"] == "pruned"
    assert not (chunks / "act_gone.jsonl").exists()
    assert "act_gone" not in json.loads((chunks / "manifest.json").read_text(encoding="utf-8"))["documents"]


def test_one_unreadable_document_does_not_stop_the_others_and_parse_report_is_ignored(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    (processed / "broken.json").write_text("{ nope", encoding="utf-8")
    (processed / "parse_report.json").write_text(json.dumps({"documents": []}), encoding="utf-8")
    result = run(processed, chunks, P)
    s = statuses(result)
    assert s["act_a"] == "chunked" and s["broken"] == "failed" and "parse_report" not in s


def test_only_flag_limits_the_run(dirs):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    put(processed, doc("act_b", TEST_ACT))
    assert statuses(run(processed, chunks, P, only=["act_b"])) == {"act_b": "chunked"}
    assert not (chunks / "act_a.jsonl").exists()


def test_cli_main_runs_against_configured_folders(dirs, monkeypatch, capsys):
    processed, chunks = dirs
    put(processed, doc("act_a", TEST_ACT))
    monkeypatch.setattr(chunk_cli, "settings", type("S", (), {"processed_data_dir": processed, "chunks_data_dir": chunks})())
    assert chunk_cli.main([]) == 0
    assert "[chunked] act_a" in capsys.readouterr().out
    assert chunk_cli.main([]) == 0
    assert "[up_to_date] act_a" in capsys.readouterr().out


def test_chunks_folder_is_configurable():
    import config
    assert config.settings.chunks_data_dir.name == "chunks"


# ---------------------------------------------------------------- the real corpus
PROCESSED = ROOT / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "bns_2023.json").exists() or not (PROCESSED / "ipc_1860.json").exists(),
                    reason="run `python -m ingestion.parse` first")
def test_real_statutes_are_chunked_by_section(tmp_path):
    chunks = tmp_path / "chunks"
    run(PROCESSED, chunks, P, only=["bns_2023", "ipc_1860"])
    for doc_id, n_min in (("bns_2023", 350), ("ipc_1860", 495)):
        recs = [json.loads(line) for line in (chunks / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        secs = {r["unit_number"] for r in recs if r["chunk_type"] == "section"}
        assert len(secs) >= n_min, (doc_id, len(secs))
        assert "1" in secs and all(r["token_estimate"] <= int(P.max_tokens * 1.15) for r in recs if r["chunk_type"] == "section")
        assert all(r["heading"] for r in recs if r["chunk_type"] == "section" and r["part_index"] == 1 and r["unit_number"] in ("1", "77", "377"))
    ipc = [json.loads(line) for line in (chunks / "ipc_1860.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["legal_status"] == "repealed" and r["usage_note"] for r in ipc)
    bns = [json.loads(line) for line in (chunks / "bns_2023.jsonl").read_text(encoding="utf-8").splitlines()]
    s77 = next(r for r in bns if r["unit_number"] == "77")
    assert s77["heading"].strip(".") == "Voyeurism" and "CHAPTER V" in s77["context"]


@pytest.mark.skipif(not (PROCESSED / "bns_2023.json").exists() or not (PROCESSED / "sc_1997_vishaka.json").exists(),
                    reason="run `python -m ingestion.parse` first")
def test_real_chunks_trace_back_to_the_source_and_no_section_is_cut_mid_clause(tmp_path):
    from ingestion.chunker import paragraph_spans
    ids = ["bns_2023", "ipc_1860", "sc_1997_vishaka", "sc_1985_shah_bano", "sc_1978_maneka_gandhi"]
    chunks = tmp_path / "chunks"
    run(PROCESSED, chunks, P, only=ids)
    for doc_id in ids:
        d = json.loads((PROCESSED / f"{doc_id}.json").read_text(encoding="utf-8"))
        paras, spans = paragraph_spans(d["text"])
        whole = set(paras)
        recs = [json.loads(line) for line in (chunks / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert recs
        for r in recs:
            a, b = r["para_start"], r["para_end"]
            span = d["text"][r["char_start"]:r["char_end"]]
            assert (r["char_start"], r["char_end"]) == (spans[a][0], spans[b][1]), (doc_id, r["seq"])
            assert all(piece in span for piece in r["text"].split("\n\n")), (doc_id, r["seq"])
            assert r["token_estimate"] <= int(P.max_tokens * 1.15), (doc_id, r["seq"], r["token_estimate"])
            if r["chunk_type"] == "section":  # a section is only ever cut between its own clauses
                assert all(piece in whole for piece in r["text"].split("\n\n")), (doc_id, r["locator"], r["part_index"])


@pytest.mark.skipif(len(list(PROCESSED.glob("sc_*.json"))) < 20, reason="needs the parsed judgments")
def test_real_judgments_get_conservative_passage_tags(tmp_path):
    chunks = tmp_path / "chunks"
    ids = [p.stem for p in PROCESSED.glob("sc_*.json")]
    run(PROCESSED, chunks, P, only=ids)
    with_order = 0
    tagged = total = 0
    for doc_id in ids:
        recs = [json.loads(line) for line in (chunks / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        total += len(recs)
        tagged += sum(bool(r["section_type"]) for r in recs)
        assert {r["section_type"] for r in recs} <= {"", "headnote", "facts", "issues", "arguments", "ratio", "order"}
        assert all(r["section_cues"] or r["section_type"] in ("", "headnote") for r in recs)  # a tag always shows its evidence
        with_order += any(r["section_type"] == "order" for r in recs[-8:])
    assert with_order >= 15  # the operative order is found at the end of most judgments
    assert 0.03 <= tagged / total <= 0.40  # tags are sparse by design: precision over recall
