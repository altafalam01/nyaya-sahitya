"""Step 7 tests: prompt, language, citation checking, grounded answering. No network, no models."""
from generation.answer import generate_answer
from generation.citations import check_citations
from generation.prompt import MESSAGES, NOT_FOUND, build_sources, detect_language, system_prompt, user_prompt
from retrieval.search import Hit


def hit(cite, text="Whoever cheats shall be punished.", status="in_force"):
    return Hit(cite, 1.0, cite, "Section 318", "bns_2023", status, "", 2023, "section", text)


H1 = hit("bns_2023, Section 318")
H2 = hit("ipc_1860, Section 420", status="repealed")


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


# ---------------------------------------------------------------- language / prompt
def test_detect_language():
    assert detect_language("What is cheating?") == "en"
    assert detect_language("धोखाधड़ी की सज़ा क्या है?") == "hi"
    assert detect_language("धारा 420 IPC क्या है") == "hi"
    assert detect_language("") == "en"


def test_prompt_contains_sources_and_status():
    s = build_sources([H1, H2])
    assert "[SOURCE 1] CITE: bns_2023, Section 318" in s and "STATUS: repealed" in s
    assert "STATUS" not in s.split("[SOURCE 2]")[0]
    assert "QUESTION: q" in user_prompt("q", [H1])
    assert "Hindi" in system_prompt("hi") and NOT_FOUND in system_prompt("en")


# ---------------------------------------------------------------- citations
def test_valid_citations_kept():
    r = check_citations("Cheating is punishable [bns_2023, Section 318]. It is cognizable [BNS_2023,  section 318].",
                        [H1.cite])
    assert not r.dropped and r.citations == [H1.cite]
    assert r.text.count("[") == 2


def test_invented_citation_sentence_dropped():
    r = check_citations("Real [bns_2023, Section 318]. Fake [bns_2023, Section 999]. Uncited claim.", [H1.cite])
    assert r.text == "Real [bns_2023, Section 318]."
    assert len(r.dropped) == 2


def test_mixed_valid_and_invalid_in_one_sentence_dropped():
    r = check_citations("Both [bns_2023, Section 318][x, Section 1].", [H1.cite])
    assert r.text == "" and r.citations == []


def test_full_stop_inside_citation_does_not_split():
    c = "State v. Kumar, Para 5"
    r = check_citations(f"The court held so [{c}].", [c])
    assert r.text == f"The court held so [{c}]." and not r.dropped


def test_hindi_danda_split():
    r = check_citations("पहला [bns_2023, Section 318]। दूसरा बिना उद्धरण।", [H1.cite])
    assert r.text == "पहला [bns_2023, Section 318]।" and len(r.dropped) == 1


# ---------------------------------------------------------------- generate_answer
def test_grounded_answer():
    llm = FakeLLM("Cheating is an offence [bns_2023, Section 318].")
    a = generate_answer("What is cheating?", [H1, H2], llm)
    assert a.grounded and a.lang == "en" and a.citations == [H1.cite] and a.sources == [H1]
    assert a.disclaimer == MESSAGES["en"]["disclaimer"]
    assert "SOURCES" in llm.calls[0][1]


def test_no_hits_refuses_without_calling_llm():
    llm = FakeLLM("anything")
    a = generate_answer("What is X?", [], llm)
    assert not a.grounded and a.text == MESSAGES["en"]["not_found"] and not llm.calls


def test_model_not_found_sentinel():
    a = generate_answer("What is X?", [H1], FakeLLM(NOT_FOUND))
    assert not a.grounded and a.text == MESSAGES["en"]["not_found"] and a.citations == []


def test_all_citations_invalid_refuses():
    a = generate_answer("What is X?", [H1], FakeLLM("Made up [fake, Section 1]."))
    assert not a.grounded and a.text == MESSAGES["en"]["not_found"] and a.dropped


def test_hindi_question_gets_hindi_refusal_and_disclaimer():
    a = generate_answer("धोखाधड़ी क्या है?", [H1], FakeLLM(NOT_FOUND))
    assert a.lang == "hi" and a.text == MESSAGES["hi"]["not_found"] and a.disclaimer == MESSAGES["hi"]["disclaimer"]


def test_lang_override():
    a = generate_answer("cheating kya hai", [H1], FakeLLM(NOT_FOUND), lang="hi")
    assert a.lang == "hi"
