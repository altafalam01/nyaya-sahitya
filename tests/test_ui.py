"""Step 9 tests: API client (mock transport) and the Streamlit app (AppTest, client patched). No network."""
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import ui.client as client
from ui.client import ApiError, Reply, Source, ask
from ui.strings import TEXT

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
BODY = {"answer": "Cheating [bns_2023, Section 318].", "lang": "en", "grounded": True,
        "citations": ["bns_2023, Section 318"], "disclaimer": "Not legal advice.",
        "sources": [{"cite": "bns_2023, Section 318", "locator": "Section 318", "doc_id": "bns_2023",
                     "legal_status": "in_force", "court": "", "year": 2023, "text": "Cheating text"}]}


def transport(status=200, body=None, seen=None):
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(req)
        return httpx.Response(status, json=body if body is not None else {})
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------- client
def test_strings_have_same_keys():
    assert TEXT["en"].keys() == TEXT["hi"].keys()


def test_ask_parses_reply_and_sends_options():
    seen = []
    r = ask("http://api", "What is cheating?", lang="hi", k=4, transport=transport(200, BODY, seen))
    assert r.grounded and r.citations == ["bns_2023, Section 318"] and r.sources[0].locator == "Section 318"
    assert seen[0].url.path == "/ask" and b'"lang":"hi"' in seen[0].content.replace(b" ", b"") and b'"k":4' in seen[0].content.replace(b" ", b"")


def test_ask_omits_unset_options():
    seen = []
    ask("http://api", "What is cheating?", transport=transport(200, BODY, seen))
    assert b"lang" not in seen[0].content and b'"k"' not in seen[0].content


@pytest.mark.parametrize("status,kind", [(503, "unconfigured"), (422, "invalid"), (500, "server")])
def test_ask_error_kinds(status, kind):
    with pytest.raises(ApiError) as e:
        ask("http://api", "What is cheating?", transport=transport(status, {"detail": "x"}))
    assert e.value.kind == kind


def test_ask_unreachable():
    def boom(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError) as e:
        ask("http://api", "What is cheating?", transport=httpx.MockTransport(boom))
    assert e.value.kind == "unreachable"


# ---------------------------------------------------------------- app
def reply(grounded=True, lang="en"):
    return Reply("Cheating [bns_2023, Section 318]." if grounded else "Not in sources.", lang, grounded,
                 ["bns_2023, Section 318"] if grounded else [],
                 [Source("bns_2023, Section 318", "Section 318", "in_force", "Cheating text")] if grounded else [],
                 "Not legal advice.")


def run_app(monkeypatch, fake):
    monkeypatch.setattr(client, "ask", fake)
    return AppTest.from_file(APP, default_timeout=15).run()


def texts(elements):
    return " ".join(e.value for e in elements)


def test_app_shows_disclaimer_and_title(monkeypatch):
    at = run_app(monkeypatch, lambda *a, **k: reply())
    assert not at.exception
    assert "not legal advice" in texts(at.info).lower()
    assert at.title[0].value == "Nyaya Sahitya"


def test_app_grounded_answer_and_sources(monkeypatch):
    calls = []

    def fake(url, q, **kw):
        calls.append((q, kw))
        return reply()

    at = run_app(monkeypatch, fake)
    at.chat_input[0].set_value("What is cheating?").run()
    assert not at.exception and calls == [("What is cheating?", {"lang": None})]
    assert any("Cheating [bns_2023" in m.value for m in at.markdown)
    assert at.expander[0].label.startswith("Sources")
    assert "Not legal advice." in texts(at.caption)


def test_app_not_grounded_shows_warning(monkeypatch):
    at = run_app(monkeypatch, lambda *a, **k: reply(grounded=False))
    at.chat_input[0].set_value("Unknown thing?").run()
    assert not at.exception and len(at.warning) == 1 and not at.expander


def test_app_hindi_choice_switches_ui_and_passes_lang(monkeypatch):
    calls = []

    def fake(url, q, **kw):
        calls.append(kw)
        return reply(lang="hi")

    at = run_app(monkeypatch, fake)
    at.segmented_control(key="lang_choice").set_value("hi").run()
    assert at.title[0].value == TEXT["hi"]["title"]
    at.chat_input[0].set_value("धोखाधड़ी क्या है?").run()
    assert calls == [{"lang": "hi"}]


def test_app_api_error_is_friendly(monkeypatch):
    def fake(*a, **k):
        raise ApiError("unreachable")

    at = run_app(monkeypatch, fake)
    at.chat_input[0].set_value("What is cheating?").run()
    assert not at.exception and at.error[0].value == TEXT["en"]["err_unreachable"]
