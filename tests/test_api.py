"""Step 8 tests: API endpoints with an injected answerer (no models, no network)."""
from fastapi.testclient import TestClient

from api.app import create_app
from generation.answer import Answer
from retrieval.search import Hit

HIT = Hit("c1", 1.0, "bns_2023, Section 318", "Section 318", "bns_2023", "in_force", "", 2023, "section", "Cheating text")


def client(fn):
    return TestClient(create_app(fn), raise_server_exceptions=False)


def test_health():
    r = client(lambda q, **o: None).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_grounded_roundtrip_and_options_forwarded():
    seen = {}

    def fn(q, **o):
        seen.update(q=q, o=o)
        return Answer("Cheating [bns_2023, Section 318].", "en", True, [HIT.cite], [HIT], [], "Not legal advice.")

    r = client(fn).post("/ask", json={"question": "  What is cheating?  ", "k": 3, "act": "BNS_2023"})
    body = r.json()
    assert r.status_code == 200
    assert body["grounded"] and body["citations"] == [HIT.cite] and body["disclaimer"] == "Not legal advice."
    assert body["sources"][0]["locator"] == "Section 318"
    assert seen == {"q": "What is cheating?", "o": {"k": 3, "act": "BNS_2023"}}


def test_ask_not_found_is_200_with_grounded_false():
    r = client(lambda q, **o: Answer("Not in sources.", "hi", False, [], [], [], "d")).post(
        "/ask", json={"question": "कुछ अजीब सवाल", "lang": "hi"})
    assert r.status_code == 200 and r.json()["grounded"] is False and r.json()["sources"] == []


def test_validation_errors():
    c = client(lambda q, **o: None)
    assert c.post("/ask", json={"question": "hi"}).status_code == 422          # too short
    assert c.post("/ask", json={}).status_code == 422
    assert c.post("/ask", json={"question": "valid question", "lang": "fr"}).status_code == 422
    assert c.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_unconfigured_llm_is_503_without_leaking_detail():
    def fn(q, **o):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    r = client(fn).post("/ask", json={"question": "What is cheating?"})
    assert r.status_code == 503 and "ANTHROPIC" not in r.text


def test_unexpected_error_is_500_generic():
    def fn(q, **o):
        raise ValueError("secret internals")

    r = client(fn).post("/ask", json={"question": "What is cheating?"})
    assert r.status_code == 500 and "secret" not in r.text
