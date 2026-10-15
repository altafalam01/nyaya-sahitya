# Nyaya Sahitya

A Retrieval-Augmented Generation (RAG) app over Indian legal literature. Answers are grounded only in
retrieved source text, every claim carries a citation, and anything not in the indexed sources is reported
as "not found" rather than guessed. Supports English and Hindi.

> **Legal information, not legal advice.** Consult a qualified advocate for your situation.

## What it covers

Constitution of India, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS),
Bharatiya Sakshya Adhiniyam (BSA), the older IPC (with IPC to BNS mapping), and a set of landmark
Supreme Court judgments.

## How it works

1. **Ingestion** (`ingestion/`): collect, parse and chunk documents by section/article/paragraph.
2. **Retrieval** (`retrieval/`): dense (multilingual-e5) + BM25, fused with RRF, reranked by a cross-encoder.
   A query pre-processor recognises references like "Section 420 IPC" and maps IPC and BNS sections.
3. **Generation** (`generation/`): Claude answers only from the retrieved sources. A citation validator removes
   any sentence that is uncited or cites something not retrieved.
4. **API** (`api/`): FastAPI, `POST /ask` and `GET /health`.
5. **UI** (`ui/`): Streamlit chat in English and Hindi, with expandable sources.

## Getting started

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then set ANTHROPIC_API_KEY in .env (never commit it)
pytest
```

The source documents and the search index are not stored in this repository (`data/` is git-ignored).
Build them with the ingestion and indexing modules before running the app.

Run the app (two terminals):

```powershell
uvicorn api.app:app
streamlit run ui/app.py
```

Configuration lives in `config.py` and is read from `.env` (see `.env.example` for all variables).

## Layout

| Folder | Purpose |
|---|---|
| `data/` | Raw, processed and chunked documents and the index (untracked) |
| `ingestion/` | Load and process documents |
| `retrieval/` | Hybrid search, reranking, query pre-processing |
| `generation/` | Grounded answers with citation validation |
| `api/` | FastAPI backend |
| `ui/` | Streamlit interface |
| `eval/` | Retrieval evaluation queries |
| `tests/` | pytest tests |

## License

MIT, see [LICENSE](LICENSE). This covers the code only; legal source texts remain subject to their own terms.
