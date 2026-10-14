# Nyaya Sahitya

A legal research tool that never makes things up: answers must be grounded in
source legal documents.

## Layout

| Folder | Purpose |
|---|---|
| `data/raw` | Original legal documents (untracked) |
| `data/processed` | Cleaned/chunked output (untracked) |
| `ingestion/` | Load and process documents |
| `retrieval/` | Find relevant passages |
| `generation/` | Produce grounded answers |
| `api/` | Backend API |
| `ui/` | User interface |
| `tests/` | pytest tests |

## Getting started

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit values
pytest
python main.py              # loads config and logs a startup message
```

Configuration lives in `config.py` and is read from `.env`
(see `.env.example` for all variables).

*Status: Step 1 (foundation) only. No retrieval or AI yet.*
