# Project: Nyaya Sahitya
A Retrieval-Augmented Generation (RAG) app over Indian legal literature: the Constitution of India, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), Bharatiya Sakshya Adhiniyam (BSA), older IPC/CrPC/Evidence Act (for old-to-new mapping), key Acts, and Supreme Court / High Court judgments.

## Non-negotiable rules
- Answers must be grounded ONLY in retrieved text. Never invent sections, cases, or citations.
- Every claim carries a citation like [Act/Case, Section/Para].
- If the answer is not in the indexed sources, say so plainly.
- The app gives legal information, not legal advice. Show a disclaimer.
- Support English and Hindi.

## Stack
Python, FastAPI, a vector DB (Qdrant or Chroma), hybrid search (dense + BM25) with a reranker, simple web UI (React or Streamlit).

## Working style
- Work on ONE step at a time. Never start the next step on your own.
- Plan first, then build. Keep code modular and typed.
- Write tests for every module. Run them before saying a step is done.
- Keep secrets in .env, never in code.
- At the end of each step, report: what you did, files changed, test results, and the "Done when" check.
