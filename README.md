# AI RAG Starter Template (React + FastAPI + Pinecone + LangChain)

A reusable **starter template** for building modern AI SaaS chatbot apps with:

- React frontend (clean, minimal, reusable)
- FastAPI backend (modular service layer)
- Pinecone vector database for document retrieval
- SQLite for chat/session history (easy local setup; replace with PostgreSQL in production)
- LangChain-based RAG pipeline
- Azure GenAI Lab-compatible model configuration

---

## 1) What you will learn

This project is intentionally structured to help a beginner become production-ready:

1. How RAG works (ingest → chunk → embed → index → retrieve → generate)
2. Difference between vector DB (semantic search) and SQL DB (transactions/history)
3. How to design API-first AI systems
4. How to build a modular backend and simple React client
5. How to evaluate and improve a RAG app with practical metrics

---

## 2) Architecture (Enterprise mindset)

```text
React UI
  -> FastAPI REST API
      -> RAG Service (LangChain)
          -> Embedding model (Azure endpoint)
          -> Pinecone index (vector retrieval)
          -> Chat model (Azure endpoint)
      -> SQLite (sessions, chat history, uploaded docs metadata)
```

### Why this design?
- **Scalable**: Pinecone handles retrieval scale independently.
- **Reliable**: SQL DB keeps deterministic records of sessions/messages.
- **Maintainable**: Service boundaries (`db.py`, `rag.py`, `main.py`) are easy to swap.

---

## 3) Project structure

```text
.
├── backend
│   ├── main.py
│   ├── db.py
│   ├── rag.py
│   ├── requirements.txt
│   └── .env.example
└── frontend
    ├── package.json
    ├── index.html
    ├── vite.config.js
    └── src
        ├── main.jsx
        └── App.jsx
```

---

## 4) Setup

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# default RAG_MODE=local works without Pinecone/LLM keys
uvicorn main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: `http://localhost:5173`
API docs: `http://localhost:8000/docs`

---

## 5) Environment variables (`backend/.env`)

Use your provided hackathon keys and endpoints.


### RAG modes

- `RAG_MODE=local` (default): fully local retrieval mode for quick end-to-end testing without external credentials.
- `RAG_MODE=pinecone`: production mode using Azure chat/embeddings + Pinecone retrieval.

- `RAG_MODE=local`
- `GENAI_BASE_URL=https://genailab.tcs.in`
- `GENAI_API_KEY=...`
- `CHAT_MODEL=azure_ai/genailab-maas-DeepSeek-V3-0324`
- `EMBED_MODEL=azure/genailab-maas-text-embedding-3-large`
- `PINECONE_API_KEY=...`
- `PINECONE_INDEX=ai-rag-starter`
- `PINECONE_CLOUD=aws`
- `PINECONE_REGION=us-east-1`
- `SQLITE_PATH=./app.db`
- `ALLOWED_ORIGINS=http://localhost:5173`

---

## 6) API flow (prototype demo)

1. Create/load a session in UI.
2. Upload a PDF.
3. Backend extracts + chunks + embeds text and stores vectors in Pinecone namespace = session.
4. Ask chat question.
5. Backend retrieves top-k relevant chunks and sends context + history to LLM.
6. Response + citations/snippets shown in UI and persisted to SQL history.

---

## 7) Performance metrics and validation strategy

This template includes a practical evaluation approach (you can automate later):

- **Retrieval Precision@k**: fraction of top-k chunks that are relevant.
- **Answer Faithfulness**: whether answer is grounded in retrieved chunks.
- **Latency**: p50/p95 for embed, retrieval, generation.
- **User feedback score**: thumbs up/down on answer quality.

### Baseline comparison
Start with:
- Baseline: direct LLM (no retrieval)
- Proposed: RAG with Pinecone

Expected outcome: RAG improves factual grounding and domain relevance.

---

## 8) Limitations and future enhancements

### Current limitations
- Local SQLite (single-node, not ideal for high concurrency)
- Basic prompting/routing (no tool-calling agent yet)
- No auth/RBAC yet

### Next enterprise upgrades
- PostgreSQL + Alembic migrations
- Redis caching for hot queries
- Multi-tenant auth (JWT + tenant isolation)
- Observability (OpenTelemetry, LangSmith, Prometheus/Grafana)
- Guardrails (PII redaction, moderation, prompt injection defenses)
- CI/CD + containerization + IaC

---

## 9) Innovation options to stand out

- Hybrid search (dense + sparse)
- Query rewriting + self-reflection loop
- Cost-aware model routing (`gpt-4o-mini` vs larger models)
- Multi-modal ingestion (images + OCR + Whisper)

---

## 10) Reusability guidance

To reuse for another AI use case, usually change only:
- system prompt in `rag.py`
- ingestion parser (PDF/CSV/HTML/audio)
- metadata filters + retrieval strategy

Everything else can stay as-is.
