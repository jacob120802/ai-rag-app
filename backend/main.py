"""FastAPI entrypoint for the AI RAG starter backend."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from db import ChatDB
from rag import RAGConfig, RAGService


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rag_mode: str = "local"
    genai_base_url: str = "https://genailab.tcs.in"
    genai_api_key: str = "replace_me"
    chat_model: str = "azure_ai/genailab-maas-DeepSeek-V3-0324"
    embed_model: str = "azure/genailab-maas-text-embedding-3-large"
    pinecone_api_key: str = "replace_me"
    pinecone_index: str = "ai-rag-starter"
    allowed_origins: str = "http://localhost:5173"
    sqlite_path: str = "./app.db"
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200


settings = Settings()
os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

app = FastAPI(title="AI RAG Starter API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = ChatDB(settings.sqlite_path)
rag = RAGService(
    RAGConfig(
        mode=settings.rag_mode,
        base_url=settings.genai_base_url,
        api_key=settings.genai_api_key,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        pinecone_index=settings.pinecone_index,
        top_k=settings.top_k,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "rag_mode": settings.rag_mode}


@app.post("/sessions")
def create_session() -> dict[str, str]:
    return {"session_id": db.create_session()}


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str) -> dict:
    return {"messages": db.get_messages(session_id)}


@app.get("/sessions/{session_id}/documents")
def get_documents(session_id: str) -> dict:
    return {"documents": db.list_documents(session_id)}


@app.post("/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    text = rag.parse_pdf(pdf_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    doc_id = db.save_document(session_id=session_id, filename=file.filename)
    chunk_count = rag.index_document(text, session_id=session_id, doc_id=doc_id, filename=file.filename)

    return {"doc_id": doc_id, "filename": file.filename, "chunks_indexed": chunk_count}


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    db.save_message(payload.session_id, "user", payload.message)
    history = db.get_messages(payload.session_id)
    result = rag.answer(payload.message, payload.session_id, history)
    db.save_message(payload.session_id, "assistant", result["answer"])
    return result
