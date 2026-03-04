"""RAG service module.

Contains all AI-related logic:
- model client creation
- PDF parsing and chunking
- vector upsert
- retrieval + answer generation
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader


@dataclass
class RAGConfig:
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    pinecone_index: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


class RAGService:
    """Encapsulates retrieval-augmented generation operations."""

    def __init__(self, config: RAGConfig) -> None:
        http_client = httpx.Client(verify=False, timeout=60.0)
        self.llm = ChatOpenAI(
            base_url=config.base_url,
            model=config.chat_model,
            api_key=config.api_key,
            http_client=http_client,
            temperature=0.2,
        )
        self.embeddings = OpenAIEmbeddings(
            base_url=config.base_url,
            model=config.embed_model,
            api_key=config.api_key,
            http_client=http_client,
        )
        self.vectorstore = PineconeVectorStore(
            index_name=config.pinecone_index,
            embedding=self.embeddings,
        )
        self.top_k = config.top_k
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def parse_pdf(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip()

    def index_document(self, text: str, session_id: str, doc_id: str, filename: str) -> int:
        chunks = self.splitter.split_text(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={"session_id": session_id, "doc_id": doc_id, "filename": filename},
            )
            for chunk in chunks
        ]
        self.vectorstore.add_documents(docs, namespace=session_id)
        return len(docs)

    def answer(self, question: str, session_id: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k, "namespace": session_id},
        )
        context_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in context_docs)

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-8:]])

        prompt = f"""
You are an enterprise assistant. Answer only from retrieved context.
If context is insufficient, say what is missing.

Conversation history:
{history_text}

Retrieved context:
{context}

User question:
{question}
""".strip()

        result = self.llm.invoke(prompt)

        sources = [
            {
                "doc_id": d.metadata.get("doc_id"),
                "filename": d.metadata.get("filename"),
                "snippet": d.page_content[:200],
            }
            for d in context_docs
        ]

        return {"answer": result.content, "sources": sources}
