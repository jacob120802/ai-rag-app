"""RAG service module.

Supports two modes:
1) pinecone: Azure LLM + Azure embeddings + Pinecone retrieval
2) local: fully local fallback for easy end-to-end development without external keys
"""

from __future__ import annotations

import io
import re
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
    mode: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    pinecone_index: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


class RAGService:
    """RAG service with production and local-dev backends."""

    def __init__(self, config: RAGConfig) -> None:
        self.mode = config.mode.lower()
        self.top_k = config.top_k
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.local_docs: dict[str, list[Document]] = {}

        if self.mode == "pinecone":
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
        else:
            self.llm = None
            self.embeddings = None
            self.vectorstore = None

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

        if self.mode == "pinecone":
            self.vectorstore.add_documents(docs, namespace=session_id)
        else:
            self.local_docs.setdefault(session_id, []).extend(docs)

        return len(docs)

    def _tokenize(self, value: str) -> set[str]:
        return set(re.findall(r"\w+", value.lower()))

    def _retrieve_local(self, question: str, session_id: str) -> list[Document]:
        question_tokens = self._tokenize(question)
        docs = self.local_docs.get(session_id, [])

        scored = []
        for doc in docs:
            overlap = len(question_tokens.intersection(self._tokenize(doc.page_content)))
            scored.append((overlap, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[: self.top_k] if score > 0] or [doc for _, doc in scored[: self.top_k]]

    def answer(self, question: str, session_id: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if self.mode == "pinecone":
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
            answer_text = result.content
        else:
            context_docs = self._retrieve_local(question, session_id)
            if not context_docs:
                answer_text = "No indexed content found for this session. Please upload a PDF first."
            else:
                excerpts = "\n".join([f"- {doc.page_content[:220]}" for doc in context_docs[:3]])
                answer_text = (
                    "[Local Dev Mode] I generated this answer from retrieved PDF chunks without external LLM calls.\n"
                    f"Question: {question}\n\n"
                    f"Most relevant excerpts:\n{excerpts}"
                )

        sources = [
            {
                "doc_id": d.metadata.get("doc_id"),
                "filename": d.metadata.get("filename"),
                "snippet": d.page_content[:200],
            }
            for d in context_docs
        ]

        return {"answer": answer_text, "sources": sources}
