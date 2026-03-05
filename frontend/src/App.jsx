import React, { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

async function parseError(response, fallbackMessage) {
  try {
    const data = await response.json();
    return data.detail || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [ragMode, setRagMode] = useState("unknown");

  const canUpload = useMemo(() => Boolean(file && sessionId && !isUploading), [file, sessionId, isUploading]);
  const canAsk = useMemo(
    () => Boolean(question.trim() && sessionId && !isThinking),
    [question, sessionId, isThinking]
  );

  useEffect(() => {
    const init = async () => {
      try {
        const healthResponse = await fetch(`${API_BASE}/health`);
        if (healthResponse.ok) {
          const healthData = await healthResponse.json();
          setRagMode(healthData.rag_mode || "unknown");
        }

        const response = await fetch(`${API_BASE}/sessions`, { method: "POST" });
        if (!response.ok) {
          throw new Error(await parseError(response, "Unable to create session."));
        }
        const data = await response.json();
        setSessionId(data.session_id);
        setStatus("Session initialized.");
      } catch (error) {
        setStatus(`Backend unavailable at ${API_BASE}. ${error.message}`);
      }
    };

    init();
  }, []);

  const uploadPdf = async () => {
    if (!canUpload) return;

    setIsUploading(true);
    setStatus("Uploading and indexing PDF...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/upload?session_id=${sessionId}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await parseError(response, "Document upload failed."));
      }

      const data = await response.json();
      setStatus(`Indexed ${data.chunks_indexed} chunks from ${data.filename}.`);
      setFile(null);
    } catch (error) {
      setStatus(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const ask = async () => {
    if (!canAsk) return;

    const userMessage = { role: "user", content: question.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setIsThinking(true);
    setStatus("Generating response...");

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error(await parseError(response, "Chat request failed."));
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
      setSources(data.sources || []);
      setQuestion("");
      setStatus("Answer ready.");
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I could not reach the backend or retrieval service. Please check API logs and credentials.",
        },
      ]);
      setStatus(`Chat failed: ${error.message}`);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="left-panel card">
        <h1>AI RAG Studio</h1>
        <p className="muted">Modern starter for enterprise GenAI assistants.</p>

        <div className="panel-block">
          <label className="label">Session ID</label>
          <div className="session-id">{sessionId || "Creating session..."}</div>
        </div>

        <div className="panel-block">
          <label className="label">Upload PDF</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <button onClick={uploadPdf} disabled={!canUpload}>
            {isUploading ? "Indexing..." : "Index Document"}
          </button>
        </div>

        <div className="panel-block status">
          <label className="label">System status</label>
          <p>{status}</p>
        </div>
      </aside>

      <main className="chat-panel card">
        <header>
          <h2>Chat Assistant</h2>
          <span className="pill">RAG mode: {ragMode}</span>
        </header>

        <section className="chat-feed">
          {messages.length === 0 ? (
            <p className="empty-state">Upload a PDF and ask your first question to begin.</p>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-role">{message.role}</div>
                <div className="message-body">{message.content}</div>
              </div>
            ))
          )}
        </section>

        <section className="composer">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about your uploaded documents..."
            onKeyDown={(event) => {
              if (event.key === "Enter") ask();
            }}
          />
          <button onClick={ask} disabled={!canAsk}>
            {isThinking ? "Thinking..." : "Send"}
          </button>
        </section>
      </main>

      <section className="sources-panel card">
        <h3>Retrieved Sources</h3>
        {sources.length === 0 ? (
          <p className="empty-state">Source snippets will appear here after a response.</p>
        ) : (
          sources.map((source, index) => (
            <article key={index} className="source-item">
              <p className="source-title">{source.filename || "Untitled document"}</p>
              <p className="source-meta">Doc ID: {source.doc_id || "N/A"}</p>
              <p className="source-snippet">{source.snippet}...</p>
            </article>
          ))
        )}
      </section>
    </div>
  );
}
