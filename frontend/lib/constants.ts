export const API_BASE_URL =
  typeof window !== "undefined" &&
  window.location.hostname !== "localhost" &&
  window.location.hostname !== "127.0.0.1"
    ? "https://chatgpt-pro-backend.onrender.com/api/v1"
    : "/api/v1";

export const STARTER_SUGGESTIONS = [
  "Explain quantum superposition with a simple analogy",
  "Write a Python FastAPI endpoint for real-time SSE streaming",
  "Summarize the key trade-offs between local and hosted LLMs",
  "How does FAISS vector search compute cosine similarity?",
];
