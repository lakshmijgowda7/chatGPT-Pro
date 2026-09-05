# Project 3: REST & Streaming API Specification

Base URL: `http://localhost:8000/api/v1`

---

## 1. Chat & Inference Endpoints

### `POST /api/v1/chat/completions`
Send a user prompt and generate an AI response.

**Request Body:**
```json
{
  "conversation_id": "conv_12345",
  "message": "Explain quantum computing in simple terms.",
  "mode": "chat",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024
}
```

### `POST /api/v1/chat/stream`
Stream token chunks in real-time using Server-Sent Events (SSE).

**Response (SSE Stream):**
```
data: {"token": "Quantum", "is_complete": false}
data: {"token": " computing", "is_complete": false}
...
data: {"token": "", "is_complete": true, "message_id": "msg_987"}
```

---

## 2. Conversation Endpoints

### `GET /api/v1/conversations`
List all saved conversations.

### `POST /api/v1/conversations`
Create a new conversation session.

### `GET /api/v1/conversations/{id}`
Retrieve a specific conversation with all message history.

### `PATCH /api/v1/conversations/{id}`
Update conversation title.

### `DELETE /api/v1/conversations/{id}`
Permanently delete a conversation.

---

## 3. Document & RAG Endpoints

### `POST /api/v1/documents/upload`
Upload a document (PDF, DOCX, TXT).

### `GET /api/v1/documents`
List all uploaded and indexed documents.

### `DELETE /api/v1/documents/{id}`
Delete a document and clear its vector index.
