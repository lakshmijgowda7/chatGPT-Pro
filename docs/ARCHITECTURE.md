# Project 3: Full-Stack ChatGPT-Style AI Platform Architecture

## System Architecture Diagram

```mermaid
graph TD
    User([Web User / Browser])
    
    subgraph Frontend [Next.js 14 / React / TypeScript]
        UI[React UI Components]
        ChatHook[useChat / useStreaming Hook]
        APIClient[Frontend API Client lib/api.ts]
        UI --> ChatHook
        ChatHook --> APIClient
    end

    subgraph Backend [FastAPI Server - Port 8000]
        Main[FastAPI Application main.py]
        Router[API v1 Router]
        ChatRouter[/api/v1/chat]
        ConvRouter[/api/v1/conversations]
        DocRouter[/api/v1/documents]
        
        ChatService[Chat Service]
        RAGService[RAG Service]
        LLMClient[Hosted LLM Client OpenAI / Groq]
        
        Main --> Router
        Router --> ChatRouter
        Router --> ConvRouter
        Router --> DocRouter
        ChatRouter --> ChatService
        DocRouter --> RAGService
        ChatService --> LLMClient
        RAGService --> LLMClient
    end

    subgraph Storage [Database & File Store]
        DB[(PostgreSQL / SQLite Database)]
        DocStore[(Local / Cloud Document Storage)]
    end

    subgraph HostedLLM [Hosted LLM Inference Cloud]
        CloudAPI[Groq / OpenRouter / OpenAI API]
    end

    User <-->|HTTP / SSE Streaming| UI
    APIClient <-->|REST API / SSE| Router
    ChatService <-->|CRUD| DB
    RAGService <-->|CRUD| DB
    RAGService <-->|Read / Write| DocStore
    LLMClient <-->|HTTPS API / SSE| CloudAPI
```

---

## Component Separation & Responsibilities

1. **Frontend (`frontend/`)**:
   - Handles all user interaction, state management, and real-time streaming display.
   - Built with Next.js 14 App Router, React 18, TypeScript, and Tailwind CSS.
   - **Zero Secrets**: Contains NO API keys. Communicates strictly with the FastAPI backend.

2. **Backend (`backend/app/`)**:
   - **`core/config.py`**: Centralized environment variable validation using Pydantic Settings.
   - **`database/`**: SQLAlchemy declarative base, session dependency, and automatic schema initialization for PostgreSQL (with SQLite local fallback).
   - **`models/`**: Declarative models for `Conversation`, `Message`, and `Document`.
   - **`schemas/`**: Pydantic v2 schemas for API validation and serialization.
   - **`llm/`**: Hosted LLM client adapter supporting OpenAI format (compatible with Groq, OpenRouter, Together AI, OpenAI).
   - **`rag/`**: Modular document extraction, chunking, and semantic vector retrieval.
   - **`api/v1/`**: Clean RESTful routes with Server-Sent Events (SSE) streaming.
