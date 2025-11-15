---
title: Agentic RAG
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🔬 Agentic RAG

**Pure LangChain/LangGraph RAG Agent** - Implementation based on [official LangChain tutorial](https://python.langchain.com/docs/tutorials/rag_agent/)

## Features

- ✅ **Intelligent Routing** - LLM decides when to retrieve documents
- ✅ **Conversation Memory** - Thread-based chat history with InMemorySaver
- ✅ **Streaming Responses** - Real-time answer generation via SSE
- ✅ **Document Upload** - PDF, TXT, MD, DOCX, IPYNB support
- ✅ **Optimized Retrieval** - k=4 similarity search with ChromaDB
- ✅ **Message Trimming** - Auto-manages context window (last 10 messages)

## Tech Stack

- **Backend**: FastAPI + LangChain + LangGraph
- **Frontend**: React 19 + Vite
- **LLM**: Claude Sonnet 4.5 (Anthropic)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB (local persistence)
- **Memory**: InMemorySaver (LangGraph checkpointer)

## Configuration

This Space requires the following secrets:

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `TAVILY_API_KEY`: Your Tavily API key for web search (optional)

## Local Development

```bash
# Clone and run with Docker
docker-compose up

# Or run manually
pip install -r requirements.txt
cd backend/api && python main.py
cd frontend && npm install && npm run dev
```

## Architecture

```
User Query
    │
    ▼
┌───────────────────┐
│   FastAPI API     │
│   (/api/query)    │
└───────────────────┘
    │
    ▼
┌───────────────────┐
│   RAG Agent       │ ← LangGraph StateGraph
│  (with Memory)    │   + InMemorySaver
└───────────────────┘
    │
    ├─ Tool Call? ──→ ChromaDB (k=4) ──→ Documents
    │                                        │
    └─ Direct Answer ←──────────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  Claude 4.5  │
            │  Generation  │
            └──────────────┘
                    │
                    ▼
            Streaming Response
            (SSE word-by-word)
```

**Key Components:**
- **LangGraph**: Orchestrates agent workflow
- **Tool Calling**: LLM decides if retrieval needed
- **InMemorySaver**: Persists conversation by thread_id
- **Streaming**: Real-time SSE for UX

## API Endpoints

- `POST /api/query` - Standard query (with memory support)
- `POST /api/rag_agent` - RAG agent endpoint
- `POST /api/query/stream` - Streaming endpoint
- `POST /api/upload` - Document upload
- `GET /api/documents` - List uploaded documents
- `GET /health` - Health check

## Documentation

📖 See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed implementation

## License

MIT
