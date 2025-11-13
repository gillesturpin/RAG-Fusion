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

**Pure LangChain/LangGraph implementation** - Advanced RAG system with agentic workflow

## Features

- ✅ **Intelligent Document Processing** - PDF, TXT, MD, DOCX support
- ✅ **Streaming Responses** - Real-time answer generation
- ✅ **Document Upload** - Drag & drop interface
- ✅ **Conversation Memory** - Session-based chat history
- ✅ **Optimized Retrieval** - k=4 with enhanced prompts

## Tech Stack

- **Backend**: FastAPI + LangChain + LangGraph
- **Frontend**: React + Vite
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **LLM**: Claude (Anthropic)

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
User Query → FastAPI → RAG Agent (LangGraph)
                         ↓
                    ChromaDB (k=4)
                         ↓
                    Claude LLM
                         ↓
                 Streaming Response
```

## License

MIT
