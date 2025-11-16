# 🔬 Agentic RAG

**Pure LangChain/LangGraph RAG Agent** - Advanced retrieval-augmented generation system with RAG Fusion and intelligent routing.

Implementation based on [official LangChain tutorial](https://python.langchain.com/docs/tutorials/rag_agent/).

## ✨ Features

- 🤖 **Intelligent Routing** - LLM autonomously decides when to retrieve documents
- ⚡ **RAG Fusion** - Multi-query retrieval + Reciprocal Rank Fusion (RRF) for better reranking
- 📊 **Stateless Mode** - Optimized for independent questions and RAGAS evaluation (no conversation memory)
- ⚡ **Streaming Responses** - Real-time answer generation via Server-Sent Events
- 📄 **Document Upload** - Supports PDF, TXT, MD, DOCX, IPYNB formats
- 🎯 **Optimized Retrieval** - k=8 documents with RAG Fusion
- 📈 **RAGAS Evaluation** - Automated quality assessment (Score: 87.4% - Grade A)

## 🛠️ Tech Stack

- **Backend**: FastAPI + LangChain + LangGraph
- **Frontend**: React 19 + Vite
- **LLM**: Claude Sonnet 4.5 (Anthropic)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB (local persistence)
- **Retrieval**: RAG Fusion (multi-query + RRF reranking, k=8)
- **Mode**: Stateless (optimized for independent questions)
- **Evaluation**: RAGAS framework

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- ANTHROPIC_API_KEY from [Anthropic](https://console.anthropic.com/)
- (Optional) TAVILY_API_KEY for web search

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd agentic-rag

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Start with Docker
./start.sh
```

The app will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

```bash
# 1. Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend/api && python main.py

# 2. Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Option 3: Development Mode

```bash
# Backend in Docker + Frontend with hot reload
./start-dev.sh
```

Frontend will be at http://localhost:5173 with hot reload enabled.

## 📋 Available Scripts

- `./start.sh` - Start all services with Docker
- `./start-dev.sh` - Backend in Docker + Frontend in dev mode
- `./stop.sh` - Stop all services
- `make build` - Build Docker images
- `make test` - Run evaluation tests
- `make clean` - Clean up containers and volumes

## 🏗️ Architecture

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
│   (Stateless)     │   + RAG Fusion
└───────────────────┘
    │
    ├─ Tool Call? ──→ RAG Fusion (k=8) ──→ Documents
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
- **LangGraph**: Orchestrates the agentic workflow
- **Tool Calling**: LLM autonomously decides if retrieval is needed
- **RAG Fusion**: Multi-query retrieval + Reciprocal Rank Fusion for better document reranking
- **Stateless Mode**: Each question processed independently (optimized for evaluation)
- **Streaming**: Real-time Server-Sent Events for better UX

## 📡 API Endpoints

- `POST /api/query` - Standard query (stateless)
- `POST /api/rag_agent` - RAG agent endpoint
- `POST /api/query/stream` - Streaming endpoint (SSE)
- `POST /api/upload` - Document upload
- `GET /api/documents` - List uploaded documents
- `DELETE /api/documents` - Delete specific document
- `DELETE /api/documents/all` - Clear all documents
- `GET /api/evaluation` - Get evaluation results
- `GET /health` - Health check

## 📊 Evaluation

The system includes automated evaluation using the RAGAS framework:

```bash
# Run evaluation on test dataset
python backend/scripts/run_evaluation.py

# Run with custom parameters
python backend/scripts/run_evaluation.py --limit 5 --no-rag-fusion --temperature 0.7
```

Metrics evaluated:
- **Context Precision** (30% weight) - Quality of retrieved documents
- **Answer Similarity** (70% weight) - Semantic similarity to ground truth

Results are saved in `backend/scripts/evaluation_report_FINAL.json`.

## 🐳 Docker Optimizations

This setup uses **CPU-only PyTorch** for optimal build times:
- ⚡ Build time: **3-5 minutes** (vs 60+ with CUDA)
- 💾 Image size: **~586 MB** backend (vs ~2.5 GB with GPU)
- 🚀 Works on any machine (no GPU required)

The optimization is configured in `requirements.txt`:
```python
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
```

## 📚 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed system architecture
- [DOCUMENTATION_TECHNIQUE.md](docs/DOCUMENTATION_TECHNIQUE.md) - Technical implementation details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 🙏 Acknowledgments

Built following the [LangChain RAG Agent tutorial](https://python.langchain.com/docs/tutorials/rag_agent/) with additional enhancements for production use.
