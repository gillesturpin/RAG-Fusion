# 🔧 Documentation Technique - Agentic RAG

**Version** : 2.0.0
**Date** : Novembre 2025
**Implémentation** : Pure LangChain/LangGraph selon [tutoriel officiel](https://python.langchain.com/docs/tutorials/rag_agent/)

## Table des matières
1. [Stack Technique](#stack-technique)
2. [Structure du Projet](#structure-du-projet)
3. [Implémentation RAG Agent](#implémentation-rag-agent)
4. [API REST](#api-rest)
5. [Mémoire Conversationnelle](#mémoire-conversationnelle)
6. [Streaming](#streaming)
7. [Upload de Documents](#upload-de-documents)
8. [Tests](#tests)
9. [Déploiement](#déploiement)

---

## 🛠️ Stack Technique

### Backend
- **Python** 3.12+
- **FastAPI** - Framework web async
- **LangChain** 0.3.x - Orchestration LLM
- **LangGraph** 0.2.x - State machine workflow
- **Anthropic** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **ChromaDB** - Vector store (local persistence)
- **HuggingFace** - Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **InMemorySaver** - Conversation memory (LangGraph checkpointer)

### Frontend
- **React** 19
- **Vite** 7.x - Build tool
- **Axios** - HTTP client
- **Framer Motion** - Animations
- **React Markdown** - Rendu markdown
- **Recharts** - Graphiques

### Infrastructure
- **Docker** & **Docker Compose**
- **Uvicorn** - ASGI server
- **CORS** - Middleware FastAPI

---

## 📁 Structure du Projet

```
agentic-rag/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI server (480 lignes)
│   │
│   └── rags/
│       ├── __init__.py           # Package exports
│       └── rag_agent.py         # RAG Agent avec mémoire (212 lignes)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Application React
│   │   └── ...
│   │
│   ├── Dockerfile               # Build React
│   └── package.json
│
├── data/
│   └── chroma_db/               # Persistance ChromaDB
│
├── docs/
│   ├── ARCHITECTURE.md          # Architecture détaillée
│   └── DOCUMENTATION_TECHNIQUE.md # Ce fichier
│
├── docker-compose.yml           # Orchestration
├── Dockerfile                   # Build backend
├── requirements.txt             # Dépendances Python
├── test_memory.py               # Tests mémoire
└── README.md                    # Documentation
```

---

## 💻 Implémentation RAG Agent

### Fichier : `backend/rags/rag_agent.py`

Le RAG Agent suit exactement le tutoriel officiel LangChain avec ajout de la mémoire conversationnelle.

### 1. Initialisation

```python
class RAGAgent:
    def __init__(self, vectorstore, checkpointer=None):
        """
        Args:
            vectorstore: ChromaDB vectorstore existant
            checkpointer: InMemorySaver pour la mémoire (optionnel)
        """
        self.vectorstore = vectorstore

        # Mémoire conversationnelle
        self.checkpointer = checkpointer or InMemorySaver()

        # LLM
        self.model = init_chat_model(
            "claude-sonnet-4-5-20250929",
            model_provider="anthropic"
        )

        # Créer le tool de retrieval
        @tool
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vectorstore.similarity_search(query, k=4)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized

        self.tools = [retrieve]

        # Bind tools au model
        self.model_with_tools = self.model.bind_tools(self.tools)

        # Build le graph LangGraph
        self.graph = self._build_graph()
```

**Points clés** :
- **Vectorstore** : Passé en paramètre (créé au startup de l'API)
- **Checkpointer** : `InMemorySaver` pour persister les conversations
- **Tool** : Fonction `retrieve` wrappée avec `@tool`
- **k=4** : Récupère 4 documents max
- **Graph** : Compilé avec le checkpointer

### 2. Build du Graph LangGraph

```python
def _build_graph(self):
    """Construit le StateGraph LangGraph"""
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", self._call_model)
    workflow.add_node("tools", ToolNode(self.tools))

    # Entry point
    workflow.add_edge(START, "agent")

    # Conditional routing
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        # Si tool_calls présents → execute tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Sinon → fin
        return END

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )

    # Loop: tools → agent
    workflow.add_edge("tools", "agent")

    # Compile avec memory
    return workflow.compile(checkpointer=self.checkpointer)
```

**Flow** :
1. START → agent
2. Agent décide : tool call ou réponse directe
3. Si tool call → tools → agent (loop)
4. Si pas de tool call → END

### 3. Call Model avec Trimming

```python
def _call_model(self, state: MessagesState):
    """Invoke LLM avec message trimming"""
    messages = state["messages"]

    # Trim pour éviter overflow du contexte
    messages = trim_messages(messages)

    # Ajouter system message si absent
    if not messages or messages[0].type != "system":
        system_msg = SystemMessage(
            "You have access to a tool that retrieves context from documents. "
            "Use the tool to help answer user queries. "
            "IMPORTANT: Provide COMPLETE and COMPREHENSIVE answers."
        )
        messages = [system_msg] + messages

    # Appel LLM avec tools
    response = self.model_with_tools.invoke(messages)
    return {"messages": [response]}
```

**Trimming** :
```python
def trim_messages(messages):
    """Garde les 10 derniers messages pour ne pas dépasser le contexte"""
    if len(messages) <= 10:
        return messages
    # Garde le premier (system) + 9 derniers
    return [messages[0]] + messages[-9:]
```

### 4. Invoke avec Thread ID

```python
def invoke(self, question: str, thread_id: str = None) -> dict:
    """
    Exécute le RAG Agent avec support mémoire

    Args:
        question: Question de l'utilisateur
        thread_id: ID du thread pour mémoire conversationnelle

    Returns:
        dict avec answer, messages, used_retrieval, thread_id
    """
    # Config pour mémoire
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    # Invoke graph
    from langchain_core.messages import HumanMessage
    result = self.graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )

    # Extraire réponse
    last_message = result["messages"][-1]

    # Détecter si retrieval utilisé
    used_retrieval = any(
        msg.type == "tool" for msg in result["messages"]
    )

    return {
        "answer": last_message.content,
        "messages": result["messages"],
        "used_retrieval": used_retrieval,
        "thread_id": thread_id
    }
```

---

## 🌐 API REST

### Fichier : `backend/api/main.py`

### Startup Event

```python
@app.on_event("startup")
async def startup_event():
    """Initialize RAG agent on startup"""
    global rag_agent, vectorstore

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load ChromaDB
    persist_dir = Path(__file__).parent.parent.parent / "data" / "chroma_db"
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

    # Create checkpointer
    checkpointer = InMemorySaver()

    # Initialize RAG agent
    rag_agent = RAGAgent(vectorstore, checkpointer=checkpointer)

    print("✅ RAG Agent initialized with memory support")
```

### Endpoints Principaux

#### 1. `POST /api/rag_agent`

Endpoint natif pour le RAG Agent.

```python
@app.post("/api/rag_agent", response_model=AgentResponse)
async def query_rag_agent(request: QueryRequest):
    """Query the RAG agent with optional memory support"""
    start_time = time.time()

    try:
        # Generate thread_id si absent
        thread_id = request.thread_id or str(uuid.uuid4())

        # Invoke
        result = rag_agent.invoke(request.question, thread_id=thread_id)
        latency = time.time() - start_time

        return AgentResponse(
            answer=result["answer"],
            used_retrieval=result.get("used_retrieval"),
            latency=latency,
            thread_id=result.get("thread_id")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Request** :
```json
{
  "question": "What is RAG?",
  "thread_id": "thread-abc123"  // Optionnel
}
```

**Response** :
```json
{
  "answer": "RAG stands for...",
  "used_retrieval": true,
  "latency": 2.45,
  "thread_id": "thread-abc123"
}
```

#### 2. `POST /api/query`

Endpoint adapter pour compatibilité frontend.

```python
@app.post("/api/query")
async def query_adapter(request: dict):
    """Adapter pour frontend original"""
    question = request.get("question", "")
    session_id = request.get("session_id", str(uuid.uuid4()))

    # Mapping session_id → thread_id
    if session_id not in session_threads:
        session_threads[session_id] = f"thread-{uuid.uuid4()}"
    thread_id = session_threads[session_id]

    # Invoke agent
    result = rag_agent.invoke(question, thread_id=thread_id)

    return {
        "answer": result["answer"],
        "method": "optimized_rag_agent",
        "latency": ...,
        "confidence": 0.90,  # Hardcodé
        "faithfulness": 0.92,  # Hardcodé
        "thread_id": thread_id
    }
```

**⚠️ Note** : `confidence` et `faithfulness` sont **hardcodés** ici.

#### 3. `POST /api/upload`

Upload et traitement de documents.

```python
@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload PDF, TXT, MD, DOCX, IPYNB"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )

    for file in files:
        # Save temp
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load selon extension
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.filename.endswith(('.txt', '.md')):
            loader = TextLoader(tmp_path)
        elif file.filename.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(tmp_path)
        elif file.filename.endswith('.ipynb'):
            loader = NotebookLoader(tmp_path)

        documents = loader.load()

        # Split
        chunks = text_splitter.split_documents(documents)

        # Add metadata
        for chunk in chunks:
            chunk.metadata["source"] = file.filename
            chunk.metadata["upload_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Add to vectorstore
        vectorstore.add_documents(chunks)

    return {"total_chunks": total_chunks}
```

---

## 🧠 Mémoire Conversationnelle

### InMemorySaver

LangGraph utilise un **checkpointer** pour persister l'état du graph entre les appels.

**Fonctionnement** :
- Chaque `thread_id` a son propre état isolé
- L'historique des messages est sauvegardé
- Pas de limite de durée (reste en RAM)

### Utilisation

```python
# Conversation 1 (thread-1)
agent.invoke("My name is Alice", thread_id="thread-1")
agent.invoke("What is my name?", thread_id="thread-1")
# → "Your name is Alice" ✅

# Conversation 2 (thread-2)
agent.invoke("What is my name?", thread_id="thread-2")
# → "I don't know your name" ✅
```

### Gestion des Sessions (API)

```python
# Store thread_id par session_id
session_threads = {}

# Mapping
if session_id not in session_threads:
    session_threads[session_id] = f"thread-{uuid.uuid4()}"

thread_id = session_threads[session_id]
```

**Note** : En production, utiliser Redis ou PostgreSQL checkpointer pour persistance.

---

## 🚀 Streaming

### Server-Sent Events (SSE)

Endpoint streaming pour réponses en temps réel.

```python
@app.post("/api/query/stream")
async def query_stream(request: dict):
    """Streaming SSE"""
    question = request.get("question", "")
    thread_id = ...

    return StreamingResponse(
        generate_sse_stream(question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
```

### Générateur SSE

```python
async def generate_sse_stream(question: str, thread_id: str):
    """Generate SSE events"""

    # Get full response
    result = rag_agent.invoke(question, thread_id)
    full_response = result["answer"]

    # Simulate streaming (mot par mot)
    words = full_response.split()
    for word in words:
        event_data = {
            "type": "token",
            "content": word + " ",
            "thread_id": thread_id
        }
        yield f"data: {json.dumps(event_data)}\n\n"
        await asyncio.sleep(0.03)  # 30ms delay

    # Completion event
    event_data = {
        "type": "complete",
        "content": full_response,
        "thread_id": thread_id
    }
    yield f"data: {json.dumps(event_data)}\n\n"
```

**Note** : LangGraph ne supporte pas encore le streaming token-by-token natif, d'où la simulation.

---

## 📤 Upload de Documents

### Formats Supportés

- **PDF** : `PyPDFLoader`
- **TXT/MD** : `TextLoader`
- **DOCX** : `UnstructuredWordDocumentLoader`
- **IPYNB** : `NotebookLoader` (avec outputs)

### Text Splitting

```python
RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Chunks plus gros pour meilleur contexte
    chunk_overlap=400,    # 20% overlap
    separators=["\n\n", "\n", " ", ""]
)
```

### Ajout Vectorstore

```python
# Add chunks avec métadonnées
vectorstore.add_documents(chunks)

# ChromaDB persiste automatiquement
```

---

## 🧪 Tests

### Test Mémoire

**Fichier** : `test_memory.py`

```bash
python test_memory.py
```

**Tests** :
- ✅ Thread différent → pas de mémoire
- ✅ Même thread → mémoire OK
- ✅ Trimming → garde 10 messages max

### Test API Manuel

```bash
# Lancer serveur
cd backend/api && python main.py

# Test sans mémoire
curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Test avec mémoire
curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "My name is Bob", "thread_id": "test-1"}'

curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my name?", "thread_id": "test-1"}'
# → "Your name is Bob"
```

---

## 🐳 Déploiement

### Variables d'Environnement

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...   # Requis
TAVILY_API_KEY=tvly-...        # Optionnel (non utilisé)
```

### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "7860:80"
    depends_on:
      - backend
```

### Lancement

```bash
docker-compose up --build
```

**Accès** :
- Backend : http://localhost:8000
- Frontend : http://localhost:7860
- Docs API : http://localhost:8000/docs

---

## 📊 Métriques

### Actuelles (Hardcodées)

Dans `/api/query` adapter :
```python
"confidence": 0.90,
"faithfulness": 0.92,
```

### Futures (RAGAS)

Pour de vraies métriques :
- **Faithfulness** : LLM-as-judge hallucination
- **Answer Relevance** : Pertinence question-réponse
- **Context Precision** : Documents utilisés vs récupérés

Voir plan d'évaluation séparé.

---

## 🔐 Sécurité

### API Key

```python
# Jamais en dur !
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not found")
```

### Input Validation

```python
class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None

    @validator('question')
    def validate_question(cls, v):
        if not v or len(v) > 5000:
            raise ValueError("Invalid question length")
        return v.strip()
```

### CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📚 Ressources

- [LangChain RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)

---

## 🚀 Roadmap

### ✅ Implémenté
- RAG Agent avec tool calling
- Mémoire conversationnelle (InMemorySaver)
- Streaming SSE
- Upload documents multi-formats
- Message trimming
- API REST complète

### ⏳ À Venir
- **Évaluation RAGAS** (faithfulness, relevance, precision)
- Agentic RAG avec grading/rewriting
- PostgreSQL checkpointer (persistance DB)
- Hybrid search (dense + sparse)
- Citation tracking
- Token usage tracking réel
- Rate limiting
- Tests automatisés complets
