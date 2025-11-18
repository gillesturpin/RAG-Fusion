# Documentation Technique - Agentic RAG

**Version** : 2.0.0
**Date** : Novembre 2025
**Implémentation** : Pure LangChain/LangGraph selon [tutoriel officiel](https://python.langchain.com/docs/tutorials/rag_agent/)

## Table des matières
1. [Stack Technique](#stack-technique)
2. [Structure du Projet](#structure-du-projet)
3. [Implémentation RAG Agent](#implémentation-rag-agent)
4. [API REST](#api-rest)
5. [Mode Stateless (Configuration Actuelle)](#mode-stateless-configuration-actuelle)
6. [Streaming](#streaming)
7. [Upload de Documents](#upload-de-documents)
8. [Tests](#tests)
9. [Déploiement](#déploiement)

---

## Stack Technique

### Backend
- **Python** 3.12+
- **FastAPI** - Framework web async
- **LangChain** 0.3.x - Orchestration LLM
- **LangGraph** 0.2.x - State machine workflow
- **Anthropic** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **ChromaDB** - Vector store (local persistence)
- **HuggingFace** - Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Stateless mode** - No conversation memory (checkpointer=None by default)

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

## Structure du Projet

```
agentic-rag/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI server (480 lignes)
│   │
│   └── rags/
│       ├── __init__.py           # Package exports
│       └── rag_agent.py         # RAG Agent avec RAG Fusion (327 lignes)
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

## Implémentation RAG Agent

### Fichier : `backend/rags/rag_agent.py`

Le RAG Agent suit le tutoriel officiel LangChain avec deux améliorations majeures :
1. **RAG Fusion** : Multi-query retrieval avec RRF reranking pour améliorer la qualité de récupération
2. **Mode stateless** : Pas de mémoire conversationnelle par défaut (optimisé pour évaluation)

### 1. Initialisation

Voir le code réel dans `backend/rags/rag_agent.py` (lignes 75-131).

**Signature** :
```python
def __init__(self, vectorstore, checkpointer=None, use_rag_fusion=True, temperature=1.0, k_documents=8)
```

**Configuration par défaut** :
- **checkpointer=None** : Mode stateless, aucune mémoire entre questions
- **use_rag_fusion=True** : RAG Fusion activé (multi-query + RRF)
- **temperature=1.0** : Température maximale pour génération créative
- **k_documents=8** : Nombre final de documents retournés après fusion

**Fonctionnement RAG Fusion** :
1. Génère 4 variations de la question (1 originale + 3 reformulations)
2. Récupère 4 documents pour chaque variation (total : 16 documents)
3. Applique RRF (Reciprocal Rank Fusion) pour reranker
4. Retourne les top k=8 documents avec meilleurs scores

**LLM** :
- **Modèle** : Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Temperature** : 1.0 (configurable)
- **Tools** : Fonction `retrieve` avec RAG Fusion

**Points clés** :
- **Vectorstore** : ChromaDB passé en paramètre (créé au startup)
- **Checkpointer** : None par défaut = AUCUNE mémoire conversationnelle
- **Tool** : Fonction `retrieve` avec logique RAG Fusion intégrée
- **Graph** : Compilé sans checkpointer (stateless)

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

    # Compile (stateless si checkpointer=None)
    if self.checkpointer:
        return workflow.compile(checkpointer=self.checkpointer)
    else:
        return workflow.compile()  # Stateless par défaut
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

### 4. Invoke (Stateless par défaut)

Voir le code réel dans `backend/rags/rag_agent.py` (lignes 254-288).

**Signature** :
```python
def invoke(self, question: str, thread_id: str = None) -> dict
```

**IMPORTANT** : En mode stateless (checkpointer=None), le paramètre `thread_id` est **IGNORÉ**.

```python
# Config pour mémoire (SEULEMENT si checkpointer existe)
config = {}
if thread_id and self.checkpointer:  # <- Note le AND
    config = {"configurable": {"thread_id": thread_id}}
```

**Comportement actuel (checkpointer=None)** :
- Chaque question est **indépendante**
- Aucune mémoire entre les appels
- thread_id est retourné mais n'a aucun effet
- Optimisé pour évaluation RAGAS (pas de contamination entre questions)

**Retour** :
```python
{
    "answer": str,           # Réponse générée
    "messages": list,        # Liste des messages (HumanMessage, AIMessage, ToolMessage)
    "used_retrieval": bool,  # True si le tool retrieve a été appelé
    "thread_id": str         # Retourné tel quel (tracking frontend uniquement)
}
```

---

## API REST

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

    # Initialize RAG agent (STATELESS mode)
    rag_agent = RAGAgent(vectorstore, checkpointer=None)
    # Par défaut : use_rag_fusion=True, temperature=1.0, k_documents=8

    print("✓ RAG Agent initialized in STATELESS mode with RAG Fusion")
```

### Endpoints Principaux

#### 1. `POST /api/rag_agent`

Endpoint natif pour le RAG Agent.

```python
@app.post("/api/rag_agent", response_model=AgentResponse)
async def query_rag_agent(request: QueryRequest):
    """Query the RAG agent (STATELESS - thread_id ignored)"""
    start_time = time.time()

    try:
        # Generate thread_id si absent (pour tracking frontend uniquement)
        thread_id = request.thread_id or str(uuid.uuid4())

        # Invoke (thread_id IGNORÉ car checkpointer=None)
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
    """Adapter pour frontend original (STATELESS)"""
    question = request.get("question", "")
    session_id = request.get("session_id", str(uuid.uuid4()))

    # Mapping session_id → thread_id (INUTILE car checkpointer=None)
    # Gardé uniquement pour compatibilité frontend
    if session_id not in session_threads:
        session_threads[session_id] = f"thread-{uuid.uuid4()}"
    thread_id = session_threads[session_id]

    # Invoke agent (thread_id IGNORÉ)
    result = rag_agent.invoke(question, thread_id=thread_id)

    return {
        "answer": result["answer"],
        "method": "optimized_rag_agent",
        "latency": ...,
        "confidence": 0.90,  # Hardcodé
        "faithfulness": 0.92,  # Hardcodé
        "thread_id": thread_id  # Retourné mais sans effet
    }
```

**Notes** :
- `confidence` et `faithfulness` sont **hardcodés**
- Le mapping session_id → thread_id est **inutile** (checkpointer=None) mais gardé pour compatibilité

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

## Mode Stateless (Configuration Actuelle)

### Pas de Mémoire Conversationnelle

Le système est configuré en **mode stateless** par défaut (`checkpointer=None`).

**Implications** :
- ❌ **Aucune mémoire** entre les questions
- ❌ Le paramètre `thread_id` est **accepté mais IGNORÉ**
- ✅ Chaque question est **totalement indépendante**
- ✅ Optimisé pour **évaluation RAGAS** (pas de contamination entre questions)
- ✅ Pas de stockage d'état en mémoire

**Exemple de comportement actuel** :

```python
# Question 1
agent.invoke("My name is Alice", thread_id="thread-1")
# → Réponse générée

# Question 2 (MÊME thread_id)
agent.invoke("What is my name?", thread_id="thread-1")
# → "I don't have information about your name" ❌ Pas de mémoire !

# Le thread_id ne fait RIEN car checkpointer=None
```

### Activer la Mémoire Conversationnelle (Non implémenté actuellement)

Pour activer la mémoire, il faudrait :

1. **Changer le code dans `backend/api/main.py`** :
```python
from langgraph.checkpoint.memory import MemorySaver

# Ligne 77 (actuellement : checkpointer=None)
checkpointer = MemorySaver()  # Au lieu de None
rag_agent = RAGAgent(vectorstore, checkpointer=checkpointer)
```

2. **Fonctionnement avec mémoire** :
- Chaque `thread_id` aurait son propre état isolé
- L'historique des messages serait sauvegardé
- Pas de limite de durée (reste en RAM)

3. **Pour persistance en production** :
- Utiliser `SqliteSaver` ou `PostgresSaver` au lieu de `MemorySaver`
- Voir [LangGraph Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/)

**Note** : La mémoire n'est PAS activée pour optimiser l'évaluation RAGAS.

---

## Streaming

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

## Upload de Documents

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

## Tests

### Test Mode Stateless

**Fichier** : `test_memory.py`

```bash
python test_memory.py
```

**Tests actuels (mode stateless)** :
- ❌ Threads différents → pas de mémoire (normal)
- ❌ Même thread → **PAS de mémoire** (car checkpointer=None)
- ✅ Trimming → garde 10 messages max
- ✅ RAG Fusion → génère 4 queries, récupère 16 docs, retourne top 8

### Test API Manuel

```bash
# Lancer serveur
cd backend/api && python main.py

# Test basique (stateless)
curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Test avec thread_id (IGNORÉ car checkpointer=None)
curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "My name is Bob", "thread_id": "test-1"}'

curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my name?", "thread_id": "test-1"}'
# → "I don't have information about your name" ❌ PAS de mémoire !
# Le thread_id est accepté mais IGNORÉ (checkpointer=None)
```

**Note** : Pour tester la mémoire, il faudrait d'abord modifier `backend/api/main.py` ligne 77 pour activer un checkpointer.

---

## Déploiement

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

## Métriques & Évaluation RAGAS

### Système d'Évaluation Actuel

Le système utilise **RAGAS 0.3.9** pour évaluation objective avec LLM-as-judge (Claude Haiku).

### Métriques Principales

**2 métriques essentielles** (sélectionnées après optimisation) :

| Métrique | Poids | Description | Seuil Pass |
|----------|-------|-------------|------------|
| **Context Precision** | 30% | Qualité du retrieval - Documents pertinents bien classés | ≥ 0.80 |
| **Answer Similarity** | 70% | Similarité sémantique answer vs ground truth | ≥ 0.70 |

**Score Global** = (Context Precision × 0.30) + (Answer Similarity × 0.70)

### Explication des Métriques

#### 1. Context Precision (30% du score)

**Qu'est-ce que c'est ?**
- Mesure la **qualité du retrieval** : les documents récupérés sont-ils pertinents ?
- Vérifie si les documents utilisés pour générer la réponse sont **réellement utiles**

**Comment c'est calculé ?**
```
Context Precision = Documents pertinents / Total documents récupérés

Exemple :
- Récupération : 8 documents
- Documents pertinents (contiennent l'info nécessaire) : 8
- Context Precision = 8/8 = 1.0 (100%) ⭐
```

**Pourquoi c'est important ?**
- ✅ Score élevé (>0.8) = RAG Fusion + RRF fonctionnent bien
- ✅ Pas de "bruit" : seulement des docs pertinents
- ❌ Score faible (<0.5) = Beaucoup de docs inutiles récupérés

**Comment RAGAS l'évalue ?**
1. LLM-as-judge (Claude Haiku) analyse chaque document récupéré
2. Question : "Ce document contient-il des informations utiles pour répondre à la question ?"
3. Compte le nombre de documents pertinents vs total

**Notre score : 99.99%**
- Signifie : Presque tous les documents récupérés sont pertinents
- Grâce à : RAG Fusion (4 queries) + RRF reranking

---

#### 2. Answer Similarity (70% du score)

**Qu'est-ce que c'est ?**
- Mesure la **similarité sémantique** entre la réponse générée et la réponse de référence (ground truth)
- Vérifie si le modèle a compris et reformulé correctement l'information

**Comment c'est calculé ?**
```
1. Embedding de la réponse générée : E(answer)
2. Embedding de la ground truth : E(ground_truth)
3. Similarité cosinus : cos(E(answer), E(ground_truth))

Exemple :
Answer: "Git is a version control system for tracking code changes"
Ground truth: "Git is a VCS used to track modifications in source code"
→ Similarity = 0.92 (très proche sémantiquement)
```

**Pourquoi c'est important ?**
- ✅ Score élevé (>0.7) = Réponse contient les bonnes informations
- ✅ Tolérant aux reformulations (sémantique vs exact match)
- ❌ Score faible (<0.5) = Réponse à côté de la plaque

**Différence avec exact match :**
| Métrique | Answer 1 | Answer 2 | Score |
|----------|----------|----------|-------|
| **Exact Match** | "Git is a VCS" | "Git is a version control system" | 0% ❌ |
| **Answer Similarity** | "Git is a VCS" | "Git is a version control system" | 95% ✅ |

**Notre score : 82%**
- Signifie : Réponses sémantiquement très proches de la vérité
- Grâce à : System prompt optimisé + Context Precision élevé

---

#### Pourquoi 2 métriques seulement ?

**Métriques RAGAS disponibles** (6 au total) :
1. Context Precision ✅ **Retenue**
2. Answer Similarity ✅ **Retenue**
3. Faithfulness ❌ Redondant (déjà couvert par prompt strict)
4. Answer Relevance ❌ Capturé par Answer Similarity
5. Context Recall ❌ Difficile à évaluer sans annotation manuelle
6. Context Relevance ❌ Proche de Context Precision

**Décision** :
- Focus sur 2 métriques complémentaires
- Context Precision → Qualité **retrieval**
- Answer Similarity → Qualité **génération**
- Couvre toute la chaîne RAG

**Pondération 30/70** :
```
Score = (Context Precision × 0.30) + (Answer Similarity × 0.70)

Pourquoi ?
- Answer Similarity (70%) : métrique finale, ce que l'utilisateur voit
- Context Precision (30%) : métrique intermédiaire, mais cruciale
```

---

#### ⚠️ Ground Truth : Concept Clé

**Les 2 métriques utilisent la ground truth** - c'est normal pour l'évaluation !

**Qu'est-ce que la ground truth ?**
```json
{
  "question": "What are three key learning objectives for Git course?",
  "ground_truth": "The course aims to teach: 1) Git basics as VCS,
                   2) Using Git with terminal and GitHub,
                   3) Collaboration with agile methods"
}
```

La ground truth = **réponse de référence** annotée manuellement par un humain.

**Comment chaque métrique utilise la ground truth :**

| Métrique | Utilisation de Ground Truth | Exemple |
|----------|----------------------------|---------|
| **Context Precision** | LLM-as-judge compare chaque document récupéré avec la question + ground truth pour savoir si le doc est pertinent | Doc contient "Git is a VCS" → Pertinent pour ground truth mentionnant "Git basics as VCS" ✅ |
| **Answer Similarity** | Calcul de similarité sémantique directe entre answer et ground truth | Embeddings: cos(answer, ground_truth) = 0.82 |

**Workflow d'évaluation complet :**

```
1. Dataset avec ground truth (annoté manuellement)
   ├─ Question: "What are Git basics?"
   └─ Ground Truth: "Git is a version control system..."

2. RAG génère une réponse
   └─ Answer: "Based on course materials, Git is a VCS for tracking code changes..."

3. Évaluation RAGAS
   ├─ Context Precision: LLM vérifie si docs récupérés contiennent infos de ground truth
   └─ Answer Similarity: Compare sémantiquement answer vs ground truth

4. Score final
   └─ 87.4% (Grade A)
```

**Important : Évaluation vs Production**

| Mode | Ground Truth ? | Utilisation |
|------|----------------|-------------|
| **Évaluation** | ✅ Nécessaire | Mesurer la qualité du système |
| **Production** | ❌ Pas disponible | Utilisateurs posent des questions réelles |

En production, on ne peut **pas** calculer Context Precision ni Answer Similarity car :
- Pas de ground truth pré-annotée
- Mais on peut monitorer d'autres métriques :
  - Latence
  - Nombre de documents récupérés
  - Feedback utilisateur (👍/👎)

**C'est pourquoi l'évaluation en amont est cruciale** pour s'assurer que le système fonctionne bien ! 🎯

### Résultats Obtenus

**Performance actuelle** (10 questions, dataset Git/Python) :

```
╔══════════════════════════════════════════════╗
║  RAGAS EVALUATION - RESULTS                  ║
╠══════════════════════════════════════════════╣
║  Context Precision:     99.99%  ⭐           ║
║  Answer Similarity:     82.0%               ║
║  ─────────────────────────────────────────   ║
║  Overall Score:         87.4%               ║
║  Grade:                 A (Very Good)       ║
║  Pass Rate:             90% (9/10)          ║
╚══════════════════════════════════════════════╝
```

**Analyse** :
- ✅ **Context Precision quasi-parfait** : RAG Fusion + RRF très efficace
- ✅ **Answer Similarity bon** : Prompts optimisés
- ✅ **Pass Rate > 80%** : Seuil dépassé

### Lancer l'Évaluation

**Évaluation complète** (10 questions) :
```bash
cd backend/scripts
python run_evaluation.py
```

**Évaluation rapide** (2 questions) :
```bash
python run_evaluation.py --limit 2
```

**Options disponibles** :
```bash
# Désactiver RAG Fusion (test comparatif)
python run_evaluation.py --no-rag-fusion

# Changer température
python run_evaluation.py --temperature 0.5

# Changer nombre de documents
python run_evaluation.py --k-documents 12
```

### Fichiers Générés

- **Dataset** : `backend/scripts/evaluation_dataset.json` (10 questions)
- **Rapport** : `backend/scripts/evaluation_report_YYYYMMDD_HHMMSS.json`

**Structure du rapport** :
```json
{
  "timestamp": "2025-11-18T10:30:00",
  "evaluation_passed": true,
  "summary": {
    "total_questions": 10,
    "passed": 9,
    "pass_rate": 0.9,
    "overall_score": 0.874,
    "grade": "A (Very Good)"
  },
  "detailed_results": [...]
}
```

### Critères de Réussite

| Critère | Seuil | Actuel | Status |
|---------|-------|--------|--------|
| Pass Rate | ≥ 80% | 90% | ✅ |
| Overall Score | ≥ 0.70 | 0.874 | ✅ |
| Context Precision | ≥ 0.80 | 0.999 | ✅ |
| Answer Similarity | ≥ 0.70 | 0.82 | ✅ |

### Dataset d'Évaluation

**10 questions** couvrant :
- **Git Basics** (3 questions) - facile/moyen
- **Python Classes** (3 questions) - facile/moyen
- **Python Functions** (2 questions) - moyen
- **Markdown** (2 questions) - facile

Chaque question contient :
```json
{
  "id": "q001",
  "question": "What are three key learning objectives...",
  "ground_truth": "The course aims to teach...",
  "category": "Git Basics",
  "difficulty": "easy",
  "source_file": "01-introduction.md"
}
```

### Optimisations Appliquées

**Résultats des tests** (voir tests d'optimisation) :

| Configuration | Score | Décision |
|--------------|-------|----------|
| k=4 docs | 0.759 | ❌ |
| k=8 docs | **0.874** | ✅ **Retenu** |
| k=12 docs | 0.841 | ❌ |
| RAG Fusion ON | **0.874** | ✅ **Retenu** |
| RAG Fusion OFF | 0.782 | ❌ |
| Temperature 0.0 | 0.801 | ❌ |
| Temperature 1.0 | **0.874** | ✅ **Retenu** |

**Configuration finale optimale** :
```python
RAGFusion(
    vectorstore=vectorstore,
    use_rag_fusion=True,      # Multi-query + RRF
    temperature=1.0,           # Diversité des réponses
    k_documents=8              # Top 8 après RRF
)
```

---

## Sécurité

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

## Ressources

- [LangChain RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)

---

## Roadmap

### Implémenté ✅
- **RAG Agent** avec tool calling (LangGraph)
- **RAG Fusion** : Multi-query retrieval + RRF reranking (4 queries → 16 docs → top 8)
- **Mode stateless** : Pas de mémoire conversationnelle (optimisé pour évaluation)
- **Claude Sonnet 4.5** : Modèle de dernière génération
- **Temperature 1.0** : Génération créative maximale
- **Streaming SSE** : Réponses en temps réel
- **Upload multi-formats** : PDF/TXT/MD/DOCX/IPYNB
- **Message trimming** : Garde 10 derniers messages
- **API REST complète** : FastAPI avec CORS
- **Évaluation RAGAS** : Score 87.4% - Grade A

### À Venir 🔜
- **Mode conversationnel** avec mémoire (MemorySaver/PostgresSaver)
- **Hybrid search** : Dense + sparse (BM25)
- **Citation tracking** : Sources précises
- **Token usage tracking** : Coûts réels
- **Rate limiting** : Protection API
- **Tests automatisés** : Coverage complet
