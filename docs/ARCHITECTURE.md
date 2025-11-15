# 🏗️ Architecture RAG Agent

## Vue d'ensemble

Ce projet implémente un **RAG Agent** basé sur le tutoriel officiel LangChain ([RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/)).

## 📊 Architecture Globale

```
┌─────────────────────────────────────────┐
│            User Question                 │
└─────────────────────────────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │    RAG Agent     │
          │  (with Memory)   │
          └──────────────────┘
                    │
                    ▼
             [Intelligent]
             [k=4 docs]
            [Conversational]
```

---

## 🔧 RAG Agent - Architecture Détaillée

### Flow Diagram
```
User Question
      │
      ▼
┌─────────────────────────┐
│   LangGraph Workflow    │
│   (with InMemorySaver)  │
└─────────────────────────┘
      │
      ▼
┌─────────────────┐
│   LLM Router    │ ← Décide si retrieval nécessaire
│  (Claude 4.5)   │    via tool calling
└─────────────────┘
      │
      ├─── Tool Call ──→ ┌──────────────┐
      │                  │   Retriever   │
      │                  │    (k=4)      │
      │                  └──────────────┘
      │                         │
      │                         ▼
      │                  ┌──────────────┐
      │                  │   Documents  │
      │                  │  + Metadata  │
      │                  └──────────────┘
      │                         │
      └─── Direct ──────────────┤
                                ▼
                         ┌──────────────┐
                         │   Generate   │
                         │    Answer    │
                         └──────────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Save State  │
                         │  (Memory)    │
                         └──────────────┘
```

### Caractéristiques Principales

- **Framework** : LangGraph `StateGraph`
- **LLM** : Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Retrieval** : Optionnel (LLM décide via tool calling)
- **Documents** : k=4 (similarity search)
- **Mémoire** : InMemorySaver (conversation persistante par thread_id)
- **Streaming** : Support SSE (Server-Sent Events)
- **Flow** : Question → Route → (Retrieve?) → Generate → Save

### Implémentation Core

**Fichier** : `backend/rags/rag_agent.py`

```python
class RAGAgent:
    def __init__(self, vectorstore, checkpointer=None):
        # Use InMemorySaver for conversation memory
        self.checkpointer = checkpointer or InMemorySaver()

        # Create retrieve tool
        @tool
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vectorstore.similarity_search(query, k=4)
            # Format documents with metadata
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools([retrieve])

        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode([retrieve]))

        # Conditional routing
        def should_continue(state):
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"  # LLM wants to retrieve
            return END  # LLM answers directly

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")  # Loop back after retrieval

        # Compile with memory
        return workflow.compile(checkpointer=self.checkpointer)

    def invoke(self, question: str, thread_id: str = None):
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        return {
            "answer": result["messages"][-1].content,
            "messages": result["messages"],
            "used_retrieval": any(msg.type == "tool" for msg in result["messages"]),
            "thread_id": thread_id
        }
```

### Nodes du Graph

#### 1. **agent** (LLM Router)
- Reçoit la question + historique (mémoire)
- Décide : retrieval nécessaire ou non ?
- Retourne : réponse directe OU appel au tool `retrieve`

#### 2. **tools** (Retriever)
- Exécute `similarity_search(k=4)` sur ChromaDB
- Formate les documents avec métadonnées
- Retourne le contexte au LLM

#### 3. **Conditional Edge**
- Si `tool_calls` présent → va vers `tools`
- Sinon → END (réponse finale)

### Mémoire Conversationnelle

**InMemorySaver** stocke l'historique par `thread_id` :

```python
# Premier message (thread-1)
agent.invoke("My name is Alice", thread_id="thread-1")

# Deuxième message (même thread)
agent.invoke("What is my name?", thread_id="thread-1")
# Réponse : "Your name is Alice" ✅

# Nouveau thread
agent.invoke("What is my name?", thread_id="thread-2")
# Réponse : "I don't know" (pas de mémoire) ✅
```

### Message Trimming

Pour éviter de dépasser la limite de contexte :

```python
def trim_messages(messages):
    """Keep only the last 10 messages to fit context window."""
    if len(messages) <= 10:
        return messages
    # Keep first (system) and last 9 messages
    return [messages[0]] + messages[-9:]
```

### Prompt System

```python
SystemMessage(
    "You have access to a tool that retrieves context from documents. "
    "Use the tool to help answer user queries. "
    "IMPORTANT: Provide COMPLETE and COMPREHENSIVE answers with ALL details. "
    "Do not omit any information. Use proper Markdown formatting."
)
```

### Cas d'Usage

✅ **Optimal pour** :
- Questions nécessitant contexte documentaire
- Conversations multi-tours
- Applications nécessitant mémoire
- Chatbots conversationnels
- Questions mixtes (in/out context)

❌ **Moins optimal pour** :
- Questions ultra-simples (overhead du graph)
- Batch processing sans mémoire
- Cas nécessitant grading strict des documents

### Output Format

```json
{
  "answer": "La réponse générée avec contexte",
  "messages": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Tool call"},
    {"role": "tool", "content": "Documents..."},
    {"role": "assistant", "content": "Réponse finale"}
  ],
  "used_retrieval": true,
  "thread_id": "thread-abc123"
}
```

---

## 📈 Métriques de Performance

| Métrique | Valeur Typique |
|----------|----------------|
| **Latence moyenne** | 2-5s |
| **Appels LLM par requête** | 1-2 |
| **Documents récupérés** | k=4 |
| **Mémoire max** | 10 messages |
| **Coût estimé par requête** | ~$0.001 |

### Breakdown Latence

- **Sans retrieval** : ~1-2s (réponse directe)
- **Avec retrieval** : ~3-5s (similarity search + génération)
- **Conversation** : +0.5s (chargement historique)

---

## 🔧 Configuration

### Variables d'Environnement

```bash
# Requis
ANTHROPIC_API_KEY=sk-ant-...  # Claude Sonnet 4.5

# Optionnel
TAVILY_API_KEY=tvly-...  # Web search (non utilisé actuellement)
```

### Paramètres Ajustables

**Dans `rag_agent.py` :**

```python
# LLM Configuration
model = "claude-sonnet-4-5-20250929"
model_provider = "anthropic"

# Retrieval
k = 4  # Nombre de documents à récupérer

# Memory
max_messages = 10  # Historique conservé par conversation

# Embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

**Dans `api/main.py` :**

```python
# Text Splitting (upload)
chunk_size = 2000
chunk_overlap = 400

# Streaming
word_delay = 0.03  # 30ms entre chaque mot (effet visuel)
```

---

## 📚 Ressources

### Documentation Officielle
- [LangChain RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/) - Base de ce projet
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Framework utilisé
- [Anthropic Claude API](https://docs.anthropic.com/) - LLM provider

### Fichiers du Projet
- `backend/rags/rag_agent.py` - Implémentation core
- `backend/api/main.py` - FastAPI endpoints
- `test_memory.py` - Tests de la mémoire conversationnelle

---

## 🧪 Tests

### Test de la Mémoire

```bash
python test_memory.py
```

**Résultats attendus** :
- ✅ Thread différent → pas de mémoire
- ✅ Même thread → mémoire fonctionnelle
- ✅ Trimming → garde 10 derniers messages

### Test de l'API

```bash
# Lancer le serveur
cd backend/api && python main.py

# Tester (autre terminal)
curl -X POST http://localhost:8000/api/rag_agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "thread_id": "test-123"}'
```

---

## 🚀 Roadmap

### Actuellement Implémenté
- ✅ RAG Agent avec tool calling
- ✅ Mémoire conversationnelle (InMemorySaver)
- ✅ Streaming SSE
- ✅ Upload documents (PDF/TXT/MD/DOCX)
- ✅ Message trimming
- ✅ ChromaDB vectorstore

### Améliorations Possibles
- ⏳ Agentic RAG avec grading (comme dans tutoriel avancé)
- ⏳ PostgreSQL checkpointer (persistance DB)
- ⏳ Hybrid search (dense + sparse)
- ⏳ Citation tracking
- ⏳ Token usage tracking réel
- ⏳ Métriques d'évaluation (RAGAS)