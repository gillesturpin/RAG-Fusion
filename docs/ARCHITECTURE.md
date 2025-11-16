# Architecture RAG Agent

## Vue d'ensemble

Ce projet implémente un **RAG Agent** optimisé basé sur le tutoriel officiel LangChain ([RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/)).

**Améliorations principales** :
- **RAG Fusion** : Multi-query retrieval + Reciprocal Rank Fusion (RRF) pour un meilleur reranking
- **Mode Stateless** : Pas de mémoire conversationnelle, optimisé pour l'évaluation RAGAS
- **Configuration optimale** : k=8 documents, temperature=1.0, pas de grading
- **Performance** : Score RAGAS 87.4% (Grade A)

## Architecture Globale

```
┌─────────────────────────────────────────┐
│            User Question                 │
└─────────────────────────────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │    RAG Agent     │
          │   (Stateless)    │
          └──────────────────┘
                    │
                    ▼
           [RAG Fusion]
           [k=8 docs]
        [No Memory/Grading]
```

---

## RAG Agent - Architecture Détaillée

### Flow Diagram
```
User Question
      │
      ▼
┌─────────────────────────┐
│   LangGraph Workflow    │
│      (Stateless)        │
└─────────────────────────┘
      │
      ▼
┌─────────────────┐
│   LLM Router    │ ← Décide si retrieval nécessaire
│  (Claude 4.5)   │    via tool calling
└─────────────────┘
      │
      ├─── Tool Call ──→ ┌──────────────────┐
      │                  │   RAG Fusion     │
      │                  │  Multi-query +   │
      │                  │  RRF reranking   │
      │                  └──────────────────┘
      │                         │
      │                         ▼
      │                  ┌──────────────────┐
      │                  │  Top k=8 Docs    │
      │                  │   + Metadata     │
      │                  └──────────────────┘
      │                         │
      └─── Direct ──────────────┤
                                ▼
                         ┌──────────────┐
                         │   Generate   │
                         │    Answer    │
                         └──────────────┘
```

### Caractéristiques Principales

- **Framework** : LangGraph `StateGraph`
- **LLM** : Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Retrieval** : RAG Fusion (multi-query + RRF reranking)
- **Documents** : k=8 (optimisé via tests - meilleur score)
- **Mode** : Stateless (pas de mémoire conversationnelle)
- **Grading** : Désactivé (coûteux sans gain de performance)
- **Streaming** : Support SSE (Server-Sent Events)
- **Flow** : Question → Route → (RAG Fusion?) → Generate

### Implémentation Core

**Fichier** : `backend/rags/rag_agent.py`

**Note**: Pour le code source complet et à jour, consultez directement le fichier `backend/rags/rag_agent.py`.

Points clés de l'implémentation :
- **Stateless mode**: `checkpointer=None` par défaut, pas de mémoire conversationnelle
- **RAG Fusion**: Multi-query retrieval + RRF reranking pour améliorer la pertinence
- **k=8 documents**: Optimisé pour maximiser le score RAGAS
- **Temperature=1.0**: Diversité des réponses
- **LangGraph workflow**: StateGraph avec conditional routing (agent → tools → agent)

### Nodes du Graph

#### 1. **agent** (LLM Router)
- Reçoit la question (mode stateless - pas d'historique entre questions)
- Décide : retrieval nécessaire ou non ?
- Retourne : réponse directe OU appel au tool `retrieve`

#### 2. **tools** (RAG Fusion Retriever)
- Génère 4 requêtes au total (1 question originale + 3 variations)
- Récupère 4 documents pour chaque requête (16 documents au total)
- Applique RRF (Reciprocal Rank Fusion) pour reranker les 16 documents
- Retourne top k=8 documents finaux avec métadonnées au LLM

#### 3. **Conditional Edge**
- Si `tool_calls` présent → va vers `tools`
- Sinon → END (réponse finale)

### Mode Stateless

**Pas de mémoire conversationnelle** : chaque question est traitée indépendamment.

```python
# Première question
agent.invoke("My name is Alice")
# Réponse basée uniquement sur les documents

# Deuxième question (indépendante)
agent.invoke("What is my name?")
# Réponse : "I don't have that information" (pas de mémoire)
```

**Avantage** : Optimisé pour l'évaluation RAGAS et les questions indépendantes.

### Message Trimming

Pour éviter de dépasser la limite de contexte (utilisé pour éviter overflow, pas pour la mémoire conversationnelle) :

```python
def trim_messages(messages):
    """Keep only the last 10 messages to fit context window.
    NOTE: This is for context length management, not conversation memory.
    The system is stateless (checkpointer=None) - no memory between questions.
    """
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

**Optimal pour** :
- Évaluation RAGAS (questions indépendantes)
- Questions nécessitant contexte documentaire
- Recherche multi-angle (RAG Fusion)
- Questions complexes nécessitant plusieurs perspectives
- Batch processing de questions indépendantes

**Moins optimal pour** :
- Conversations multi-tours (pas de mémoire)
- Chatbots conversationnels
- Applications nécessitant contexte de conversation

### Output Format

```json
{
  "answer": "La réponse générée avec contexte",
  "messages": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Tool call"},
    {"role": "tool", "content": "Top 8 documents via RAG Fusion"},
    {"role": "assistant", "content": "Réponse finale"}
  ],
  "used_retrieval": true,
  "num_rewrites": 0
}
```

---

## Métriques de Performance

| Métrique | Valeur |
|----------|--------|
| **Score RAGAS** | 87.4% (Grade A) |
| **Context Precision** | 0.937 |
| **Answer Similarity** | 0.811 |
| **Latence moyenne** | 3-6s |
| **Appels LLM par requête** | 4-5 (multi-query + génération) |
| **Documents récupérés** | k=8 (via RAG Fusion) |
| **Coût estimé par requête** | ~$0.002 |

### Breakdown Latence

- **Sans retrieval** : ~1-2s (réponse directe)
- **Avec RAG Fusion** : ~4-6s (3 queries + RRF + génération)
- **Mode stateless** : Pas de surcoût mémoire

---

## Configuration

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
temperature = 1.0  # Optimisé pour diversité des réponses

# RAG Fusion
use_rag_fusion = True  # Multi-query + RRF reranking
k_documents = 8  # Nombre final de documents (optimisé)

# Mode
checkpointer = None  # Stateless mode (pas de mémoire)

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

## Ressources

### Documentation Officielle
- [LangChain RAG Agent Tutorial](https://python.langchain.com/docs/tutorials/rag_agent/) - Base de ce projet
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Framework utilisé
- [Anthropic Claude API](https://docs.anthropic.com/) - LLM provider

### Fichiers du Projet
- `backend/rags/rag_agent.py` - Implémentation core
- `backend/api/main.py` - FastAPI endpoints
- `test_memory.py` - Tests de la mémoire conversationnelle

---

## Tests

### Test de la Mémoire

```bash
python test_memory.py
```

**Résultats attendus** :
-  Thread différent → pas de mémoire
-  Même thread → mémoire fonctionnelle
-  Trimming → garde 10 derniers messages

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

## Roadmap

### Actuellement Implémenté
- RAG Agent avec tool calling
- RAG Fusion (multi-query + RRF reranking)
- Mode stateless (optimisé pour évaluation)
- Streaming SSE
- Upload documents (PDF/TXT/MD/DOCX/IPYNB)
- Message trimming
- ChromaDB vectorstore
- Évaluation RAGAS complétée (Score 87.4% - Grade A)

### Améliorations Possibles
- Mode conversationnel avec mémoire (checkpointer)
- PostgreSQL checkpointer (persistance DB)
- Hybrid search (dense + sparse)
- Citation tracking
- Token usage tracking réel
- Document grading (non nécessaire - coûteux sans gain)