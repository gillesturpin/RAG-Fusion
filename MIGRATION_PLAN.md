# Plan de Migration vers LangChain 1.0 + RAGAS 0.3

## Contexte

Pour utiliser RAGAS 0.3+ avec support Anthropic/Claude (sans dépendance OpenAI obligatoire), nous devons migrer vers LangChain 1.0+.

RAGAS 0.3.9 requiert `langchain-core>=1.0`, mais les versions actuelles du projet (0.3.x) créent des conflits de dépendances.

## Packages à mettre à jour avec langchain-core 1.0+

Voici le mapping complet des packages nécessitant une mise à jour :

| Package | Version actuelle | Version cible | Raison |
|---------|------------------|---------------|---------|
| `langchain-core` | 0.3.17 | **1.0.5** | Requis par RAGAS 0.3+ |
| `langchain` | 0.3.7 | **1.0.7** | Dépend de langchain-core 1.0+ |
| `langchain-community` | 0.3.7 | **1.0.4** | Dépend de langchain-core 1.0+ |
| `langchain-anthropic` | 0.2.4 | **1.0.4** | Compatible avec langchain-core 1.0+ |
| `langchain-huggingface` | 0.1.2 | **1.0.1** | Compatible avec langchain-core 1.0+ |
| `langchain-text-splitters` | 0.3.2 | **1.0.0** | Compatible avec langchain-core 1.0+ |
| `langgraph` | 0.2.45 | **0.2.62** | Version stable récente |
| `ragas` | 0.1.9 → 0.2.0 | **0.3.9** | Support Anthropic natif |

## Fichiers concernés

### 1. requirements.txt ✅ (MODIFIÉ)

```txt
# LangChain - UPGRADED TO 1.0+ for RAGAS compatibility
langchain==1.0.7
langchain-community==1.0.4
langchain-core==1.0.5
langgraph==0.2.62
langchain-anthropic==1.0.4
langchain-text-splitters==1.0.0
langchain-huggingface==1.0.1

# Evaluation - RAGAS 0.3+ with Claude/Anthropic support
ragas==0.3.9
```

### 2. backend/Dockerfile

Le Dockerfile construit l'image backend avec `pip install -r requirements.txt`.
Lors du prochain build Docker, les nouvelles versions seront installées.

### 3. backend/rags/evaluator.py ⚠️ (À VÉRIFIER)

Le code d'évaluation utilise RAGAS. Avec la version 0.3+, la configuration des LLMs pourrait changer légèrement.

**Action requise** : Tester après migration pour vérifier compatibilité API.

### 4. backend/scripts/run_certification.py ⚠️ (À VÉRIFIER)

Script d'évaluation utilisant l'evaluator.

**Action requise** : Test complet après migration.

## Procédure de Migration

### Étape 1 : Backup de l'environnement actuel

```bash
# Sauvegarder l'état actuel du backend Docker
docker compose exec backend pip freeze > backend/requirements.backup.txt
```

### Étape 2 : Rebuild du backend avec nouvelles dépendances

```bash
# Stop les conteneurs
docker compose down

# Rebuild l'image backend (forcé, sans cache)
docker compose build --no-cache backend

# Redémarrer
docker compose up -d backend
```

### Étape 3 : Vérifier les imports LangChain

```bash
# Tester les imports critiques
docker compose exec backend python -c "
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy, answer_correctness
print('✅ All imports successful')
"
```

### Étape 4 : Tester le RAG Agent

```bash
cd backend
docker compose exec backend python -c "
import sys
sys.path.append('/app')
from rags.rag_agent import RAGAgent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma(persist_directory='/data/chroma_db', embedding_function=embeddings)
checkpointer = InMemorySaver()
agent = RAGAgent(vectorstore, checkpointer=checkpointer)

result = agent.invoke('What are Python classes?', thread_id='test-migration')
print(f'✅ RAG Agent OK: {result[\"answer\"][:100]}...')
"
```

### Étape 5 : Tester l'évaluateur RAGAS

```bash
cd backend/scripts
docker compose exec backend python -c "
import sys
sys.path.append('/app')
from scripts.run_certification import run_certification

# Test sur 1 seule question
exit_code = run_certification(limit=1)
print(f'✅ Evaluation test completed with exit code: {exit_code}')
"
```

### Étape 6 : Évaluation complète

```bash
# Lancer l'évaluation complète sur les 10 questions
docker compose exec backend python scripts/run_certification.py
```

## Risques et Mitigations

### Risque 1 : Breaking changes API LangChain 1.0

**Probabilité** : FAIBLE (LangChain 1.0 annonce "no breaking changes")

**Mitigation** :
- LangChain 1.0 suit semantic versioning strict
- Migration guide officielle : https://python.langchain.com/docs/versions/v0_3/

### Risque 2 : RAGAS 0.3 API incompatible avec code actuel

**Probabilité** : MOYENNE

**Mitigation** :
- Vérifier documentation RAGAS 0.3 : https://docs.ragas.io/en/latest/
- Ajuster evaluator.py si nécessaire (configuration LLM)

### Risque 3 : Problèmes avec ChromaDB/Embeddings

**Probabilité** : FAIBLE

**Mitigation** :
- Les embeddings sont déjà générés dans ChromaDB
- langchain-huggingface 1.0.1 maintient compatibilité avec sentence-transformers

## Rollback si nécessaire

Si la migration échoue, revenir à l'état précédent :

```bash
# Restaurer requirements.txt original
git restore requirements.txt

# Rebuild avec anciennes versions
docker compose build --no-cache backend
docker compose up -d backend
```

## Validation Post-Migration

Checklist complète :

- [ ] Backend démarre sans erreur
- [ ] Frontend peut communiquer avec backend
- [ ] Upload de documents fonctionne
- [ ] RAG Agent répond aux questions
- [ ] RAGAS Evaluator peut évaluer 1 question
- [ ] Évaluation complète des 10 questions réussit
- [ ] Rapport de certification généré

## Notes

- La migration est **nécessaire** pour utiliser RAGAS avec Claude/Anthropic
- LangChain 1.0 est stable (released Nov 2024)
- Tous les packages cibles sont disponibles sur PyPI
- Temps estimé de migration : **30-60 minutes**
