# Quick Start - Guide pour Collègues

**Guide rapide pour cloner et démarrer le projet RAG Fusion**

---

## 🚀 Démarrage Rapide (5 minutes)

### 1. Cloner le Repository

```bash
# Cloner le repo
git clone https://github.com/gillesturpin/Agentic-RAG.git

# Aller dans le dossier
cd Agentic-RAG
```

### 2. Configuration de l'API Key

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer le fichier .env
nano .env  # ou vim, code, etc.
```

Ajouter votre clé Anthropic :
```
ANTHROPIC_API_KEY=sk-ant-...votre-clé...
```

**Obtenir une clé** : https://console.anthropic.com/

### 3. Démarrage avec Docker (Recommandé)

```bash
# Démarrer tout (backend + frontend)
./start.sh

# Ou en mode développement (frontend avec hot reload)
./start-dev.sh
```

**URLs** :
- Frontend : http://localhost:3000 (ou http://localhost:5173 en mode dev)
- Backend API : http://localhost:8000
- API Docs : http://localhost:8000/docs

### 4. Démarrage Manuel (Alternative)

**Backend** :
```bash
# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt

# Lancer l'API
cd backend/api
python main.py
```

**Frontend** (dans un autre terminal) :
```bash
cd frontend
npm install
npm run dev
```

---

## 📚 Utilisation

### Upload de Documents

1. Ouvrir http://localhost:3000 (ou 5173)
2. Cliquer "Upload Document"
3. Sélectionner un PDF, DOCX, TXT, MD, ou IPYNB
4. Attendre la confirmation

### Poser une Question

```
Exemple : "What are the key learning objectives for Git?"
```

Le système va :
1. Générer 4 variations de la question
2. Récupérer 16 documents (4 par query)
3. Appliquer RRF reranking → Top 8
4. Générer la réponse

---

## 🧪 Lancer l'Évaluation RAGAS

```bash
# Activer l'environnement
source venv/bin/activate

# Évaluation complète (10 questions)
cd backend/scripts
python run_evaluation.py

# Évaluation rapide (2 questions)
python run_evaluation.py --limit 2
```

**Résultat attendu** :
- Context Precision : ~99.99%
- Answer Similarity : ~82%
- Overall Score : ~87.4% (Grade A)

---

## 📖 Documentation

**Documents essentiels** :
- `README.md` : Vue d'ensemble
- `docs/ARCHITECTURE.md` : Architecture détaillée + explication RRF
- `docs/DOCUMENTATION_TECHNIQUE.md` : Métriques et évaluation
- `docs/RESUME_PRESENTATION.md` : Guide de présentation orale

**Code principal** :
- `backend/rags/rag_fusion.py` : Implémentation RAG Fusion (~180 lignes)
- `backend/api/main.py` : API FastAPI
- `backend/scripts/run_evaluation.py` : Évaluation RAGAS

---

## 🎯 Points Clés du Projet

### Architecture Simplifiée

**RAG Fusion = Multi-Query + RRF Reranking**

```
1 question → 4 query variations
          ↓
4 queries × 4 docs = 16 documents
          ↓
RRF reranking → Top 8 documents
          ↓
Génération de réponse
```

### Performance

| Métrique | Score | Signification |
|----------|-------|---------------|
| Context Precision | 99.99% | Retrieval quasi-parfait |
| Answer Similarity | 82% | Réponses très proches de la vérité |
| Overall Score | 87.4% | Grade A |
| Pass Rate | 90% | 9/10 questions réussies |

### Gains vs Architecture Agentique

| Aspect | Avant | Après | Gain |
|--------|-------|-------|------|
| API calls | 3 | 2 | -33% |
| Latence | ~3-4s | ~2-3s | -1s |
| Code | ~330 lignes | ~180 lignes | -45% |
| Complexité | LangGraph + Tools | Simple Chain | Simplifié |

---

## 🔧 Configuration Optimale (Déjà Appliquée)

```python
RAGFusion(
    use_rag_fusion=True,      # Multi-query + RRF
    temperature=1.0,           # Diversité des réponses
    k_documents=8              # Top 8 après RRF
)
```

**Tests d'optimisation effectués** :
- ✅ k=8 optimal (vs k=4: 75.9%, vs k=12: 84.1%)
- ✅ RAG Fusion ON (vs OFF: 78.2%, soit -9.2%)
- ✅ Temperature 1.0 (vs 0.0: 80.1%)

---

## 🐛 Troubleshooting

### Erreur "ANTHROPIC_API_KEY not found"
```bash
# Vérifier que .env existe et contient la clé
cat .env | grep ANTHROPIC_API_KEY

# Si absent, ajouter :
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Erreur de port déjà utilisé
```bash
# Backend (port 8000)
lsof -ti:8000 | xargs kill -9

# Frontend (port 3000 ou 5173)
lsof -ti:3000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

### ChromaDB vide (0 documents)
```bash
# Uploader des documents via l'interface
# Ou vérifier que data/chroma_db/ existe
ls -la data/chroma_db/
```

### Dépendances manquantes
```bash
# Réinstaller tout
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Tester Rapidement

**Test 1 : API fonctionne ?**
```bash
curl http://localhost:8000/health
# Devrait retourner : {"status":"healthy"}
```

**Test 2 : RAG Fusion fonctionne ?**
```bash
source venv/bin/activate
python backend/scripts/test_rag_fusion.py
# Devrait afficher : "✅ Test fonctionnel réussi"
```

**Test 3 : Comparer avec ancienne version**
```bash
python backend/scripts/compare_implementations.py
# Compare RAGAgent (old) vs RAGFusion (new)
```

---

## 🎤 Préparer la Présentation

1. **Lire le guide** : `docs/RESUME_PRESENTATION.md`
2. **Démarrer le système** : `./start.sh`
3. **Préparer un document PDF** à uploader (ex: cours Git)
4. **Tester les questions** :
   - Simple : "What is Git?"
   - Complexe : "Explain Git branching and merging workflows"
5. **Regarder les logs backend** pour montrer les 4 queries générées

**Timing suggéré** :
- Introduction : 2 min
- Problème : 2 min
- Solution : 3 min
- **Démo** : 5 min ⭐ (moment clé)
- Résultats : 3 min
- Conclusion : 2 min

---

## 📞 Ressources

**GitHub** : https://github.com/gillesturpin/Agentic-RAG

**Learning LangChain** (source d'inspiration) :
- Ch3 : RAG Fusion pattern (simple chains)
- Ch6 : Agents & tools
- Ch8 : Production & streaming
- Ch10 : Évaluation RAGAS

**Anthropic Claude** : https://console.anthropic.com/
**RAGAS Documentation** : https://docs.ragas.io/

---

## ✅ Checklist Avant Présentation

- [ ] Le système démarre sans erreur (`./start.sh`)
- [ ] API répond sur http://localhost:8000/health
- [ ] Frontend accessible sur http://localhost:3000
- [ ] Document de test uploadé avec succès
- [ ] Question de test retourne une réponse
- [ ] Logs backend visibles (montrent les 4 queries)
- [ ] Score RAGAS vérifié (~87.4%)
- [ ] Slides préparées (voir RESUME_PRESENTATION.md)
- [ ] Timing répété (15-20 min max)

---

**Bonne chance pour la présentation ! 🚀**

Si problème : vérifier les logs, relancer Docker, ou lire DOCUMENTATION_TECHNIQUE.md
