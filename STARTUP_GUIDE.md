# 🚀 Guide de Démarrage - Agentic RAG

## 📋 Scripts Disponibles

### **Mode Développement** (Recommandé pour travailler sur le code)
```bash
./start-dev.sh
```
**Caractéristiques** :
- ✅ Backend Docker (stable)
- ✅ Frontend mode dev avec hot reload (modifications instantanées)
- ✅ Frontend sur http://localhost:5173
- ✅ Backend sur http://localhost:8000
- ✅ Parfait pour développer l'interface d'évaluation

**Utilisation** :
- Modifiez les fichiers dans `frontend/src/`
- Les changements apparaissent automatiquement (pas de rebuild)
- Appuyez sur Ctrl+C pour arrêter le frontend (backend continue)

---

### **Mode Production** (Pour démo/certification)
```bash
./start-prod.sh
```
**Caractéristiques** :
- ✅ Backend + Frontend en Docker (optimisé)
- ✅ Version compilée et minifiée (nginx)
- ✅ Frontend sur http://localhost:3000
- ✅ Backend sur http://localhost:8000
- ✅ Performance maximale

**Utilisation** :
- Le script demande si vous voulez rebuilder (y/N)
- Répondez 'y' si vous avez modifié le code frontend
- Tout tourne en containers isolés

---

### **Arrêt du Système**
```bash
./stop.sh
```
Arrête tous les containers Docker proprement.

---

## 🎯 Quel Mode Choisir ?

| Situation | Script à utiliser |
|-----------|------------------|
| **Développer l'évaluation RAGAS** | `./start-dev.sh` |
| **Modifier l'interface frontend** | `./start-dev.sh` |
| **Démo pour la certification** | `./start-prod.sh` |
| **Tests de performance** | `./start-prod.sh` |
| **Upload de documents** | Les deux fonctionnent |

---

## 🔧 Commandes Utiles

### Voir les logs
```bash
# Tous les containers
docker compose logs -f

# Backend uniquement
docker compose logs -f backend

# Frontend uniquement (mode prod)
docker compose logs -f frontend
```

### Statut des containers
```bash
docker compose ps
```

### Rebuild après modifications
```bash
# Backend
docker compose build backend

# Frontend
docker compose build frontend

# Tout
docker compose build
```

### Redémarrer un service
```bash
docker compose restart backend
docker compose restart frontend
```

---

## 📍 URLs de l'Application

| Service | Dev Mode | Prod Mode |
|---------|----------|-----------|
| **Frontend** | http://localhost:5173 | http://localhost:3000 |
| **Backend API** | http://localhost:8000 | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health | http://localhost:8000/health |

---

## ⚠️ Troubleshooting

### Le frontend ne se met pas à jour (mode prod)
```bash
# Rebuilder l'image frontend
docker compose down
docker compose build frontend
docker compose up -d
```

### Port déjà utilisé
```bash
# Vérifier ce qui utilise le port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend prod
lsof -i :5173  # Frontend dev

# Arrêter tous les containers
docker compose down
```

### Backend ne démarre pas
```bash
# Vérifier les logs
docker compose logs backend

# Vérifier la clé API
cat .env | grep ANTHROPIC_API_KEY
```

---

## 💡 Workflow Recommandé pour Certification

1. **Phase développement** (ajout évaluation RAGAS)
   ```bash
   ./start-dev.sh
   # Développer dans frontend/src/
   # Hot reload automatique
   ```

2. **Phase tests** (avant certification)
   ```bash
   ./start-prod.sh
   # Tester la version production
   ```

3. **Jour de certification**
   ```bash
   ./start-prod.sh
   # Version optimisée et stable
   ```

---

## 📚 Next Steps

Après avoir démarré le système :
1. Uploader vos documents via http://localhost:5173 (dev) ou :3000 (prod)
2. Tester quelques requêtes
3. Implémenter l'évaluation RAGAS (voir docs/EVALUATION.md)
