#!/bin/bash
# Script de démarrage DÉVELOPPEMENT pour Agentic RAG
# Backend Docker + Frontend en mode dev (hot reload)

echo "Starting Démarrage d'Agentic RAG (Development Mode)"
echo ""

# Arrêter le frontend Docker s'il tourne
echo "Stopping Arrêt du frontend Docker (si actif)..."
docker compose stop frontend 2>/dev/null

# Démarrer uniquement le backend
echo "Starting Démarrage du backend Docker..."
docker compose up -d backend

if [ $? -eq 0 ]; then
    echo "OK Backend Docker démarré"
else
    echo "ERROR Erreur lors du démarrage du backend"
    exit 1
fi

echo ""
echo "⏳ Attente du démarrage du backend (5 secondes)..."
sleep 5

# Vérifier la santé
echo "Health check Vérification de l'API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "OK Backend est opérationnel"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "WARNING  Backend démarre encore... (patientez quelques secondes)"
fi

echo ""
echo "Starting Démarrage du frontend en mode développement..."
echo ""
echo "=" * 60
echo " URLs disponibles:"
echo "   - Frontend:  http://localhost:5173  (DEV - hot reload)"
echo "   - Backend:   http://localhost:8000"
echo "   - API Docs:  http://localhost:8000/docs"
echo ""
echo "Note Appuyez sur Ctrl+C pour arrêter le frontend"
echo "   (le backend Docker continuera de tourner)"
echo "=" * 60
echo ""

# Vérifier que npm est installé
if ! command -v npm &> /dev/null; then
    echo "ERROR npm n'est pas installé. Installer Node.js d'abord."
    exit 1
fi

# Démarrer le frontend en mode dev
cd frontend && npm run dev
