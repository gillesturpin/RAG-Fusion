#!/bin/bash
# Script de démarrage pour Agentic RAG
# Usage: ./start.sh

echo "🚀 Démarrage d'Agentic RAG..."
echo ""

# Démarrer Docker Compose
echo "📦 Démarrage des containers Docker..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Containers Docker démarrés"
else
    echo "❌ Erreur lors du démarrage de Docker"
    exit 1
fi

echo ""
echo "⏳ Attente du démarrage du backend (5 secondes)..."
sleep 5

echo ""
echo "🎨 Démarrage du frontend..."
echo ""
echo "📍 URLs disponibles:"
echo "   - Frontend: http://localhost:5173"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Appuie sur Ctrl+C pour arrêter le frontend"
echo "   (les containers Docker continueront de tourner)"
echo ""

cd frontend && npm run dev
