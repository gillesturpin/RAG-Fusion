#!/bin/bash
# Script d'arrêt pour Agentic RAG
# Usage: ./stop.sh

echo "🛑 Arrêt d'Agentic RAG..."
echo ""

# Arrêter Docker Compose
echo "📦 Arrêt des containers Docker..."
docker-compose down

if [ $? -eq 0 ]; then
    echo "✅ Containers Docker arrêtés"
else
    echo "❌ Erreur lors de l'arrêt de Docker"
    exit 1
fi

echo ""
echo "✨ Tout est arrêté proprement !"
echo ""
echo "💡 Pour redémarrer, utilise: ./start.sh"
