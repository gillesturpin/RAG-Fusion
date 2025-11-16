#!/bin/bash
# Script d'arrêt pour Agentic RAG
# Usage: ./stop.sh

echo "Stopping Arrêt d'Agentic RAG..."
echo ""

# Arrêter Docker Compose
echo "Starting Arrêt des containers Docker..."
docker-compose down

if [ $? -eq 0 ]; then
    echo "OK Containers Docker arrêtés"
else
    echo "ERROR Erreur lors de l'arrêt de Docker"
    exit 1
fi

echo ""
echo "✨ Tout est arrêté proprement !"
echo ""
echo "Note Pour redémarrer, utilise: ./start.sh"
