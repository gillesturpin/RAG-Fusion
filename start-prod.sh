#!/bin/bash
# Script de démarrage PRODUCTION pour Agentic RAG
# Utilise Docker Compose pour backend + frontend (version optimisée)

echo "🚀 Démarrage d'Agentic RAG (Production Mode)"
echo ""

# Demander si rebuild nécessaire
read -p "Rebuild les images Docker? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 Building Docker images..."
    docker compose build
    echo ""
fi

# Démarrer les containers
echo "📦 Démarrage des containers Docker..."
docker compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Containers Docker démarrés"
else
    echo "❌ Erreur lors du démarrage de Docker"
    exit 1
fi

echo ""
echo "⏳ Attente du démarrage du backend (5 secondes)..."
sleep 5

# Vérifier la santé
echo "🏥 Vérification de l'API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend est opérationnel"
else
    echo "⚠️  Backend démarre encore... (patientez quelques secondes)"
fi

echo ""
echo "=" * 60
echo "✅ Agentic RAG démarré en mode PRODUCTION"
echo "=" * 60
echo ""
echo "📍 URLs disponibles:"
echo "   - Frontend:  http://localhost:3000"
echo "   - Backend:   http://localhost:8000"
echo "   - API Docs:  http://localhost:8000/docs"
echo ""
echo "💡 Commandes utiles:"
echo "   docker compose logs -f        # Voir les logs"
echo "   docker compose ps             # Statut des containers"
echo "   ./stop.sh                     # Arrêter tout"
echo ""
