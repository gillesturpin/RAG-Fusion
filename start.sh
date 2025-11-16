#!/bin/bash
# Docker startup script for Agentic RAG
# Bootcamp Evaluation Project

echo "🐳 Starting Agentic RAG with Docker"
echo "===================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from template..."
    cp .env.example .env
    echo "Please edit .env and add your ANTHROPIC_API_KEY"
    exit 1
fi

# Check if ANTHROPIC_API_KEY is set
if ! grep -q "ANTHROPIC_API_KEY=sk-" .env; then
    echo "❌ ANTHROPIC_API_KEY not configured in .env"
    echo "Please add your key to .env file"
    exit 1
fi

echo "✅ Environment configured"

# Build and start containers
echo "🔨 Building containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Services started!"
echo "===================================="
echo "📍 Backend API: http://localhost:8000"
echo "📍 Frontend: http://localhost:3000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "  docker-compose logs -f     # View logs"
echo "  docker-compose down        # Stop services"
echo "  docker-compose restart     # Restart services"
echo "  ./stop.sh                  # Stop all services"
echo ""
