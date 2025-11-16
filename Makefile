# Makefile for Agentic RAG

.PHONY: help build up down restart logs clean test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs
	docker-compose logs -f

clean: ## Clean up containers and volumes
	docker-compose down -v
	rm -rf data/chroma_db/*

test: ## Run evaluation tests
	./venv/bin/python backend/scripts/run_evaluation.py --limit 3

dev: ## Run in development mode (without Docker)
	cd backend/api && python main.py

install: ## Install dependencies locally
	pip install -r requirements.txt
	cd frontend && npm install

docker-clean: ## Remove all Docker artifacts
	docker-compose down -v
	docker system prune -f