.PHONY: help build run dev stop test lint seed seed-docker clean deploy logs restart

help:
	@echo "AI Ticket Router - Available Commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make dev         - Start FastAPI in development mode (local)"
	@echo "    make test        - Run pytest suite"
	@echo "    make lint        - Run ruff linter"
	@echo "    make eval        - Run offline evaluation over labeled dataset"
	@echo "    make seed        - Load seed documents into ChromaDB (local)"
	@echo ""
	@echo "  Docker:"
	@echo "    make build       - Build Docker images"
	@echo "    make run         - Start services in detached mode"
	@echo "    make stop        - Stop all services"
	@echo "    make restart     - Restart all services"
	@echo "    make logs        - Tail container logs"
	@echo "    make seed-docker - Seed KB inside Docker"
	@echo "    make clean       - Remove containers and volumes"
	@echo ""
	@echo "  Deployment:"
	@echo "    make deploy      - Deploy to VPS via SSH"

# --- Development ---

dev:
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

eval:
	python -m scripts.run_evaluation

seed:
	python -m scripts.seed_kb

seed-test:
	python -m scripts.seed_kb --test

# --- Docker ---

build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose stop

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

seed-docker:
	docker compose run --rm seed

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# --- Deployment ---

VPS_HOST ?= your-vps-ip
VPS_USER ?= root
VPS_DIR ?= ~/projects/ai-ticket-router

deploy:
	@echo "Deploying to $(VPS_USER)@$(VPS_HOST):$(VPS_DIR)..."
	ssh $(VPS_USER)@$(VPS_HOST) "\
		cd $(VPS_DIR) && \
		git pull origin main && \
		docker compose down && \
		docker compose up -d --build && \
		echo 'Waiting for services...' && \
		sleep 10 && \
		docker compose run --rm seed && \
		echo 'Deployment complete!'"
