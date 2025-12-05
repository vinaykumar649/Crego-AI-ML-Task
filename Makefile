.PHONY: help setup install run test lint format typecheck clean build run_examples ci docker-build docker-up docker-down

help:
	@echo "JSON Logic Rule Generator - Available Commands"
	@echo ""
	@echo "setup          Install dependencies and initialize project"
	@echo "install        Install Python dependencies"
	@echo "run            Start the FastAPI server"
	@echo "run_examples   Run example prompts"
	@echo "test           Run pytest tests"
	@echo "lint           Run flake8 linting"
	@echo "format         Format code with black"
	@echo "typecheck      Run mypy type checking"
	@echo "ci             Run lint, format check, tests"
	@echo "clean          Remove build artifacts and cache"
	@echo "docker-build   Build Docker image"
	@echo "docker-up      Start Docker containers"
	@echo "docker-down    Stop Docker containers"

setup: install
	@echo "✓ Project setup complete"
	@echo "Next: make run_examples or make run"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

run:
	@echo "Starting FastAPI server..."
	python -m uvicorn src.main:app --host ${API_HOST:=0.0.0.0} --port ${API_PORT:=8000} --reload

run_examples:
	@echo "Running example prompts..."
	python -m src.examples.run_examples

test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Tests completed"

lint:
	@echo "Running linting..."
	flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__,venv
	@echo "✓ Linting passed"

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=120
	@echo "✓ Code formatted"

format-check:
	@echo "Checking code format..."
	black src/ tests/ --line-length=120 --check
	@echo "✓ Code format is correct"

typecheck:
	@echo "Running type checking..."
	mypy src/ --ignore-missing-imports
	@echo "✓ Type checking passed"

ci: format-check lint test
	@echo "✓ CI checks passed"

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .DS_Store -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage 2>/dev/null || true
	@echo "✓ Cleanup complete"

docker-build:
	@echo "Building Docker image..."
	docker-compose build
	@echo "✓ Docker image built"

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d
	@echo "✓ Containers started"
	@echo "API available at http://localhost:${API_PORT:=8000}"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✓ Containers stopped"

docker-logs:
	docker-compose logs -f api

.DEFAULT_GOAL := help
