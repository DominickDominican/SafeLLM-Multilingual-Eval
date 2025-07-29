.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs

help:
	@echo "SafeLLM Multilingual Evaluation - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode with all dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy" 
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"

install:
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock black flake8 mypy isort pre-commit bandit safety
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=safellm_eval --cov-report=html --cov-report=term --cov-report=xml -v

test-unit:
	pytest tests/ -m "unit" -v

test-integration:
	pytest tests/ -m "integration" -v

lint:
	flake8 safellm_eval tests
	bandit -r safellm_eval

format:
	black safellm_eval tests
	isort safellm_eval tests

type-check:
	mypy safellm_eval --ignore-missing-imports

security-check:
	bandit -r safellm_eval
	safety check

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build: clean
	python -m build

docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# CLI testing commands
cli-test:
	@echo "Testing CLI commands..."
	safellm-eval --help
	safellm-eval info
	safellm-eval init --output test_config.yaml
	safellm-eval validate --config test_config.yaml
	safellm-eval list-models --config test_config.yaml
	@rm -f test_config.yaml

# Dataset validation
validate-datasets:
	@echo "Validating datasets..."
	safellm-eval inspect datasets/comprehensive_prompts.jsonl
	safellm-eval inspect datasets/benign_prompts.jsonl

# Performance testing
perf-test:
	@echo "Running performance tests..."
	python -m pytest tests/ -m "not slow" --benchmark-only

# Full CI pipeline locally
ci-local: format lint type-check security-check test-cov
	@echo "✅ All CI checks passed locally!"

# Release preparation
prepare-release:
	@echo "Preparing release..."
	python -m build
	twine check dist/*
	@echo "✅ Release ready! Run 'twine upload dist/*' to publish."

# Docker commands (if needed)
docker-build:
	docker build -t safellm-eval:latest .

docker-test:
	docker run --rm safellm-eval:latest pytest tests/ -v