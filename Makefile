# RDT Trading System Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev install-hooks lint format test test-fast test-cov \
        security build run clean docs docker-build docker-run docker-push \
        pre-commit update-deps check-deps audit all

# Default target
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := rdt-trading-system
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
PYLINT := $(PYTHON) -m pylint
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit
COVERAGE := $(PYTHON) -m coverage

# Docker configuration
DOCKER_REGISTRY := ghcr.io
DOCKER_IMAGE := $(DOCKER_REGISTRY)/rdt-trading/$(PROJECT_NAME)
DOCKER_TAG := $(shell git describe --tags --always --dirty 2>/dev/null || echo "latest")

# Directories
SRC_DIRS := scanner automation shared alerts config agents portfolio risk backtesting monitoring brokers
TEST_DIR := tests
DATA_DIR := data

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

# =============================================================================
# Help
# =============================================================================
help:  ## Show this help message
	@echo "$(BLUE)RDT Trading System - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make <target>"
	@echo ""
	@echo "$(GREEN)Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Installation
# =============================================================================
install:  ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Development installation complete!$(NC)"

install-hooks:  ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	$(PIP) install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit install --hook-type pre-push
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

install-all: install-dev install-hooks  ## Install all dependencies and hooks

# =============================================================================
# Code Quality
# =============================================================================
lint:  ## Run all linters (black check, isort check, pylint, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "\n$(YELLOW)Checking code formatting with Black...$(NC)"
	$(BLACK) --check --diff .
	@echo "\n$(YELLOW)Checking import sorting with isort...$(NC)"
	$(ISORT) --check-only --diff .
	@echo "\n$(YELLOW)Running Pylint...$(NC)"
	$(PYLINT) --rcfile=pyproject.toml $(SRC_DIRS) || true
	@echo "\n$(YELLOW)Running MyPy type checking...$(NC)"
	$(MYPY) --config-file=pyproject.toml $(SRC_DIRS) || true
	@echo "\n$(GREEN)Linting complete!$(NC)"

lint-fix:  ## Run linters and auto-fix issues
	@echo "$(BLUE)Auto-fixing lint issues...$(NC)"
	$(BLACK) .
	$(ISORT) .
	@echo "$(GREEN)Auto-fix complete!$(NC)"

format:  ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) .
	$(ISORT) .
	@echo "$(GREEN)Formatting complete!$(NC)"

format-check:  ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(BLACK) --check --diff .
	$(ISORT) --check-only --diff .
	@echo "$(GREEN)Format check complete!$(NC)"

pylint:  ## Run pylint only
	@echo "$(BLUE)Running Pylint...$(NC)"
	$(PYLINT) --rcfile=pyproject.toml $(SRC_DIRS)

mypy:  ## Run mypy type checking only
	@echo "$(BLUE)Running MyPy...$(NC)"
	$(MYPY) --config-file=pyproject.toml $(SRC_DIRS)

# =============================================================================
# Testing
# =============================================================================
test:  ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR)/ \
		--cov=. \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--cov-fail-under=70 \
		-v
	@echo "$(GREEN)Tests complete! Coverage report: htmlcov/index.html$(NC)"

test-fast:  ## Run tests without coverage (faster)
	@echo "$(BLUE)Running tests (fast mode)...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short
	@echo "$(GREEN)Tests complete!$(NC)"

test-parallel:  ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -n auto -v --tb=short
	@echo "$(GREEN)Tests complete!$(NC)"

test-unit:  ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -m "unit" -v
	@echo "$(GREEN)Unit tests complete!$(NC)"

test-integration:  ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -m "integration" -v
	@echo "$(GREEN)Integration tests complete!$(NC)"

test-cov:  ## Run tests and generate coverage report
	@echo "$(BLUE)Running tests with detailed coverage...$(NC)"
	$(COVERAGE) run -m pytest $(TEST_DIR)/ -v
	$(COVERAGE) report --fail-under=70
	$(COVERAGE) html
	$(COVERAGE) xml
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(NC)"

test-watch:  ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(PYTEST) $(TEST_DIR)/ --watch -v

# =============================================================================
# Security
# =============================================================================
security:  ## Run security scans (bandit, safety, pip-audit)
	@echo "$(BLUE)Running security scans...$(NC)"
	@echo "\n$(YELLOW)Running Bandit security linter...$(NC)"
	$(BANDIT) -r $(SRC_DIRS) -x $(TEST_DIR),$(DATA_DIR) --severity-level medium || true
	@echo "\n$(YELLOW)Running Safety dependency check...$(NC)"
	safety check -r requirements.txt --full-report || true
	@echo "\n$(YELLOW)Running pip-audit...$(NC)"
	pip-audit -r requirements.txt || true
	@echo "\n$(GREEN)Security scans complete!$(NC)"

bandit:  ## Run bandit security scan only
	@echo "$(BLUE)Running Bandit...$(NC)"
	$(BANDIT) -r $(SRC_DIRS) -x $(TEST_DIR),$(DATA_DIR) --severity-level low -f txt

safety-check:  ## Run safety dependency vulnerability check
	@echo "$(BLUE)Running Safety check...$(NC)"
	safety check -r requirements.txt --full-report

pip-audit:  ## Run pip-audit dependency check
	@echo "$(BLUE)Running pip-audit...$(NC)"
	pip-audit -r requirements.txt

detect-secrets:  ## Scan for secrets in code
	@echo "$(BLUE)Scanning for secrets...$(NC)"
	detect-secrets scan --all-files --baseline .secrets.baseline
	@echo "$(GREEN)Secrets scan complete!$(NC)"

# =============================================================================
# Docker
# =============================================================================
build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):$(DOCKER_TAG) \
		-t $(PROJECT_NAME):latest \
		--build-arg BUILD_DATE=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ") \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
		--build-arg VERSION=$(DOCKER_TAG) \
		.
	@echo "$(GREEN)Docker image built: $(PROJECT_NAME):$(DOCKER_TAG)$(NC)"

docker-build: build  ## Alias for build

docker-build-no-cache:  ## Build Docker image without cache
	@echo "$(BLUE)Building Docker image (no cache)...$(NC)"
	docker build --no-cache -t $(PROJECT_NAME):$(DOCKER_TAG) \
		-t $(PROJECT_NAME):latest \
		.
	@echo "$(GREEN)Docker image built: $(PROJECT_NAME):$(DOCKER_TAG)$(NC)"

docker-run:  ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/.env:/app/.env:ro \
		-p 5000:5000 \
		$(PROJECT_NAME):latest

docker-run-scanner:  ## Run scanner in Docker
	@echo "$(BLUE)Running scanner in Docker...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/.env:/app/.env:ro \
		$(PROJECT_NAME):latest scanner

docker-shell:  ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in Docker container...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/.env:/app/.env:ro \
		--entrypoint /bin/bash \
		$(PROJECT_NAME):latest

docker-push:  ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image to registry...$(NC)"
	docker tag $(PROJECT_NAME):$(DOCKER_TAG) $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker tag $(PROJECT_NAME):latest $(DOCKER_IMAGE):latest
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest
	@echo "$(GREEN)Docker image pushed: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-scan:  ## Scan Docker image for vulnerabilities
	@echo "$(BLUE)Scanning Docker image...$(NC)"
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image $(PROJECT_NAME):latest

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit:  ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit complete!$(NC)"

pre-commit-update:  ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate
	@echo "$(GREEN)Pre-commit hooks updated!$(NC)"

# =============================================================================
# Dependencies
# =============================================================================
update-deps:  ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt
	@echo "$(GREEN)Dependencies updated!$(NC)"

freeze-deps:  ## Freeze current dependencies
	@echo "$(BLUE)Freezing dependencies...$(NC)"
	$(PIP) freeze > requirements-frozen.txt
	@echo "$(GREEN)Dependencies frozen to requirements-frozen.txt$(NC)"

check-deps:  ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)Dependency check complete!$(NC)"

audit:  ## Run full dependency audit
	@echo "$(BLUE)Running dependency audit...$(NC)"
	$(PIP) check
	safety check -r requirements.txt
	pip-audit -r requirements.txt
	@echo "$(GREEN)Dependency audit complete!$(NC)"

# =============================================================================
# Application
# =============================================================================
run:  ## Run the application (scanner mode)
	@echo "$(BLUE)Starting RDT Trading Scanner...$(NC)"
	$(PYTHON) main.py scanner

run-bot:  ## Run the trading bot
	@echo "$(BLUE)Starting RDT Trading Bot...$(NC)"
	$(PYTHON) main.py bot

run-backtest:  ## Run backtesting
	@echo "$(BLUE)Starting backtesting...$(NC)"
	$(PYTHON) main.py backtest

run-dashboard:  ## Run the web dashboard
	@echo "$(BLUE)Starting web dashboard...$(NC)"
	$(PYTHON) main.py dashboard

# =============================================================================
# Documentation
# =============================================================================
docs:  ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if [ -d docs ]; then \
		cd docs && make html; \
		echo "$(GREEN)Documentation generated: docs/_build/html/index.html$(NC)"; \
	else \
		echo "$(YELLOW)No docs directory found. Skipping documentation generation.$(NC)"; \
	fi

docs-serve:  ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@if [ -d docs/_build/html ]; then \
		cd docs/_build/html && $(PYTHON) -m http.server 8080; \
	else \
		echo "$(YELLOW)Documentation not built. Run 'make docs' first.$(NC)"; \
	fi

# =============================================================================
# Cleanup
# =============================================================================
clean:  ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf coverage.json
	rm -rf .tox/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage.*" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-docker:  ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker artifacts...$(NC)"
	docker rmi $(PROJECT_NAME):latest 2>/dev/null || true
	docker rmi $(PROJECT_NAME):$(DOCKER_TAG) 2>/dev/null || true
	docker system prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

clean-all: clean clean-docker  ## Clean everything

# =============================================================================
# CI/CD Helpers
# =============================================================================
ci-lint:  ## CI-specific lint command
	@echo "$(BLUE)Running CI lint checks...$(NC)"
	$(BLACK) --check --diff .
	$(ISORT) --check-only --diff .
	$(PYLINT) --rcfile=pyproject.toml --exit-zero $(SRC_DIRS)
	$(MYPY) --config-file=pyproject.toml --ignore-missing-imports $(SRC_DIRS) || true
	@echo "$(GREEN)CI lint checks complete!$(NC)"

ci-test:  ## CI-specific test command
	@echo "$(BLUE)Running CI tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ \
		--cov=. \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-fail-under=70 \
		--junitxml=test-results.xml \
		-v
	@echo "$(GREEN)CI tests complete!$(NC)"

ci-security:  ## CI-specific security command
	@echo "$(BLUE)Running CI security checks...$(NC)"
	$(BANDIT) -r $(SRC_DIRS) -x $(TEST_DIR),$(DATA_DIR) -f json -o bandit-report.json --severity-level medium || true
	safety check -r requirements.txt --json > safety-report.json 2>&1 || true
	pip-audit -r requirements.txt --format json --output pip-audit-report.json 2>&1 || true
	@echo "$(GREEN)CI security checks complete!$(NC)"

# =============================================================================
# Composite Commands
# =============================================================================
all: lint test security  ## Run all checks (lint, test, security)
	@echo "$(GREEN)All checks passed!$(NC)"

check: lint test-fast  ## Quick check (lint + fast tests)
	@echo "$(GREEN)Quick checks passed!$(NC)"

release-check: all build docker-scan  ## Full release check
	@echo "$(GREEN)Release checks passed!$(NC)"
