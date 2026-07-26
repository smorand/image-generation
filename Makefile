.PHONY: sync run run-image run-video run-dev test test-cov lint lint-fix format format-check typecheck security check build install uninstall clean clean-all info help

PROJECT_NAME=image-gen
SRC_DIR=src
PYTHON_VERSION=$(shell python3 --version 2>/dev/null | cut -d' ' -f2)
HAS_UV=$(shell command -v uv >/dev/null 2>&1 && echo "yes" || echo "no")
DOCKER_TAG ?= latest

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

## sync: Install/update project dependencies using uv
sync:
ifeq ($(HAS_UV),yes)
	@echo "Syncing dependencies with uv..."
	@uv sync
	@echo "Dependencies synced!"
else
	@echo "Error: uv not found. Install it from https://docs.astral.sh/uv/"
	@exit 1
endif

# ============================================================================
# RUNNING
# ============================================================================

## run: Alias for run-image
run: run-image

## run-image: Run image-gen CLI
run-image: sync
ifdef ARGS
	@uv run image-gen $(ARGS)
else
	@uv run image-gen --help
endif

## run-video: Run video-gen CLI
run-video: sync
ifdef ARGS
	@uv run video-gen $(ARGS)
else
	@uv run video-gen --help
endif

## run-dev: Run entry point directly (useful during development)
run-dev:
ifdef ARGS
	@uv run python $(SRC_DIR)/image_gen/cli.py $(ARGS)
else
	@uv run python $(SRC_DIR)/image_gen/cli.py --help
endif

# ============================================================================
# TESTING
# ============================================================================

## test: Run tests with pytest
test:
	@echo "Running tests..."
ifdef ARGS
	@uv run python -m pytest -v $(ARGS)
else
	@uv run python -m pytest -v
endif
	@echo "Tests complete!"

## test-cov: Run tests with coverage report
test-cov:
	@echo "Running tests with coverage..."
	@uv run python -m pytest -v --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "Tests complete!"

# ============================================================================
# CODE QUALITY
# ============================================================================

## lint: Check code style with Ruff
lint:
	@echo "Running Ruff linter..."
	@uv run ruff check .
	@echo "Lint check complete!"

## lint-fix: Auto-fix lint issues with Ruff
lint-fix:
	@echo "Running Ruff linter with auto-fix..."
	@uv run ruff check --fix .
	@echo "Lint fix complete!"

## format: Format code with Ruff
format:
	@echo "Formatting code with Ruff..."
	@uv run ruff format .
	@echo "Format complete!"

## format-check: Check code formatting without changes
format-check:
	@echo "Checking code format..."
	@uv run ruff format --check .
	@echo "Format check complete!"

## typecheck: Run type checking with mypy
typecheck:
	@echo "Running mypy type checker..."
	@uv run mypy $(SRC_DIR)/
	@echo "Type check complete!"

## security: Run bandit security scanner
security:
	@echo "Running bandit security scanner..."
	@uv run bandit -r $(SRC_DIR)/ -c pyproject.toml 2>/dev/null || uv run bandit -r $(SRC_DIR)/
	@echo "Security scan complete!"

## check: Run all quality checks (lint, format, typecheck, security, tests+coverage)
check: lint format-check typecheck security test-cov
	@echo "All checks passed!"

# ============================================================================
# BUILD & INSTALL
# ============================================================================

## build: Build wheel and sdist packages
build: sync
	@echo "Building package..."
	@uv build
	@echo "Build complete! Artifacts in dist/"

## install: Install image-gen and video-gen as uv tools (system-wide)
install:
	@echo "Installing image-gen and video-gen as uv tools..."
	@uv tool install . --reinstall --force
	@echo "Install complete! Run 'image-gen' or 'video-gen' from anywhere."

## uninstall: Remove uv tools
uninstall:
	@echo "Uninstalling image-gen..."
	@uv tool uninstall image-gen 2>/dev/null || echo "image-gen not installed"
	@echo "Uninstall complete!"

# ============================================================================
# CLEANUP
# ============================================================================

## clean: Remove caches and build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@rm -rf dist build *.egg-info
	@rm -rf .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete!"

## clean-all: Remove everything including venv and lock file
clean-all: clean
	@echo "Removing virtual environment and lock file..."
	@rm -rf .venv
	@rm -f uv.lock
	@echo "Full clean complete!"

# ============================================================================
# INFORMATION
# ============================================================================

## info: Show project information
info:
	@echo "Project Information"
	@echo "==================="
	@echo "Project name:    $(PROJECT_NAME)"
	@echo "Source dir:      $(SRC_DIR)/"
	@echo "Python version:  $(PYTHON_VERSION)"
	@echo "uv available:    $(HAS_UV)"
	@echo ""
	@echo "Entry points:"
	@echo "  image-gen  ->  src/image_gen/cli.py"
	@echo "  video-gen  ->  src/video_gen/cli.py"

## help: Show this help message
help:
	@echo "Image Generation Makefile"
	@echo "========================="
	@echo ""
	@echo "Dependency Management:"
	@echo "  sync             - Install/update dependencies with uv"
	@echo ""
	@echo "Running:"
	@echo "  run              - Alias for run-image"
	@echo "  run-image        - Run image-gen CLI (ARGS='...' for arguments)"
	@echo "  run-video        - Run video-gen CLI (ARGS='...' for arguments)"
	@echo "  run-dev          - Run image-gen entry point directly"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run tests with pytest"
	@echo "  test-cov         - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             - Check code style with Ruff"
	@echo "  lint-fix         - Auto-fix lint issues"
	@echo "  format           - Format code with Ruff"
	@echo "  format-check     - Check formatting without changes"
	@echo "  typecheck        - Run mypy type checking"
	@echo "  security         - Run bandit security scanner"
	@echo "  check            - Run all quality checks"
	@echo ""
	@echo "Build & Install:"
	@echo "  build            - Build wheel and sdist packages"
	@echo "  install          - Install image-gen + video-gen as uv tools"
	@echo "  uninstall        - Remove uv tools"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            - Remove caches and build artifacts"
	@echo "  clean-all        - Remove everything including venv"
	@echo ""
	@echo "Examples:"
	@echo "  make sync"
	@echo "  make install"
	@echo "  make run-image ARGS='generate --help'"
	@echo "  make run-video ARGS='generate --help'"
	@echo "  make test ARGS='-k test_foo'"
	@echo "  make check"
