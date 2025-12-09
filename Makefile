.PHONY: all lint test cov clean upgrade build

all: lint test

lint: .venv/bin/activate
	uvx ruff format --check
	uvx ruff check src tests
	uvx pyrefly check

test: .venv/bin/activate
	uv run pytest

cov:
	uv run coverage report

.venv/bin/activate: pyproject.toml
	uv sync

upgrade:
	uv lock --upgrade
	uv sync

build:
	uv build

clean:
	find src tests -type d -name "__pycache__" -exec rm -r {} +
	rm -rf dist tests/tmp src/*.egg-info .coverage
