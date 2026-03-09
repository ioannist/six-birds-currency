.PHONY: setup test lint format

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

format:
	python -m black .
	python -m ruff check . --fix
