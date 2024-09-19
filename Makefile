.PHONY:
	install-dev
	install
	lint
	format
	run

install-dev: install
	pip install -e ".[dev]"

install:
	pip install --upgrade pip
	pip install .

lint:
	python -m ruff format --diff
	python -m ruff check
	python -m mypy .

format:
	python -m ruff check --fix
	python -m ruff check --select I --fix
	python -m ruff format

run:
	python src/impact_explorer/main.py
