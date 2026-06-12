.PHONY: install lint format test precommit run help

help:
	@echo "Available targets:"
	@echo "  install    - Sync dependencies (uv sync --all-extras)"
	@echo "  lint       - Run ruff lint checks on src/ and tests/"
	@echo "  format     - Auto-format src/ and tests/ with ruff"
	@echo "  test       - Run the test suite with coverage"
	@echo "  precommit  - Run all pre-commit hooks on every file"
	@echo "  run        - Launch the project notebook"

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest

precommit:
	uv run pre-commit run --all-files

run:
	uv run jupyter notebook notebooks/ex03_fitness_rl.ipynb
