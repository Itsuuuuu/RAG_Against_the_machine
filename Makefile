.PHONY: install run debug clean fclean re lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -name __pycache__ -type d -not -path "./.venv/*" -not -path "./data/*" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

fclean: clean
	rm -rf .venv

re: fclean install

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
