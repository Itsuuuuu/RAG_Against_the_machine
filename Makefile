.PHONY: install run debug clean fclean lint lint_strict

install:
	pip uv install
	uv sync
	python3 -m venv .venv

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__
	rm -rf src/__pyache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache

fclean:
	rm -rf .venv

re: fclean install

lint:
	uv run flake8 . 
	uv run mypy  . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint_strict:
	uv run flake8 .
	uv run mypy . --strict