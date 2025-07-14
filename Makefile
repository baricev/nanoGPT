.PHONY: all install lint format typecheck test

# Default command: run all checks
all: lint typecheck test

# Install all development and runtime dependencies
install:
	uv pip install --no-cache-dir ruff mypy types-requests pytest pytest-cov black torch numpy transformers datasets tiktoken wandb tqdm requests


# Format the code using ruff
format:
	ruff format .

# Lint the code using ruff
lint:
	ruff check . --ignore E721,E741

# Run the type checker using mypy
typecheck:
	mypy . --ignore-missing-imports --exclude data --exclude train.py --exclude bench.py --exclude sample.py



# Run tests using pytest
test:
	pytest -v -m "not heavy"