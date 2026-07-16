SOURCES := furnace tests

.PHONY: check lint typecheck test

check: lint typecheck test

lint:
	uv run ruff check $(SOURCES)

typecheck:
	uv run mypy $(SOURCES) --strict

test:
	uv run pytest tests/ -q \
		--cov=furnace --cov=tests --cov-branch \
		--cov-report=term-missing:skip-covered \
		--cov-fail-under=100
