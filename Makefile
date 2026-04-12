SOURCES := furnace tests

.PHONY: check lint typecheck test install-hooks

check: lint typecheck test

install-hooks:
	git config core.hooksPath .githooks

lint:
	uv run ruff check $(SOURCES)

typecheck:
	uv run mypy $(SOURCES) --strict

test:
	uv run pytest tests/ -q \
		--cov=furnace --cov-branch \
		--cov-report=term-missing \
		--cov-fail-under=100
