# Furnace — Developer Guide

## Versioning (SemVer)

Format: `MAJOR.MINOR.PATCH`

- **MAJOR** (1.0.0, 2.0.0) — breaking changes: incompatible JSON plan format, CLI interface changes, removed features
- **MINOR** (0.2.0, 0.3.0) — new features: new mkvmerge flags, new codec support, new CLI options, TUI changes
- **PATCH** (0.2.1, 0.2.2) — bugfixes: fix broken behavior, fix display issues, fix edge cases

**When to bump:**
- Every commit that changes user-facing behavior MUST bump the version
- Multiple bugfixes can share one PATCH bump if committed together
- Version lives in TWO places — both must be updated together:
  - `furnace/__init__.py` → `VERSION = "X.Y.Z"`
  - `pyproject.toml` → `version = "X.Y.Z"`

## Quality Gates

Before committing:
```
uv run ruff check furnace/
uv run mypy furnace/ --strict
uv run pytest tests/ -q
```

All three must pass clean.

## Architecture

Hexagonal (Ports & Adapters). Dependency direction:

```
UI --> Services --> Core <-- Adapters
```

- **Core** (`furnace/core/`) — pure Python, no I/O. Models, rules, quality, detect, ports (Protocol interfaces).
- **Services** (`furnace/services/`) — orchestration. Scanner, analyzer, planner, executor.
- **Adapters** (`furnace/adapters/`) — external tool wrappers. Implement Protocol interfaces from core.
- **UI** (`furnace/ui/`) — Textual TUI + Rich progress. ASCII-only, no Unicode borders (Windows compatibility).

Core MUST NOT import from services, adapters, or ui.

## External Tools

Paths configured via `furnace.toml` (not committed, in .gitignore). Never hardcode paths or rely on PATH.

## Testing

- `tests/core/` — unit tests, pure functions, no mocks
- `tests/services/` — service tests with mocked Protocol adapters
- `tests/test_plan.py` — JSON plan serialization round-trip
- `tests/test_mkvmerge_color.py` — mkvmerge color/HDR flag generation

## Key Conventions

- eac3to: always pass `-removeDialnorm` explicitly (user's eac3to.ini has `-keepDialnorm`)
- mkvmerge: duplicate color/HDR metadata at container level (for Plex/Jellyfin/TV compatibility)
- TUI: no Textual Button widgets, use keyboard shortcuts only (Windows cmd.exe compatibility)
- Plan JSON: atomic write (write-to-temp-then-rename) for crash safety
