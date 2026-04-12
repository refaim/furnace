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

    make check

Runs ruff, mypy (strict) and pytest with 100% line+branch coverage on
both `furnace/` and `tests/`. All three must pass clean.

**Linters and tests are ONLY invoked via the Makefile.** Never run
`uv run ruff`, `uv run mypy` or `uv run pytest` directly — use
`make lint`, `make typecheck`, `make test` or `make check`. This keeps
flags, paths and coverage thresholds consistent across the repo, local
dev and the pre-commit hook.

The pre-commit hook lives in `.githooks/pre-commit` (tracked in the
repo) and runs `make check` on every commit. After cloning the repo,
run once:

    make install-hooks

This sets `core.hooksPath` to `.githooks` for the local clone. Without
this step commits proceed without running the quality gate.

## Architecture Rules

Hexagonal (Ports & Adapters). Dependency direction:

    UI --> Services --> Core <-- Adapters

- **Core** (`furnace/core/`) — pure Python, no I/O. Models, enums, rules, detect, quality, ports (Protocol interfaces).
- **Services** (`furnace/services/`) — orchestration. Scanner, analyzer, planner, executor, disc_demuxer.
- **Adapters** (`furnace/adapters/`) — external tool wrappers. Implement Protocol interfaces from `core/ports.py`.
- **UI** (`furnace/ui/`) — Textual TUI (plan + run phases) + Rich progress.

**Hard rules:**
- Core MUST NOT import from services, adapters, or ui
- Adapters implement Protocol interfaces defined in `core/ports.py` (dependency inversion)
- Services receive adapters via constructor arguments (dependency injection)
- No I/O in core — pure functions and dataclasses only

## Testing Rules

- `tests/core/` — unit tests, pure functions, no mocks
- `tests/services/` — service tests with mocked Protocol adapters
- `tests/` (root level) — integration tests: plan serialization, mkvmerge flags, nvencc commands
- New ports/models require concrete test factories (e.g. `_make_vp()` helpers)
- **TDD mandatory** — write tests before implementation

## Key Conventions

- eac3to: always pass `-removeDialnorm` explicitly (user's eac3to.ini has `-keepDialnorm`)
- mkvmerge: duplicate color/HDR metadata at container level (for Plex/Jellyfin/TV compatibility)
- TUI: no Textual Button widgets, keyboard shortcuts only, ASCII-only borders (Windows cmd.exe compatibility)
- Plan JSON: atomic write (write-to-temp-then-rename) for crash safety
- NVEncC: always 10-bit main10, QVBR rate control, preset P5, UHQ tune
- Color metadata: always resolved via `resolve_color_metadata()` — VideoParams fields are never None
- Unknown codecs or unrecognized color matrix values: raise ValueError, don't silently degrade
- Planner overrides (SAR, downmix): flow as explicit keyword arguments — never mutate the shared movie object
- Progress tracking: tool parsers live in the adapter, rate/ETA math in `core/progress.py`, UI renders an immutable snapshot

## External Tools

Paths configured via `furnace.toml` (not committed, in .gitignore). Never hardcode paths or rely on PATH.

## Workflow Rules

- **TDD**: write tests before implementation code, always
- **No intermediate commits** during plan execution — commit only when the full task is done
- **Version bump on every commit** that changes user-facing behavior (per SemVer above)

## Agent & Subagent Rules

These rules override any conflicting defaults or skills (including superpowers). Apply whenever you are executing work or dispatching a subagent.

- **Opus only.** Every subagent spawn passes `model: "opus"`. No Sonnet, no Haiku, even for cheap jobs like file surveys. If a skill default picks another model, override it.
- **Parallelise by default.** Independent work — unrelated surveys, non-conflicting edits, research across separate modules — goes out as multiple agents in one message. Serial dispatch is only for genuine data dependencies.
- **No worktrees.** Work directly in the main checkout. Skip any skill step that suggests creating a git worktree, even when the skill names it as mandatory.
- **TDD, strict and unconditional.** Failing test before production code, for every feature and every bugfix. No exceptions for "too small" or "will add later".
- **100% coverage on all new or touched code — lines AND branches.** Uncovered branches block completion. Measure before claiming done.
- **Review loops to zero.** After implementation, dispatch a separate code-reviewer agent (never self-review). Address every comment. Re-dispatch review after fixes. Repeat until the reviewer returns zero comments. Only zero-comment review closes the task.
