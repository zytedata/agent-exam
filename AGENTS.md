# AGENTS.md

`agent-exam` is an eval framework for agent skills. See `README.md` for the
user-facing overview.

## Layout

- `agent_exam/` — the engine. Assertions in `assertions/`, subcommands in
  `commands/`, harness adapters in `providers/`, judge dispatch in `judge/`.
- `tests/` — unit tests, mirroring the package layout.
- `examples/config.yaml` — annotated reference for `evals/config.yaml`.

## Architecture

A plain Python CLI — it does **not** use the Agent SDK. Per task it spawns the
host agent as a **subprocess** (for Claude Code:
`claude -p <prompt> --output-format stream-json`), parses the streamed events,
locates the session transcript, and runs the task's assertions against it.

Harnesses sit behind a `Provider` adapter (`claude_code`, `codex_cli`,
`copilot_cli`, `opencode`, plus `dummy` for testing the runner without an LLM).
Everything above the adapter — assertions, scoring, reports — works on the
normalized `RunResult`, so a new harness is an adapter, not a core change.

Skills are staged hermetically into each attempt's working directory, which is
an opaque temp path so nothing in it tips the agent off that it is being
evaluated.

## How consumers wire it up

A repo under evaluation declares `[tool.agent-exam]` in its `pyproject.toml` and
puts suites under `evals/`. Skills come from `skills_dirs` in
`evals/config.yaml`, or — when they must be built first — from a `pre_run_hook`
returning `PreRunResult(skills_dirs=[...])`.

## Working on this repo

```bash
uv sync
uv run pytest                  # unit tests
uvx tox -e min                 # tests against the minimum dependency versions
uvx tox                        # full matrix + pre-commit + packaging check
```

Nothing here needs an agent CLI installed: the `dummy` provider covers the
runner paths, and every harness adapter is tested against recorded event
streams.

## Gotchas

- `pre-commit run --all-files` only covers files git already tracks, so a new
  file that has never been added is silently skipped. Check `git status` for
  untracked files before trusting a green run.
- The `min` env pins every dependency floor. If a change needs a newer API,
  raise the floor in `pyproject.toml` *and* the pin in `tox.ini` together.
- `tomllib` is 3.11+; the package supports 3.10 through a `tomli` fallback in
  `agent_exam/config.py`. Import it from there rather than adding new direct
  imports.
- Pydantic and dataclass fields resolve annotations at run time, so their
  imports must stay out of `TYPE_CHECKING` blocks. Ruff knows this via
  `runtime-evaluated-base-classes`; do not "fix" it by moving them.
- Versioning is managed by `bump-my-version`
  (`uvx bump-my-version bump patch|minor|major`), which also stamps the date
  onto the `(unreleased)` heading in `CHANGES.rst`. Do not hand-edit versions.
