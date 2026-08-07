# agent-exam

Eval framework for agent skills. Point it at a repo of skills and a suite of
YAML tasks, and it runs them against a real agent CLI, grades the resulting
transcript, and reports verdicts plus cost and token metrics.

Supported harnesses: **Claude Code**, **Codex CLI**, **Copilot CLI** and
**OpenCode**. Each runs as a subprocess against your existing subscription or
login — no pay-per-token API key is required.

## Install

```bash
pip install agent-exam
```

## Quick start

In the repo holding your skills:

```yaml
# evals/config.yaml
default_harness: claude_code
```

Skills are picked up from `./skills` by default; set `skills_dirs` to point
elsewhere.

```yaml
# evals/suites/my-suite/tasks/greets.yaml
description: |
  The skill should greet by name.
kind: execute
prompt: |
  Greet Ada.
assertions:
  - skill_invoked: my-skill
  - judge: |
      The reply greets Ada by name.
  - no_permission_errors:
```

```bash
agent-exam doctor          # preflight checks
agent-exam my-suite        # run the suite
agent-exam show latest     # read the report
```

`examples/config.yaml` documents every configuration key.

The evals directory is found relative to the nearest `pyproject.toml` and
defaults to `evals/`. Add a `[tool.agent-exam]` section only to point somewhere
else, or to register a `pre_run_hook`:

```toml
# pyproject.toml
[tool.agent-exam]
evals_dir = "qa/evals"
pre_run_hook = "evals.hooks:pre_run_hook"
```

## What it gives you

- **Multi-harness by design.** Assertions and the normalized transcript are
  harness-neutral, so adding another agent CLI is a small adapter rather than a
  core rewrite.
- **Trigger evals are first-class.** Test whether the right skill fires for a
  given prompt, separately from whether it then does the right thing.
- **Reality-check modes.** `--without-skill` re-runs a suite with one skill
  removed and `--no-skills` with the whole library removed, so you can see
  whether a skill earns its place against the bare agent.
- **Deterministic and LLM-judge assertions**, per-attempt fixtures and isolated
  working directories, parallel execution with concurrency groups, and
  regression reports across runs (`runs`, `show`, `history`, `diff`).
- **Rescoring without re-running.** `agent-exam rescore` re-grades archived
  attempts against current assertions.

## Intended use

Interactive development of skills — edit a skill, run the relevant suite or a
single task, read the report, iterate. Automating runs against
subscription-backed CLIs may stretch a harness's terms, and LLM evals are slow
and expensive enough that running every suite on every change buys little extra
signal.

## Contributing

```bash
uv sync
uv run pytest                  # unit tests
tox                            # full matrix, linting, packaging checks
pre-commit install
```

Pull requests run a secrets scan that must pass before merging;
`pre-commit install` runs the same check locally. If it flags a value that is
genuinely not a secret, audit it into `.secrets.baseline` and mention it in the
PR.

## License

Apache-2.0. See [LICENSE](LICENSE).
