===========
File layout
===========

Where things live on disk, both the inputs you write and the outputs runs
produce.

Inputs
======

agent-exam finds the project root by walking up from the current directory to
the nearest :file:`pyproject.toml`, and reads its suites from :file:`evals/`
next to it. Point it somewhere else with ``evals_dir``:

.. code-block:: toml

    [tool.agent-exam]
    evals_dir = "qa/evals"

.. code-block:: text

    evals/
      config.yaml                 # default harness, judge model, concurrency
                                  # groups, default timeout, skills_dirs, …
      config.local.yaml           # personal overrides — gitignored

      suites/<suite>/             # typically one suite per skill
        suite.yml                 # optional suite-level config
        tasks/
          <task>.yaml             # one task per file

      fixtures/<fixture>/         # starting working directory for tasks that
        ...                       # opt in via setup.fixture

:file:`config.local.yaml` is deep-merged over :file:`config.yaml` at load time.
Dict keys merge recursively, so you only need to write what you want to
override; lists are replaced outright rather than appended. Typical uses are
changing the harness for your own runs:

.. code-block:: yaml

    # evals/config.local.yaml
    default_harness: opencode

or overriding a single key inside a provider block without repeating the whole
block:

.. code-block:: yaml

    providers:
      claude_code:
        default_model: claude-opus-4-7

Suites and fixtures are committed. **All fixtures live in version control** —
evals have to be reproducible across machines and team members, so fixtures
cannot be gitignored or regenerated locally. Prefer small plain-text fixtures
where that is realistic, but do not reject a large one that is genuinely needed
(a 5 MB HTML page to test how a skill handles big input, say). If repository
size becomes a problem, Git LFS is a reasonable escape hatch for the largest
files.

Fixtures are shared across suites, so give them descriptive, scoped names to
avoid collisions.

``suite.yml``
-------------

A suite may carry an optional :file:`suite.yml` declaring suite-level metadata.
Today the only supported key is ``evaluated_skills``:

.. code-block:: yaml

    # evals/suites/<suite>/suite.yml
    evaluated_skills:
      - github-release

``evaluated_skills``
    The skills this suite evaluates. ``--without-skill`` excludes them from the
    bundle handed to the harness. Defaults to the suite name alone.

Using :file:`suite.yml` is worthwhile even for a single-skill suite: it makes
the skill-to-suite mapping explicit. Use it in earnest when a suite covers
several related skills — a primary skill plus internal helpers — so the reality
check removes the full set. ``--no-skills`` ignores the field entirely and
removes every skill under ``skills_dirs``.

Outputs
=======

Each run creates one directory under :file:`evals/runs/<run-id>/`, with
timestamp-based run ids:

.. code-block:: text

    evals/runs/<run-id>/
      run.json                    # run-level metadata: timestamps, config.
                                  # written once

      artifacts/                  # what the skill produced. frozen once the
                                  # run completes; rescoring does not touch it
        <suite>/<task>/attempt-1/
          trajectory.json         # normalized per-turn record
          cwd/                    # archived working tree the skill wrote
          attempt.json            # harness, model, timestamps, metrics, and
                                  # the raw transcript path

      reports/                    # how the run was graded. one file per report
        2026-04-23-1423.json      # initial report — every attempt
        2026-04-23-1500.json      # rescore — only the attempts in scope

The raw harness-specific transcript is not copied, since it is too
format-dependent; :file:`attempt.json` stores its path as
``raw_transcript_path``. For Claude Code, run ``claude-measure-usage <path>``
on it.

:file:`evals/runs/` is **gitignored**: runs are your local lab notebook, not a
shared artifact. They are mostly reproducible from committed inputs and usually
fine to share, but be a little careful with transcripts — skills are written to
keep secrets out of the agent's context, and mistakes happen. Skim before
posting one externally.

Nothing prunes :file:`evals/runs/` automatically. Delete old runs when your
disk complains. They feed the per-task ``history`` view, so they stay useful
for trend analysis until then.
