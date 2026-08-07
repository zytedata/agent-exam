===========
Quick start
===========

The shortest path to a working eval: install, wire :file:`evals/` into the repo
that holds your skills, run a task, read the report, and then turn a vague "the
skill does the wrong thing" into a failing assertion you can iterate against.

Install
=======

.. code-block:: bash

    pip install agent-exam

Set up ``evals/``
=================

agent-exam runs from the repo that holds your skills. It finds the project root
by walking up from the current directory to the nearest :file:`pyproject.toml`,
and reads its suites from :file:`evals/` next to it.

The minimum viable project is four files:

.. code-block:: text

    pyproject.toml
    skills/github-release/SKILL.md
    evals/config.yaml
    evals/suites/github-release/tasks/list-drafts.yaml

:file:`pyproject.toml` needs nothing added — it only marks the project root.
Add a ``[tool.agent-exam]`` section when :file:`evals/` lives somewhere else,
or when your skills have to be built before they can be evaluated:

.. code-block:: toml

    [tool.agent-exam]
    evals_dir = "qa/evals"
    pre_run_hook = "evals.hooks:pre_run_hook"

:file:`evals/config.yaml` can be nearly empty. Skills are picked up from
:file:`skills/` at the project root when that directory exists, and the default
harness is Claude Code, so this is enough:

.. code-block:: yaml

    # evals/config.yaml
    default_harness: claude_code

Everything else has a default. See :doc:`reference/config-yaml` for the full
schema, and :doc:`reference/file-layout` for where suites, fixtures and run
artifacts live.

Then check the wiring:

.. code-block:: bash

    agent-exam doctor

``doctor`` runs preflight checks — config, harness auth, skill discovery,
static validation of every suite, and a small real round-trip probe. If it is
green, you are ready.

A task at a glance
==================

A task is one YAML file under a suite's :file:`tasks/` directory:

.. code-block:: yaml

    # evals/suites/github-release/tasks/list-drafts.yaml
    description: |
      List draft releases for the repository in the working directory. The
      skill should discover the repository from the git remote rather than
      asking for it.
    kind: execute
    concurrency_group: github_api
    timeout_seconds: 120

    claude_code:
      allowed_tools:
        - "Bash(gh*)"

    setup:
      fixture: repo-with-changelog

    prompt: |
      Which releases are still in draft?

    assertions:
      - judge: |
          The agent discovered the repository from the git remote rather than
          asking the user which repository to look at.
      - tool_called: Bash
        providers: [claude_code]
      - no_permission_errors:

The fields you touch on every task:

``kind``
    ``execute`` (run the skill end to end and grade the outcome) or ``trigger``
    (only check that the right skill fires).

``prompt``
    What the agent is asked to do, in the natural language a real user would
    type.

``setup.fixture``
    A directory under :file:`evals/fixtures/` copied into the attempt's working
    directory before the agent starts. Optional.

``assertions``
    The checks that grade the attempt — deterministic ones (``tool_called``,
    ``file_exists``, ``no_permission_errors``, …) mixed with ``judge:``, which
    is graded by an LLM.

The ``claude_code:`` block above is harness-specific; a task can carry one
block per harness side by side. :doc:`reference/task-yaml` has the complete
schema.

Execute vs trigger evals
========================

Two task kinds, two different jobs:

``execute``
    Runs the agent through to completion and grades the outcome — file changes,
    judge verdicts on the response, tool-use checks. Slow and full-cost, gives
    you the most signal.

``trigger``
    Stops the agent right after the first skill invocation. The only thing it
    checks is *which* skill fired for a given prompt. One file per skill,
    batched: many short prompts split into ``positive:`` (should fire this
    skill) and ``negative:`` (should not).

Trigger evals tune skill descriptions; execute evals tune behavior. Most
user-facing skills get one trigger file plus a handful of execute tasks. See
:doc:`design/triggers`.

Run it
======

A single execute task with the default ``-k 1`` is one attempt: the cheapest
run shape and the most informative, since you get a real agent trajectory plus
a handful of assertions of different kinds.

.. code-block:: bash

    agent-exam github-release::list-drafts

That uses the harness set as ``default_harness`` in :file:`evals/config.yaml`.
To run against another one, pass ``--provider``:

.. code-block:: bash

    agent-exam github-release::list-drafts --provider codex_cli

Or set a personal default:

.. code-block:: yaml

    # evals/config.local.yaml  (gitignored — personal overrides)
    default_harness: codex_cli

Make sure the target harness is installed and
``agent-exam doctor --provider <name>`` passes before switching.

Output::

    Run:      run-2026-04-25-1432
    Suite:    github-release::list-drafts (1 task × k=1 = 1 attempts)
    Provider: claude_code (claude-sonnet-4-6)
    Parallel: up to 1

    PASS              github-release::list-drafts attempt-1  (38s total, 4s scoring)

    Run dir:     evals/runs/run-2026-04-25-1432/
    Report:      .../reports/2026-04-25-1432.json
    Inspect:     agent-exam show latest

Triggers are runnable the same way (``agent-exam github-release::trigger``),
but they make a less interesting first read: each case is just a prompt and a
yes/no on which skill fired, so you see no fixtures, judge reasoning or
tool use in the report.

Read the report
===============

Drill into the task to see what each assertion did:

.. code-block:: bash

    agent-exam show latest::github-release::list-drafts

::

    === Summary ===
    Attempt: github-release::list-drafts attempt-1
    Passed:  4/4   (1 known-issue)
    Metrics: cost $0.0823, wall 38s, peak ctx 24.1k, turns 4

    === Artifacts ===
    attempt.json:    .../attempt-1/attempt.json
    raw transcript:  ~/.claude/projects/.../<session>.jsonl

    === Prompt & response ===
    Prompt: Which releases are still in draft?

    === Assertions (4/4, 1 known-issue) ===
    [PASS]       judge: The agent discovered the repository from the git
                 remote rather than asking the user which repository to
                 look at.
                 YES — the agent read the origin remote and passed
                 acme/widgets to gh directly.
    [PASS]       tool_called: Bash
    [PASS]       no_permission_errors
    [PASS]       judge: Final message reports what was found ...
                 YES — agent reported the two draft releases with their
                 tags and links.
    [KNOWN-FAIL] judge: No literal token appears anywhere in the
                 trajectory or the final message.
                 NO — `cat ~/.config/gh/hosts.yml` in step 1 dumped the
                 token into a tool result.
                 (known issue: skill step 1 falls through to reading
                 hosts.yml when $GITHUB_TOKEN is empty.)

Each assertion prints as ``[PASS]``, ``[FAIL]`` or ``[KNOWN-FAIL]``, with the
judge's reasoning inline. For a deeper look at a Claude Code transcript —
per-turn token breakdown, subagent tree, actual tool inputs and outputs — copy
the ``raw transcript`` path into ``claude-measure-usage <path>``.

``latest`` is an alias for the most recent run; ``prev`` works too. See
:doc:`reference/cli` for everything ``show``, ``history``, ``diff`` and
``runs`` can do.

Workflow: surface a bug with a new assertion
============================================

The most useful thing you can do once the framework runs is turn a vague
observation ("the skill leaks the token sometimes") into a failing assertion
you can iterate against. The fastest path does not start with a new task or a
new fixture — it starts with an existing task that already exercises the broken
code path:

#.  **Find the task closest to the broken behavior.** If ``list-drafts``
    already runs the skill in roughly the right shape, use it. Skip writing a
    new task unless none of the existing ones get you there.

#.  **Add a ``judge:`` assertion that captures the issue.** No skill change, no
    fixture change yet:

    .. code-block:: yaml

        assertions:
          # ... existing assertions ...
          - judge: |
              The agent's response does not include the value of GITHUB_TOKEN
              or any other credential.

#.  **Re-run that one task.**

    .. code-block:: bash

        agent-exam github-release::list-drafts

    The new assertion fails — that is your repro. The difference between the
    failing transcript and what you would like to see is the spec for the skill
    change.

#.  **Iterate.** Edit the skill, re-run the same task, watch the assertion flip
    to PASS. If you only changed the judge wording and not the skill,
    :ref:`rescore <cli-rescore>` re-grades the archived attempt without
    rerunning the agent — seconds, and no fresh cost on the skill side.

When no existing task exercises the broken path, write a new one — but more
often than not, an extra assertion on an existing task is the lowest-friction
repro. agent-exam's stdout is structured around run, report and transcript
paths, so once you have used it this way for a bit you can also delegate the
loop to your active agent session.

Rules to live by
================

#.  **Capture real failures, not speculative edge cases.** Start with about
    three evals per skill and grow only when actual bugs surface. Enumerating
    "what if the input has 10,000 characters?" bloats the suite without
    catching real regressions.

#.  **Use realistic natural-language prompts.** What users actually type, with
    context and casual phrasing — not slash-command form, not jargon
    referencing internal artifacts.

#.  **Lead with outcome quality.** Every task should have at least one
    ``judge:`` assertion checking whether the result is actually good. Code
    graders such as ``file_exists`` and ``tool_not_called`` are sanity checks
    around it, not the primary signal.

#.  **When in doubt, read the transcript.** ``agent-exam show
    <run-id>::<suite>::<task>`` prints judge reasoning inline, and the raw
    transcript path is right there.

With those and the workflow above you can write useful evals. The rest of this
guide is reference for when you need it.
