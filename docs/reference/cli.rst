=============
CLI reference
=============

Every ``agent-exam`` subcommand and its flags. Run any of them with ``--help``
for the same information inline.

Conventions
===========

Suite spec
    ``<suite>`` runs every task in the suite. ``<suite>::<task>`` runs one
    task. ``<suite>::<task>::<n>`` runs a single fanned-out trigger case
    (0-based, positives first and then negatives — the ``<task>-<n>`` shown in
    reports). A suite of ``*`` stands for every suite that matches, so ``*``
    alone runs everything and ``*::trigger`` runs the trigger task of every
    suite that has one.

Run-id aliases
    ``runs``, ``show``, ``history``, ``diff`` and ``rescore`` all accept
    ``latest`` (the most recent run) and ``prev`` (the one before it) in place
    of a run id.

A bare ``agent-exam`` prints help. A first positional argument that is not a
reserved verb is treated as a suite spec and dispatched to ``run``, so
``agent-exam my-suite`` and ``agent-exam run my-suite`` are the same thing.

``run`` — execute tasks
=======================

.. code-block:: bash

    agent-exam <suite>                      # one suite, k=1, default model
    agent-exam <suite>::<task>              # one specific task
    agent-exam suite-a suite-b::task-x      # several specs at once
    agent-exam '*'                          # every suite

``--provider <name>``
    Harness to invoke. Defaults to ``default_harness`` in
    :file:`evals/config.yaml`. One of ``claude_code``, ``codex_cli``,
    ``copilot_cli``, ``opencode`` or ``dummy``.

``--model <id>``
    Model id, or an alias declared in ``model_aliases``. Defaults to the
    provider's ``default_model``.

``-k <int>``
    Attempts per task. Defaults to 1.

``-n <int>``
    Parallelism cap. Defaults to 10; use ``-n 1`` for a fully serial run.

``--without-skill``
    Reality-check mode: run the suite with the suite's evaluated skills removed
    from the bundle. Defaults to ``-k 1``, skips trigger tasks, and always
    exits 0. The skills to exclude come from ``evaluated_skills`` in
    :file:`suite.yml`, or the suite name when that file is absent.

``--no-skills``
    Reality-check mode with *every* skill removed, not just the suite's, so the
    harness runs as a plain agent. Same semantics otherwise. Mutually exclusive
    with ``--without-skill``.

``--no-mcp``
    Reality-check mode with the skills in place but no MCP server attached, so
    what the servers were doing for the suite shows up as the difference. Skips
    trigger tasks with a ``tool`` target, which have no MCP call left to make.
    Exits 0 whatever the verdicts. Refused when no server is declared, and
    mutually exclusive with the two above — a run that withholds both cannot
    say which of the two the difference is down to.

``--no-triggers``
    Skip ``kind: trigger`` tasks and run only execute tasks. Implied by the two
    skill-withholding modes; pass it explicitly on a normal run to produce a
    with-skill run over the same task set for comparison.

``--tag <name>``
    Include the tasks wearing a tag configured ``exclude_by_default``.
    Repeatable.

``--exclude-tag <name>``
    Drop the tasks wearing ``<name>``, whether or not the tag is excluded by
    default. Repeatable, and it applies to every spec, including a task named
    on the command line.

``--all-tags``
    Lift every default exclusion at once. ``--exclude-tag`` still applies.

A run skips the tasks wearing a tag configured ``exclude_by_default`` in
:file:`evals/config.yaml`, the more narrowly it asked the less of that applies —
see :ref:`tags` for the three cases. The header line reports what was skipped,
and so does ``config.tasks_excluded_by_tag`` in the run's :file:`run.json`.

``runs`` — list recent runs
===========================

.. code-block:: bash

    agent-exam runs

One row per run, newest first: run id, age, mode (``run``, ``without-skill``,
``no-skills`` or ``no-mcp``), the pass/fail/known-issue/timeout tally, total
cost and duration.

``--limit <int>``
    Maximum rows to show. Defaults to 20.

``show`` — inspect a run or attempt
===================================

.. code-block:: bash

    agent-exam show <run-id>                                    # run summary
    agent-exam show latest                                      # most recent
    agent-exam show <run-id>::<suite>                           # suite summary
    agent-exam show <run-id>::<suite>::<task>                   # every attempt
    agent-exam show <run-id>::<suite>::<task>::attempt-2        # one attempt

The spec defaults to ``latest``.

The summary table includes a ``Δ vs prev`` column comparing each attempt to the
previous run of the same task; it is empty on a first run. The per-task view
prints each assertion's verdict and reason, judge reasoning inline, and the
path to the raw harness transcript.

``--report <timestamp>``
    Inspect a specific report file instead of the latest one. Useful after
    ``rescore`` has written new reports.

``--no-issues``
    Do not list problematic tool calls (permission denials, rejections,
    errors).

``history`` — task trends across runs
=====================================

.. code-block:: bash

    agent-exam history <suite>::<task>    # this task's latest attempt, per run
    agent-exam history <suite>            # all tasks in the suite, per run

Medians and ranges across recent runs, with delta highlights. For trends across
every suite, use ``runs``.

``--limit <int>``
    Maximum rows to show. Defaults to 10.

``--all``
    Show every run that graded this scope, ignoring ``--limit``.

``diff`` — compare two runs
===========================

.. code-block:: bash

    agent-exam diff <run-a> <run-b>
    agent-exam diff prev latest

A text diff of verdict changes, metric deltas and grader changes.

``--report-a <timestamp>`` / ``--report-b <timestamp>``
    Pick a specific report within each run.

``--scope <spec>``
    Restrict the comparison to ``<suite>`` or ``<suite>::<task>``.

Both ``show`` and ``diff`` flag deltas above hard thresholds (±15% cost, ±20%
peak context, ±25% wall) as warnings even when assertions pass. Reality-check
runs are excluded from lift-style comparisons against normal runs.

.. _cli-rescore:

``rescore`` — re-grade archived attempts
========================================

.. code-block:: bash

    agent-exam rescore <run-id>                     # whole run
    agent-exam rescore <run-id>::<suite>            # one suite
    agent-exam rescore latest::<suite>::<task>      # one task

Re-grades archived attempts against the *current* assertion definitions,
without invoking the skill again. Each rescore writes a new
:file:`reports/<timestamp>.json` alongside the original; the latest report is
what ``show``, ``history`` and ``diff`` treat as current, and their
``--report`` flags reach the older ones.

What replay costs:

- Deterministic assertions are pure functions of the archived state, so they
  are free.
- Judge assertions need an LLM call per judge per attempt in scope, but
  verdicts are cached per run on ``(criterion, output, model)``. Unchanged
  judges are cache hits, so rescoring after changing one criterion only pays
  for that criterion, even when you rescore the whole run.

Rescoring works on reality-check runs too.

``doctor`` — preflight checks
=============================

.. code-block:: bash

    agent-exam doctor              # full checks
    agent-exam doctor --no-llm     # no token cost

Verifies the project root, the config, harness auth and the judge model, and
makes a real round-trip call to confirm the harness is reachable. Run it first
after installing.

It also validates every suite statically: each task's YAML parses, every
assertion's type and config are well formed, and every referenced fixture
exists on disk. The runner makes the same call and aborts before spending
tokens when a suite fails it.

If ``pre_run_hook`` is configured in :file:`pyproject.toml`, doctor invokes it
the same way the runner does, which is what projects that build their skills on
demand need. A "skills available" check then reports how many skills were
discovered across the configured ``skills_dirs``; it fails when the hook ran
but produced none, and warns when no ``skills_dirs`` are configured at all.

``--no-llm``
    Skip only the round-trip probe. Everything else still runs, so this is the
    cheap, fast form: no token cost, exit ``0`` or ``2`` for pass or fail.

``--provider <name>``
    Harness to check. Defaults to ``default_harness``.
