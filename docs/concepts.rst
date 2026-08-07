========
Concepts
========

A short tour of the vocabulary used throughout the rest of this guide.

Suites
======

A **suite** is the collection of evaluation tasks for one skill. Each suite is
a directory under :file:`evals/suites/<name>/`, conventionally named after the
skill it evaluates.

Tasks
=====

A **task** is one test, defined as a YAML file under the suite's
:file:`tasks/` directory. A task has:

- A **kind** — either ``execute`` (run the skill end to end and grade the
  outcome) or ``trigger`` (stop after the first skill invocation and only check
  which skill fired).

- A **prompt** — what the agent is asked to do.

- A list of **assertions** — checks on what the agent did and what it produced.
  They can look at the response text, the transcript (which tools were called,
  in what order, with what arguments) and the working directory the skill wrote
  to. Two families: *deterministic* (``tool_called``, ``file_exists``,
  ``no_permission_errors``, …) and *judge* (a free-form criterion graded by an
  LLM — ``judge`` for transcript-only, ``judge_agent`` for criteria that need
  to read the working directory).

- An optional **fixture** — a directory under :file:`evals/fixtures/` copied
  into the attempt's working directory before the agent starts. Reusable across
  tasks.

Runs and attempts
=================

A **run** is one invocation of ``agent-exam``. A run executes one or more tasks
— a single task with ``<suite>::<task>``, a whole suite with ``<suite>``, every
suite with ``*``. For each task it produces one or more **attempts**: ``-k 3``
makes three attempts per task to expose non-determinism. Run artifacts land
under :file:`evals/runs/<run-id>/`.

The runtime that actually executes the skill is the **harness** — Claude Code,
Codex CLI, Copilot CLI or OpenCode. agent-exam stages the same skill bundle
into each harness's project-skill discovery path and normalizes the resulting
transcript. You will see the word *provider* in a few literal YAML and config
keys; that is the internal name for the adapter that talks to a harness, so
read it as "harness".

Each attempt saves two things:

- A **transcript** — the per-turn record of what happened: user and assistant
  turns, tool calls, tool results. Stored in a normalized form so assertions
  and judges work the same on every harness, alongside a path to the harness's
  raw form.

- The **working directory** — the filesystem state the skill wrote to during
  the attempt, archived as-is. This is what ``file_exists`` and
  ``file_contains`` assertions read from.

Eval results
============

Each attempt ends with one of ``pass``, ``fail``, ``timeout``, ``error``,
``known_issue`` or ``unexpected_pass``. The full results of a run — outcomes
plus per-assertion pass/fail — land in JSON under
:file:`evals/runs/<run-id>/reports/`.

Two outcomes are worth calling out:

``known_issue``
    Marks an assertion, or a whole task, as expected to fail today — a known
    skill bug you have not fixed yet. It does not gate the suite. See
    :ref:`assertion-meta-fields`.

``unexpected_pass``
    Flips up when a ``known_issue`` assertion starts passing; a hint that the
    annotation can be removed. Informational.
