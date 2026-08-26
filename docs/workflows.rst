=========
Workflows
=========

Day-to-day workflows during skill development: adding an eval, triaging
failures, reading what the skill actually did, iterating on graders without
rerunning the agent, acting on metric drift, the reality-check modes, and
caveats when iterating fast.

agent-exam's output is designed to be useful both to humans and to an agent
reading stdout — it includes the run directory, report paths, transcript paths
and structured assertion verdicts. So everything below also works as a one-line
delegation to your active agent session: *"run the github-release evals and fix
anything that is failing"*. The agent shells out, opens the failing reports,
follows the transcript paths, reads the skill source and proposes edits.

Adding an eval
==============

Two reasons to add a task:

- **Capture a real failure** so it does not regress — a bug hit during
  development, an issue reported by a user. The eval is a regression test.

- **Pin a new use case** the skill should support, usually paired with feature
  work on the skill itself. The eval defines what "supported" means; if it does
  not pass yet, that is the work to do.

Both follow the same loop:

#.  **Start from a concrete situation.** For a regression, the source is a
    transcript — a session where something went wrong. For a new use case, it
    is a description of what a user should be able to ask for. Transcribe the
    prompt and build the fixture by hand.

#.  **Write a realistic prompt** — what a real user would type. See
    :doc:`design/suite-design` for the principles.

#.  **Reuse a fixture if you can; build a minimal one if you cannot.** Many
    tasks within a suite share starting state. Reuse keeps fixture maintenance
    bounded and makes it obvious when something genuinely needs new state. A
    new fixture should hold just the input the skill actually reads, nothing
    extra. Fixtures live under :file:`evals/fixtures/<fixture-name>/` and are
    shared across suites, so give them descriptive, scoped names.

#.  **Draft assertions, leading with outcome quality.** Start with a ``judge``
    assertion on whether the result is actually good; every task should have at
    least one. Add code graders (``file_exists``, ``file_contains``,
    ``tool_not_called``) as sanity checks around it. See :doc:`design/judges`
    and :ref:`assertion-types`.

    Before committing a task, walk through `OpenAI's eval-skills 4-angle
    taxonomy <https://developers.openai.com/blog/eval-skills>`_ (outcome,
    process, style, efficiency). Efficiency is the angle most people forget — a
    check that the skill stops cleanly without retrying or looping. Any
    assertion type can cover any angle; the taxonomy is a thinking tool.

#.  **Verify the eval discriminates.** A passing assertion on broken code, or
    on a missing feature, measures nothing. Confirm the eval fails against the
    not-yet-correct state of the skill: for a regression test, check out the
    commit before the fix; for a new feature, run the eval before implementing
    it.

#.  **Iterate on flakiness.** Run ``-k 3``. If the verdict flips across
    attempts, tighten the assertion or criterion before committing. See
    :ref:`judge-flake` when the flake is a judge.

#.  **Commit task and fixture together.** Tasks referencing missing fixtures
    are noise in the diff.

Triage when an eval fails
=========================

When an assertion fails, one of four things is true:

.. list-table::
   :header-rows: 1

   * - Symptom
     - Cause
     - Fix
   * - Skill did the wrong thing
     - Genuine regression
     - Fix the skill
   * - Skill did the right thing, assertion rejects
     - Assertion too strict
     - Loosen it, or the expected output was wrong
   * - Skill behaved reasonably, judge says NO
     - Criterion ambiguous or judge off
     - Tighten the criterion, add a counter-example, or bump the judge model
   * - Pass and fail flip on reruns
     - Non-determinism
     - Investigate; ``-k 3`` and pass^k reveal it

Rule of thumb: judge disagreement above roughly 10% on the same task across
``-k 3`` attempts means the criterion needs tightening before you ship the
eval.

When the failure is in the assertion rather than the skill, edit the task YAML
and re-grade with :ref:`rescore <rescore>` — no need to rerun the agent.

Reading results
===============

Failures and surprising metric changes are only useful if you can see *why*.
Two surfaces matter.

**Transcript drill-down.** ``agent-exam show <run-id>::<suite>::<task>`` prints
each assertion with its verdict and reason, judge reasoning included. Reading
the transcript itself — per-turn assistant text, tool calls, token usage — goes
through one of two paths:

- ``claude-measure-usage`` for Claude Code transcripts: a per-turn table with
  token breakdowns and a subagent tree, the richest view available. Copy
  ``raw_transcript_path`` from ``show``'s output, or from :file:`attempt.json`,
  and run ``claude-measure-usage <path>``.

- The normalized :file:`trajectory.json` under
  :file:`evals/runs/<run-id>/artifacts/<suite>/<task>/attempt-N/`. Plain JSON,
  the same shape on every harness, readable by anything — including your
  current agent session. Codex CLI, Copilot CLI and OpenCode also archive
  their own machine stream next to it, as :file:`raw_stream.jsonl`.

**Judge reasoning.** Every judge verdict ships with its reasoning: inline in
the terminal on failure, inline next to each verdict in ``agent-exam show``,
and as ``details.reasoning`` in
:file:`evals/runs/<run-id>/reports/<timestamp>.json` for scripting.

.. _rescore:

Iterating on graders without rerunning the skill
================================================

When you are tweaking a judge criterion or a deterministic assertion, running
the full eval every time is wasteful — the skill's output has not changed, only
the grading has. :ref:`agent-exam rescore <cli-rescore>` re-grades archived
attempts against the *current* assertion definitions, without invoking the
skill.

#.  ``agent-exam <suite>`` — a real run, producing outputs and initial
    verdicts.

#.  Spot a too-strict or flaky assertion on one task, and edit the task YAML.

#.  ``agent-exam rescore latest::<suite>::<task>`` — re-grade that task's
    archived outputs under the new rules. Takes seconds.

#.  Repeat 2 and 3 until the verdicts match your judgment.

#.  When the graders are stable, ``agent-exam <suite>`` again to test the skill
    afresh.

Each rescore writes a new report alongside the original; the initial verdicts
stay intact.

Acting on metric changes
========================

Pass/fail is the loudest signal but not the only one. A change that keeps every
eval green while doubling tokens or wall time is still a regression in most
contexts.

Single runs are noisy
---------------------

Model outputs vary run to run. One-shot comparisons catch large swings —
parallel subagents no longer firing, task duration tripling — but smaller
deltas can be variance. Use ``-k 3`` to tell them apart:

- ``agent-exam <suite> -k 3`` — three attempts per case.
- ``agent-exam history <suite>::<task>`` — medians and ranges across recent
  runs, with delta highlights.
- The verdict summary reports both **pass@k** (at least one attempt passed) and
  **pass^k** (all of them did). A drop in pass^k while pass@k stays high means
  a reliability problem that single runs would hide.

``show`` and ``diff`` flag deltas above hard thresholds (±15% cost, ±20% peak
context, ±25% wall) as warnings even when assertions pass.

**Cost numbers reflect warm-cache behavior.** The k attempts of a task run
close in time, so all but the first hit a warm prompt cache — which does not
necessarily match real usage, a mix of cold first invocations and warm
follow-ups. Treat cost as good for *comparing runs*, not as an answer to "what
does this cost" in isolation. The ``tokens.cache_read`` metric shows how much
actually hit cache; a sudden drop means a cache-breaking prompt change.

Decision guide for metric deltas
--------------------------------

Justified by new functionality
    The skill now handles an edge case it did not before. Accept it and take a
    new baseline.

Unexplained
    Investigate. Longer prompt? Broken caching? Re-reading large files?

Variance
    Unchanged skill, metrics moving across attempts. Ignore it within a normal
    range; fix the non-determinism if the variance is high.

pass^k drops while pass@k stays
    A reliability problem — a borderline prompt, or a judge flipping.

Peak context climbing over time
    Documentation creep in the skill files. Worth catching early.

Nothing gates automatically; you decide.

Leaving the expensive evals out of a wide run
=============================================

Some tasks do not belong in every run: a suite graded by ``judge_agent`` at a
dollar a task, a task that mutates a live account, a task that hits the real
network. Tag them, and declare the tag ``exclude_by_default``:

.. code-block:: yaml

    # evals/config.yaml
    tags:
      expensive: {exclude_by_default: true}

.. code-block:: yaml

    # evals/suites/analyze-page-quality/suite.yml
    tags: [expensive]

A task wears its own tags the same way, with ``tags:`` in its YAML. Either way
a wide run now leaves those tasks out and reports how many it left. Asking more
narrowly brings them back: naming the suite runs what the suite is tagged as,
and naming a task runs that task:

.. code-block:: bash

    agent-exam '*'                        # every suite, cheap tasks only
    agent-exam '*' --tag expensive        # ... and the expensive ones too
    agent-exam '*' --all-tags             # ... and anything else excluded by default
    agent-exam analyze-page-quality       # the suite is the expensive one — run it
    agent-exam zyte::scrapy-cloud-deploy  # one tagged task out of an untagged suite
    agent-exam '*' --exclude-tag network  # drop a tag that is not excluded by default

The middle two differ in an easy-to-miss way: ``agent-exam zyte`` skips the
``remote-account`` tasks inside ``zyte``, because that tag is on the tasks
rather than on the suite. :ref:`tags` has the full table.

Reality check: running without the skill
========================================

``agent-exam <suite> --without-skill`` runs the suite with the suite's
configured skills removed from the bundle, to see what the agent does without
them. The skills to exclude come from ``evaluated_skills`` in
:file:`suite.yml`, or the suite name when that file is absent. Use it for:

- Deciding whether a new skill earns its place against the bare agent.
- Revisiting whether an old skill is still needed.
- Debugging — is the skill doing what you think, or would the agent figure it
  out anyway?
- Comparing resource usage. A skill that produces the same outcome at three
  times the cost is worth a look.

``--no-skills`` is the stronger variant: it removes **every** skill under
``skills_dirs``, not just the suite's. Reach for it when the skill under test
is part of a workflow — ``--without-skill`` leaves its callers and helpers
staged, so the agent can still be routed into skill-shaped behavior by whatever
remains. The two flags are mutually exclusive.

This is **not a routine check** and **not expected to pass**. It defaults to
``-k 1``, skips ``kind: trigger`` tasks (no skill, nothing to fire), and heads
the report with "REALITY CHECK — verdicts informational" to make clear that
pass/fail is not the point. ``show`` and ``diff`` exclude reality-check runs
from lift-style comparison against normal runs, since they answer a different
question.

To compare the two halves like for like, pass ``--no-triggers`` on the
with-skill run so it covers exactly the same task set:

.. code-block:: bash

    agent-exam <suite> --no-triggers      # with the skill
    agent-exam <suite> --without-skill    # without it

``--no-triggers`` on its own is just the trigger filter: normal run mode
otherwise, skills staged, verdicts still gating the exit code.

``--no-mcp`` asks the same question about the other half of the setup: the
skills stay staged, but none of the declared :ref:`MCP servers <mcp-servers>`
is attached, so whatever the servers were doing for the suite shows up as the
difference. Skill trigger tasks still run — routing does not need the tools it
routes to — while trigger tasks with a ``tool`` target are skipped, there being
no MCP call left to make. The run exits 0 like the other reality checks.

Caveats for fast loops
======================

Two things to watch when you — or an agent — iterate quickly across many
failing cases:

- **Do not over-fit to current failures.** Seeing every failing case at once
  makes it tempting to patch each one literally rather than catch the
  underlying intent the skill is supposed to encode. The effect is mild on a
  single iteration and compounds across many, and it is worst in the
  agent-driven loop, where the agent sees the whole failure set at once.

- **Review the skill as a whole at the end of the loop.** Once everything
  passes, read the skill end to end, not just the diff stream. Small per-edit
  changes can add up to a shifted intent that no individual diff flagged. Ask
  whether the skill still describes the intended behavior cleanly to an agent
  that has never seen it. If it reads like workarounds bolted on to satisfy
  particular evals, restructure — possibly with tighter judge criteria.
