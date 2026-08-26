=============
Trigger evals
=============

A *trigger eval* tests whether the right skill fires for a given
natural-language prompt, separately from whether it then executes correctly.
The runner kills the agent immediately after the first skill invocation, so
each trigger case is one short turn without skill execution.

Per case that is cheap, but a trigger file is many cases times many attempts,
and sweeps add up. The :ref:`batched, cache-friendly dispatch
<trigger-dispatch>` keeps the aggregate manageable; do not assume "trigger"
means "free".

Trigger evals use a **batched format**: one file per skill, containing many
short prompts, some marked should-trigger and some should-not. That is more
compact than one file per case, since trigger cases share most of their context
— just a prompt and the expectation.

Which skills need trigger evals
===============================

Not every skill is meant to be triggered by a user prompt. Some are
**implementation details** invoked by other skills. For those:

- **Skip positive trigger evals entirely** — no realistic user prompt should
  make them fire on their own.
- **Add negative-only trigger evals** if there is a risk of accidental
  triggering: prompts that look adjacent to the skill's domain should not fire
  it.
- Or skip trigger evals altogether if the skill is unambiguous and unlikely to
  be confused with anything user-facing.

The general question is who is expected to invoke the skill. If it is a user
typing prompts, you want full positive and negative coverage. If it is another
skill, positive triggers make no sense and negative triggers only matter when
there is a confusion risk.

Format
======

.. code-block:: yaml

    # evals/suites/github-release/tasks/trigger.yaml
    kind: trigger
    skill: github-release   # the skill expected to fire (or not)

    positive:
      # Cutting a release
      - Cut a release for this repo.
      - Ship v2.1 and publish the release notes.
      # Drafts
      - Which releases are still in draft?
      - Publish the draft release for 2.0.
      # Assets
      - Attach the built wheel to the latest GitHub release.

    negative:
      # Local git, not GitHub releases
      - Tag this commit as v2.1 and push the tag.
      - What changed since the last tag?
      # Changelog authoring (a different skill)
      - Write the changelog entry for this pull request.
      # Domain-overlap distractor
      - How do GitHub release assets differ from artifacts?

Each list entry is a plain string prompt. Per-case structural overrides such as
a fixture or a timeout are not supported; split into separate files if you need
different setup.

A file-level ``setup.fixture:`` is allowed and applies to every case. It is
useful when a skill's description tells the agent to read a file before
deciding to fire — reads are exempt from the routing early-kill. Fixtured
triggers forfeit the :ref:`shared-cwd cache win <trigger-dispatch>`.

.. code-block:: yaml

    kind: trigger
    skill: github-release
    setup:
      fixture: repo-with-changelog
    positive:
      - Ship it.
      - Cut the release.

Harness blocks work the same as on execute tasks; see
:ref:`harness-specific-blocks`.

Run trigger evals with ``-k 3`` or higher when checking for routing flakiness —
the default ``k=1`` gives one shot per prompt and can miss a skill that fires
nine times out of ten. During rapid iteration ``k=1`` is fine as a smoke test.

**Coverage target** for skills that get a trigger file: three to five positive
and three to five negative cases is a reasonable starting point. Grow when
triggering bugs actually appear, not pre-emptively. skill-creator's anchor of
eight to ten of each is reasonable once you are already hitting triggering
ambiguities.

The negatives are the valuable part — they tune description precision. Do
*not* use obviously irrelevant negatives ("what's the weather?"); they never
trigger and teach the description nothing. Save the negative slots for
near-misses that share real domain language.

Triggering a tool instead of a skill
====================================

Swap ``skill:`` for ``tool:`` when the routing decision under test is a tool
call — an :ref:`MCP server's <mcp-servers>` tool, typically, where the thing
being evaluated is the tool description rather than a skill of yours.

.. code-block:: yaml

    kind: trigger
    tool: mcp__files__search   # the tool expected to be called (or not)

    positive:
      - Find the invoice we sent in March.

    negative:
      - What does an invoice number look like?

Cases fan out as before, graded with ``first_tool`` and ``tool_not_called``.
Positive cases are about which tool the agent picks, so the first MCP call cuts
the attempt whichever tool it is, and a case that reaches for another server's
tool first fails — the tool-target counterpart of grading skills on
``first_skill``. Native tools are ignored throughout: an agent greps and reads
before deciding, and cutting on that would settle every case before the routing
decision is observable. For a negative case even an MCP call decides nothing —
the agent can call one tool and still reach for the target afterwards — so the
only decisive signal is the turn ending without the call.

Claude Code and Copilot CLI see a call announced before it runs; Codex CLI cuts
as it starts and OpenCode once it is over, so on those two the tool does run —
keep that in mind for a tool with side effects. Any tool the agent reaches for
on the way runs regardless.

Tool cases get the whole ``default_task_timeout_seconds`` rather than the
60-second trigger default. That default assumes a skill fires immediately;
here the agent looks around first, and an ``npx``-booted stdio server can spend
a good part of a minute just starting.

Author bias is a real problem
=============================

Twenty prompts from one author tend to look alike. A trigger suite biased that
way passes when the description matches the author's style and misses
descriptions that do not. Counter-measures, in order of preference:

- **Source from real transcripts** — your own sessions, user reports, support
  tickets. Highest signal, and not biased towards the eval author.
- **Have a second author review** the trigger file and add cases. Fresh
  phrasings catch what the original normalized away.
- **Use LLM-generated variants** of seed prompts as a starting point. They have
  biases of their own, so treat the output as a candidate list.

None of these is mandatory. Just know that "prompts I wrote in 20 minutes" is a
thin sample of how users actually phrase requests.

.. _trigger-dispatch:

How the framework runs them
===========================

Trigger evals get three optimizations on top of the normal attempt loop. They
are transparent — nothing to configure — but knowing they exist explains some
otherwise surprising behavior.

**Early termination scoped to the routing decision.** Positive cases kill the
agent the moment it invokes the target skill. Negative cases kill as soon as
the routing is observably elsewhere: either the agent reaches for a non-skill
tool, or its first turn ends with no skill fired. On Claude Code, ``Read`` is
exempt, because some skills inspect files before routing. Codex CLI exposes
skill use as an announcement and a read of
:file:`.agents/skills/<name>/SKILL.md`, and its provider uses those stream
signals for the same early kill. Either way the skill body normally does not
execute, and you pay for one short routing turn. The negative half of this is a
skill-target optimization: with a ``tool:`` target, positives cut on the first
MCP call and negatives run the turn out.

**Shared working directory across attempts, for fixtureless triggers.** All
fixtureless trigger attempts in a run share one working directory. Claude Code
embeds the working-directory path in its system prompt, so a shared directory
means a shared cache prefix and a sharp drop in input cost. Fixtured triggers
need isolation and get a per-attempt directory like execute tasks, forfeiting
the cache win.

**Batched dispatch in fixtureless trigger-only runs.** For each attempt index
*K*, the runner dispatches "attempt *K* of every task" as a single parallel
batch and waits for it before starting *K+1*. Across tasks the batches fan out
fully; within a task, attempt 2 hits the cache attempt 1 populated. This is
skipped for mixed runs and for runs containing fixtured triggers, where
per-task paths kill the shared prefix.
