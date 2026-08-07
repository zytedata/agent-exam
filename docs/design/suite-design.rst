============
Suite design
============

What good looks like at the suite level — sizing, coverage, prompt realism,
task clarity, and how the suite doubles as a no-skill baseline.

**Keep the suite small and grounded.** The goal is a safety net you run
routinely, not an exhaustive specification.

**Prompts must look like real user requests.** Anthropic's `skill-creator
<https://github.com/anthropics/skills/tree/main/skills/skill-creator>`_ puts it
well:

    Queries must be realistic and something a Claude Code or Claude.ai user
    would actually type. Not abstract requests, but requests that are concrete
    and specific and have a good amount of detail — file paths, personal
    context about the user's job or situation, column names and values, company
    names, URLs. A little bit of backstory. Some might be in lowercase or
    contain abbreviations or typos or casual speech.

Eval prompts should mostly read as prose, even when some users invoke skills
directly. Direct invocation is worth reserving for two cases: testing
implementation-detail skills that no realistic user prompt targets, and
debugging a specific execution path in isolation from triggering.

Building the suite
==================

**Per skill, start with about three execute tasks** covering the intended use
cases:

- **Happy paths** — what the skill is for, in a couple of realistic shapes.

- **Real-world failures** — bugs hit during development or reported by users,
  transcribed by hand from the failing session. Each becomes a regression test.

- **Behaviors the description explicitly implies** — if the skill says it asks
  for confirmation before doing something, a task exercising that is fair game.

Grow as new real failures accumulate. Ten or more tasks per skill should be the
*consequence* of that, not a target — enumerating cases pre-emptively builds
eval debt. Plus one trigger file per user-facing skill, with three to five
positive and three to five negative cases; see :doc:`triggers`.

Add a :file:`suite.yml` with ``evaluated_skills`` to make the skill-to-suite
mapping explicit, even for single-skill suites.

What stays out:

- **Speculative edge cases** — wait for an actual failure.
- **Behavior the skill does not claim.**
- **Redundant tasks** — if three tasks exercise the same fixture and code path,
  prune to one or two.

The pruning question is whether an eval pins a use case anyone actually cares
about. Failing evals usually earn their keep: they are either catching real
regressions or pointing at brittleness worth fixing. The pruning target is the
opposite — **evals that always pass and do not matter**, adding minutes to
every run without surfacing a signal. Treat wall time, cost and parallelism as
design constraints from the start: a suite too slow to run routinely stops
getting run.

Task clarity
============

    Getting task quality right is harder than it seems. A good task is one
    where two domain experts would independently reach the same pass/fail
    verdict. Could they pass the task themselves? If not, the task needs
    refinement. Ambiguity in task specifications becomes noise in metrics. The
    same applies to criteria for model-based graders: vague rubrics produce
    inconsistent judgments.

— Anthropic, `Demystifying evals for AI agents
<https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>`_

Two practical tests when authoring:

Two-expert agreement
    Could two people grade the same attempt by hand and reach the same verdict?
    If not, the judge criterion or assertion is ambiguous and will flake when
    the model runs it.

Could you pass it yourself?
    If you could not write a passing response from the prompt and fixture
    alone, the task is under-specified — either the prompt is too vague or the
    assertions ask for something the input does not support.

No-skill baseline
=================

A useful side effect of writing realistic, user-style prompts: the same suite
doubles as a way to see what the agent does *without* the skill.
``agent-exam <suite> --without-skill`` runs everything with the suite's
configured skills removed from the bundle, and ``--no-skills`` removes the
whole library for a true plain-agent baseline.

It does not need to pass — that is not the point. The point is to understand
what the skill actually brings:

- Cases the bare agent already handles fine, where the skill may be
  over-claiming.
- Cases it handles worse, incorrectly, or expensively.
- The resource difference: does the skill reduce tokens, wall time and peak
  context, or add overhead?

Treat the no-skill comparison as a standard development check rather than a
niche tool. Tasks that lean on internal commands or private artifacts are
useless here — the bare agent does not recognize them, so the comparison says
nothing.
