==========
agent-exam
==========

agent-exam grades agent skills. You give it a library of skills and a set of
YAML tasks; it runs each task against a real agent CLI — Claude Code, Codex
CLI, Copilot CLI or OpenCode — and reports per-assertion verdicts alongside
cost, token and wall-time metrics.

The harnesses run as subprocesses against your existing subscription or login,
so no pay-per-token API key is involved.

Why evals
=========

You cannot develop a skill if you cannot tell whether it is getting better or
worse. Without evals you tweak a :file:`SKILL.md`, the case in front of you
starts working, and you ship — without knowing whether you broke three other
cases, whether the skill still triggers on the right prompts, or whether the
new phrasing doubled token cost.

Manual dogfooding catches the loud failures and misses everything else. Skill
behavior is non-deterministic, so one good run is not a green light, and "feels
worse after my edit" is not a regression report. The moment you start changing
skills based on real user feedback, you need a frozen set of tasks you can
re-run on every change and read the diff off.

Evals also encode what "good" means. Two people reading the same
:file:`SKILL.md` often disagree about edge-case behavior; a task that fails on
the disputed case forces the question into the open.

Cross-harness portability adds another reason. "I tested it in Claude Code"
tells you nothing about how the skill behaves in Codex CLI. The only realistic
way to keep skills portable is to run the same task suite on every harness you
care about and compare the results.

The cost of evals is upfront — writing them, maintaining them. The benefits
compound: every regression caught automatically, every model swap shipped in a
day, every "did this prompt change save tokens?" question answered concretely.

Intended use
============

Interactive skill development: edit a skill, run the relevant suite or a single
task, read the report, iterate. Automating runs against subscription-backed
CLIs may stretch a harness's terms, and LLM evals are slow and expensive enough
that running every suite on every change buys little extra signal.

.. toctree::
   :caption: Getting started
   :hidden:

   quick-start
   concepts
   workflows

.. toctree::
   :caption: Design
   :hidden:

   design/suite-design
   design/triggers
   design/network
   design/judges

.. toctree::
   :caption: Reference
   :hidden:

   reference/cli
   reference/task-yaml
   reference/config-yaml
   reference/file-layout

.. toctree::
   :caption: All the rest
   :hidden:

   changelog

:doc:`quick-start`
    Install, wire :file:`evals/` into your repo, run a task, read the report.

:doc:`concepts`
    Suite, task, assertion, fixture, run, attempt, harness, transcript.

:doc:`workflows`
    Adding evals, triaging failures, rescoring, acting on metric drift.

:doc:`design/suite-design`
    What good looks like: sizing, coverage, prompt realism, task clarity.

:doc:`design/triggers`
    Testing whether the right skill fires for a prompt.

:doc:`design/network`
    Credentials, shared services and per-harness permission settings.

:doc:`design/judges`
    What ``judge:`` and ``judge_agent:`` see, and what good criteria look like.

:doc:`reference/cli`
    Every subcommand and flag.

:doc:`reference/task-yaml`
    The complete task schema.

:doc:`reference/config-yaml`
    Every :file:`evals/config.yaml` field.

:doc:`reference/file-layout`
    What lives on disk, in and out.
