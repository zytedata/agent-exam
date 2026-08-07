======================
``config.yaml``
======================

Complete schema for :file:`evals/config.yaml`, and for its gitignored personal
override :file:`evals/config.local.yaml`. For where those files live and how
the override is merged, see :doc:`file-layout`.

Every field has a default, so the file can be empty, or absent altogether.

Top-level fields
================

``default_harness``
    The harness used when a run does not pass ``--provider``. Must match a key
    in ``providers``. Defaults to ``claude_code``.

``skills_dirs``
    Directories the skills under evaluation are staged from. See
    :ref:`skills-dirs` below.

``default_task_timeout_seconds``
    Per-attempt wall-clock timeout applied to every task, overridden per task
    by ``timeout_seconds:``. Defaults to 300.

``concurrency_groups``
    Named concurrency caps. See :ref:`concurrency-groups` below.

``providers``
    Per-harness configuration blocks. See :ref:`providers` below.

``judge``
    Configuration for the LLM judge. See :ref:`judge-config` below.

.. _skills-dirs:

``skills_dirs``
===============

The directories skills are staged from into each attempt's temporary
workspace. Paths are absolute, or relative to the project root:

.. code-block:: yaml

    skills_dirs:
      - ./skills

It defaults to :file:`skills/` at the project root when that directory exists,
so most projects can leave it out. When it resolves to nothing — no
:file:`skills/` directory, nothing configured, and no pre-run hook supplying it
— the runner raises an error.

Each :file:`<dir>/<skill-name>/` is copied under the harness's own skill
discovery path, so the agent loads them without a plugin manifest:

.. list-table::
   :header-rows: 1

   * - Harness
     - Discovery path
   * - ``claude_code``
     - :file:`.claude/skills/`
   * - ``codex_cli``
     - :file:`.agents/skills/`
   * - ``copilot_cli``
     - :file:`.github/skills/`
   * - ``opencode``
     - :file:`.opencode/skills/`

Real directory copies are used rather than symlinks, because some harnesses'
bash sandboxes cannot traverse symlinks that point outside the working tree.

Pre-run hook
============

Skills that have to be *built* before they can be evaluated — compiled into
harness-specific variants, assembled from templates — need the build to happen
inside the run. That is what the pre-run hook is for, and it keeps agent-exam
out of your build tooling.

Register it in :file:`pyproject.toml`:

.. code-block:: toml

    [tool.agent-exam]
    pre_run_hook = "evals.hooks:pre_run_hook"

The callable takes a single ``PreRunRequest`` and returns a ``PreRunResult``,
or ``None`` to skip any override. ``PreRunRequest`` carries one field,
``harness``, naming the harness about to run; ``PreRunResult`` carries
``skills_dirs``, the directories to stage from for this run:

.. code-block:: python

    from pathlib import Path

    from agent_exam.config import PreRunRequest, PreRunResult


    def pre_run_hook(request: PreRunRequest) -> PreRunResult:
        build_skills(harness=request.harness, out="build/skills")
        return PreRunResult(skills_dirs=[Path("build/skills")])

Both the runner and ``doctor`` invoke it, so ``doctor`` sees the same skills a
run would. The value it returns overrides ``skills_dirs`` from
:file:`config.yaml`, but not one set in :file:`config.local.yaml`.

.. _providers:

``providers``
=============

Each key is a harness identifier. Harnesses that are not selected for a run may
still be present, but the fields inside each block are strict: an unknown field
name fails config validation.

Common fields
-------------

``default_model``
    Model id, or alias, sent to the harness when ``--model`` is not passed.

``judge_model``
    Model used by ``judge:`` and ``judge_agent:`` assertions on this harness.
    When unset, the harness's own default model is used where supported;
    configure it explicitly for stable judge behavior.

``model_aliases``
    Short alias to full model id. Used to resolve both ``--model`` and
    ``default_model``.

``extra_args``
    Additional CLI flags appended verbatim to every harness invocation.

``claude_code``
---------------

``permission_mode``
    Default permission mode. One of ``auto``, ``bypassPermissions``,
    ``acceptEdits``, ``dontAsk``, ``default`` or ``plan``. Omit it to leave the
    flag off entirely. A task's own ``permission_mode:`` overrides this.

``blocked_plugins``
    Plugin names whose presence in the active Claude Code session would
    invalidate eval results — typically a plugin shipping the same skills you
    stage from ``skills_dirs``. Claude Code loads user-enabled plugins
    additively, so the plugin's copy would load alongside the staged one and
    ``--without-skill`` could not exclude it. The runner warns at run start if
    a blocked plugin is enabled, and ``doctor`` warns if one appears in the
    loaded skill listing.

``opencode``
------------

``pure``
    Run OpenCode with ``--pure``, which disables external plugins. Defaults to
    ``true``, and keeps evals hermetic the way ``blocked_plugins`` does from
    the other direction.

``codex_cli``
-------------

``writable_roots``
    Extra paths every task's ``workspace-write`` sandbox may write to; Codex
    expands ``~``. Use it for tools with home-directory caches — uv fails on
    :file:`~/.cache/uv` inside the default sandbox, which trips
    ``no_permission_errors``. A task-level ``writable_roots`` replaces this
    list; tasks using permission profiles ignore it.

``network_access``
    Default for ``sandbox_workspace_write.network_access`` across all tasks.
    Unset means Codex's own default, which is off — but that makes Codex runs
    stricter than Claude Code and OpenCode, which run with full network, so
    setting it to ``true`` is usually what you want. A task-level
    ``network_access`` overrides it, and judge invocations are unaffected since
    they are read-only with network off.

Codex runs with fixed headless defaults: ``--ask-for-approval never``,
``--ignore-user-config``, ``--ignore-rules``, and the ``workspace-write``
sandbox. Two things worth knowing:

- ``--ignore-user-config`` does not remove Codex's user-level skill roots, so
  the provider warns when skills under :file:`$CODEX_HOME/skills` or
  :file:`~/.agents/skills` clash with the staged ones.

- Codex session files are what completed-run transcripts are built from. The
  stdout stream drives real-time trigger detection, but it can omit tool calls
  and is not treated as authoritative.

``copilot_cli``
---------------

No harness-specific fields beyond the common ones; the per-task
``allowed_tools`` is the knob that matters. Three things worth knowing:

- Model names use dots (``claude-sonnet-4.6``), not hyphens.
- ``cost_usd`` is always null, displayed as ``?``, because Copilot CLI does not
  report cost. Output token counts are tracked per turn, and
  ``metrics.raw["premium_requests"]`` holds the number of premium LLM requests
  the session consumed.
- Skills are staged for walk-up discovery, so no plugin manifest is required.

.. _judge-config:

``judge``
=========

``timeout_seconds``
    How long a single ``judge:`` call may take before it is aborted. Defaults
    to 60.

``agent_timeout_seconds``
    The same, for ``judge_agent:``, whose multi-turn tool loop needs more
    headroom. Defaults to 300. Tune it up when your criteria need many tool
    calls, and down to fail faster on stuck judges.

``include_trajectory``
    Global default for whether judges see the full transcript, overridden per
    assertion. Defaults to ``true``.

``pass_on``
    The judge's verdict is compared against this list with a case-insensitive
    prefix match, and the assertion passes when it starts with any entry.
    Defaults to ``["YES"]``.

.. _concurrency-groups:

``concurrency_groups``
======================

Named caps for tasks that share a limited external resource, such as a live API
account. Each value is the maximum number of tasks in that group that may run
in parallel within a single run:

.. code-block:: yaml

    concurrency_groups:
      github_api: 1

Tasks opt in with ``concurrency_group:`` in their YAML.

A full example
==============

:file:`examples/config.yaml` in the repository is an annotated file exercising
every section:

.. literalinclude:: ../../examples/config.yaml
   :language: yaml
