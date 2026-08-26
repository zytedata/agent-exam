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

``tags``
    Labels suites and tasks may wear, and which of them are skipped by
    default. See :ref:`tags` below.

``mcp_servers``
    MCP servers to attach to the agent under evaluation. See
    :ref:`mcp-servers` below.

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

.. _tags:

``tags``
========

Every tag a suite or a task may wear. A tag that is not declared here fails
validation, so a typo cannot quietly change what a run covers:

.. code-block:: yaml

    tags:
      expensive:      {exclude_by_default: true}
      remote-account: {exclude_by_default: true}
      network:        {}

``exclude_by_default``
    Keeps the tasks wearing this tag out of runs that cast a wide net.
    Defaults to ``false``, which leaves the tag a plain label — still useful
    with ``--exclude-tag``.

The name means nothing to agent-exam: ``expensive`` is not measured, it is
what you chose to call the tasks you do not want in every run.

Tasks wear tags through ``tags:`` in their own YAML, whole suites through
``tags:`` in :file:`suite.yml`; the two union.

What a run does with a default-excluded tag depends on how narrowly it asked:

.. list-table::
   :header-rows: 1

   * - Spec
     - Effect
   * - ``'*'``, or several suites
     - Every task wearing the tag is skipped.
   * - One suite
     - Skipped, except for the tags that suite itself declares — naming a
       suite asks for what that suite is, so a suite tagged ``expensive``
       runs, while a single ``remote-account`` task inside it stays out.
   * - ``<suite>::<task>``
     - Runs. Naming a task asks for that task.

``--tag``, ``--exclude-tag`` and ``--all-tags`` override all of it — see
:doc:`cli`.

.. _mcp-servers:

``mcp_servers``
===============

MCP servers attached to the agent under evaluation — for skills that depend on
one, and for evaluating a server's own tools and descriptions. Each entry is
the standard MCP JSON, so a block copies over from the server's README:

.. code-block:: yaml

    mcp_servers:
      files:
        command: mcp-files
        args: ["--root", "."]
        env:
          FILES_TOKEN: "${FILES_TOKEN}"
      tickets:
        type: http
        url: https://tickets.example.com/mcp
        headers:
          Authorization: "Bearer ${TICKETS_TOKEN}"

A remote server needs only its ``url``; ``type`` is ``http`` unless the server
speaks ``sse``.

``${VAR}`` in a stdio server's ``command`` or ``args``, or in an ``env`` or
``headers`` value, is substituted from the environment agent-exam itself runs
in, so credentials stay out of the file. A variable that is not set fails the
run at its start, before any trial — only for the servers that run's tasks
attach, so a credential is needed by the runs that use it and not by every
run. ``${PROJECT_ROOT}`` is a builtin, resolved to this project's own root
regardless of the environment — useful for a local stdio server that runs a
module out of this same checkout.

``codex_cli`` sends no header of its own — it reads a bearer token out of the
environment at launch — so ``Authorization: "Bearer ${VAR}"`` is the only
header it can carry, and a server needing any other has to be scoped away from
it with ``providers:``.

A server whose token comes from an OAuth 2.0 client credentials grant runs it
via ``oauth`` instead of a pre-obtained token in ``env``/``headers``:

.. code-block:: yaml

    mcp_servers:
      tickets:
        type: http
        url: https://tickets.example.com/mcp
        oauth:
          token_url: https://auth.example.com/oauth/token
          client_id: "${TICKETS_CLIENT_ID}"
          client_secret: "${TICKETS_CLIENT_SECRET}"
          env_var: TICKETS_TOKEN
        headers:
          Authorization: "Bearer ${TICKETS_TOKEN}"

The grant runs once per run, before the server it belongs to is resolved, and
the access token it returns is exported into ``env_var`` — the server's own
``env``/``headers`` then reference it with ``${VAR}`` like a token obtained
any other way. ``scope`` is optional.

Tasks attach every configured server unless they name a subset with their own
``mcp_servers:`` — see :doc:`task-yaml`. Definitions belong here rather than in
a task file because reports serialize task files verbatim.

Runs are hermetic: the servers configured here are the only ones the agent
sees, and the ones set up in the developer's own harness config are left out.

A per-task ``allowed_tools`` allowlist covers every attached server on its
own, since a list that named only native tools would deny the tools the task
is about.

A server that fails to connect leaves the agent silently without its tools, so
an attempt whose harness reports one is an ``error`` rather than a graded
result. The statuses land in the attempt's :file:`attempt.json`. ``claude_code``,
``copilot_cli`` and ``opencode`` each report them; ``codex_cli`` reports
nothing, and under it a server that never came up surfaces as the task
failing.

Assertions grade MCP tool calls through the ordinary ``tool_called``,
``tool_not_called`` and ``tool_count`` types, naming the tool as
``mcp__<server>__<tool>`` whichever harness ran, so a server name is limited
to letters, digits, ``-`` and ``_``, the last of which can neither be doubled
nor sit at either end.

A full example
==============

:file:`examples/config.yaml` in the repository is an annotated file exercising
every section:

.. literalinclude:: ../../examples/config.yaml
   :language: yaml
