===================
Task YAML reference
===================

Complete schema for the task files in :file:`evals/suites/<suite>/tasks/`. Two
task kinds: ``execute`` (run the skill end to end and grade the outcome) and
``trigger`` (only check which skill fired). Most fields apply to both.

For design principles, see :doc:`../design/suite-design`,
:doc:`../design/judges`, :doc:`../design/triggers` and
:doc:`../design/network`.

Top-level fields
================

These appear at the top of any task file.

``kind``
    ``execute`` or ``trigger``, defaulting to ``execute``. Determines which
    other fields apply.

``description``
    Free-text description. Not graded; it is there for the reader.

``setup.fixture``
    Name of a directory under :file:`evals/fixtures/`, whose contents are
    copied into the attempt's working directory before the agent starts. Works
    for both kinds; fixtured triggers forfeit the :ref:`shared-cwd cache
    optimization <trigger-dispatch>`.

``setup.env``
    Environment-variable overrides for this task. String values set or
    override; ``null`` removes the variable from the inherited environment.

``concurrency_group``
    Name of a concurrency limiter declared in :file:`evals/config.yaml`. Tasks
    in the same group serialize within a run, up to that group's cap.

``timeout_seconds``
    Per-attempt wall-clock timeout, overriding
    ``default_task_timeout_seconds``.

``known_issue``
    Marks the whole task as an expected failure. See
    :ref:`assertion-meta-fields`.

``<harness>``
    A harness-specific block, e.g. ``claude_code:``. See
    :ref:`harness-specific-blocks`.

Execute-task fields
===================

``prompt`` (required)
    The natural-language prompt sent to the agent as the first user turn.

``assertions`` (required)
    The checks that grade the attempt. Each entry is one of the
    :ref:`assertion types <assertion-types>`, optionally carrying
    :ref:`meta-fields <assertion-meta-fields>`.

Trigger-task fields
===================

``skill`` (required)
    The skill expected to fire, or expected not to, depending on which list the
    case appears in.

``positive``
    User prompts that should fire ``skill``.

``negative``
    User prompts that should not fire ``skill``.

At least one of ``positive`` and ``negative`` must be non-empty. Each entry is
a plain prompt string; per-case structural overrides are not supported, though
a file-level ``setup.fixture:`` applies to every case.

``assertions`` is not used on trigger tasks — the framework generates the
appropriate assertion per case, ``first_skill`` for positives and
``skill_not_invoked`` for negatives.

.. _assertion-meta-fields:

Assertion meta-fields
=====================

These wrap an assertion entry without changing what it checks.

``known_issue``
    Marks an expected failure. The assertion is still evaluated, but its result
    is excluded from the task's aggregate verdict. It renders as
    ``[KNOWN-FAIL]`` while it fails as expected, and ``[UNEXPECTED-PASS]`` once
    it flips to passing. The same field at the file's top level marks a
    whole-task expected failure.

``providers``
    Restricts the assertion to specific harnesses. When the active harness is
    not in the list the assertion is skipped: it appears in the report with a
    dim ``[SKIPPED]`` tag and is excluded from the aggregate.

.. code-block:: yaml

    assertions:
      - judge: |
          The response does not leak the API token.
        known_issue: |
          Skill step 1 reads the CLI's own credentials file; tracking a fix.
      - tool_called: Bash
        providers: [claude_code]

.. _assertion-types:

Assertion types
===============

Every built-in assertion. Most accept either a scalar shorthand, when one
parameter is enough, or an explicit mapping.

.. _assertion-judge:

``judge``
---------

An LLM-graded criterion against the agent's response and, optionally, the
transcript.

.. code-block:: yaml

    - judge: The response reports what was released and where to find it.
    # or:
    - judge:
        criterion: The response does not include fabricated URLs.
        include_trajectory: false

``criterion`` (required)
    The plain-text criterion the judge evaluates.

``include_trajectory``
    Whether the judge sees the full (truncated) transcript, or only the final
    response. Defaults to ``true``.

``judge_agent``
---------------

Like ``judge``, but the judge runs against a copy of the attempt's archived
working directory, with the harness's read-only file tools exposed. Use it when
the criterion needs to inspect generated files — comparing two artifacts,
checking that a value in one file matches a value in another, validating
structure beyond what a single regex captures.

.. code-block:: yaml

    - judge_agent: |
        The version in CHANGELOG.md matches the version in pyproject.toml.
    # or:
    - judge_agent:
        criterion: |
          The generated release notes cover every merged pull request.
        include_trajectory: false

``criterion`` (required)
    The plain-text criterion the judge evaluates.

``include_trajectory``
    Whether the judge sees the full transcript in addition to the working
    directory. Defaults to ``true``.

It costs more than ``judge`` — a multi-turn tool loop rather than a one-shot
call — and its cache hits are scoped per working directory. See
:doc:`../design/judges` for when it is worth it.

``file_exists``
---------------

Checks that a file exists in the attempt's working directory.

.. code-block:: yaml

    - file_exists: dist/notes.md
    # or:
    - file_exists:
        path: dist/notes.md

``path``
    Path relative to the attempt's working directory.

``file_contains``
-----------------

Checks that a file contains a substring, or matches a regular expression.

.. code-block:: yaml

    - file_contains:
        path: CHANGELOG.md
        pattern: '## 2\.1\.0'
        regex: true

``path`` (required)
    Path relative to the attempt's working directory.

``pattern`` (required)
    Substring, or regular expression, to look for.

``regex``
    Treat ``pattern`` as a regular expression. Defaults to ``false``.

``tool_called``
---------------

Checks that a tool was called at least once, anywhere in the trajectory,
including in subagents.

.. code-block:: yaml

    - tool_called: Bash
    # or:
    - tool_called:
        name: Bash

``name``
    Tool name. These are harness-specific, so usually pair this with
    ``providers:``.

``tool_not_called``
-------------------

The inverse of ``tool_called``, with the same config shape.

.. code-block:: yaml

    - tool_not_called: WebFetch

``tool_count``
--------------

Checks how many times a tool was called.

.. code-block:: yaml

    - tool_count:
        name: Bash
        exactly: 2
    # or:
    - tool_count:
        name: WebFetch
        min: 1
        max: 5

``name`` (required)
    Tool name.

``exactly``
    Required count. Mutually exclusive with ``min`` and ``max``.

``min`` / ``max``
    Inclusive bounds.

``first_skill``
---------------

Asserts that the first skill invocation in the trajectory matches the given
name. This is what trigger evals generate for their positive cases.

.. code-block:: yaml

    - first_skill: github-release
    # or:
    - first_skill:
        skill: github-release

``skill``
    Expected skill name.

``skill_not_invoked``
---------------------

Asserts that a skill was never invoked anywhere in the trajectory. This is what
trigger evals generate for their negative cases.

.. code-block:: yaml

    - skill_not_invoked: github-release

``skill_invoked``
-----------------

Asserts that a skill was invoked at least once anywhere in the trajectory,
subagents included. The inverse of ``skill_not_invoked``.

.. code-block:: yaml

    - skill_invoked: github-release

**Behavior in reality-check runs.** A reality-check run drops skills from the
bundle, so a literal ``skill_invoked: <dropped-skill>`` would always fail and
add noise to the report. To stay useful, the assertion flips when the asserted
skill is one the run excluded:

.. list-table::
   :header-rows: 1

   * - Run mode
     - Skill invoked
     - Skill absent
   * - Normal
     - **pass**
     - fail
   * - Reality check, asserted skill excluded
     - fail (it leaked in somehow)
     - **pass** (removal worked)
   * - Reality check, asserted skill not excluded
     - **pass**
     - fail

Under ``--no-skills`` every skill is excluded, so every ``skill_invoked``
assertion takes the flipped column. The flip only affects assertions naming an
excluded skill.

``no_permission_errors``
------------------------

Asserts that no tool call in the trajectory was blocked by the harness's
permission system. It catches skill-side bugs that would nag a real user with
approval prompts. No tunables.

.. code-block:: yaml

    - no_permission_errors:

.. _harness-specific-blocks:

Harness-specific blocks
=======================

A top-level mapping named after a harness holds configuration that is only
meaningful for that harness: ``claude_code:``, ``codex_cli:``, ``copilot_cli:``
and ``opencode:``. Unknown keys inside one fail at task-load time, which
catches typos like ``allow_tools``.

The literal YAML keyword ``providers:`` used in :ref:`meta-fields
<assertion-meta-fields>` refers to the same thing — "provider" is the internal
name for the harness adapter.

``claude_code``
---------------

.. code-block:: yaml

    claude_code:
      allowed_tools:
        - "Bash(gh*)"
        - "Bash(curl*)"
      permission_mode: auto

``allowed_tools``
    Patterns forwarded to ``claude --allowed-tools``, such as ``Bash(curl*)``
    or ``WebFetch(domain:books.toscrape.com)``. Anything outside the list goes
    through normal approval, which auto-rejects in headless runs. The ``Skill``
    tool is appended automatically when the list is non-empty: without it the
    agent's natural-language route to a skill fails, because the
    skill-invocation redirect comes back as an error and the agent treats it as
    a real one. Leave the list unset to fall back to Claude Code's default of
    all tools available.

``permission_mode``
    One of ``auto``, ``bypassPermissions``, ``acceptEdits``, ``dontAsk``,
    ``default`` or ``plan``. Defaults to the harness config in
    :file:`evals/config.yaml`, except on trigger tasks, which default to
    ``bypassPermissions``.

``opencode``
------------

.. code-block:: yaml

    opencode:
      permission:
        bash:
          "*": "ask"        # block all bash (auto-rejects under opencode run)
          "gh *": "allow"   # except gh commands

``permission``
    Per-tool permission rules, injected through ``OPENCODE_CONFIG_CONTENT``.
    Tool names are lowercase (``bash``, ``edit``, ``read``, …), and each entry
    is either a bare action or a mapping of pattern to action.

Valid actions are ``allow`` (approve), ``ask`` (prompt the user, which
**auto-rejects under** ``opencode run`` and so is equivalent to blocking the
tool) and ``deny`` (forbidden, with the model seeing the rule in the error). A
bare ``"deny"`` on ``bash``, ``edit``, ``write`` or ``read`` can hang OpenCode
in headless mode and is rejected at load time; use the mapping form instead,
e.g. ``bash: {"*": "deny"}``.

Without a ``permission`` key the task inherits OpenCode's defaults: most tools
allow, while ``external_directory`` and ``doom_loop`` ask, and therefore
auto-reject in headless runs.

.. _codex-cli-task-block:

``codex_cli``
-------------

.. code-block:: yaml

    codex_cli:
      network_access: true
      sandbox: workspace-write
      prefix_rules:
        - pattern: ["gh", "release"]
          decision: allow
          justification: "Release command under test"

``sandbox``
    Per-task Codex sandbox mode: ``read-only``, ``workspace-write`` or
    ``danger-full-access``. Defaults to ``workspace-write``.

``network_access``
    Per-task override for ``sandbox_workspace_write.network_access``. Set it to
    ``false`` to test offline behavior.

``writable_roots``
    Extra paths the ``workspace-write`` sandbox may write to; Codex expands
    ``~``. This *replaces* the provider-level default rather than extending it.
    Needed for tools with home-directory caches — uv, for instance, needs
    :file:`~/.cache/uv`.

``prefix_rules``
    Task-local Codex execpolicy rules. Each has a ``pattern`` (a list of
    command tokens, where a nested list means alternatives at that position),
    an optional ``decision`` (``allow``, ``prompt`` or ``forbidden``, defaulting
    to ``allow``) and an optional ``justification``. They apply to command
    requests outside the sandbox, not to the visible tool inventory.

Tool assertions run against the normalized trajectory and should be
Codex-scoped:

.. code-block:: yaml

    - tool_called: command_execution
      providers: [codex_cli]

Normalized tool names:

``command_execution``
    All of Codex's shell-exec variants collapse to this one name, so a shell
    assertion is stable regardless of which variant a model or Codex version
    emits.

``web_search``
    Codex's native web search, with the query or URL in ``input``. The tool
    must be enabled; it is off in restricted and judge runs.

``image_generation``
    Codex's image tool. The base64 result is elided.

Custom and freeform tools, and MCP-server tools, keep their own names.
``spawn_agent`` and ``wait`` cover subagent spawning and waiting.

``copilot_cli``
---------------

.. code-block:: yaml

    copilot_cli:
      allowed_tools:
        - bash
        - write

``allowed_tools``
    Exact tool names passed to ``--available-tools`` and ``--allow-tool``. Only
    those tools, plus ``skill`` and ``report_intent``, are visible to the model
    — anything else is hidden entirely, with no prompt and no hang. When the
    list is absent the harness passes ``--allow-all-tools``, so headless runs
    never block on permission prompts.

Copilot CLI tool names are lowercase plain identifiers (``bash``, ``read``,
``write``, ``web_search``, …) — no glob patterns and no parenthesised
sub-commands like Claude Code's ``Bash(curl*)``.
