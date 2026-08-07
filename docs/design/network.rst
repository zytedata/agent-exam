=============================
Evals that touch the network
=============================

Some skills need real network access to be meaningful. If you can write a
useful eval without it, do; reserve network-using evals for the skills that
genuinely invoke remote services.

Remote websites
===============

**Do not write evals that hit arbitrary third-party websites.** Sites change,
they block traffic, and it is not polite to hammer them.

Point evals at a site meant for it, such as `books.toscrape.com
<https://books.toscrape.com>`_, both for requests the skill makes itself and
for anything the skill generates and then runs.

Such sites work for happy paths, or when you just need "a website", but they
are too simple for testing logic against a realistic complex page. For those
cases, save the page to a fixture and refactor the skill so the complex logic
is testable without network requests.

A third option is to replace network calls with cached responses transparently,
via a caching layer with committed responses. It is valid, but it needs careful
design to stay ergonomic and maintainable.

Remote services and accounts
============================

When a skill talks to a hosted service, every eval for it should hit **one
account, project or repository that you control and share across the team**.
Two things to never do:

- **Invent or guess resource identifiers.** A wrong digit can land your eval in
  someone else's real data. Use the shared identifier; if a new task needs a
  differently shaped resource, set one up first and use *that* identifier.

- **Point an eval at your personal sandbox.** A shared resource keeps results
  comparable across the team and means fixture-encoded identifiers behave the
  same regardless of who runs the suite.

Credentials come from the environment the run inherits, so log in with the
service's own CLI, or export its API key, before running. To exercise the
missing-credential path, unset the variable on the specific task:

.. code-block:: yaml

    setup:
      env:
        GITHUB_TOKEN: null

``null`` removes a variable from the inherited environment; a string value sets
or overrides it. The same mechanism works for any other variable, such as
forcing a log level or swapping an endpoint.

Two limitations are worth knowing about. Two people running the same evals at
the same moment can collide, because the concurrency cap below applies within a
run and not across runs — rerun if you see suspicious failures. And the
missing-credential path is only exercisable through ``setup.env``, since every
run otherwise sees a logged-in environment.

Concurrency groups
==================

Parallel operations against the same remote resource interfere with each other:
two deploys cannot both win cleanly, a listing sees a confused mix of in-flight
state. Declare a concurrency group on the task:

.. code-block:: yaml

    concurrency_group: github_api

and cap it in :file:`evals/config.yaml`:

.. code-block:: yaml

    concurrency_groups:
      github_api: 1

Tasks in that group then serialize within a run, so each one sees a clean
state.

.. _permission-mode:

Letting the agent run shell commands unattended
===============================================

Network-using evals typically need the agent to run shell commands without a
human approving each one. These settings are harness-specific, which is why
they sit under a per-harness block rather than at the task's top level.

**Default: allow only the specific tools the task needs.**

For Claude Code, use ``allowed_tools``:

.. code-block:: yaml

    claude_code:
      allowed_tools:
        - "Bash(gh*)"        # the CLI this eval needs

Patterns use Claude Code's own ``--allowed-tools`` syntax: ``Bash(curl*)``,
``WebFetch(domain:books.toscrape.com)``, and so on.

For OpenCode, use ``permission`` with pattern-to-action mappings. ``"ask"``
auto-rejects under ``opencode run``, so it is the headless equivalent of
blocking a tool:

.. code-block:: yaml

    opencode:
      permission:
        bash:
          "*": "ask"      # block all bash by default
          "gh *": "allow" # except the CLI this eval needs

Codex CLI has no per-command allowlist. agent-exam runs ``codex exec``
non-interactively with ``--ask-for-approval never``, so the shell tool always
runs unattended and nothing pauses for confirmation. What a command may *do* is
governed entirely by Codex's sandbox, and network is a sub-capability of the
``workspace-write`` sandbox. So there is no ``gh`` to allow — you just enable
network:

.. code-block:: yaml

    codex_cli:
      network_access: true

Leave it unset and the ``workspace-write`` default disables network: the
command still runs, but its requests fail, with no prompt. See Codex's `agent
approvals and security
<https://developers.openai.com/codex/agent-approvals-security>`_ documentation
for the upstream model, and :ref:`codex-cli-task-block` for the remaining
fields.

A task can carry blocks for several harnesses side by side. Only the block for
the harness actually running is used:

.. code-block:: yaml

    claude_code:
      allowed_tools:
        - "Bash(gh*)"
    opencode:
      permission:
        bash:
          "*": "ask"
          "gh *": "allow"
    codex_cli:
      network_access: true

For harnesses with per-tool allowlists, anything outside the allowed list is
blocked or hidden. For Codex, the sandbox and ``network_access`` are the
confinement boundary.

**Last resort, Claude Code only:** ``permission_mode: bypassPermissions``. Use
it only when the path under test genuinely needs unattended shell that cannot
be enumerated as patterns, and you have verified the commands are safe against
the target account:

.. code-block:: yaml

    claude_code:
      permission_mode: bypassPermissions

Set it per task, inside the ``claude_code:`` block, rather than globally.

Agent-decided fetches
=====================

Tools like ``WebFetch`` are agent-initiated: whether and where they are called
is part of the behavior under evaluation, so they are neither blocked nor
mocked. Pin the expected behavior with ``tool_called``, ``tool_not_called`` or
``tool_count`` assertions. If usage drifts — a skill suddenly fetching on every
run — the assertions surface it.
