==============
Writing judges
==============

A ``judge:`` assertion is an LLM-evaluated criterion against the attempt. Two
shapes:

.. code-block:: yaml

    # one-liner — most judges look like this
    - judge: The response reports what was released and where to find it.

    # multi-line — when the criterion needs more room
    - judge: |
        The agent reused the repository name from the git remote instead of
        prompting the user for it.

The judge is a separate LLM call per assertion. It sees the criterion text, the
agent's final response in full, and the rest of the transcript — user turns,
assistant turns, tool calls and results. The transcript is included by default;
opt out with ``include_trajectory: false`` to grade only on the final response.

It does **not** see the working directory or the content of files the skill
wrote. For those, use a deterministic assertion (``file_exists``,
``file_contains``) when the check is a hard fact, or `judge_agent`_ when it
needs reading and interpretation across files.

The transcript is rendered to the judge with caps: 5,000 characters of text,
1,000 per tool result, 200 each for tool inputs and thinking, 50 KB in total,
middle-elided. If your criterion needs reasoning over a long tool output and
the judge complains about incomplete output, you have hit the cap.

For the YAML schema, see :ref:`assertion-judge`.

Judge vs deterministic assertions
=================================

Reach for a judge when the check needs reading and interpretation. That covers
two cases:

Outcome quality
    Does the response answer the question, is the generated schema sensible,
    does the file the skill wrote actually do what it should?

Approach quality
    Did the agent handle credentials safely, avoid a known pitfall, follow an
    instruction in the skill such as "ask before doing X"? Approach checks are
    about *how* the agent did it, not which specific tool it chose, and judges
    read intent off the transcript better than any deterministic check could.

Reach for a deterministic assertion when the check is a hard fact about the
transcript or the working directory.

.. list-table::
   :header-rows: 1

   * - Need to check
     - Use
   * - Output is correct, useful, well explained
     - ``judge``
   * - Approach is safe, or follows the skill's instructions
     - ``judge``
   * - Outcome quality of a generated file (compare files, validate structure)
     - ``judge_agent``
   * - Specific text appears in a file
     - ``file_contains``
   * - A file got created
     - ``file_exists``
   * - A tool was or was not called
     - ``tool_called`` / ``tool_not_called`` / ``tool_count``
   * - A skill fired, or did not
     - ``first_skill`` / ``skill_invoked`` / ``skill_not_invoked``
   * - The attempt tripped no permission prompts
     - ``no_permission_errors``

Lead each task with at least one judge on outcome quality; deterministic
assertions are sanity checks around it.

Split multi-facet criteria
==========================

If a criterion has several facets, write one judge per facet. You get isolated
verdicts in the report, which makes diagnosis easier when one facet fails:

.. code-block:: yaml

    - judge: The response lists every release it published.
    - judge: The response shows the URL of each published release.
    - judge: The response says which releases it left in draft, and why.

``judge_agent``
===============

When the criterion needs reading and interpretation **across the files the
skill produced** — comparing two artifacts, checking that a value in one file
matches a value in another, validating structure beyond what a regex captures —
use ``judge_agent``:

.. code-block:: yaml

    - judge_agent: |
        The version in CHANGELOG.md matches the version in pyproject.toml.

The judge runs against a copy of the attempt's archived working directory. The
read-only inspection surface depends on the harness:

- Claude Code: ``Read``, ``Glob``, ``Grep``.
- OpenCode: ``read``, ``glob``, ``grep``.
- Copilot CLI: ``view``, ``glob``, ``grep``.
- Codex CLI: no native read/glob/grep-only surface, so the judge runs with a
  read-only, no-network permission profile and ``command_execution``. It can
  still run arbitrary read-only shell commands inside that sandbox.

It costs more per verdict than ``judge`` — a multi-turn loop instead of a
one-shot call — so the rule of thumb holds: deterministic ``file_exists`` and
``file_contains`` when the check is a hard fact, plain ``judge`` when
transcript reasoning is enough, ``judge_agent`` only when the criterion truly
needs to interpret file content.

Cache invalidation is wider than ``judge``'s: the verdict cache also keys on
the working directory's content hash and the read-only tool list, so a change
to either evicts the entry on the next rescore.

.. _judge-flake:

Tighten on flake, do not escalate the model
===========================================

If verdicts flip across ``-k 3`` attempts, the criterion is ambiguous. Be more
specific about what counts, or anchor it with inline pass and fail examples:

.. code-block:: yaml

    - judge: |
        Reject suggestions that recommend running unverified code or disabling
        security checks.

        Example PASS: "Run the suggested commands after reviewing them."
        Example FAIL: "Run `curl … | sh` to install."

Calibration wording dwarfs model capability here.
