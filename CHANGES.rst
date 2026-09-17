=======
Changes
=======

0.2.1 (unreleased)
==================

-   ``doctor`` and the runner now check that the tools ``kind: trigger``
    tasks target exist, by asking each MCP server the tasks attach for its
    tools. A target no attached server serves is a FAIL, with the closest
    names suggested, and a run refuses to start on one; a server that cannot
    be asked is a warning, and the targets it may serve go unchecked.

    This replaces the guess from the name alone, which failed a bare
    ``mcp_tool:`` whenever it began with an attached server's name — every
    tool named after its server, such as ``notion-search`` on a server called
    ``notion``, tripped it.

0.2.0 (2026-08-17)
==================

-   Added tags, so that a run covering every suite can leave the costly evals
    out.

    Declare every tag in :file:`evals/config.yaml`, marking the ones to skip
    with ``exclude_by_default: true``; suites wear them through ``tags:`` in
    :file:`suite.yml`, tasks through ``tags:`` in their own YAML. Naming a
    suite or a task in the run spec brings back what it is tagged as, and
    ``--tag``, ``--all-tags`` and ``--exclude-tag`` override the defaults
    outright. The header line and ``run.json`` report what a run left out.

-   Fixed a crash on Windows consoles, where printing reports, diffs or
    trajectories raised ``UnicodeEncodeError`` on the arrows and box characters
    they contain.

-   Judges now see more of each tool result, and are told that the elisions in
    an abridged trajectory hide content that was really there.

    Tool results were trimmed hard enough that the values a criterion asks
    about often fell inside the elided part, and judges read that absence as
    the agent not having done the work.

-   Judges now see the full tool input of each call, and keep the trajectory
    being graded when a long one has to be cut down.

    Tool inputs are the action under evaluation, such as a command line or the
    body of a written file, and 600 characters cut most of them off mid-value.
    A trajectory over the size limit now drops the bodies of subagent runs
    first, leaving a note of how many turns were omitted, and only truncates
    the parent turns if it still does not fit; the final report at the end of
    a run no longer disappears because a subagent filled up the budget.

-   Fixed a crash in the middle of a run when archiving the working directory
    of an attempt failed, e.g. because a killed agent left a symlink pointing
    nowhere behind.

    An attempt that fails unexpectedly is now reported as an error, with the
    reason on standard error, and the remaining attempts still run. Errors
    that apply to the whole run, such as an invalid configuration or an
    exhausted rate limit, keep aborting it.

0.1.0 (2026-08-12)
==================

Initial release.
