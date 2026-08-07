from __future__ import annotations


def build_prompt(
    criterion: str,
    final_output: str,
    formatted_trajectory: str,
    include_trajectory: bool = True,
    *,
    inspect_cwd: bool = False,
) -> str:
    """Build the judge prompt from a criterion, final output, and trajectory.

    Order is trajectory → final message → criterion. The criterion
    intentionally comes *last* because it's the only section that varies
    across judges on the same trial — putting it at the end maximizes
    Claude's prefix-cache hit rate when a task has multiple judge
    assertions (the common case). For N judges, only criterion N is a
    cache miss after the first; trajectory + final message hit the cache.

    Changing the wording is fine, but any change that alters the reasoning
    Claude produces changes verdicts on unchanged criteria, so prefer
    stability.

    ``inspect_cwd`` is set by the ``judge_agent`` assertion — without
    an explicit directive judges tend to answer UNCLEAR when the
    criterion references a file and the trajectory doesn't quote its
    contents (the model sees the tools the harness exposes but
    doesn't know it's expected to use them for this verdict). The
    plain ``judge`` assertion leaves it ``False``.
    """
    parts = [
        "You are evaluating an AI agent's response against a criterion.",
    ]
    if inspect_cwd:
        parts.append(
            "You are running in the agent's working directory and have"
            " read-only file tools. When the criterion refers to a file"
            " or its contents, use the tools to read it before deciding."
        )
    if include_trajectory:
        parts.extend(
            [
                "",
                "AGENT'S TRAJECTORY:",
                formatted_trajectory or "(empty trajectory)",
            ]
        )
    parts.extend(
        [
            "",
            "AGENT'S FINAL MESSAGE:",
            final_output or "(empty)",
            "",
            "CRITERION:",
            criterion,
            "",
            "Respond with reasoning (1-2 lines citing specific evidence), then on",
            "a separate final line:",
            "VERDICT: YES | NO | UNCLEAR",
        ]
    )
    return "\n".join(parts)
