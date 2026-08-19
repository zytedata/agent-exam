from __future__ import annotations

import sys

import click

from .commands import diff as diff_cmd_impl
from .commands import doctor as doctor_cmd_impl
from .commands import history as history_cmd_impl
from .commands import rescore as rescore_cmd_impl
from .commands import runs as runs_cmd_impl
from .commands import show as show_cmd_impl
from .config import load_config
from .errors import AgentExamError
from .runner import RunRequest, run

RESERVED_VERBS = {"run", "runs", "show", "history", "diff", "rescore", "doctor", "ui"}


def _parse_suite_spec(spec: str) -> tuple[str, str | None]:
    if ":" in spec and "::" not in spec:
        raise click.UsageError(f"use '::' to separate suite and task (got {spec!r})")
    if "::" in spec:
        parts = spec.split("::")
        if len(parts) not in (2, 3) or not all(parts):
            raise click.UsageError(
                f"bad suite spec {spec!r}; expected <suite>, <suite>::<task>, "
                f"or <suite>::<task>::<n>"
            )
        if len(parts) == 3:
            # <suite>::<task>::<n> selects a single fanned-out trigger case.
            # It maps to the case task name the fan-out assigns ("<task>-<n>").
            suite, task, case = parts
            if not case.isdigit():
                raise click.UsageError(
                    f"bad suite spec {spec!r}; trigger case index must be a "
                    f"non-negative integer (got {case!r})"
                )
            return suite, f"{task}-{int(case)}"
        return parts[0], parts[1]
    return spec, None


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """agent-exam — eval framework for agent skills."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _run_cmd(
    suite_specs: tuple[str, ...],
    provider: str | None,
    model: str,
    k: int,
    n_parallel: int,
    without_skill: bool,
    no_skills: bool,
    no_mcp: bool,
    no_triggers: bool,
    tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    all_tags: bool = False,
) -> int:
    cfg = load_config()
    specs = [_parse_suite_spec(spec) for spec in suite_specs]
    req = RunRequest(
        specs=specs,
        provider=provider or cfg.default_harness,
        model=model,
        k=k,
        n_parallel=n_parallel,
        without_skill=without_skill,
        no_skills=no_skills,
        no_mcp=no_mcp,
        no_triggers=no_triggers,
        tags=list(tags),
        exclude_tags=list(exclude_tags),
        all_tags=all_tags,
    )
    return run(cfg, req)


@cli.command("runs", help="List recent runs (newest first).")
@click.option(
    "--limit", type=int, default=20, show_default=True, help="Max rows to show."
)
def runs_cmd(limit: int) -> None:
    cfg = load_config()
    sys.exit(runs_cmd_impl.run(cfg.evals_dir, limit=limit))


@cli.command(
    "show",
    help=(
        "Show a run, attempt, or assertion detail.\n\n"
        "SPEC forms:\n"
        "  <run-id>                             run summary\n"
        "  <run-id>::<suite>                    suite summary\n"
        "  <run-id>::<suite>::<task>            all attempts of a task\n"
        "  <run-id>::<suite>::<task>::attempt-N one specific attempt\n\n"
        "Run-id accepts 'latest' and 'prev'."
    ),
)
@click.argument("spec", required=False, default="latest")
@click.option(
    "--report", "report_ts", default=None, help="Pick a specific report by timestamp."
)
@click.option(
    "--no-issues",
    is_flag=True,
    help="Don't list problematic tool calls (permission denials, rejections, errors).",
)
def show_cmd(spec: str, report_ts: str | None, no_issues: bool) -> None:
    cfg = load_config()
    sys.exit(
        show_cmd_impl.run(
            cfg.evals_dir, spec, report_ts=report_ts, show_issues=not no_issues
        )
    )


@cli.command(
    "history",
    help=(
        "Trends across runs.\n\n"
        "SPEC forms:\n"
        "  <suite>::<task>  per-run row of this task's latest attempt\n"
        "  <suite>          per-run row aggregating all tasks in the suite\n\n"
        "Default shows the last 10 runs that graded the scope; pass --all "
        "for every run. (For trends across every suite, use `runs`.)"
    ),
)
@click.argument("spec")
@click.option(
    "--limit", type=int, default=10, show_default=True, help="Max rows to show."
)
@click.option(
    "--all", "all_runs", is_flag=True, help="Show every run that graded this task."
)
def history_cmd(spec: str, limit: int, all_runs: bool) -> None:
    cfg = load_config()
    sys.exit(history_cmd_impl.run(cfg.evals_dir, spec, limit=limit, all_runs=all_runs))


@cli.command(
    "diff",
    help=(
        "Diff two reports.\n\n"
        "Each arg accepts a run-id, 'latest', or 'prev'. Typical call:\n"
        "  agent-exam diff prev latest"
    ),
)
@click.argument("a")
@click.argument("b")
@click.option("--report-a", default=None, help="Pick a specific report in run A.")
@click.option("--report-b", default=None, help="Pick a specific report in run B.")
@click.option("--scope", default=None, help="Restrict to <suite> or <suite>::<task>.")
def diff_cmd(
    a: str, b: str, report_a: str | None, report_b: str | None, scope: str | None
) -> None:
    cfg = load_config()
    sys.exit(
        diff_cmd_impl.run(
            cfg.evals_dir,
            a,
            b,
            report_a=report_a,
            report_b=report_b,
            scope=scope,
        )
    )


@cli.command(
    "rescore",
    help=(
        "Re-grade archived attempts against current assertions / judge criteria.\n\n"
        "SCOPE forms (run-id accepts 'latest'):\n"
        "  <run-id>                        whole run\n"
        "  <run-id>::<suite>               one suite\n"
        "  <run-id>::<suite>::<task>       one task (the common case)\n\n"
        "Does NOT re-run the agent. Writes a new reports/<timestamp>.json\n"
        "in the source run; judge verdicts are cached per-run so unchanged\n"
        "criteria are free to rescore."
    ),
)
@click.argument("scope")
def rescore_cmd(scope: str) -> None:
    cfg = load_config()
    sys.exit(rescore_cmd_impl.run(cfg.evals_dir, scope))


@cli.command(
    "doctor",
    help=(
        "Preflight checks: project + config + provider + a real round-trip "
        "to verify auth and the full pipeline end-to-end.\n\n"
        "Pass --no-llm to skip the round-trip probe (no token cost)."
    ),
)
@click.option(
    "--no-llm",
    "no_llm",
    is_flag=True,
    help="Skip the round-trip probe — no token cost. "
    "Everything else (incl. suite validation) still runs.",
)
@click.option(
    "--provider",
    default=None,
    help="Provider to check (default: config default_harness).",
)
def doctor_cmd(no_llm: bool, provider: str | None) -> None:
    sys.exit(doctor_cmd_impl.run(no_llm=no_llm, provider=provider))


@cli.command(
    "run",
    help=(
        "Run one or more suites (or specific tasks within them).\n\n"
        "SUITE_SPEC forms:\n"
        "  <suite>                  run every task in the suite\n"
        "  <suite>::<task>          run one task\n"
        "  <suite>::<task>::<n>     run one fanned-out trigger case (0-based)\n"
        "  *::trigger               run the trigger task from every suite that has one\n\n"
        "Multiple specs are accepted: agent-exam run suite-a suite-b::task-x\n\n"
        "Shortcut: `agent-exam <suite>` dispatches here implicitly.\n\n"
        "Tasks wearing a tag configured exclude_by_default are skipped unless "
        "the spec asks for them: naming a suite ignores the tags that suite "
        "declares, naming a task runs it. --tag and --all-tags override."
    ),
    context_settings={"ignore_unknown_options": True},
)
@click.argument("suite_specs", nargs=-1, required=True)
@click.option(
    "--provider",
    default=None,
    help="Harness to invoke. Defaults to default_harness in config.",
)
@click.option("--model", default="", help="Model id or alias.")
@click.option("-k", "k", type=int, default=1, help="Attempts per task.")
@click.option("-n", "n_parallel", type=int, default=10, help="Parallel attempts.")
@click.option(
    "--without-skill",
    is_flag=True,
    help="Reality-check mode: drop the suite's evaluated skills.",
)
@click.option(
    "--no-skills",
    "no_skills",
    is_flag=True,
    help="Reality-check mode: drop every skill, not just the suite's.",
)
@click.option(
    "--no-mcp",
    is_flag=True,
    help="Reality-check mode: attach none of the configured MCP servers.",
)
@click.option(
    "--no-triggers",
    is_flag=True,
    help="Skip 'kind: trigger' tasks — run only execute tasks. "
    "Implied by the skill-withholding modes; pass it on the with-skill run "
    "to compare like for like.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Include the tasks wearing this tag even though it is "
    "excluded by default. Repeatable.",
)
@click.option(
    "--exclude-tag",
    "exclude_tags",
    multiple=True,
    help="Drop the tasks wearing this tag. Repeatable, and applies to "
    "every spec — even a task named on the command line.",
)
@click.option(
    "--all-tags",
    is_flag=True,
    help="Ignore every exclude_by_default tag, i.e. run the expensive ones too.",
)
def run_cmd(
    suite_specs: tuple[str, ...],
    provider: str,
    model: str,
    k: int,
    n_parallel: int,
    without_skill: bool,
    no_skills: bool,
    no_mcp: bool,
    no_triggers: bool,
    tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
    all_tags: bool,
) -> None:
    code = _run_cmd(
        suite_specs,
        provider,
        model,
        k,
        n_parallel,
        without_skill,
        no_skills,
        no_mcp,
        no_triggers,
        tags=tags,
        exclude_tags=exclude_tags,
        all_tags=all_tags,
    )
    sys.exit(code)


def main() -> None:
    """Entry point with implicit-run-mode dispatch.

    If argv[1] is not a reserved verb / flag and not empty, treat it as a
    suite spec and invoke the `run` command.
    """
    # Reports, diffs and trajectories are full of arrows and box characters,
    # and a Windows console hands Python a cp1252 stdout unless UTF-8 mode is
    # on, which turns any of them into a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in RESERVED_VERBS:
        argv = ["run", *argv]
    try:
        cli.main(args=argv, prog_name="agent-exam", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        click.echo(f"error: {exc.format_message()}", err=True)
        sys.exit(2)
    except click.exceptions.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except AgentExamError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(exc.exit_code)
    except SystemExit:
        raise
