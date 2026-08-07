"""`agent-exam show <scope>` — run summary / attempt detail."""

from __future__ import annotations

import json
from shutil import get_terminal_size
from typing import TYPE_CHECKING

import click

from ..ids import list_run_ids
from ..providers.claude_code.transcripts import (
    _is_harness_rejection,
    _is_permission_denial,
)
from ..run_modes import banner_lines
from ._format import (
    THRESHOLDS,
    _visual_width,
    delta_marker,
    fmt_cost,
    fmt_ctx,
    fmt_pass_ratio,
    fmt_wall,
    iso_duration,
    render_table,
)
from ._loader import RunData, load_run, parse_run_spec

if TYPE_CHECKING:
    from pathlib import Path


def run(
    evals_dir: Path,
    scope_spec: str,
    report_ts: str | None = None,
    show_issues: bool = True,
) -> int:
    scope = parse_run_spec(scope_spec)
    # `show <suite>::<task>` without a run-id is ambiguous — which run?
    # Silently resolving to `latest` would strip context from an
    # identifier the user copied out of a specific-run view. Point them
    # at the two unambiguous forms.
    existing_ids = list_run_ids(evals_dir / "runs")
    if (
        scope.run_id not in ("latest", "prev")
        and scope.run_id not in existing_ids
        and "::" in scope_spec
    ):
        click.echo(
            f"{scope_spec!r} needs a run context.\n"
            f"  one run:     uv run agent-exam show latest::{scope_spec}\n"
            f"  across runs: uv run agent-exam history {scope_spec}",
            err=True,
        )
        return 2
    data = load_run(evals_dir, scope.run_id)
    report = data.report_for(report_ts)
    if report is None:
        if report_ts:
            click.echo(f"no report {report_ts!r} in run {data.run_id}", err=True)
            return 2
        click.echo(f"run {data.run_id} has no reports yet", err=True)
        return 2

    if scope.suite is None:
        _show_run_summary(evals_dir, data, report)
        return 0
    if scope.task is None:
        # User typed `<run>::<X>`. X is usually a suite, but if no suite
        # matches and X uniquely matches a task name across all suites in
        # the report, interpret as `<run>::<suite>::<X>` — saves authors
        # from typing the suite in the common single-suite case.
        suites_in_report = {a["suite"] for a in report.attempts}
        if scope.suite not in suites_in_report:
            task_matches = [a for a in report.attempts if a["task"] == scope.suite]
            suites_matched = sorted({a["suite"] for a in task_matches})
            if len(suites_matched) == 1:
                return _show_attempt_detail(
                    data,
                    report,
                    suites_matched[0],
                    scope.suite,
                    None,
                    show_issues=show_issues,
                )
            if len(suites_matched) > 1:
                click.echo(
                    f"{scope.suite!r} is ambiguous — matches a task in multiple "
                    f"suites ({', '.join(suites_matched)}). Use "
                    f"<run>::<suite>::<task>.",
                    err=True,
                )
                return 2
        _show_suite_summary(data, report, scope.suite)
        return 0
    return _show_attempt_detail(
        data,
        report,
        scope.suite,
        scope.task,
        scope.attempt,
        show_issues=show_issues,
    )


def _show_run_summary(evals_dir: Path, data: RunData, report) -> None:
    rj = data.run_json
    dur = iso_duration(rj.get("started_at", ""), rj.get("finished_at", ""))
    banner = banner_lines(
        rj.get("run_mode", "run"),
        rj.get("config", {}).get("skills_excluded") or [],
    )
    if banner:
        for line in banner:
            click.echo(line)
        click.echo("")
    header_lines = [
        f"Run:       {data.run_id}",
        f"Mode:      {rj.get('run_mode', '?')}",
        f"Models:    {_resolve_models(rj, data, report)}",
        f"Started:   {rj.get('started_at', '')}  (duration {fmt_wall(dur) if dur else '—'})",
        f"Report:    {_fmt_report_line(data, report)}",
    ]
    click.echo("\n".join(header_lines))
    click.echo("")

    prev_report = _load_prev_report(evals_dir, data.run_id)

    rows = []
    for a in report.attempts:
        meta = data.load_attempt_metadata(a["suite"], a["task"], a["attempt"])
        metrics = (meta or {}).get("metrics") or {}
        prev_meta = _load_prev_metadata(prev_report, evals_dir, a)

        cost = metrics.get("cost_usd")
        wall = metrics.get("wall_time_seconds")
        ctx = metrics.get("peak_context")

        prev_metrics = (prev_meta or {}).get("metrics") or {}
        deltas = _format_deltas(metrics, prev_metrics)

        rows.append(
            [
                f"{a['suite']}::{a['task']}",
                f"attempt-{a['attempt']}",
                fmt_pass_ratio(a),
                fmt_cost(cost),
                fmt_wall(wall),
                fmt_ctx(ctx),
                _fmt_tools_compact(metrics),
                deltas,
            ]
        )

    click.echo(
        render_table(
            rows,
            [
                "SUITE::TASK",
                "ATTEMPT",
                "PASSED",
                "COST",
                "WALL",
                "CTX",
                "TOOLS",
                "Δ vs prev",
            ],
        )
    )
    click.echo("")
    click.echo(f"Run dir:   {data.paths.run_dir}")
    click.echo(f"Report:    {report.path}")
    # Concrete drill-down: first attempt's suite::task, against THIS run
    # (not the abstract `latest::<suite>::<task>` — copying that from the
    # footer to the shell strips the run context).
    if report.attempts:
        sample = report.attempts[0]
        example = f"{data.run_id}::{sample['suite']}::{sample['task']}"
        click.echo(f"Inspect:   uv run agent-exam show {example}")


def _show_suite_summary(data: RunData, report, suite: str) -> None:
    matching = [a for a in report.attempts if a["suite"] == suite]
    if not matching:
        click.echo(f"no attempts for suite {suite!r} in {data.run_id}", err=True)
        return
    click.echo(f"Run:     {data.run_id}")
    click.echo(f"Suite:   {suite}")
    click.echo(f"Report:  {_fmt_report_line(data, report)}")
    click.echo("")
    rows = []
    for a in matching:
        meta = data.load_attempt_metadata(a["suite"], a["task"], a["attempt"])
        metrics = (meta or {}).get("metrics") or {}
        rows.append(
            [
                a["task"],
                f"attempt-{a['attempt']}",
                fmt_pass_ratio(a),
                _fmt_tools_compact(metrics),
            ]
        )
    click.echo(render_table(rows, ["TASK", "ATTEMPT", "PASSED", "TOOLS"]))


def _show_attempt_detail(
    data: RunData,
    report,
    suite: str,
    task: str,
    attempt: int | None,
    show_issues: bool = True,
) -> int:
    attempt_entries = [
        a
        for a in report.attempts
        if a["suite"] == suite
        and a["task"] == task
        and (attempt is None or a["attempt"] == attempt)
    ]
    if not attempt_entries:
        avail = sorted(
            {
                a["attempt"]
                for a in report.attempts
                if a["suite"] == suite and a["task"] == task
            }
        )
        click.echo(
            f"no attempt for {suite}::{task}"
            + (f" attempt-{attempt}" if attempt is not None else "")
            + f" in report {report.timestamp}. "
            + (f"Available attempts: {avail}" if avail else "(task not in report)"),
            err=True,
        )
        return 2

    run_mode = data.run_json.get("run_mode", "run")
    banner = banner_lines(
        run_mode, data.run_json.get("config", {}).get("skills_excluded") or []
    )
    if banner:
        for line in banner:
            click.echo(line)
        click.echo("")

    # Unified label width across the Summary block so all values line
    # up on the same left edge. Longest label is "Provider".
    label_w = len("Provider") + 1

    def kv(label: str, value) -> None:
        click.echo(f"{(label + ':').ljust(label_w)} {value}")

    for entry in attempt_entries:
        meta = data.load_attempt_metadata(
            entry["suite"], entry["task"], entry["attempt"]
        )
        metrics = (meta or {}).get("metrics") or {}
        prompt, _thinking, final_msg = _extract_prompt_and_final(data, entry)
        _section("Summary")
        kv("Attempt", f"{entry['suite']}::{entry['task']} attempt-{entry['attempt']}")
        kv("Run", data.run_id)
        kv("Mode", run_mode)
        # Provider/model live on the attempt artifact (so a rescore
        # against a different model would show up here), with a
        # fallback to the run-level config when the attempt was
        # written before this field existed.
        provider = (meta or {}).get("provider") or data.run_json.get("config", {}).get(
            "provider", "?"
        )
        model = (
            (meta or {}).get("model")
            or ", ".join(data.run_json.get("config", {}).get("models") or [])
            or "?"
        )
        kv("Provider", provider)
        kv("Model", model)
        kv("Report", _fmt_report_line(data, report))
        kv("Passed", fmt_pass_ratio(entry))
        kv(
            "Metrics",
            f"cost {fmt_cost(metrics.get('cost_usd'))}, "
            f"wall {fmt_wall(metrics.get('wall_time_seconds'))}, "
            f"peak ctx {fmt_ctx(metrics.get('peak_context'))}, "
            f"turns {metrics.get('turn_count', '—')}",
        )
        kv("Tools", _fmt_tools(metrics))
        click.echo("")
        _section("Artifacts")
        attempt_json = data.paths.attempt_json(
            entry["suite"], entry["task"], entry["attempt"]
        )
        click.echo(click.style(f"attempt.json:    {attempt_json}", dim=True))
        raw_path = (meta or {}).get("raw_transcript_path")
        if raw_path:
            click.echo(click.style(f"raw transcript:  {raw_path}", dim=True))
        if prompt or final_msg:
            click.echo("")
            _section("Prompt & response")
            if prompt:
                click.echo(click.style("Prompt:", bold=True))
                click.echo(prompt)
            if final_msg:
                click.echo("")
                click.echo(click.style("Response:", bold=True))
                click.echo(final_msg)
        click.echo("")
        _section("Assertions", fmt_pass_ratio(entry))
        assertions = entry.get("assertions", [])
        for i, a in enumerate(assertions):
            if i > 0:
                click.echo("")  # blank line between assertions
            _render_assertion(a)
        if show_issues:
            _render_tool_issues(data, entry)
    return 0


_THINKING_BLOCK_LIMIT = 400  # chars shown per thinking block


def _extract_prompt_and_final(
    data: RunData, entry: dict
) -> tuple[str | None, list[str], str | None]:
    """Pull the user prompt, all thinking blocks (truncated), and the agent's
    final text response from the archived trajectory.
    Returns (None, [], None) if the trajectory is absent.
    """
    traj_path = data.paths.trajectory_json(
        entry["suite"], entry["task"], entry["attempt"]
    )
    if not traj_path.exists():
        return None, [], None
    turns = json.loads(traj_path.read_text()).get("turns", [])
    prompt = None
    thinking: list[str] = []
    final = None
    for turn in turns:
        if turn.get("role") == "user":
            for block in turn.get("content") or []:
                if block.get("type") == "text" and block.get("text") and prompt is None:
                    prompt = block["text"].strip()
                    break
        elif turn.get("role") == "assistant":
            for block in turn.get("content") or []:
                if block.get("type") == "thinking":
                    text = (block.get("text") or "").strip()
                    if text:
                        truncated = text[:_THINKING_BLOCK_LIMIT]
                        if len(text) > _THINKING_BLOCK_LIMIT:
                            truncated += (
                                f"… (+{len(text) - _THINKING_BLOCK_LIMIT} chars)"
                            )
                        thinking.append(truncated)
    for turn in reversed(turns):
        if turn.get("role") != "assistant":
            continue
        texts = [
            (block.get("text") or "").strip()
            for block in (turn.get("content") or [])
            if block.get("type") == "text" and (block.get("text") or "").strip()
        ]
        if texts:
            final = "\n".join(texts)
            break
    return prompt, thinking, final


def _trunc_inline(text: str, n: int = 200) -> str:
    """Collapse a potentially-multi-line string onto one line, then
    truncate with an ellipsis when too long for an inline kv value.
    """
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= n else one_line[: n - 1] + "…"


def _resolve_models(rj: dict, data, report) -> str:
    """Return a display string for the model(s) used in this run.

    run.json config.models may be [""] when the provider resolves the model
    at runtime (e.g. opencode). In that case fall back to the unique non-empty
    models recorded in individual attempt metadata.
    """
    from_config = [m for m in (rj.get("config", {}).get("models") or []) if m]
    if from_config:
        return ", ".join(from_config)
    seen: dict[str, None] = {}  # ordered set
    for a in report.attempts:
        meta = data.load_attempt_metadata(a["suite"], a["task"], a["attempt"])
        model = (meta or {}).get("model") or ""
        if model:
            seen[model] = None
    return ", ".join(seen)


def _fmt_report_line(data: RunData, current) -> str:
    """Render the current report's timestamp; when the run has more than
    one report (the rescore case), append a compact list of the others
    and a hint about `--report <ts>`. Silent when there's only one.
    """
    all_reports = data.reports
    if len(all_reports) <= 1:
        return current.timestamp
    others = [
        r.timestamp for r in reversed(all_reports) if r.timestamp != current.timestamp
    ]
    return (
        f"{current.timestamp} ({len(all_reports)} total; "
        f"others: {', '.join(others)}; use --report <ts> to pick)"
    )


def _section(title: str, stats: str | None = None) -> None:
    """Pytest-style section banner — `=` padded to terminal width with
    the title centered. Plain text (no bold, no color) since the `===`
    shape already carries the visual weight; weight + color are saved
    for content semantics below. Content follows immediately; breathing
    room comes from the blank line above.

    Optional `stats` is appended in parens so section headers double as
    at-a-glance landing zones when scrolling — e.g.
    `=== Assertions (4/4 +1) ===`. `stats` may contain ANSI color
    codes from the content it summarizes (fmt_pass_ratio, _fmt_tools);
    banner width is computed from the visible-char count.
    """
    width = get_terminal_size((80, 20)).columns
    label = title if stats is None else f"{title} ({stats})"
    middle = f" {label} "
    middle_visible = _visual_width(middle)
    bars = max(3, (width - middle_visible) // 2)
    left = "=" * bars
    right = (
        "=" * (width - middle_visible - bars) if width > middle_visible + bars else left
    )
    click.echo(f"{left}{middle}{right}")


def _render_tool_issues(data: RunData, entry: dict) -> None:
    """List any tool calls with permission_denied / rejected / error status.

    Surfaces issues the counters already summarize, so the user doesn't
    have to open attempt.json or the trajectory to see what actually
    went wrong. Skipped entirely when everything was clean.

    Reclassifies on the fly from the result text — older trajectories
    recorded plain `status: "error"` before the permission/rejection
    detectors existed, so we can't rely on the stored status alone.
    """
    traj_path = data.paths.trajectory_json(
        entry["suite"], entry["task"], entry["attempt"]
    )
    if not traj_path.exists():
        return
    turns = json.loads(traj_path.read_text()).get("turns", [])
    issues: list[
        tuple[str, str, str, str, str]
    ] = []  # (status, turn, name, input, result)

    def _walk(ts, parent_turn: str | None = None):
        for i, turn in enumerate(ts):
            turn_label = parent_turn if parent_turn is not None else f"turn {i}"
            for block in turn.get("content", []) or []:
                if block.get("type") != "tool_call":
                    continue
                if block.get("name") == "Skill":
                    continue
                status = block.get("status", "ok")
                result = block.get("result", "") or ""
                # Lift error→permission_denied / rejected where pattern
                # matches — matches the backfill / counter logic.
                if status == "error":
                    if _is_permission_denial(result):
                        status = "permission_denied"
                    elif _is_harness_rejection(result):
                        status = "rejected"
                if status in ("permission_denied", "rejected", "error", "aborted"):
                    inp = block.get("input", {})
                    cmd = inp.get("command") if isinstance(inp, dict) else None
                    if not cmd and isinstance(inp, dict):
                        cmd = inp.get("file_path") or inp.get("pattern")
                    cmd = (cmd or json.dumps(inp, default=str))[:120]
                    issues.append(
                        (status, turn_label, block.get("name", "?"), cmd, result[:200])
                    )
                if block.get("subagent"):
                    # Subagent tool calls stay attributed to the parent turn
                    # (the outer Agent tool_use lives at that turn) with a
                    # "(subagent)" suffix to signal the nesting.
                    _walk(block["subagent"], parent_turn=f"{turn_label} (subagent)")

    _walk(turns)
    if not issues:
        return

    click.echo("")
    metrics = (entry.get("metrics") if isinstance(entry, dict) else None) or {}
    # Load metrics from attempt.json if not inline (the entry is the
    # report's attempt entry, which doesn't carry metrics — those live
    # alongside in attempt.json).
    meta = data.load_attempt_metadata(entry["suite"], entry["task"], entry["attempt"])
    metrics = (meta or {}).get("metrics") or {}
    _section("Tool issues", _fmt_tools(metrics))
    for status, turn_label, name, cmd, result in issues:
        tag_color = {
            "permission_denied": "red",
            "rejected": "magenta",
            "error": "yellow",
            "aborted": "yellow",
        }.get(status)
        label = {
            "permission_denied": "[DENIED]",
            "rejected": "[REJECTED]",
            "error": "[ERROR]",
            "aborted": "[ABORTED]",
        }.get(status, f"[{status.upper()}]")
        tag = click.style(label, fg=tag_color) if tag_color else label
        prefix = click.style(f"{turn_label}  ", dim=True)
        click.echo(f"{prefix}{tag} {click.style(name, fg='cyan')}: {cmd}")
        if result:
            click.echo(click.style(result, dim=True))


def _render_assertion(a: dict) -> None:
    """One assertion block, flush-left. Terminal wraps don't break anything
    since there's no per-line indent to disagree with. Structure is
    carried by color (PASS/FAIL tag, cyan type, dim reasoning) and by
    the blank line between blocks.

    Known-issue assertions get a [KNOWN-FAIL] / [UNEXPECTED-PASS] tag
    instead of the plain PASS/FAIL: yellow for a still-failing known
    issue (expected), cyan for one that's now passing (hint: the
    annotation can be removed).

    Provider-filtered assertions (skipped on this run's provider) get a
    dim [SKIPPED] tag.
    """
    passed = a.get("pass", False)
    known_issue = a.get("known_issue")
    skipped_reason = a.get("skipped_reason")
    if skipped_reason:
        tag = click.style("[SKIPPED]", fg="bright_blue")
        kind = click.style(a["type"], dim=True)
        criterion = _truncate_inline_config(a.get("config"))
        header = f"{tag} {kind}"
        if criterion:
            header += click.style(f": {criterion}", dim=True)
        # Normalise legacy verbose format stored in older reports
        short = skipped_reason
        for prefix in ("skipped: provider ", "skipped: "):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        click.echo(f"{header}{click.style(f'  ({short})', dim=True)}")
        return
    if known_issue:
        if passed:
            # Kept in the cyan family (same "hint/info" semantics as
            # task-level unexpected_pass) but distinct from the cyan
            # type label next to it.
            tag = click.style("[UNEXPECTED-PASS]", fg="bright_cyan")
        else:
            tag = click.style("[KNOWN-FAIL]", fg="yellow")
    else:
        tag = (
            click.style("[PASS]", fg="green")
            if passed
            else click.style("[FAIL]", fg="red")
        )
    kind = click.style(a["type"], fg="cyan")
    criterion = _truncate_inline_config(a.get("config"))
    header = f"{tag} {kind}"
    if criterion:
        header += f": {criterion}"
    click.echo(header)
    if known_issue:
        # Yellow (not dim) so it doesn't blend with judge reasoning, which
        # is also printed dim below.
        click.echo(click.style(f"known_issue: {known_issue}", fg="yellow"))

    # For judge assertions the structured verdict + reasoning belong together;
    # the `reason` field ("judge said YES") is redundant with the verdict
    # that leads the reasoning block. Non-judge assertions print the `reason`
    # as the single detail line.
    if a["type"] in ("judge", "judge_agent"):
        details = a.get("details") or {}
        verdict = details.get("verdict", "")
        reasoning = details.get("reasoning") or ""
        if reasoning or verdict:
            prefix = click.style(f"{verdict} — ", dim=True) if verdict else ""
            click.echo(f"{prefix}{click.style(reasoning, dim=True)}")
    else:
        reason = a.get("reason", "")
        if reason:
            click.echo(click.style(reason, dim=True))


def _fmt_tools_compact(metrics: dict) -> str:
    """Table-column-friendly: `5` when clean, `5 (2e 1d)` with color if not.

    Keeps the clean case as a single number so the column stays narrow,
    adds a parenthesized e/d suffix when something's off.
    """
    total = metrics.get("n_tool_calls", 0)
    errs = metrics.get("n_tool_errors", 0)
    denied = metrics.get("n_permission_denied", 0)
    rejected = metrics.get("n_tool_rejected", 0)
    if not total:
        return "—"
    if not errs and not denied and not rejected:
        return str(total)
    suffix_parts: list[str] = []
    if errs:
        suffix_parts.append(click.style(f"{errs}e", fg="yellow"))
    if denied:
        suffix_parts.append(click.style(f"{denied}d", fg="red"))
    if rejected:
        suffix_parts.append(click.style(f"{rejected}r", fg="magenta"))
    return f"{total} ({' '.join(suffix_parts)})"


def _fmt_tools(metrics: dict) -> str:
    """Compact tool-call counter: `5 total, 0 err, 0 denied, 0 rejected`.

    Error/denied/rejected segments are colored when non-zero so a
    single glance surfaces skill-side issues. Red denials are the
    clearest signal — anything in that bucket is a permission prompt
    the skill shouldn't have caused.
    """
    total = metrics.get("n_tool_calls", 0)
    errs = metrics.get("n_tool_errors", 0)
    denied = metrics.get("n_permission_denied", 0)
    rejected = metrics.get("n_tool_rejected", 0)
    parts = [f"{total} total"]
    err_part = f"{errs} err"
    if errs:
        err_part = click.style(err_part, fg="yellow")
    parts.append(err_part)
    denied_part = f"{denied} denied"
    if denied:
        denied_part = click.style(denied_part, fg="red")
    parts.append(denied_part)
    rejected_part = f"{rejected} rejected"
    if rejected:
        rejected_part = click.style(rejected_part, fg="magenta")
    parts.append(rejected_part)
    return ", ".join(parts)


def _truncate_inline_config(cfg, n: int = 200) -> str:
    """Single-line summary of an assertion's config for inline display.

    Collapses all whitespace (including newlines from YAML block scalars)
    to a single space so multi-line judge criteria don't break the table.
    """
    if cfg is None:
        return ""
    if isinstance(cfg, str):
        text = " ".join(cfg.split())
    else:
        text = json.dumps(cfg, default=str)
    return text if len(text) <= n else text[:n] + "..."


def _load_prev_report(evals_dir: Path, current_run_id: str):
    """Walk back for the previous run whose `run_mode` matches the current's.

    Mixing normal and reality-check runs would otherwise produce misleading
    metric deltas — two runs with different loaded-skill sets aren't a
    fair comparison.
    """
    ids = list_run_ids(evals_dir / "runs")
    if current_run_id not in ids:
        return None
    try:
        current = load_run(evals_dir, current_run_id)
    except Exception:
        return None
    current_mode = current.run_json.get("run_mode", "run")
    idx = ids.index(current_run_id)
    for prev_id in reversed(ids[:idx]):
        try:
            prev_run = load_run(evals_dir, prev_id)
        except Exception:  # noqa: S112 -- a half-written or corrupt run dir is skipped, not fatal
            continue
        if prev_run.run_json.get("run_mode", "run") == current_mode:
            return prev_run
    return None


def _load_prev_metadata(prev_run, evals_dir: Path, attempt_entry: dict) -> dict | None:
    if prev_run is None:
        return None
    return prev_run.load_attempt_metadata(
        attempt_entry["suite"], attempt_entry["task"], attempt_entry["attempt"]
    )


def _format_deltas(current: dict, prev: dict) -> str:
    if not prev:
        return ""
    parts = []
    for metric in THRESHOLDS:
        new = current.get(metric)
        old = prev.get(metric)
        marker = delta_marker(new, old, metric)
        if marker:
            parts.append(f"{_short(metric)} {marker}")
    return ", ".join(parts)


def _short(metric: str) -> str:
    return {
        "cost_usd": "cost",
        "peak_context": "ctx",
        "wall_time_seconds": "wall",
    }.get(metric, metric)
