"""`agent-exam doctor` — preflight checks.

Intended as the first command a dev runs after install, and the one to
re-run when "it used to work" turns into "something broke". Two groups
of checks:

- **Framework side** (always run): project root, config parses, evals
  dir exists + writable.
- **Provider side**: provider preflight checks plus a real harness round-trip
  against the configured judge model to verify auth + the full stream
  pipeline. Costs a fraction of a cent; skip with ``--no-llm``.

Exit codes: 0 all-OK, 2 any FAIL. WARN is informational.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

from ..config import Config, find_project_root, load_config
from ..errors import UsageError
from ..hooks import call_pre_run_hook
from ..mcp import preflight as mcp_preflight
from ..providers import get_provider
from ..providers.skill_staging import discover_skills
from ..schemas import CheckResult
from ..tasks import list_suites
from ..validation import validate_suite
from ._format import fmt_cost


def run(no_llm: bool = False, provider: str | None = None) -> int:
    results: list[CheckResult] = []

    cfg = _discover(results)
    if cfg is not None:
        effective_provider = provider or cfg.default_harness
        results.extend(_framework_checks(cfg, effective_provider))

        hook_checks, cfg, hook_supplied_skills_dirs = _pre_run_hook_check(
            cfg, effective_provider
        )
        results.extend(hook_checks)
        results.extend(_skills_dirs_check(cfg, hook_supplied_skills_dirs))

        results.extend(_suite_checks(cfg))
        results.extend(_provider_checks(cfg, effective_provider))
        if not no_llm:
            results.extend(_round_trip_check(cfg, effective_provider))

    _render(results)

    if any(r.status == "FAIL" for r in results):
        return 2
    return 0


def _discover(results: list[CheckResult]) -> Config | None:
    try:
        root = find_project_root()
        results.append(CheckResult(name="project root", status="OK", hint=str(root)))
    except UsageError as exc:
        results.append(
            CheckResult(
                name="project root",
                status="FAIL",
                hint=str(exc),
            )
        )
        return None

    try:
        cfg = load_config(root)
        results.append(
            CheckResult(
                name="config",
                status="OK",
                hint=f"evals_dir={cfg.evals_dir.relative_to(root)}",
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                name="config",
                status="FAIL",
                hint=f"{type(exc).__name__}: {exc}",
            )
        )
        return None

    return cfg


def _framework_checks(cfg: Config, provider_name: str) -> list[CheckResult]:
    checks: list[CheckResult] = []

    if not cfg.evals_dir.is_dir():
        checks.append(
            CheckResult(
                name="evals dir",
                status="FAIL",
                hint=f"{cfg.evals_dir} missing — run 'uv run agent-exam <suite>' from the project root",
            )
        )
        return checks
    checks.append(
        CheckResult(
            name="evals dir",
            status="OK",
            hint=str(cfg.evals_dir),
        )
    )

    runs_dir = cfg.evals_dir / "runs"
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe = runs_dir / ".doctor-probe"
        probe.write_text("x")
        probe.unlink()
        checks.append(
            CheckResult(name="runs dir writable", status="OK", hint=str(runs_dir))
        )
    except OSError as exc:
        checks.append(
            CheckResult(
                name="runs dir writable",
                status="FAIL",
                hint=f"{runs_dir}: {exc}",
            )
        )

    # Config-yaml must declare the default provider.
    if cfg.default_harness not in cfg.providers:
        checks.append(
            CheckResult(
                name="default harness configured",
                status="WARN",
                hint=f"default_harness={cfg.default_harness} has no providers.{cfg.default_harness} block",
            )
        )
    else:
        checks.append(
            CheckResult(
                name="default harness configured",
                status="OK",
                hint=cfg.default_harness,
            )
        )

    provider_cfg = cfg.provider(provider_name)
    omitted_model_label = _omitted_model_label(provider_name)
    # Judge model configured (soft: some providers can omit --model, but
    # eval scoring should not depend on a developer's local harness default).
    if not provider_cfg.judge_model:
        if provider_cfg.default_model:
            hint = (
                f"providers.{provider_name}.judge_model unset — judge calls "
                f"will use default_model={provider_cfg.default_model}"
            )
        elif omitted_model_label:
            hint = (
                f"providers.{provider_name}.judge_model unset — judge calls "
                f"will use {omitted_model_label}; set judge_model for stable "
                "results"
            )
        else:
            hint = (
                f"providers.{provider_name}.judge_model unset and no "
                "default_model configured — judge assertions may fail or use "
                "harness defaults"
            )
        checks.append(
            CheckResult(
                name="judge model configured",
                status="WARN",
                hint=hint,
            )
        )
    else:
        checks.append(
            CheckResult(
                name="judge model configured",
                status="OK",
                hint=provider_cfg.judge_model,
            )
        )

    return checks


def _suite_checks(cfg: Config) -> list[CheckResult]:
    """Static validation of every suite — tasks parse, referenced
    fixtures exist. Shares `validate_suite` with the runner (which fails
    fast on any FAIL before spending tokens).
    """
    results: list[CheckResult] = []
    for suite in list_suites(cfg.evals_dir):
        results.extend(validate_suite(cfg, suite))
    return results


def _provider_checks(cfg: Config, provider_name: str) -> list[CheckResult]:
    try:
        provider = get_provider(provider_name)
    except ValueError as exc:
        return [
            CheckResult(
                name=f"{provider_name} provider",
                status="FAIL",
                hint=str(exc),
            )
        ]
    return [*provider.preflight(cfg), *mcp_preflight(cfg, provider)]


def _provider_or_none(provider_name: str):
    try:
        return get_provider(provider_name)
    except ValueError:
        return None


def _omitted_model_label(provider_name: str) -> str | None:
    provider = _provider_or_none(provider_name)
    if provider is None:
        return None
    return getattr(provider, "omitted_model_label", None)


def _round_trip_check(cfg: Config, provider_name: str) -> list[CheckResult]:
    """Spawn a real provider round-trip to verify auth + stream pipeline end-to-end."""
    provider_cfg = cfg.provider(provider_name)
    model = provider_cfg.resolve_model(
        provider_cfg.judge_model or provider_cfg.default_model or ""
    )
    provider = _provider_or_none(provider_name)
    omitted_model_label = getattr(provider, "omitted_model_label", None)
    if not model and not omitted_model_label:
        return [
            CheckResult(
                name=f"{provider_name} round-trip",
                status="WARN",
                hint=(
                    f"no default_model or judge_model under providers.{provider_name} "
                    "— skipping round-trip probe"
                ),
            )
        ]
    if provider is None:
        return [
            CheckResult(
                name=f"{provider_name} round-trip",
                status="FAIL",
                hint=f"unknown provider {provider_name!r}",
            )
        ]

    with tempfile.TemporaryDirectory(prefix="agent-exam-doctor-") as tmp:
        # Laid out like an attempt: the cwd is a directory of its own under a
        # tmp root, and whatever a provider stages beside it — an MCP config,
        # a raw stream — is a sibling rather than something the agent sees.
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()
        try:
            # Attach the configured servers so the probe's own session
            # reports whether each one connects — the cheapest place to
            # catch a server that dies on startup.
            probe_options = {
                "extra_args": list(provider_cfg.extra_args),
                "permission_mode": provider_cfg.permission_mode,
            }
            if cfg.mcp_servers:
                probe_options.update(provider.stage_mcp_config(Path(tmp), cfg))
            result = provider.invoke(
                prompt="Respond with just the two characters: ok",
                model=model,
                cwd=cwd,
                provider_options=probe_options,
                stop_on_first_skill=False,
                timeout_seconds=60,
            )
        except Exception as exc:
            return [
                CheckResult(
                    name=f"{provider_name} round-trip",
                    status="FAIL",
                    hint=f"{type(exc).__name__}: {exc}",
                )
            ]

    probe_checks = provider.probe_checks(result, cfg) if result is not None else []
    model_label = model or omitted_model_label or "<provider default>"
    return [
        CheckResult(
            name=f"{provider_name} round-trip",
            status="OK",
            hint=(
                f"{result.metrics.turn_count} turns, "
                f"wall {result.metrics.wall_time_seconds:.2f}s, "
                f"cost {fmt_cost(result.metrics.cost_usd)}, "
                f"model={model_label}"
            ),
        ),
        *probe_checks,
    ]


def _pre_run_hook_check(
    cfg: Config, provider_name: str
) -> tuple[list[CheckResult], Config, bool]:
    """Invoke the project's pre_run_hook (if configured) and surface
    success/failure as a doctor check. On success, fold the hook's
    `skills_dirs` into cfg so downstream checks see the built skills.

    Returns (checks, possibly-updated cfg, hook_supplied_skills_dirs).
    The boolean records whether the hook actually returned skills_dirs,
    so `_skills_dirs_check` can later tell an empty build (hook gave dirs
    but they contain no SKILL.md → FAIL) from a hook that legitimately
    returned nothing / no hook at all (→ WARN).
    """
    if not cfg.pre_run_hook:
        return (
            [CheckResult(name="pre-run hook", status="OK", hint="not configured")],
            cfg,
            False,
        )

    try:
        result = call_pre_run_hook(cfg, provider_name)
    except Exception as exc:
        return (
            [
                CheckResult(
                    name="pre-run hook",
                    status="FAIL",
                    hint=f"cannot run hook: {type(exc).__name__}: {exc}",
                )
            ],
            cfg,
            False,
        )

    if result is None or result.skills_dirs is None:
        return (
            [
                CheckResult(
                    name="pre-run hook",
                    status="OK",
                    hint=f"{cfg.pre_run_hook} returned no skills_dirs",
                )
            ],
            cfg,
            False,
        )

    # Honour an explicit local override — same rule the runner uses so
    # doctor and runner see the same effective skills_dirs.
    if not cfg._skills_dirs_locked:
        cfg = cfg.model_copy(update={"skills_dirs": result.skills_dirs})

    return (
        [
            CheckResult(
                name="pre-run hook",
                status="OK",
                hint=f"{cfg.pre_run_hook} → {len(result.skills_dirs)} dir(s)",
            )
        ],
        cfg,
        True,
    )


def _skills_dirs_check(
    cfg: Config, hook_supplied_skills_dirs: bool
) -> list[CheckResult]:
    """Confirm skills_dirs is set and actually contains skills.

    If configured dirs contain no SKILL.md, FAIL only when those dirs
    came from the hook; otherwise WARN. A hook that returns no
    skills_dirs is not considered a failure here.
    """
    if cfg.skills_dirs is None:
        return [
            CheckResult(
                name="skills available",
                status="WARN",
                hint="skills_dirs not configured; set it in evals/config.yaml or return it from your pre_run_hook",
            )
        ]

    missing = [str(p) for p in cfg.skills_dirs if not p.is_dir()]
    if missing:
        return [
            CheckResult(
                name="skills available",
                status="FAIL",
                hint=f"missing skills_dirs: {', '.join(missing)}",
            )
        ]

    discovered = discover_skills(cfg.skills_dirs)
    if not discovered:
        status = "FAIL" if hook_supplied_skills_dirs else "WARN"
        return [
            CheckResult(
                name="skills available",
                status=status,
                hint=f"no SKILL.md found under {[str(p) for p in cfg.skills_dirs]}",
            )
        ]

    return [
        CheckResult(
            name="skills available",
            status="OK",
            hint=f"{len(discovered)} skills in {len(cfg.skills_dirs)} dir(s)",
        )
    ]


_COLOR = {"OK": "green", "WARN": "yellow", "FAIL": "red"}


def _render(results: list[CheckResult]) -> None:
    name_width = max((len(r.name) for r in results), default=0)
    for r in results:
        tag = click.style(f"[{r.status}]", fg=_COLOR.get(r.status))
        line = f"{tag}  {r.name.ljust(name_width)}"
        if r.hint:
            line += f"  {r.hint}"
        click.echo(line)

    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    click.echo("")
    if fails:
        click.echo(
            click.style(f"{fails} FAIL, {warns} WARN, {len(results)} total", fg="red")
        )
    elif warns:
        click.echo(click.style(f"{warns} WARN, {len(results)} total", fg="yellow"))
    else:
        click.echo(click.style(f"all {len(results)} checks OK", fg="green"))
