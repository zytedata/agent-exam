"""Tests for `agent-exam doctor`.

Covers the framework-check surface: project discovery, evals dir,
runs-writable, judge model configured, provider preflight. The `--no-llm` path
skips the real round-trip; tests don't exercise it either (no subprocess),
but verify provider.preflight is called and its output rendered.
"""

from __future__ import annotations

import json
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from agent_exam.commands import doctor
from agent_exam.providers.claude_code.doctor_probes import hermetic_check
from agent_exam.schemas import Metrics, RunResult, Tokens

if TYPE_CHECKING:
    from pathlib import Path

_PYPROJECT = """\
[tool.agent-exam]
evals_dir = "evals"
"""


def _make_project(tmp_path: Path, config: str) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "evals").mkdir()
    (root / "evals" / "config.yaml").write_text(config)
    return root


def _run(
    monkeypatch, root: Path, omitted_model_labels: dict[str, str] | None = None
) -> tuple[int, list]:
    monkeypatch.chdir(root)
    # Replace the provider so doctor does not shell out to the real harness.
    from agent_exam.schemas import CheckResult

    class _FakeProvider:
        def __init__(self, name: str):
            self.name = name
            self.omitted_model_label = (omitted_model_labels or {}).get(name)

        def preflight(self, cfg=None):
            return [CheckResult(name="fake provider ok", status="OK", hint="stubbed")]

        def probe_checks(self, result, cfg=None):
            return []

        def pre_run_warnings(self, cfg=None):
            return []

        def invoke(self, *a, **kw):
            raise AssertionError("invoke should not run under --no-llm")

    monkeypatch.setattr("agent_exam.commands.doctor.get_provider", _FakeProvider)
    results: list = []
    # Capture what `_render` was fed by wrapping it.
    real_render = doctor._render

    def capturing_render(items):
        results.extend(items)
        real_render(items)

    monkeypatch.setattr(doctor, "_render", capturing_render)
    return doctor.run(no_llm=True), results


def test_happy_path(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                judge_model: haiku
            """
        ),
    )
    (root / "build" / "claude_code" / "skills" / "dummy").mkdir(parents=True)
    (root / "build" / "claude_code" / "skills" / "dummy" / "SKILL.md").write_text(
        "# dummy"
    )

    exit_code, results = _run(monkeypatch, root)
    assert exit_code == 0
    statuses = {r.name: r.status for r in results}
    assert statuses["project root"] == "OK"
    assert statuses["config"] == "OK"
    assert statuses["evals dir"] == "OK"
    assert statuses["runs dir writable"] == "OK"
    assert statuses["default harness configured"] == "OK"
    assert statuses["judge model configured"] == "OK"
    assert statuses["fake provider ok"] == "OK"


def test_missing_evals_dir(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(_PYPROJECT)
    # No evals/ dir at all.
    exit_code, results = _run(monkeypatch, root)
    assert exit_code == 2
    names = {r.name: r.status for r in results}
    assert names.get("evals dir") == "FAIL"


def test_missing_judge_model_is_warn(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                default_model: sonnet
            """
        ),
    )
    (root / "build" / "claude_code" / "skills").mkdir(parents=True)
    exit_code, results = _run(monkeypatch, root)
    statuses = {r.name: r.status for r in results}
    assert statuses["judge model configured"] == "WARN"
    # WARN alone does not fail the run.
    assert exit_code == 0


def test_missing_judge_model_warns_about_provider_omitted_model(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: custom_cli
            providers:
              custom_cli: {}
            """
        ),
    )
    exit_code, results = _run(
        monkeypatch,
        root,
        omitted_model_labels={"custom_cli": "custom CLI default model"},
    )
    by_name = {r.name: r for r in results}
    assert by_name["judge model configured"].status == "WARN"
    assert "custom CLI default model" in by_name["judge model configured"].hint
    assert exit_code == 0


def test_unknown_default_harness_is_warn(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: codex
            providers:
              claude_code:
                judge_model: haiku
            """
        ),
    )
    (root / "build" / "codex" / "skills").mkdir(parents=True)
    # get_provider("codex") fails → provider check FAILs, doctor exits 2.

    def fake_get_provider(name):
        raise ValueError(f"unknown provider {name!r}")

    monkeypatch.chdir(root)
    monkeypatch.setattr("agent_exam.commands.doctor.get_provider", fake_get_provider)
    exit_code = doctor.run(no_llm=True)
    assert exit_code == 2


def test_round_trip_check_not_invoked_under_no_llm(tmp_path, monkeypatch):
    """Regression guard: --no-llm must skip the real provider.invoke."""
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                judge_model: haiku
            """
        ),
    )
    (root / "build" / "claude_code" / "skills" / "x").mkdir(parents=True)
    (root / "build" / "claude_code" / "skills" / "x" / "SKILL.md").write_text("# x")

    exit_code, results = _run(monkeypatch, root)
    names = {r.name for r in results}
    assert "claude_code round-trip" not in names
    assert exit_code == 0


def test_config_parse_error_is_fail(tmp_path, monkeypatch):
    root = _make_project(tmp_path, "default_harness: [unterminated")
    exit_code, results = _run(monkeypatch, root)
    assert exit_code == 2
    assert any(r.name == "config" and r.status == "FAIL" for r in results)


def test_round_trip_check_handles_unknown_cost(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: copilot_cli
            providers:
              copilot_cli:
                judge_model: gpt-5-mini
            """
        ),
    )
    monkeypatch.chdir(root)

    class _FakeProvider:
        def invoke(self, *a, **kw):
            return RunResult(
                trajectory=[],
                metrics=Metrics(
                    wall_time_seconds=1.25,
                    tokens=Tokens(),
                    cost_usd=None,
                    peak_context=0,
                    turn_count=1,
                ),
                model="gpt-5-mini",
            )

        def probe_checks(self, result, cfg=None):
            return []

    monkeypatch.setattr(
        "agent_exam.commands.doctor.get_provider", lambda name: _FakeProvider()
    )

    results = doctor._round_trip_check(doctor.load_config(root), "copilot_cli")
    assert results[0].status == "OK"
    assert "cost    ?   " in results[0].hint


def test_round_trip_check_uses_provider_omitted_model_label(tmp_path, monkeypatch):
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: custom_cli
            providers:
              custom_cli: {}
            """
        ),
    )
    monkeypatch.chdir(root)

    class _FakeProvider:
        omitted_model_label = "custom CLI default model"

        def invoke(self, *a, **kw):
            assert kw["model"] == ""
            return RunResult(
                trajectory=[],
                metrics=Metrics(
                    wall_time_seconds=0.5,
                    tokens=Tokens(),
                    cost_usd=0,
                    peak_context=0,
                    turn_count=1,
                ),
                model=None,
            )

        def probe_checks(self, result, cfg=None):
            return []

    monkeypatch.setattr(
        "agent_exam.commands.doctor.get_provider", lambda name: _FakeProvider()
    )

    results = doctor._round_trip_check(doctor.load_config(root), "custom_cli")
    assert results[0].status == "OK"
    assert "model=custom CLI default model" in results[0].hint


# --- Suite checks ----------------------------------------------------------

_GOOD_CONFIG = dedent(
    """\
    default_harness: claude_code
    skills_dirs:
      - ./skills
    providers:
      claude_code:
        judge_model: haiku
    """
)


def test_suite_checks_pass_for_clean_suite(tmp_path, monkeypatch):
    root = _make_project(tmp_path, _GOOD_CONFIG)
    (root / "skills").mkdir()
    tasks_dir = root / "evals" / "suites" / "skill-a" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "basic.yaml").write_text("kind: execute\nprompt: x\nassertions: []\n")

    exit_code, results = _run(monkeypatch, root)
    by_name = {r.name: r for r in results}
    assert by_name["skill-a: task files parse"].status == "OK"
    assert exit_code == 0


def test_suite_checks_catch_bad_task(tmp_path, monkeypatch):
    """A malformed task fails the suite's parse check — and fails doctor."""
    root = _make_project(tmp_path, _GOOD_CONFIG)
    (root / "skills").mkdir()
    tasks_dir = root / "evals" / "suites" / "skill-a" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "bad.yaml").write_text(
        "kind: execute\nprompt: x\nassertions:\n  - judege: typo\n"
    )

    exit_code, results = _run(monkeypatch, root)
    assert any(
        r.name == "skill-a: task files parse" and r.status == "FAIL" for r in results
    )
    assert exit_code == 2


def test_suite_checks_catch_missing_fixture(tmp_path, monkeypatch):
    root = _make_project(tmp_path, _GOOD_CONFIG)
    (root / "skills").mkdir()
    tasks_dir = root / "evals" / "suites" / "skill-a" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "fix.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            setup:
              fixture: does-not-exist
            assertions: []
            """
        )
    )

    exit_code, results = _run(monkeypatch, root)
    fixture_check = [r for r in results if "fixtures exist" in r.name]
    assert fixture_check
    assert fixture_check[0].status == "FAIL"
    assert exit_code == 2


# --- Hermetic leak-detection tests -----------------------------------------


def _write_transcript(tmp_path, entries):
    p = tmp_path / "session.jsonl"
    with p.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return p


def test_hermetic_clean_transcript(tmp_path, monkeypatch):
    tr = _write_transcript(
        tmp_path,
        [
            {"type": "attachment", "attachment": {"type": "skill_listing"}},
            {"type": "attachment", "attachment": {"type": "deferred_tools_delta"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            },
        ],
    )
    # No memory root at all.
    monkeypatch.setenv("HOME", str(tmp_path))
    result = hermetic_check(tr)
    assert result.status == "OK"


def test_hermetic_detects_memory_attachment(tmp_path, monkeypatch):
    tr = _write_transcript(
        tmp_path,
        [
            {"type": "attachment", "attachment": {"type": "skill_listing"}},
            {
                "type": "attachment",
                "attachment": {"type": "auto_memory", "content": "..."},
            },
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    result = hermetic_check(tr)
    assert result.status == "WARN"
    assert "auto_memory" in result.hint


def test_hermetic_detects_memory_phrase_injection(tmp_path, monkeypatch):
    # Simulate a home with auto-memory containing a distinctive phrase.
    home = tmp_path / "home"
    memory_dir = home / ".claude" / "projects" / "-some-proj" / "memory"
    memory_dir.mkdir(parents=True)
    sentinel_phrase = (
        "A distinctive sentinel phrase long enough to be unique across builds."
    )
    (memory_dir / "feedback_sample.md").write_text(
        f"---\nname: sample\n---\n\n{sentinel_phrase}\n"
    )
    monkeypatch.setenv("HOME", str(home))

    # Transcript contains that exact phrase (as if Claude leaked it).
    tr = _write_transcript(
        tmp_path,
        [
            {
                "type": "user",
                "message": {"content": f"system note: {sentinel_phrase} (leaked)"},
            }
        ],
    )
    result = hermetic_check(tr)
    assert result.status == "WARN"
    assert "auto-memory phrase" in result.hint


def test_hermetic_skipped_when_transcript_missing():
    result = hermetic_check(None)
    assert result.status == "WARN"
    assert "missing" in result.hint.lower() or "skipped" in result.hint.lower()


def test_hermetic_skipped_when_no_memory_dir(tmp_path, monkeypatch):
    # No auto-memory anywhere → sentinel picker returns None, only attachment
    # check runs.
    tr = _write_transcript(tmp_path, [])
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    result = hermetic_check(tr)
    assert result.status == "OK"


# --- pre-run hook + skills_dirs checks -------------------------------------


def _make_project_with_hook(
    tmp_path: Path, hook_body: str, *, extra_pyproject: str = ""
) -> Path:
    """Same as `_make_project`, but also drops a hook module at project root
    and points `pyproject.toml [tool.agent-exam] pre_run_hook` at it."""
    # Evict any cached `my_hook` from a previous test — otherwise importlib
    # returns the old module and ignores the file we just wrote.
    sys.modules.pop("my_hook", None)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[tool.agent-exam]\n"
        'evals_dir = "evals"\n'
        'pre_run_hook = "my_hook:hook"\n' + extra_pyproject
    )
    (root / "evals").mkdir()
    (root / "evals" / "config.yaml").write_text(
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                judge_model: haiku
            """
        )
    )
    (root / "my_hook.py").write_text(hook_body)
    return root


def test_pre_run_hook_not_configured(tmp_path, monkeypatch):
    """No `pre_run_hook` in pyproject → check renders OK 'not configured'."""
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                judge_model: haiku
            """
        ),
    )
    _exit_code, results = _run(monkeypatch, root)
    statuses = {r.name: r.status for r in results}
    hints = {r.name: r.hint for r in results}
    assert statuses["pre-run hook"] == "OK"
    assert "not configured" in hints["pre-run hook"]


def test_pre_run_hook_runs_and_populates_skills_dirs(tmp_path, monkeypatch):
    """Hook returns a real skills_dirs → both checks OK, count surfaces."""
    skills_dir = tmp_path / "built-skills"
    (skills_dir / "fake-skill").mkdir(parents=True)
    (skills_dir / "fake-skill" / "SKILL.md").write_text("# fake")

    hook = dedent(
        f"""\
        from pathlib import Path
        from agent_exam.config import PreRunRequest, PreRunResult

        def hook(req: PreRunRequest) -> PreRunResult:
            return PreRunResult(skills_dirs=[Path({str(skills_dir)!r})])
        """
    )
    root = _make_project_with_hook(tmp_path, hook)

    _exit_code, results = _run(monkeypatch, root)
    statuses = {r.name: r.status for r in results}
    hints = {r.name: r.hint for r in results}
    assert statuses["pre-run hook"] == "OK"
    assert "1 dir(s)" in hints["pre-run hook"]
    assert statuses["skills available"] == "OK"
    assert "1 skills" in hints["skills available"]


def test_pre_run_hook_failure_is_fail(tmp_path, monkeypatch):
    """Hook raises → FAIL with 'cannot run hook' wording, exit code 2."""
    hook = dedent(
        """\
        from agent_exam.config import PreRunRequest

        def hook(req: PreRunRequest):
            raise RuntimeError("boom")
        """
    )
    root = _make_project_with_hook(tmp_path, hook)

    exit_code, results = _run(monkeypatch, root)
    statuses = {r.name: r.status for r in results}
    hints = {r.name: r.hint for r in results}
    assert exit_code == 2
    assert statuses["pre-run hook"] == "FAIL"
    assert "cannot run hook" in hints["pre-run hook"]
    assert "RuntimeError" in hints["pre-run hook"]


def test_pre_run_hook_does_not_override_locked_skills_dirs(tmp_path, monkeypatch):
    """When `skills_dirs` is set in evals/config.local.yaml the hook's
    returned dirs must NOT replace it — same rule the runner enforces."""
    local_skills = tmp_path / "local-skills"
    (local_skills / "local-only-skill").mkdir(parents=True)
    (local_skills / "local-only-skill" / "SKILL.md").write_text("# local")

    other_skills = tmp_path / "hook-skills"
    (other_skills / "hook-only-skill").mkdir(parents=True)
    (other_skills / "hook-only-skill" / "SKILL.md").write_text("# hook")

    hook = dedent(
        f"""\
        from pathlib import Path
        from agent_exam.config import PreRunRequest, PreRunResult

        def hook(req: PreRunRequest) -> PreRunResult:
            return PreRunResult(skills_dirs=[Path({str(other_skills)!r})])
        """
    )
    root = _make_project_with_hook(tmp_path, hook)

    (root / "evals" / "config.local.yaml").write_text(
        f'skills_dirs:\n  - "{local_skills}"\n'
    )

    _exit_code, results = _run(monkeypatch, root)
    hints = {r.name: r.hint for r in results}
    assert "1 skills" in hints["skills available"]
    assert "local-skills" in str(local_skills)  # guard against fixture typo


def test_skills_dirs_missing_is_warn_when_no_hook(tmp_path, monkeypatch):
    """No hook, no skills_dirs in config → WARN, exit 0 (informational)."""
    root = _make_project(
        tmp_path,
        dedent(
            """\
            default_harness: claude_code
            providers:
              claude_code:
                judge_model: haiku
            """
        ),
    )
    exit_code, results = _run(monkeypatch, root)
    statuses = {r.name: r.status for r in results}
    hints = {r.name: r.hint for r in results}
    assert exit_code == 0
    assert statuses["skills available"] == "WARN"
    assert "skills_dirs not configured" in hints["skills available"]
