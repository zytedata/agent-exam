"""Assertion config validation — the per-assertion `validate` functions
and the registry's `validate_config`.

Each assertion's own test file already covers the `check`-side behavior
(bad config → fail-result, not a crash). These tests cover the load-time
path: `validate` raises `UsageError`, so a malformed config fails before
an agent run rather than silently during scoring.
"""

from __future__ import annotations

import pytest

from agent_exam.assertions.registry import validate_config
from agent_exam.errors import UsageError

# (type, config) pairs that must validate cleanly.
_VALID = [
    ("file_exists", "out/x.py"),
    ("file_exists", {"path": "out/x.py"}),
    ("file_contains", {"path": "a.py", "pattern": "x"}),
    ("file_contains", {"path": "a.py", "pattern": "x", "regex": True}),
    ("tool_called", "Bash"),
    ("tool_called", {"name": "Bash"}),
    ("tool_not_called", "WebFetch"),
    ("tool_count", {"name": "Bash", "exactly": 2}),
    ("tool_count", {"name": "Bash", "min": 1, "max": 3}),
    ("tool_count", {"name": "Bash", "min": 0}),
    ("first_skill", "scrape-codegen"),
    ("judge", "some criterion"),
    ("judge", {"criterion": "x", "include_trajectory": False}),
    ("judge", {"criterion": "x", "pass_on": ["YES", "NO"]}),
    ("judge_agent", "some criterion"),
    ("judge_agent", {"criterion": "x"}),
    ("judge_agent", {"criterion": "x", "include_trajectory": True, "pass_on": ["YES"]}),
    ("skill_invoked", "scrape-codegen"),
    ("skill_not_invoked", "scrape-codegen"),
    ("no_permission_errors", None),
    ("no_permission_errors", {}),
]

# (type, config, substring-expected-in-error) that must be rejected.
_INVALID = [
    # Pydantic-migrated assertions: error messages come from pydantic.
    # Non-dict shorthand is wrapped to {<key>: <value>} first, so bad
    # scalars produce field-level errors (e.g. "path: ... valid string")
    # rather than a top-level "valid dictionary".
    ("file_exists", 42, "valid string"),
    ("file_exists", {}, "Field required"),
    ("file_exists", {"path": "x", "typo": "y"}, "Extra inputs are not permitted"),
    ("file_contains", "scalar", "valid dictionary"),
    ("file_contains", {"path": "a.py"}, "pattern"),
    ("tool_called", 1, "valid string"),
    ("tool_called", {}, "Field required"),
    ("tool_called", {"name": "Bash", "tyop": "x"}, "Extra inputs"),
    ("first_skill", {"skill": 123}, "valid string"),
    ("skill_invoked", None, "valid string"),
    ("tool_count", "Read", "valid dictionary"),
    ("tool_count", {"name": "Bash"}, "needs"),
    ("tool_count", {"name": "Bash", "exactly": 1, "min": 1}, "both"),
    ("tool_count", {"exactly": 1}, "name"),
    ("tool_count", {"name": 123, "exactly": 1}, "valid string"),
    ("tool_count", {"name": ""}, "at least 1 character"),
    ("tool_count", {"name": "Bash", "exactly": -1}, "greater than or equal to 0"),
    ("tool_count", {"name": "Bash", "min": "two"}, "valid integer"),
    ("tool_count", {"name": "Bash", "exactly": True}, "valid integer"),
    ("tool_count", {"name": "Bash", "min": 5, "max": 2}, "<="),
    ("tool_count", {"name": "Bash", "min": 1, "typo": 2}, "Extra inputs"),
    ("file_contains", {"path": "a.py", "pattern": "x", "rexex": True}, "Extra inputs"),
    ("file_contains", {"path": 1, "pattern": "x"}, "valid string"),
    (
        "file_contains",
        {"path": "a.py", "pattern": "x", "regex": "yes"},
        "valid boolean",
    ),
    ("file_contains", {"path": "a.py", "pattern": "[", "regex": True}, "invalid regex"),
    ("judge", {}, "criterion"),
    ("judge", {"criterion": "   "}, "empty or whitespace"),
    ("judge", 42, "valid string"),
    ("judge", {"criterion": "x", "pass_on": "YES"}, "valid list"),
    ("judge", {"criterion": "x", "include_trajectory": "no"}, "valid boolean"),
    ("judge", {"criterion": "x", "critereon": "y"}, "Extra inputs"),
    ("judge_agent", {}, "criterion"),
    ("judge_agent", {"criterion": "x", "pass_on": []}, "at least 1 item"),
    ("judge_agent", {"criterion": "x", "pass_on": [""]}, "non-empty"),
    ("no_permission_errors", {"ignore": ["Bash"]}, "Extra inputs"),
    ("no_permission_errors", "oops", "valid dictionary"),
]


@pytest.mark.parametrize(("type_name", "config"), _VALID)
def test_valid_configs_pass(type_name, config):
    validate_config(type_name, config)  # must not raise


@pytest.mark.parametrize(("type_name", "config", "expected"), _INVALID)
def test_invalid_configs_rejected(type_name, config, expected):
    with pytest.raises(UsageError, match=expected):
        validate_config(type_name, config)


def test_unknown_type_rejected():
    with pytest.raises(UsageError, match="unknown assertion type"):
        validate_config("bogus", "x")
