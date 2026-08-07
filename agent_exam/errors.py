class AgentExamError(Exception):
    """Base class for all framework-raised exceptions."""

    exit_code = 2


class UsageError(AgentExamError):
    """Bad user input — config, CLI args, task YAML."""

    exit_code = 2


class FrameworkError(AgentExamError):
    """Internal framework problem — provider crash, corrupted artifacts, etc."""

    exit_code = 2


class RateLimitError(AgentExamError):
    """Transient 429/529 from the harness. Caught and retried by the retry loop."""

    exit_code = 3

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitExhausted(AgentExamError):
    """Too many consecutive 429/529s. Aborts the run with exit code 3."""

    exit_code = 3


class ProviderTimeout(AgentExamError):
    """Raised when the harness didn't exit within its wall-clock budget.

    Carries `partial_run_result` when the provider managed to recover
    a usable trajectory from the killed process (session_id seen, raw
    transcript readable). Callers that record artifacts should prefer
    the partial over `None` — the transcript up to the kill point is
    the difference between diagnosable and not.
    """

    exit_code = 1

    def __init__(self, message: str, partial_run_result=None):
        super().__init__(message)
        self.partial_run_result = partial_run_result
