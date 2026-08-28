"""
Shared pytest setup.

Two jobs, and the second one matters more than it looks.

Every suite here records results in a module-level `FAIL` list and only its
`__main__` block inspects it. Under pytest nothing did, so a test function that
recorded five failed checks and returned normally was reported as PASSING. That
is exactly what happened when the agent became a multi-agent network: the
standalone runner said `FAILED: exposes triage_film`, and `pytest -q` said
8 passed.

`_fail_on_recorded_failures` closes that: after each test, whatever the module
recorded is asserted on.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail a test whose module recorded a failed check while it ran."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.failed:
        return
    failures = getattr(item.module, "FAIL", None)
    if not failures:
        return
    # Only the ones this test added: suites accumulate across the module.
    seen = getattr(item.session, "_ff_seen_failures", 0)
    new = failures[seen:]
    item.session._ff_seen_failures = len(failures)
    if new:
        report.outcome = "failed"
        report.longrepr = "recorded check failures:\n  - " + "\n  - ".join(new)
