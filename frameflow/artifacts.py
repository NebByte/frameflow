"""
artifacts -- what a finished job is called on disk.

One place knows these names, because two things need them and they disagree
about history: new runs write `frameflow_*`, and every job rendered before the
project took that name is sitting in `jobs/` under `screenx_*`. A rename that
orphaned those would have thrown away the measured runs the README quotes.

So writes always use the current name, reads accept either, and the legacy
constants stay until there is nothing left that predates them.
"""
from __future__ import annotations

from pathlib import Path

SUMMARY = "frameflow_summary.json"
DEMO = "frameflow_demo.mp4"

# Written by every run before the project was called Frameflow.
LEGACY_SUMMARY = "screenx_summary.json"
LEGACY_DEMO = "screenx_demo.mp4"


def summary_path(job_dir) -> Path:
    """
    The run summary for a job, whichever name it was written under.

    Returns the CURRENT name when neither exists, so a caller creating one
    writes the new name and a caller reporting a missing file names the file
    somebody would now expect to see.
    """
    job = Path(job_dir)
    new = job / SUMMARY
    if new.exists():
        return new
    old = job / LEGACY_SUMMARY
    return old if old.exists() else new


def demo_path(job_dir) -> Path:
    """The review cut for a job, whichever name it was written under."""
    job = Path(job_dir)
    new = job / DEMO
    if new.exists():
        return new
    old = job / LEGACY_DEMO
    return old if old.exists() else new


def has_summary(job_dir) -> bool:
    """Whether this directory holds a finished (or checkpointed) run."""
    return summary_path(job_dir).exists()
