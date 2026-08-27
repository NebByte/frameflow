"""
tools -- what the agent is allowed to do.

Every one of these is a thin wrapper over code that already exists and is
already tested. That is deliberate: an agent that reimplements the pipeline
inside its own tool layer is an agent whose numbers nobody has checked. The
measurements a tool returns are the same objects `screenx_render`, `gating`
and `polish` produce on the command line, and if the two ever disagree the
tests catch it, not a demo.

Tools return plain dicts, and they return failures as dicts too. A tool that
raises hands the model a stack trace and invites it to invent a summary of what
went wrong; a tool that returns {"ok": false, "error": ...} lets it say so.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
JOBS = HERE / "jobs"
MEDIA_DIRS = [HERE / "demos", HERE / "media", JOBS]

# Long renders run in a thread and report progress here rather than blocking a
# chat turn for two hours.
_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _fail(msg, **extra):
    return dict(ok=False, error=str(msg), **extra)


def _resolve(video: str) -> Path | None:
    """A caller-supplied path, checked against the places films actually live."""
    p = Path(video)
    if p.is_file():
        return p
    for d in MEDIA_DIRS:
        cand = d / video
        if cand.is_file():
            return cand
    return None


# ------------------------------------------------------------------ discovery

def list_films() -> dict:
    """
    List the video files available to work on, and the finished jobs on disk.

    Use this first when the user refers to a film by name rather than by path,
    or when they ask what there is to look at.
    """
    films = []
    for d in MEDIA_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.mp4")):
            films.append(dict(name=f.name, path=str(f),
                              megabytes=round(f.stat().st_size / 1e6, 1)))
    jobs = []
    if JOBS.is_dir():
        for j in sorted(p for p in JOBS.iterdir() if p.is_dir()):
            s = j / "screenx_summary.json"
            if not s.exists():
                continue
            try:
                d = json.loads(s.read_text(encoding="utf-8"))
            except ValueError:
                continue
            jobs.append(dict(job=j.name, source=d.get("source"),
                             shots=d.get("shots"),
                             mean_real_wing=d.get("mean_real_wing"),
                             partial=bool(d.get("partial"))))
    return dict(ok=True, films=films, jobs=jobs)


# ------------------------------------------------------------------ the answer

def triage_film(video: str, max_shots: int = 0, working_width: int = 480) -> dict:
    """
    Decide, per shot, whether 270-degree side walls can be recovered from the
    film's own footage -- WITHOUT rendering anything.

    This is the cheap question and usually the one worth asking first: it runs
    the same motion classifier, geometry hold-out and gate that a real render
    uses, on a window of each shot, and returns the verdict each shot would
    get. Seconds per shot instead of hours per film.

    Every figure it returns is a floor. A full render holds more footage than
    the window does and recovers more, never less.

    Args:
        video: file name or path of the film.
        max_shots: stop after this many shots. 0 means every shot.
        working_width: pixel width to analyse at. 480 is fast and enough.

    Returns:
        Per-shot verdicts (FULL / NARROW / BORROWED / GEN / OFF / LOCKED) with
        geometry dB and effective coverage, plus how much of the running time
        can be widened from the film's own photography.
    """
    path = _resolve(video)
    if path is None:
        return _fail(f"no such film: {video}", hint="call list_films first")
    try:
        import triage as tr
        rep = tr.triage_film(str(path), maxw=int(working_width),
                             max_shots=int(max_shots) or None, verbose=False)
        return dict(ok=True, **rep)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ rendering

def render_film(video: str, working_width: int = 480, frames_per_shot: int = 200,
                max_shots: int = 1) -> dict:
    """
    Actually convert a film to 270-degree three-wall output. Runs in the
    background and returns a job id immediately.

    This is the expensive operation -- minutes to hours depending on width and
    frame count. Prefer triage_film when the user is asking WHETHER something
    can be converted. Use this when they want the film itself.

    Args:
        video: file name or path of the film.
        working_width: pixel width to render at. Higher is sharper and slower.
        frames_per_shot: cap on frames per shot. This is a CAP, not a target --
            a shot longer than this is cut off at it.
        max_shots: how many shots to convert. 0 means all of them.

    Returns:
        A job id. Poll it with render_status.
    """
    path = _resolve(video)
    if path is None:
        return _fail(f"no such film: {video}", hint="call list_films first")

    job_id = f"agent-{uuid.uuid4().hex[:8]}"
    outdir = JOBS / job_id
    state = dict(ok=True, job=job_id, state="running", log=[], summary=None,
                 error="")
    with _LOCK:
        _RUNS[job_id] = state

    def work():
        try:
            import screenx_render as sx
            sx.run(str(path), outdir=str(outdir), maxw=int(working_width),
                   max_shots=int(max_shots) or None,
                   frames_per_shot=int(frames_per_shot),
                   deliver="deliverable")
            s = outdir / "screenx_summary.json"
            if s.exists():
                state["summary"] = json.loads(s.read_text(encoding="utf-8"))
            state["state"] = "done"
        except Exception as e:                    # a render must not kill the agent
            state["state"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return dict(ok=True, job=job_id, state="running",
                note="rendering in the background; call render_status(job)")


def render_status(job: str) -> dict:
    """
    How a background render is doing, and its results once it finishes.

    Args:
        job: the job id returned by render_film.
    """
    with _LOCK:
        st = _RUNS.get(job)
    if st is None:
        d = JOBS / job / "screenx_summary.json"
        if d.exists():
            return dict(ok=True, job=job, state="done",
                        summary=json.loads(d.read_text(encoding="utf-8")))
        return _fail(f"no such job: {job}")
    out = dict(ok=True, job=job, state=st["state"], error=st["error"])
    s = st.get("summary")
    if s:
        out["shots"] = s.get("shots")
        out["mean_real_wing"] = s.get("mean_real_wing")
        out["wings_on"] = s.get("wings_on")
        out["wings_generated"] = s.get("wings_generated")
        out["per_shot"] = s.get("per_shot")
    return out


# ------------------------------------------------------------------ finishing

def inspect_walls(job: str) -> dict:
    """
    Measure what is actually wrong with a converted film's side walls.

    Reports four things the conversion gate cannot see: thin dark lines,
    how much the walls shimmer relative to the picture, whether the wall joins
    the centre or is cut against it, and streaking.

    Args:
        job: a job id, or the name of a directory under jobs/.
    """
    d = JOBS / job
    if not (d / "screenx_summary.json").exists():
        return _fail(f"no rendered job at {d}")
    try:
        import polish
        rep = polish.inspect(str(d), vision=lambda *a, **k: [], verbose=False)
        return dict(ok=True, job=job, findings=rep.get("findings"),
                    faulted=rep.get("repairable"))
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


def settle_walls(job: str) -> dict:
    """
    Fix a converted film's walls using only its own photography. Free, and it
    invents nothing, so the real-footage figure does not move.

    Removes thin dark lines and damps the shimmer by medianing each frame's
    wall against its own aligned neighbours. Always prefer this before
    suggesting anything that generates pixels.

    Args:
        job: a job id, or the name of a directory under jobs/.
    """
    d = JOBS / job
    if not (d / "screenx_summary.json").exists():
        return _fail(f"no rendered job at {d}")
    try:
        import polish
        done = polish.settle(str(d), verbose=False)
        return dict(ok=True, job=job, settled=done,
                    note="nothing was invented; mean_real_wing is unchanged")
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ the ledger

def record_run(job: str) -> dict:
    """
    Write a finished conversion into the ClickHouse ledger: one row per shot,
    carrying its verdict, where every pixel came from, and how the wall
    measures. Refused shots are recorded too.

    Do this after a render so the film can be compared with everything else
    that has been analysed.

    Args:
        job: a job id, or the name of a directory under jobs/.
    """
    d = JOBS / job
    if not (d / "screenx_summary.json").exists():
        return _fail(f"no rendered job at {d}")
    try:
        import ledger
        run_id, n = ledger.write_run(str(d), verbose=False)
        return dict(ok=True, job=job, run_id=run_id, rows=n)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


def ledger_examples() -> dict:
    """
    The questions the ledger was built to answer, with the SQL for each.

    Useful when the user asks what can be queried, or when composing a new
    question and you want to see the column names in context.
    """
    import ledger
    return dict(ok=True, table=ledger.TABLE,
                columns=list(ledger.COLUMNS),
                examples={k: v for k, v in ledger.EXAMPLES.items()})
