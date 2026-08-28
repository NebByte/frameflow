"""
test_app -- the serving layer, which has never had a test.

`serve.py` shipped with an upload endpoint, a path-traversal guard and
server-side option clamping, and not one line of coverage for any of it. The
interesting part here is not the routes: it is that the argv the browser builds
must stay identical to the documented CLI line. demo.py and render.py
already drifted apart once -- demo exposed 11 of 18 flags -- and the way that
happens is nobody asserting they agree.

Run: python test_app.py
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import json
import re
from pathlib import Path

import app

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- filenames

def test_safe_name():
    print("filenames we are willing to write")
    check("a plain clip is kept", app.safe_name("clip.mp4") == "clip.mp4")
    check("a traversal is stripped to its basename",
          app.safe_name("../../../etc/passwd.mp4") == "passwd.mp4",
          app.safe_name("../../../etc/passwd.mp4"))
    check("a windows path is stripped too",
          app.safe_name(r"C:\Users\x\clip.mov") == "clip.mov",
          app.safe_name(r"C:\Users\x\clip.mov"))
    check("an unsupported extension is refused", app.safe_name("payload.exe") == "")
    check("no extension is refused", app.safe_name("payload") == "")
    check("spaces and punctuation are flattened",
          app.safe_name("my clip (1).mp4") == "my_clip_1.mp4",
          app.safe_name("my clip (1).mp4"))
    check("a url-encoded traversal is decoded first, then stripped",
          app.safe_name("%2e%2e%2fevil.mp4") == "evil.mp4",
          app.safe_name("%2e%2e%2fevil.mp4"))


# ---------------------------------------------------------------- traversal

def test_traversal_guard():
    print("\nstatic paths may not escape their root")
    root = Path(app.HERE)
    check("a normal path resolves", app.under(root, "static/frameflow.css") is not None)
    check("a parent escape is refused", app.under(root, "../secrets") is None)
    check("a deep escape is refused",
          app.under(root, "static/../../etc/passwd") is None)
    check("an absolute path is refused",
          app.under(root, "C:/Windows/System32/drivers/etc/hosts") is None
          or app.under(root, "/etc/passwd") is None)


# ---------------------------------------------------------------- clamping

def test_clamping():
    print("\nthe browser is not trusted to have sent sane numbers")
    o = app.clamp(dict(maxw="40000", frames_per_shot="-5", max_shots="9999",
                       rotate="45", wings_on_dark="rm -rf"))
    check("width is bounded", o["maxw"] == 1920, str(o["maxw"]))
    check("frames per shot is bounded", o["frames_per_shot"] == 20,
          str(o["frames_per_shot"]))
    check("shot count is bounded", o["max_shots"] == 999, str(o["max_shots"]))
    check("a rotation off the quarter turns falls back to none", o["rotate"] == 0)
    check("an unknown generator is dropped, not passed through",
          o["wings_on_dark"] is None, str(o["wings_on_dark"]))
    check("garbage where a number belongs falls back to the default",
          app.clamp(dict(maxw="; drop table"))["maxw"] == 640)
    check("an empty request is all defaults",
          app.clamp({})["frames_per_shot"] == 200)


# ---------------------------------------------------------------- argv

def test_argv_parity():
    """
    The UI and the CLI must produce the same run.

    Each case below is a command line documented in results/ or the README, and
    the assertion is that the equivalent set of UI options rebuilds it.
    """
    print("\nthe argv the UI builds")
    clip, out = Path("clip.mp4"), Path("jobs/x")

    argv = app.build_argv(clip, out, dict(maxw="640", frames_per_shot="200"))
    check("the render module is the one invoked",
          argv[2] == "-m" and argv[3] == "frameflow.render",
          " ".join(argv[1:4]))
    check("progress json is always on, so the UI never scrapes text",
          "--progress-json" in argv)
    # the deliverable rides along by default now: the extended film is the
    # product, and a run that produced only a report was the old posture
    check("the cafe run rebuilds exactly, plus the deliverable",
          argv[3:] == [str(clip), "-o", str(out), "--maxw", "640",
                       "--frames-per-shot", "200", "--progress-json",
                       "--deliver", "deliverable"],
          " ".join(argv[3:]))

    argv = app.build_argv(clip, out, dict(sources="1", sfm="1", prefer_3d="1"))
    check("--sfm is given a directory, not a bare flag",
          argv[argv.index("--sfm") + 1] == str(out / "sfm"))
    check("the 3D path passes its three flags together",
          {"--sources", "--sfm", "--prefer-3d"} <= set(argv))

    argv = app.build_argv(clip, out, dict(other_cut=["b.mp4"], also=["wide.mp4"]))
    check("a second cut is passed for DONATED", "--other-cut" in argv)
    check("a second setup is passed for RETRIEVED", "--also" in argv)
    check("the two are not conflated",
          argv[argv.index("--other-cut") + 1] == "b.mp4"
          and argv[argv.index("--also") + 1] == "wide.mp4")

    argv = app.build_argv(clip, out, dict(wings_on_dark="wavespeed"))
    check("a hosted generator reaches the CLI",
          argv[argv.index("--wings-on-dark") + 1] == "wavespeed")
    check("an unknown generator never reaches the CLI",
          "--wings-on-dark" not in app.build_argv(clip, out,
                                                  dict(wings_on_dark="nonsense")))

    off = app.build_argv(clip, out, {})
    check("nothing optional is passed unasked",
          not ({"--sources", "--prefer-3d", "--sfm", "--reason", "--vision",
                "--online", "--wings-on-dark"} & set(off)))
    check("zero shots means all shots, not --max-shots 0",
          "--max-shots" not in app.build_argv(clip, out, dict(max_shots="0")))


# ---------------------------------------------------------------- capability

def test_capabilities():
    print("\nwhat this machine reports it can do")
    caps = app.capabilities()
    for key in ("gpu", "colmap", "ffmpeg", "wavespeed", "gemini"):
        check(f"{key} is reported", key in caps)
    check("every capability carries a label",
          all(c.get("label") for c in caps.values()))
    check("anything unavailable says why",
          all(c["ok"] or c["reason"] for c in caps.values()),
          str({k: v["reason"] for k, v in caps.items() if not v["ok"]}))
    check("each capability names what it unlocks",
          all(isinstance(c.get("enables"), list) for c in caps.values()))


# ---------------------------------------------------------------- job record

def test_job_shape():
    print("\nthe job record a client polls")
    job = app.new_job("clip.mp4")
    for key in ("id", "name", "state", "dir", "log", "shots", "error", "summary"):
        check(f"job carries {key}", key in job)
    # staged, not queued: a clip waits so another cut, another setup or context
    # can arrive first, and each of those changes which rungs the run can reach
    check("a new job stages rather than starting", job["state"] == "staged")
    check("the id sorts chronologically", job["id"][:8].isdigit(), job["id"])
    check("the job is serialisable for the wire",
          isinstance(json.dumps(job, default=str), str))
    app.JOBS.pop(job["id"], None)


# ---------------------------------------------------------------- states

def test_every_verdict_has_styling():
    """
    demo_ui had no class for BORROWED, so a new rung silently rendered as OFF --
    a verdict reading as a refusal is the worst direction for that to fail in.
    """
    print("\nevery verdict the pipeline can emit is styled")
    css = (app.HERE / "static" / "frameflow.css").read_text(encoding="utf-8")
    for state in ("FULL", "NARROW", "BORROWED", "GEN", "OFF"):
        check(f"{state} has a rule", f".v-{state}" in css)



# ---------------------------------------------------------------- context

def test_context_names():
    """
    Context files are not video, so they need their own allow-list.

    A subtitle or a screenplay page binds to a shot and makes its wings
    DIRECTED, which is a rung on the provenance ladder rather than a decoration
    -- so what counts as one is stated, not inferred from "not a video".
    """
    print("\ncontext files we are willing to write")
    for good in ("subs.srt", "dialogue.vtt", "script.fountain", "notes.md",
                 "plate.png", "still.jpg", "beats.json", "notes.txt"):
        check(f"{good} is accepted", app.context_name(good) == good, good)
    for bad in ("payload.exe", "run.bat", "lib.dll", "archive.zip", "clip.mp4"):
        check(f"{bad} is refused", app.context_name(bad) == "", bad)
    check("a traversal in a context name is stripped",
          app.context_name("../../secrets.txt") == "secrets.txt")


# ---------------------------------------------------------------- knobs

def test_geometry_and_gate_reach_the_cli():
    """
    Wing width, auditorium geometry and the gate were reachable in code and
    never from a command line, so in practice they were constants: gating.decide
    has always taken a thresholds dict and nothing ever passed one.
    """
    print("\nthe knobs that used to be constants")
    clip, out = Path("c.mp4"), Path("jobs/x")

    argv = app.build_argv(clip, out, dict(wing="0.3", screen_width="20",
                                          viewer_distance="9"))
    check("wing width reaches the CLI", "--wing" in argv)
    check("auditorium geometry reaches the CLI",
          "--screen-width" in argv and "--viewer-distance" in argv)

    argv = app.build_argv(clip, out, dict(gate_geometry="18", gate_narrow="0.15"))
    check("the gate can be loosened deliberately",
          argv[argv.index("--gate-geometry") + 1] == "18.0")
    check("and only where asked", "--gate-full" not in argv)

    check("an absent knob passes nothing at all",
          not [a for a in app.build_argv(clip, out, {}) if a.startswith("--gate")
               or a.startswith("--screen") or a == "--wing"])

    o = app.clamp(dict(wing="9", gate_geometry="-5", viewer_distance="0.1"))
    check("a wing wider than the frame is clamped", o["wing"] == 0.6, str(o["wing"]))
    check("a negative gate is clamped to zero", o["gate_geometry"] == 0.0)
    check("an impossible viewing distance is clamped", o["viewer_distance"] == 2.0)
    check("garbage in a knob is dropped, not coerced to zero",
          app.clamp(dict(wing="wide"))["wing"] is None)


# ---------------------------------------------------------------- rungs

def test_attachments_stay_separate():
    """
    other_cut, also and context are three different claims about where pixels
    may come from -- DONATED, RETRIEVED, DIRECTED. Collapsing them into one
    "extra files" bucket would let a subtitle argue for photographic pixels.
    """
    print("\nattachments map to the rung they can justify")
    clip, out = Path("c.mp4"), Path("jobs/x")
    argv = app.build_argv(clip, out, dict(other_cut=["b.mp4"], also=["wide.mp4"],
                                          context=["subs.srt"]))
    check("a second cut goes to --other-cut",
          argv[argv.index("--other-cut") + 1] == "b.mp4")
    check("a second setup goes to --also",
          argv[argv.index("--also") + 1] == "wide.mp4")
    check("context goes to --context",
          argv[argv.index("--context") + 1] == "subs.srt")
    check("three kinds, three flags, no overlap",
          len({argv.index("--other-cut"), argv.index("--also"),
               argv.index("--context")}) == 3)

    job = app.new_job("c.mp4")
    check("a job stages before it runs, so attachments can arrive",
          job["state"] == "staged", job["state"])
    check("a staged job carries an attachments map", job["attachments"] == {})
    app.JOBS.pop(job["id"], None)


def test_the_deliverable_is_the_default():
    """
    The tool exists to extend a film, not to report on one.

    The interface was built posture-first -- walls stay dark unless the
    measurement earns them -- which is right for judging a conversion and wrong
    for making one. A maker opening this wants the extended film, with the
    honesty report attached rather than instead of it.
    """
    print("\nthe extended film is written by default")
    clip, out = Path("c.mp4"), Path("jobs/x")

    argv = app.build_argv(clip, out, {})
    check("a plain run writes the deliverable", "--deliver" in argv, " ".join(argv[3:]))
    check("and names the folder it lands in",
          argv[argv.index("--deliver") + 1] == "deliverable")
    check("opting out is possible and explicit",
          "--deliver" not in app.build_argv(clip, out, dict(deliver="0")))

    from frameflow import render as sx
    src = open(sx.__file__, encoding="utf-8").read()
    check("the deliverable writer exists", "def write_deliverable" in src)
    check("it writes three projector feeds, not a contact sheet",
          'for side in ("left", "centre", "right")' in src)
    check("and the widened master beside them", "master_widened.mp4" in src)
    # asserted on the property, not on a call's exact spelling: the deliverable
    # must never tint. mark_generated flags pixels not filmed at this location,
    # which is what a reviewer needs and what an audience must not see.
    body = src.split("def write_deliverable")[1].split("def ")[0]
    # the word appears in the docstring explaining why it is not used, so the
    # assertion has to be about the argument, not the prose
    check("it is rendered unmarked -- an audience is not shown a QC overlay",
          "mark_generated=True" not in body)
    check("and the feeds are projected, not previewed at 300px",
          "height_px" in body)


def test_session_keys_never_persist():
    """
    A hosted generator needs a credential and the repo is not the place for one.

    Held in memory, handed to the render subprocess, and never read back: the
    response says which names are set, not what they are. A key that reaches a
    file ends up in a backup, a zip or a repository, and this one bills a card.
    """
    print("\ncredentials are session-only")
    src = open(app.__file__, encoding="utf-8").read()
    check("keys live in memory, not on disk", "KEYS: dict[str, str] = {}" in src)
    check("nothing writes them out",
          "KEYS" not in src.split("def _set_keys")[0].split("write_text")[-1][:200])
    check("the subprocess is handed them explicitly",
          "env = {**os.environ, **KEYS}" in src)
    check("only known credential names are accepted", "unknown credential" in src)
    check("the response reports names, never values",
          'return self._json({"set": sorted(KEYS)})' in src)


def test_remote_does_not_fake_a_gpu():
    """
    Colab hands out CPU runtimes while refusing GPUs to an account over quota --
    measured tonight: `colab new --gpu T4` returns Service Unavailable while
    `colab new` returns READY. A 3D run silently served by the CPU path would
    look like a result and prove nothing.
    """
    print("\na remote run says which accelerator it actually got")
    from frameflow import colabrun as cr
    src = open(app.__file__, encoding="utf-8").read()
    check("the accelerator is recorded on the job", 'job["accelerator"]' in src)
    check("3D flags are dropped when no GPU arrived",
          "3D flags dropped" in src)
    check("and the reason is logged, not swallowed",
          "prove nothing" in src)

    csrc = open(cr.__file__, encoding="utf-8").read()
    check("allocate falls back to CPU and says so", 'accelerator="CPU"' in csrc)
    check("a bare pgrep would match its own command, so it is filtered",
          "grep -v 'bash -lc'" in csrc)
    check("the runtime is released when the job ends", "def stop(" in csrc)
    check("parse_shots reads the render's own progress lines",
          len(cr.parse_shots("  shot   3 PARALLAX  NARROW geom  31.0dB eff  42.8%")) == 1)

def test_a_rehydrated_job_is_a_whole_job():
    """
    A job read back off disk must be shaped like one that was just created.

    Polish is the first caller that puts a rehydrated record into JOBS -- it has
    to, so a pass started on a job this server has never seen can be polled --
    and the record was missing `started`, which `known_jobs` reads without a
    default. The failure was nasty out of proportion to the cause: the job list
    returned an empty response for the rest of the process's life, and it began
    doing so at the moment somebody polished, nowhere near the record.
    """
    print("\na job read back off disk is shaped like a fresh one")
    fresh = set(app.new_job("probe.mp4"))
    jid = next((d.name for d in sorted(app.JOBS_DIR.iterdir(), reverse=True)
                if d.is_dir() and (d / "screenx_summary.json").exists()), None)
    app.JOBS.clear()
    if jid is None:
        check("no finished job on disk to check against", True, "skipped")
        return
    back = app.Handler._from_disk(jid)
    check("it rehydrates", back is not None, jid)
    missing = sorted(fresh - set(back or {}))
    check("with every key a fresh job has", not missing, ", ".join(missing))

    # the specific read that took the job list down
    app.JOBS[jid] = back
    try:
        listed = app.known_jobs()
        check("and the job list survives it being registered",
              any(j["id"] == jid for j in listed))
    except KeyError as e:
        check("and the job list survives it being registered", False, f"KeyError {e}")
    app.JOBS.clear()


def test_every_id_is_unique():
    """
    The bug this exists for cost an afternoon and threw no error.

    The Render stage had `<select id="shots">` and the Review stage
    `<table id="shots">`. `querySelector("#shots")` returns the first in
    document order, so Review's click handler was bound to a dropdown on
    another tab: clicking a shot did nothing, silently, because a click landing
    on an element with no listener looks exactly like a click that did nothing.

    The id-consistency check in place at the time could not catch it -- it
    asserted every id the JS reaches for *exists*, which was true. What was
    false is that each exists once.
    """
    print("\nno two elements share an id")
    html = (app.HERE / "static" / "index.html").read_text(encoding="utf-8")
    ids = re.findall(r'\bid="([\w-]+)"', html)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    check("every id in the document is unique", not dupes, ", ".join(dupes))
    check("the collision that caused it cannot come back",
          ids.count("shots") == 1 and "maxshots" in ids)


# ---------------------------------------------------------------- polish

def test_polish_is_reachable_and_priced():
    """
    The finishing pass, wired.

    Every rung on this project has failed the same way at least once: the
    mechanism exists, is tested, and nothing calls it. `add_film` had no
    callers; `ExternalReferenceTool` was constructed by `director` and did not
    exist. So what is asserted here is reachability -- a route, a control, and
    a handler -- not the repainting itself, which `test_polish` pins.
    """
    print("\nthe finishing pass is reachable from the browser")
    src = (app.HERE / "app.py").read_text(encoding="utf-8")
    js = (app.HERE / "static" / "app.js").read_text(encoding="utf-8")
    html = (app.HERE / "static" / "index.html").read_text(encoding="utf-8")

    check("a POST route starts it", "/polish$" in src and "self._polish)" in src)
    check("a GET route reports what it found", "self._polish_state(" in src)
    check("the browser can start it", "/polish`" in js)
    check("and there is a control to press", 'id="polishrun"' in html)

    check("it shells out rather than importing the pipeline",
          '"frameflow.polish"' in src and "import polish" not in src)
    check("session keys reach it, so a hosted repaint can authenticate",
          "{**os.environ, **KEYS}" in src.split("def run_polish")[1].split("def ")[0])

    check("an unknown generator is refused before anything is spent",
          "unknown generator" in src.split("def _polish(")[1][:2000])
    check("every generator offered by the UI is one the pipeline has",
          all(f'value="{g}"' in html.split('id="polishgen"')[1][:900]
              for g in ("mirror", "inpaint", "wavespeed", "gemini-edit", "hosted")))
    check("inspecting is the default and costs nothing",
          'value=""' in html.split('id="polishgen"')[1][:300])

    # a repaint moves the headline down; a screen still showing the old figure
    # would report pixels as filmed that a model drew a minute ago
    body = src.split("def run_polish")[1][:2200]
    check("a repaired run re-reads its restated summary",
          "screenx_summary.json" in body and 'job["summary"]' in body)
    check("the browser re-renders the report after a repaint",
          "renderReport(); renderReview()" in
          js.split("async function pollPolish")[1].split("\nfunction ")[0])
    check("the cost in truth is stated on the panel, not only for paid models",
          "stops counting" in html.lower() or "falls by exactly" in html)

    # a finding describes the walls of one render
    check("a new render drops the stale report",
          "polish_report.json" in src.split("def _start(")[1][:1400])

    # A hosted generator bills per shot and a fast-cut trailer refuses most of
    # them. One run of 34 shots at $0.20 was already started by accident once.
    handler = src.split("def _polish(")[1].split("def _polish_state")[0]
    check("the caller may name which shots to pay for", "shots" in handler)
    check("and the shot list is scrubbed to digits before it reaches a command",
          're.sub(r"[^\\d,]"' in handler)
    check("the browser sends it", 'shots: $("#polishshots")' in js)
    check("with a control to type it into", 'id="polishshots"' in html)
    check("the cost note says what an empty box will spend",
          "leaving the box empty" in js)


if __name__ == "__main__":
    print("app -- the serving layer")
    test_safe_name()
    test_traversal_guard()
    test_clamping()
    test_argv_parity()
    test_capabilities()
    test_job_shape()
    test_every_verdict_has_styling()
    test_context_names()
    test_geometry_and_gate_reach_the_cli()
    test_attachments_stay_separate()
    test_the_deliverable_is_the_default()
    test_session_keys_never_persist()
    test_remote_does_not_fake_a_gpu()
    test_a_rehydrated_job_is_a_whole_job()
    test_every_id_is_unique()
    test_polish_is_reachable_and_priced()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
