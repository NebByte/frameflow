"""
test_app -- the serving layer, which has never had a test.

`serve.py` shipped with an upload endpoint, a path-traversal guard and
server-side option clamping, and not one line of coverage for any of it. The
interesting part here is not the routes: it is that the argv the browser builds
must stay identical to the documented CLI line. demo.py and screenx_render.py
already drifted apart once -- demo exposed 11 of 18 flags -- and the way that
happens is nobody asserting they agree.

Run: python test_app.py
"""
from __future__ import annotations

import json
import sys
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
    check("a normal path resolves", app.under(root, "static/screenx.css") is not None)
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
    check("the render script is the one invoked",
          argv[2].endswith("screenx_render.py"), argv[2])
    check("progress json is always on, so the UI never scrapes text",
          "--progress-json" in argv)
    check("the cafe run rebuilds exactly",
          argv[3:] == [str(clip), "-o", str(out), "--maxw", "640",
                       "--frames-per-shot", "200", "--progress-json"],
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
    css = (app.HERE / "static" / "screenx.css").read_text(encoding="utf-8")
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

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
