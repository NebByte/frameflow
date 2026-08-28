"""
test_context -- taking any file as context, and the human in the loop.

Run: python test_context.py

The assertions that matter here are not about parsing. They are about BINDING
and LABELLING: that a subtitle lands on the shot it was spoken over, that an
unbound script page does not, that a human note beats both, and that none of it
can reach the coverage number.

All CPU, no network, no model.
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import json
import tempfile
from pathlib import Path

from frameflow import context as cx
from frameflow import provenance as P
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


SRT = """1
00:00:01,000 --> 00:00:04,000
Get to the roof, now!

2
00:00:12,500 --> 00:00:15,000
<i>The helicopter is on the east side.</i>

3
00:01:40,000 --> 00:01:43,000
We lost him.
"""

SCRIPT = """INT. WAREHOUSE - NIGHT

A cavernous room. Rain through the skylights.

EXT. ROOFTOP - CONTINUOUS

Wind. The city below.
"""


def fixture(d: Path):
    (d / "subs.srt").write_text(SRT, encoding="utf-8")
    (d / "script.fountain").write_text(SCRIPT, encoding="utf-8")
    (d / "notes.txt").write_text("second unit plates are in the drive", encoding="utf-8")
    (d / "extra.json").write_text(json.dumps(
        [{"kind": "note", "text": "practical neon left of frame", "t_start": 12.0,
          "t_end": 16.0, "source": "supervisor"}]), encoding="utf-8")
    (d / "mystery.xyz").write_text("???", encoding="utf-8")


def test_loading():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        fixture(d)
        b = cx.ContextBundle.from_paths([d])
        s = b.summary()

        check("reads a mixed folder", s["items"] >= 7, str(s["kinds"]))
        check("subtitles parsed with timecodes",
              s["kinds"].get("dialogue") == 3, str(s["kinds"].get("dialogue")))
        check("screenplay split on scene headings",
              s["kinds"].get("scene") == 2, str(s["kinds"].get("scene")))
        check("json items keep their timing",
              any(i.t_start == 12.0 for i in b.items))
        check("an unreadable format is recorded, not swallowed",
              s["unusable"] >= 1, f"{s['unusable']} unusable")

        d0 = [i for i in b.items if i.kind == "dialogue"][0]
        check("timecode decoded", abs(d0.t_start - 1.0) < 1e-6 and abs(d0.t_end - 4.0) < 1e-6,
              f"{d0.t_start}-{d0.t_end}")
        check("subtitle markup stripped",
              all("<" not in i.text for i in b.items if i.kind == "dialogue"))


def test_binding():
    """The whole point: which context belongs to which shot."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        fixture(d)
        b = cx.ContextBundle.from_paths([d])
        fps = 24.0

        # shot covering 12.0-15.5s should pick up subtitle 2 and not subtitle 1
        got = b.for_shot(3, start_frame=int(12.0 * fps), n_frames=int(3.5 * fps), fps=fps)
        texts = " ".join(i.text for i in got)
        check("dialogue spoken over the shot binds to it", "helicopter" in texts)
        check("dialogue from elsewhere does not", "roof, now" not in texts)
        check("timed json note binds too", "neon" in texts)

        early = b.for_shot(0, start_frame=0, n_frames=int(2 * fps), fps=fps)
        check("a different shot gets its own line",
              "roof, now" in " ".join(i.text for i in early))

        check("unbound screenplay is offered to every shot",
              any(i.kind == "scene" for i in got) and any(i.kind == "scene" for i in early))
        check("unbound screenplay is not marked bound",
              all(not i.bound for i in b.items if i.kind == "scene"))


def test_directions():
    with tempfile.TemporaryDirectory() as td:
        store = cx.DirectionStore(td)
        store.add(7, "fire escape and a lit window, camera left", author="alon")
        store.add(9, "empty alley, no people")

        check("note pinned to its shot", len(store.for_shot(7)) == 1)
        check("does not leak to another shot", len(store.for_shot(8)) == 0)

        again = cx.DirectionStore(td)
        check("survives a restart", len(again.for_shot(7)) == 1,
              f"{len(again.items)} on disk")
        check("records who said it", again.for_shot(7)[0].author == "alon")
        check("a note is bound by construction", again.for_shot(7)[0].bound)


def test_prompt_priority():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        fixture(d)
        b = cx.ContextBundle.from_paths([d])
        fps = 24.0
        items = b.for_shot(3, int(12.0 * fps), int(3.5 * fps), fps)

        p_no_note = cx.build_prompt(items, fallback="dim blue exterior background plate")
        b.add(cx.ContextItem("note", "fire escape, camera left", shot=3, author="alon"))
        items2 = b.for_shot(3, int(12.0 * fps), int(3.5 * fps), fps)
        p_note = cx.build_prompt(items2)

        check("context beats the pixel-only fallback",
              "dim blue" not in p_no_note, p_no_note[:60])
        check("a human note leads the prompt",
              p_note.index("fire escape") < p_note.index("helicopter")
              if "helicopter" in p_note else True, p_note[:80])
        check("prompt stays inside a sane length", len(p_note) <= 420, f"{len(p_note)} chars")
        check("the shot's dialogue is in there", "helicopter" in p_note)


def test_provenance_rules():
    """The line that keeps every one of these features honest."""
    bound = [cx.ContextItem("note", "fire escape", shot=3)]
    loose = [cx.ContextItem("scene", "INT. WAREHOUSE - NIGHT")]
    empty = [cx.ContextItem("note", "", shot=3)]

    check("a bound assertion makes it DIRECTED",
          cx.provenance_for(bound) == P.DIRECTED)
    check("unbound screenplay alone does not",
          cx.provenance_for(loose) == P.GENERATED)
    check("an empty note does not either",
          cx.provenance_for(empty) == P.GENERATED)
    check("nothing at all is GENERATED", cx.provenance_for([]) == P.GENERATED)

    check("DIRECTED is outside the headline number",
          P.DIRECTED not in P.PHOTOGRAPHIC and P.DIRECTED not in P.REAL_LEVELS)
    check("DIRECTED is counted as invention",
          P.DIRECTED in P.NOT_THIS_PLACE)
    check("DIRECTED ranks above free generation", P.DIRECTED < P.GENERATED)
    check("DIRECTED ranks below real photons", P.DIRECTED > P.REFERENCED)


def test_no_metric_movement():
    """A wing full of directed pixels must not move real_same_camera."""
    import numpy as np
    from frameflow import agent as ag
    wing_w, w, h = 40, 120, 60
    prov = np.full((h, w + 2 * wing_w), P.DIRECTED, np.uint8)
    prov[:, wing_w:wing_w + w] = P.PRIMARY
    r = ag.WingAgent.report(prov, wing_w, w)

    check("a fully directed wing reports 0% real", r["real_same_camera"] == 0.0,
          str(r["real_same_camera"]))
    check("and 0% photographic", r["photographic"] == 0.0, str(r["photographic"]))
    check("it is reported, not hidden", r.get("directed", 0) == 1.0,
          str(r.get("directed")))


if __name__ == "__main__":
    print("loading any file")
    test_loading()
    print("binding to shots")
    test_binding()
    print("human in the loop")
    test_directions()
    print("prompt construction")
    test_prompt_priority()
    print("provenance rules")
    test_provenance_rules()
    print("the metric holds")
    test_no_metric_movement()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
