"""
test_reasoning -- does the plan follow from the evidence, and say which is which.

Run: python test_reasoning.py

The excursions here are the real ones, recovered by `offscreen.py` from the
ground-truth fixture, not hand-written. So these assertions cover the whole
chain: pixels -> detections -> tracks -> excursion -> plan.

The assertion that matters most is `test_modes_differ`. Interpolating mode may
use the observed return because the film shows it; causal mode may not, because
that is the mode the harness scores. If those two ever produce the same plan,
either the harness is being handed the answer or the renderer is throwing away
evidence it is entitled to.
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from frameflow import context as cx
from frameflow import offscreen as off
from frameflow import provenance as P
from frameflow import reasoning as rz
import test_offscreen as tof

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def real_excursion():
    """One genuine excursion, recovered from pixels by the Tier 3.1 stack."""
    frames, truth = tof.make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)
    return ex, truth, w, h, len(frames)


def a_brief(ex, w, h, n, items=(), wing_frac=0.22):
    return rz.brief_for(dict(shot=3, motion="LOCKED"), n_frames=n,
                        wing_w=int(w * wing_frac), frame_w=w, frame_h=h,
                        excursions=ex, context_items=items)


def test_measured_from_excursion():
    ex, truth, w, h, n = real_excursion()
    check("the fixture yielded an excursion to reason about", len(ex) == 1)
    if not ex:
        return
    plan = rz.LocalReasoner().plan(a_brief(ex, w, h, n))
    figs = plan.by_support("measured")
    check("the excursion became a measured element", len(figs) == 1)
    if not figs:
        return
    e = figs[0]
    check("placed on the wall it left by", e.side == ex[0].side, e.side)
    check("height taken from where it left",
          abs(e.y_frac - ex[0].exit_y / h) < 0.01, f"{e.y_frac:.3f}")
    check("carries its arithmetic", "px/frame" in e.because, e.because[:52])
    check("it is not called asserted", e.support == "measured")


def test_modes_differ():
    """Causal may not see the return. Interpolating may."""
    ex, truth, w, h, n = real_excursion()
    if not ex:
        return
    r = rz.LocalReasoner(depth_frac=0.22)
    inter = r.plan(a_brief(ex, w, h, n), mode="interpolating").by_support("measured")[0]
    causal = r.plan(a_brief(ex, w, h, n), mode="causal").by_support("measured")[0]

    check("interpolating cites the observed return",
          "returned" in inter.because, inter.because[:60])
    check("causal refuses to", "return not used" in causal.because,
          causal.because[:60])
    check("and they reach different depths",
          abs(inter.depth_frac - causal.depth_frac) > 0.05,
          f"{inter.depth_frac:.2f} vs {causal.depth_frac:.2f}")


def test_path_lands_when_the_film_says():
    """Interpolating placement must return to the edge on the observed frame."""
    ex, truth, w, h, n = real_excursion()
    if not ex:
        return
    e = rz.LocalReasoner().plan(a_brief(ex, w, h, n)).by_support("measured")[0]
    last = e.frames[1]
    check("the figure is back at the edge when it was filmed returning",
          abs(last - ex[0].entry_frame) <= 1, f"plan f{last}, filmed f{ex[0].entry_frame}")
    check("and it is not out there before it left",
          e.frames[0] > ex[0].exit_frame, f"starts f{e.frames[0]}")


def test_depth_beyond_the_wall_is_visible():
    """
    If the figure goes further out than the wall reaches, say so.

    Wings are 22% of frame width. Measured on the fixture the action goes out
    about 37%, so for most of its absence the figure is beyond the projectable
    wall entirely and no amount of generation should put it there.
    """
    ex, truth, w, h, n = real_excursion()
    if not ex:
        return
    e = rz.LocalReasoner().plan(a_brief(ex, w, h, n)).by_support("measured")[0]
    check("depth is reported in wing-widths", e.depth_frac > 0,
          f"{e.depth_frac:.2f} wing-widths")
    check("a figure that outruns the wall is detectable from the plan",
          e.depth_frac > 1.0, f"{e.depth_frac:.2f} -- beyond the wall")


def test_context_becomes_asserted():
    ex, truth, w, h, n = real_excursion()
    items = [cx.ContextItem("note", "fire escape, camera left", shot=3, author="alon"),
             cx.ContextItem("dialogue", "The helicopter is on the east side.",
                            t_start=1.0, t_end=4.0),
             cx.ContextItem("scene", "INT. WAREHOUSE - NIGHT")]     # unbound
    plan = rz.LocalReasoner().plan(a_brief(ex, w, h, n, items))
    asserted = plan.by_support("asserted")
    check("bound context became asserted elements", len(asserted) == 2,
          f"{len(asserted)} of 3 items")
    check("unbound screenplay was not promoted",
          all("WAREHOUSE" not in e.text for e in asserted))
    check("a side named in the text is picked up",
          any(e.side == "L" for e in asserted))
    check("who said it is recorded", any("alon" in e.because for e in asserted))


def test_nothing_at_all():
    plan = rz.LocalReasoner().plan(a_brief([], 320, 180, 60))
    check("with no evidence it plans continuation only",
          [e.support for e in plan.elements] == ["inferred"])
    check("and admits why", "nothing was measured" in plan.elements[0].because)
    check("pixels from that are GENERATED", plan.label() == P.GENERATED)


def test_labels():
    ex, truth, w, h, n = real_excursion()
    measured = rz.LocalReasoner().plan(a_brief(ex, w, h, n))
    check("measured evidence earns DIRECTED", measured.label() == P.DIRECTED)
    check("DIRECTED is still outside the headline number",
          measured.label() not in P.PHOTOGRAPHIC)
    check("and still counted as invention", measured.label() in P.NOT_THIS_PLACE)


def test_prompt_and_explanation():
    ex, truth, w, h, n = real_excursion()
    items = [cx.ContextItem("note", "smoke and sirens", shot=3)]
    plan = rz.LocalReasoner().plan(a_brief(ex, w, h, n, items))
    p = plan.prompt()
    check("measured facts lead the prompt",
          p.index("figure") < p.index("smoke"), p[:90])
    check("prompt is bounded", len(p) <= 420, f"{len(p)} chars")
    x = plan.explain()
    check("the reasoning is readable back",
          "[measured" in x and "[asserted" in x)


def test_api_reasoner():
    ex, truth, w, h, n = real_excursion()

    def dead(prompt, image):
        raise RuntimeError("no credentials")

    p = rz.ApiReasoner(call=dead).plan(a_brief(ex, w, h, n))
    check("a dead model degrades instead of crashing",
          any(e.support == "inferred" for e in p.elements))
    check("and the measured elements survive it",
          len(p.by_support("measured")) == 1)

    seen = {}

    def live(prompt, image):
        seen["prompt"] = prompt
        return ["wet asphalt and a fire escape", ""]

    p2 = rz.ApiReasoner(call=live).plan(a_brief(ex, w, h, n))
    check("model claims land as asserted, not measured",
          any(e.support == "asserted" and "asphalt" in e.text for e in p2.elements))
    check("empty claims are dropped",
          all(e.text for e in p2.elements))
    check("the model is told what it must not contradict",
          "must not" in seen.get("prompt", "") and "figure" in seen.get("prompt", ""))


if __name__ == "__main__":
    print("measured elements")
    test_measured_from_excursion()
    print("causal vs interpolating")
    test_modes_differ()
    print("placement")
    test_path_lands_when_the_film_says()
    test_depth_beyond_the_wall_is_visible()
    print("context")
    test_context_becomes_asserted()
    print("no evidence")
    test_nothing_at_all()
    print("provenance")
    test_labels()
    print("output")
    test_prompt_and_explanation()
    print("api reasoner")
    test_api_reasoner()

    ex, truth, w, h, n = real_excursion()
    if ex:
        plan = rz.LocalReasoner().plan(
            a_brief(ex, w, h, n, [cx.ContextItem("note", "smoke and sirens", shot=3)]))
        print("\nwhat the reasoner worked out, on a real recovered excursion:")
        print(plan.explain())
        print("\nprompt it would send:")
        print(" ", plan.prompt())

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
