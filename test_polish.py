"""
test_polish -- the pass that repaints finished walls, and what it must confess.

Polish is the one place a model is allowed to overwrite photographed pixels. It
exists because a streaked recovered wall is a real defect and a maker should be
able to fix it. The whole safety of it rests on one property: what gets
repainted stops counting as photography, exactly and only to the extent it
changed.

Run: python test_polish.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import polish
import provenance as P

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def test_a_note_leads_the_brief():
    """
    A person looking at the frame beats a model looking at the frame.

    The note says what belongs there; the model says what looks wrong. Wanting
    outranks complaining -- and the note is also what makes the result DIRECTED
    rather than merely GENERATED.
    """
    print("what the generator is told to do")
    entry = dict(claims=["Left: streaking", "Right: acceptable"])

    b = polish.brief_from(entry, ["a fire escape, camera left"])
    check("the note leads", b.startswith("a fire escape"), b[:50])
    check("and the complaint still reaches the model", "streaking" in b)
    check("an 'acceptable' side is not passed as a defect", "acceptable" not in b)

    b2 = polish.brief_from(entry, [])
    check("with no note the brief is generic plus the complaint",
          b2.startswith("extend the scene") and "streaking" in b2, b2[:50])

    check("nothing to say means no repair",
          polish.brief_from(dict(claims=["Left: acceptable"]), []) is None)


def test_repaint_is_relabelled():
    """
    The accounting that lets polish touch a photographed wall at all.

    A repainted pixel is no longer photography, so the label has to follow the
    paint. If this drifted, a model could raise the headline number by drawing
    over the evidence.
    """
    print("\nrepainting a photographed wing costs exactly what it repaints")
    h, w, ww = 20, 100, 20
    prov = np.full((h, w), P.RECOVERED, np.uint8)
    prov[:, ww:w - ww] = P.PRIMARY

    import agent as ag
    before = ag.WingAgent.report(prov, ww, w - 2 * ww)
    check("the wing starts fully photographed",
          abs(before["photographic"] - 1.0) < 1e-6, str(before["photographic"]))

    # a model repaints half of the left wall
    changed = np.zeros((h, w), bool)
    changed[:, :ww // 2] = True
    after_map = prov.copy()
    after_map[changed] = P.GENERATED
    after = ag.WingAgent.report(after_map, ww, w - 2 * ww)

    check("photographed falls by the repainted share",
          abs(after["photographic"] - 0.75) < 1e-6, str(after["photographic"]))
    check("and the repaint shows up as generated",
          abs(after["generated"] - 0.25) < 1e-6, str(after["generated"]))
    check("a note-driven repaint lands on DIRECTED, not GENERATED",
          P.DIRECTED != P.GENERATED and P.DIRECTED not in P.PHOTOGRAPHIC)
    check("neither label can be counted as filmed",
          P.GENERATED in P.NOT_THIS_PLACE and P.DIRECTED in P.NOT_THIS_PLACE)


def test_threshold_ignores_recompression():
    """
    A re-encode moves pixels a level or two. Calling that a repaint would
    relabel photographed pixels nobody touched, which is the same lie in the
    other direction.
    """
    print("\nre-encoding is not repainting")
    check("the bar sits above codec noise", polish.REPAINT_THRESHOLD >= 8,
          str(polish.REPAINT_THRESHOLD))
    check("and below a redrawn wall", polish.REPAINT_THRESHOLD <= 30,
          str(polish.REPAINT_THRESHOLD))


def test_streak_score_is_measured_not_asked():
    """
    The model's opinion is asserted. This number is not, so a reader can see
    whether the two agree.
    """
    print("\na measured companion to the model's opinion")
    h, w, ww = 30, 90, 20
    streaky = np.zeros((h, w, 3), np.uint8)
    streaky[:, :ww] = np.repeat(np.linspace(0, 255, h).astype(np.uint8), 3).reshape(h, 1, 3)
    check("a wall of identical columns scores high",
          polish.streak_score(streaky, ww) > 0.5,
          str(polish.streak_score(streaky, ww)))

    rs = np.random.RandomState(0)
    noisy = rs.randint(0, 255, (h, w, 3), dtype=np.uint8)
    check("a detailed wall scores low", polish.streak_score(noisy, ww) < 0.1,
          str(polish.streak_score(noisy, ww)))
    check("a degenerate wing does not raise", polish.streak_score(noisy, 2) == 0.0)


def test_an_unmeasured_cost_is_charged_not_excused():
    """
    The worst bug this pass has had, and the one the ladder exists to prevent.

    `repair` divided repainted photographed pixels by total photographed pixels.
    With no provenance map on disk the divisor was zero, so it reported 0.0 --
    a measured-looking figure for something it had not measured. A WaveSpeed
    pass that redrew 94% of a wall was recorded as having cost nothing.

    Absence of evidence is charged now, at the share of wing that actually
    moved. That can only overstate the loss, and the direction is the whole
    point: overstating costs coverage, understating reports invented pixels as
    photographed.
    """
    print("\nwhat a repaint costs when nothing can measure it")
    lost, basis = polish._charge(1000, 250, 0.9, 0.98)
    check("with a map, the cost is exact", abs(lost - 0.25) < 1e-9, basis)
    check("and says so", basis == "provenance map")

    lost, basis = polish._charge(0, 0, 0.94, 0.9869)
    check("with no map, an unmeasured repaint is not free",
          abs(lost - 0.94) < 1e-9, f"{lost} ({basis})")
    check("and the report says the number is an upper bound",
          "upper bound" in basis, basis)

    lost, basis = polish._charge(0, 0, 0.94, 0.0)
    check("a wing that held nothing photographed costs nothing to repaint",
          lost == 0.0, basis)

    check("the charge can never exceed the whole wing",
          polish._charge(0, 0, 5.0, 1.0)[0] == 1.0)

    prov = dict(photographic=0.9869, recovered=0.9869, generated=0.0131,
                directed=0.0)
    after = polish._charge_report(prov, 0.94, P.DIRECTED)
    check("the shot's photographic share falls by the charge",
          abs(after["photographic"] - 0.9869 * 0.06) < 1e-3,
          str(after["photographic"]))
    check("and what it lost lands on the rung the repaint earned",
          after["directed"] > 0.9, str(after["directed"]))
    check("a model-driven repaint lands on generated instead",
          polish._charge_report(prov, 0.5, P.GENERATED)["generated"] > 0.4)


def test_only_the_shots_you_pay_for():
    """
    A hosted generator bills per shot.

    `repair` used to take every shot the inspection faulted, which on a
    fast-cut trailer is most of them -- 34 shots at $0.20 each, spent on
    one-second fragments, and that run was started by accident once.
    """
    print("\nrepainting is per shot, and so is the bill")
    import inspect as _inspect
    sig = _inspect.signature(polish.repair)
    check("repair takes a shot list", "shots" in sig.parameters)
    check("and defaults to the old behaviour when none is given",
          sig.parameters["shots"].default is None)
    src = Path(polish.__file__).read_text(encoding="utf-8")
    check("a shot outside the list is skipped before anything is submitted",
          "if only is not None and int(entry[\"shot\"]) not in only" in src)
    check("the CLI exposes it", "--shots" in src)


def test_restate_moves_the_headline():
    """A polished run that kept its old number would be reporting pixels as
    photographed that a model drew a minute ago."""
    print("\nthe run summary follows the shots")
    by_shot = {
        0: dict(shot=0, provenance=dict(photographic=0.5, generated=0.5,
                                        recovered=0.5, directed=0.0)),
        1: dict(shot=1, provenance=dict(photographic=1.0, generated=0.0,
                                        recovered=1.0, directed=0.0)),
    }
    out = polish._restate(dict(mean_real_wing=1.0), by_shot)
    check("mean_real_wing is recomputed, not carried over",
          abs(out["mean_real_wing"] - 0.75) < 1e-6, str(out["mean_real_wing"]))
    check("the run is marked as polished", out.get("polished") is True)
    check("rungs_fired reflects what is actually there",
          "generated" in out["rungs_fired"] and "recovered" in out["rungs_fired"],
          str(out["rungs_fired"]))
    check("a rung with nothing in it is not claimed",
          "directed" not in out["rungs_fired"])


def test_a_warp_never_leaves_a_dark_edge():
    """
    The bug that put a thin black line down every recovered wall.

    A bilinear warp against a zero border blends its outermost pixels toward
    black; a nearest-neighbour mask of the same warp calls those pixels valid.
    So a column at roughly half brightness composited into the wing and was
    labelled RECOVERED -- one per donor, 2.3% of wing columns in a delivered
    film. Flat grey in must mean flat grey out wherever the mask says valid.
    """
    print("a warp must not invent darkness")
    import wingcoverage as wc
    src = np.full((64, 64, 3), 180, np.uint8)

    worst_old, worst_new = 180, 180
    for tx in (10.5, 7.25, 3.0, 20.75, 13.4):
        M = np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]], np.float64)
        old = cv2.warpPerspective(src, M, (128, 64))
        om = cv2.warpPerspective(np.full((64, 64), 255, np.uint8), M, (128, 64),
                                 flags=cv2.INTER_NEAREST) > 128
        if om.any():
            worst_old = min(worst_old, int(old[om].min()))
        got, gm = wc.warp_with_mask(src, M, (128, 64))
        if gm.any():
            worst_new = min(worst_new, int(got[gm].min()))

    check("the old spelling did darken mask-valid pixels", worst_old < 170,
          f"darkest valid pixel was {worst_old} of 180")
    check("warp_with_mask does not", worst_new == 180,
          f"darkest valid pixel is {worst_new} of 180")

    # and it must still survive a warp that is not a pure translation
    R = np.vstack([cv2.getRotationMatrix2D((32, 32), 7.0, 1.03), [0, 0, 1]])
    R[0, 2] += 20
    got, gm = wc.warp_with_mask(src, R, (128, 64))
    check("including under rotation and scale",
          gm.any() and int(got[gm].min()) == 180)


def test_settle_refines_without_inventing():
    """
    The property that lets the free pass run by default.

    settle_wings medians a wall against its own neighbours. It may change what
    a recovered pixel LOOKS like; it may not change what is recovered. If it
    ever filled an empty pixel, the coverage metric and every provenance rung
    under it would start counting pixels nobody photographed.
    """
    print("settling refines values, never coverage")
    import wingcoverage as wc

    rng = np.random.default_rng(7)
    h, ww, w = 40, 20, 60
    cw = w + 2 * ww
    base = rng.integers(60, 200, (h, cw, 3), dtype=np.uint8)
    packed = []
    for _ in range(7):
        c = base.copy()
        c += rng.integers(-6, 7, c.shape).astype(np.uint8)   # per-frame boil
        f = np.zeros((h, cw), bool)
        f[:, ww:ww + w] = True
        f[:, 5:ww] = True                     # part of the left wing is real
        packed.append((c, f, None))

    # settle_wings writes back into the stack it was given, to keep peak memory
    # to one copy -- so the 'before' has to be taken first
    def wobble(seq):
        return float(np.mean([np.abs(a[0][:, 5:ww].astype(float)
                                     - b[0][:, 5:ww].astype(float)).mean()
                              for a, b in zip(seq, seq[1:])]))
    was = wobble(packed)
    snap = [(c.copy(), f.copy()) for c, f, _ in packed]

    out = wc.settle_wings(packed, lambda i, t: np.eye(3), ww, k=2)
    check("every frame comes back", len(out) == len(snap))
    for (c0, f0), (c1, f1, _) in zip(snap, out):
        check("the coverage mask is untouched", np.array_equal(f0, f1))
        check("the centre is untouched",
              np.array_equal(c0[:, ww:ww + w], c1[:, ww:ww + w]))
        check("nothing outside the recovered wing moved",
              np.array_equal(c0[:, :5], c1[:, :5]))
        break

    check("and the wing is steadier than it was", wobble(out) < was,
          f"{was:.2f} -> {wobble(out):.2f}")


def test_repair_aims_at_the_defect():
    """
    Why a 99%-photographed wall stopped costing 62% of itself to fix.

    The first version kept the centre and took the whole wing from the model.
    defect_mask is what makes a repaint proportionate: dead pixels, hairlines
    and already-invented pixels are fair game, and photography is not.
    """
    print("a repaint is aimed, not sprayed")
    h, ww, w = 30, 12, 40
    cw = w + 2 * ww
    canvas = np.full((h, cw, 3), 120, np.uint8)
    canvas[:, 4] = 40                       # a hairline in the left wing
    canvas[:, cw - 3:] = 0                  # dead pixels at the right edge

    bad = polish.defect_mask(canvas, ww)
    check("the hairline is found", bool(bad[:, 4].all()))
    check("the dead edge is found", bool(bad[:, cw - 1].all()))
    check("ordinary photographed wall is left alone",
          not bad[:, 8].any(), f"{int(bad[:, 8].sum())} pixels flagged")
    check("the centre is never a target",
          not bad[:, ww:cw - ww].any())
    check("and it is a small fraction of the wall",
          bad.mean() < 0.15, f"{bad.mean() * 100:.1f}% of the canvas")

    # with a provenance map, anything already invented is fair game too
    import agent as ag
    prov = np.full((h, cw), ag.RECOVERED, np.uint8)
    prov[:, :3] = ag.GENERATED
    bad2 = polish.defect_mask(canvas, ww, prov)
    check("invented pixels are repaintable", bool(bad2[:, 0].all()))
    check("recovered pixels still are not", not bad2[:, 8].any())

    a = polish.feather(bad)
    check("the mask is feathered, not a hard rectangle",
          0.0 < float(a[:, 6].max()) < 1.0,
          f"alpha just outside the defect is {float(a[:, 6].max()):.2f}")


def test_the_finishing_pass_keeps_the_frame_rate():
    """
    The bug that made every finished film play a quarter slow.

    The renderer knew the source was 30fps. write_shot_segment was called with
    a hardcoded 24.0, and rebuild fell back to 24.0 because no summary carried
    an fps at all -- so pressing the button re-timed the film. `dims` reads
    what the renderer recorded, and only guesses for jobs that predate it.
    """
    print("frame rate survives being finished")
    ww, fps = polish.dims(dict(fps=30.0, wing_w=105))
    check("the recorded rate is used", fps == 30.0, f"got {fps}")
    check("the recorded wing width is used", ww == 105, f"got {ww}")

    ww2, fps2 = polish.dims(dict(fps=29.97, wing_w=64), canvas_w=888)
    check("a non-integer rate survives", abs(fps2 - 29.97) < 1e-6)
    check("the recorded width wins over re-deriving it", ww2 == 64)

    # a job rendered before either was recorded still has to work
    ww3, fps3 = polish.dims(dict(wing_ratio=0.22), canvas_w=690)
    check("an older job falls back rather than failing", ww3 == 105 and fps3 == 24.0,
          f"ww={ww3} fps={fps3}")


def test_a_note_reaches_the_wall():
    """
    What a person asks for must actually arrive.

    Two ways this quietly failed. `repair` skipped any shot the model called
    acceptable, so a note on a clean-looking wall did nothing at all -- you
    wrote down what belongs there, pressed the button, and watched a model's
    shrug overrule you. And once repaints were aimed at defects, a note could
    only ever reach the few percent of the wing that was broken, which is not
    honouring "a fire escape, camera left" -- it is ignoring it with extra
    steps.
    """
    print("a note is a request, not a second opinion")
    src = Path(polish.__file__).read_text(encoding="utf-8")

    check("a shot with a note is repaired even when the model is content",
          "if not entry.get(\"bad\") and not notes:" in src)
    check("and a note opens the whole wing, not just its blemishes",
          "if full or notes:" in src)

    # the labelling that pays for it
    check("what a note drives is DIRECTED, not GENERATED",
          "label = P.DIRECTED if notes else P.GENERATED" in src)

    # and the brief still leads with the person
    b = polish.brief_from(dict(claims=["Left: acceptable"]),
                          ["a fire escape, camera left"])
    check("a note alone is enough to brief the generator", b is not None, str(b))
    check("and it leads", b.startswith("a fire escape"), str(b)[:40])

    # the defect mask is still what a FAULT gets -- narrow
    h, ww, w = 30, 12, 40
    cw = w + 2 * ww
    canvas = np.full((h, cw, 3), 120, np.uint8)
    canvas[:, 4] = 40
    check("a fault on its own still aims narrowly",
          polish.defect_mask(canvas, ww).mean() < 0.15,
          f"{polish.defect_mask(canvas, ww).mean() * 100:.1f}% of the canvas")


def test_the_metric_does_not_punish_resolution():
    """
    The bug that delivered a film with black walls.

    detail_weight measured texture in a fixed NINE-PIXEL window. A physical
    texture spread over twice as many pixels varies less within any nine of
    them, so rendering sharper scored LOWER -- the same frame read 0.62 at
    480px and 0.34 at 1280px. A 1024px render of a clip that passed comfortably
    at 480px came back at 24.57% effective against a 25% bar, was gated OFF,
    and delivered black wings. The operator had asked for nothing but more
    resolution.

    The window is a fraction of the frame now, so it spans the same piece of
    the world at any working width.
    """
    print("more resolution must not score as less detail")
    import wingcoverage as wc

    rng = np.random.default_rng(3)
    base = rng.integers(40, 210, (270, 480, 3), dtype=np.uint8)
    for _ in range(60):                       # hard edges, like real footage
        x, y = int(rng.integers(0, 440)), int(rng.integers(0, 230))
        cv2.rectangle(base, (x, y), (x + 30, y + 26),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)

    got = {}
    for wdt in (480, 960, 1280):
        im = cv2.resize(base, (wdt, int(270 * wdt / 480)),
                        interpolation=cv2.INTER_AREA)
        got[wdt] = float(wc.detail_weight(im).mean())
    spread = max(got.values()) - min(got.values())
    check("the same content scores the same at any working width",
          spread < 0.08, "  ".join(f"{k}px={v:.3f}" for k, v in got.items()))

    # and the old spelling is what it is being compared against
    old = {wdt: float(wc.detail_weight(
              cv2.resize(base, (wdt, int(270 * wdt / 480)),
                         interpolation=cv2.INTER_AREA), win=9).mean())
           for wdt in (480, 1280)}
    check("a fixed pixel window is what caused it",
          old[480] - old[1280] > spread,
          f"fixed-9 drops {old[480]:.3f}->{old[1280]:.3f}")

    check("the window scales with the frame",
          getattr(wc, "DETAIL_WIN_AT", None) == 480)


if __name__ == "__main__":
    print("polish -- inspect, repaint, and confess")
    test_a_note_leads_the_brief()
    test_repaint_is_relabelled()
    test_threshold_ignores_recompression()
    test_streak_score_is_measured_not_asked()
    test_an_unmeasured_cost_is_charged_not_excused()
    test_only_the_shots_you_pay_for()
    test_restate_moves_the_headline()
    test_a_warp_never_leaves_a_dark_edge()
    test_settle_refines_without_inventing()
    test_repair_aims_at_the_defect()
    test_the_finishing_pass_keeps_the_frame_rate()
    test_a_note_reaches_the_wall()
    test_the_metric_does_not_punish_resolution()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
