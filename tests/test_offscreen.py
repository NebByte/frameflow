"""
test_offscreen -- Tier 3.1 tracking and the 3.2 verification harness.

Run: python test_offscreen.py

The fixture is a synthetic shot with EXACT off-screen truth: a disc crosses
frame, leaves by a known edge on a known frame, is genuinely absent for a known
number of frames, and returns at a known height. So every claim the tracker
makes can be checked against a number rather than against a screenshot.

The point of the harness is that a predictor never sees the re-entry it is
scored on. `test_no_leakage` is the assertion that keeps that true, because a
harness that quietly hands the answer to the predictor would report excellent
accuracy forever.

All CPU. No detection model, no weights.
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from frameflow import offscreen as off
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- fixture

def make_excursion(w=320, h=180, n=90, exit_at=28, gap=26, speed=9.0,
                   return_y=None, seed=5):
    """
    A disc leaves by the right edge and comes back through it.

    Locked camera, textured static background, one moving object. Returns
    (frames, truth) where truth is exact by construction, not annotated.
    """
    r = np.random.default_rng(seed)
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        bg[y, :] = (28 + y * 0.25, 34 + y * 0.2, 46 + y * 0.15)
    for _ in range(70):                     # corners, so homographies can lock
        x0, y0 = int(r.integers(0, w - 22)), int(r.integers(0, h - 22))
        c = tuple(int(v) for v in r.integers(40, 190, 3))
        cv2.rectangle(bg, (x0, y0), (x0 + int(r.integers(8, 20)),
                                     y0 + int(r.integers(8, 20))), c, -1)
    bg = np.clip(bg.astype(np.int16) + r.normal(0, 3, (h, w, 1)), 0, 255).astype(np.uint8)

    return_y = h * 0.42 if return_y is None else return_y
    exit_y = h * 0.5
    entry_at = exit_at + gap
    frames, truth_path = [], {}

    for i in range(n):
        f = bg.copy()
        x = y = None
        if i <= exit_at:                    # travelling out to the right
            x = w * 0.25 + speed * i
            y = exit_y
        elif i >= entry_at:                 # coming back in from the right
            x = w + 14 - speed * (i - entry_at)
            y = return_y
        if x is not None and -20 < x < w + 20:
            cv2.circle(f, (int(x), int(y)), 11, (60, 220, 250), -1)
            cv2.circle(f, (int(x), int(y)), 11, (20, 90, 110), 2)
            truth_path[i] = (float(x), float(y))
        frames.append(f)

    visible = sorted(truth_path)
    outs = [i for i in visible if i <= exit_at]
    ins = [i for i in visible if i >= entry_at]
    last_out = max(outs) if outs else None
    first_in = min(ins) if ins else None          # no return: a valid fixture
    truth = dict(side="R", exit_frame=last_out, entry_frame=first_in,
                 entry_y=truth_path[first_in][1] if first_in is not None else None,
                 gap=(first_in - last_out) if (first_in is not None and
                                               last_out is not None) else None,
                 speed=speed, path=truth_path)
    return frames, truth


def make_multi_excursion(w=360, h=200, n=110, depth=118.0, seed=11):
    """
    Three objects, three excursions, ONE shared off-screen depth.

    Different colours so pairing is unambiguous, different speeds so the gaps
    differ, different heights so they never touch. Because every one of them
    travels the same distance out and back, a depth prior fitted on two of them
    should predict the third -- which is exactly what leave-one-out asks.
    """
    r = np.random.default_rng(seed)
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        bg[y, :] = (30 + y * 0.2, 36 + y * 0.16, 44 + y * 0.12)
    for _ in range(90):
        x0, y0 = int(r.integers(0, w - 22)), int(r.integers(0, h - 22))
        c = tuple(int(v) for v in r.integers(40, 190, 3))
        cv2.rectangle(bg, (x0, y0), (x0 + int(r.integers(8, 18)),
                                     y0 + int(r.integers(8, 18))), c, -1)
    bg = np.clip(bg.astype(np.int16) + r.normal(0, 3, (h, w, 1)), 0, 255).astype(np.uint8)

    x0 = w * 0.55                            # close enough that all three fit in n
    specs = [dict(colour=(60, 220, 250), y=h * 0.22, speed=11.0, start=2),
             dict(colour=(80, 120, 245), y=h * 0.52, speed=8.0, start=6),
             dict(colour=(120, 245, 130), y=h * 0.80, speed=6.0, start=10)]
    for sp in specs:
        # frames needed to carry the centre from x0 clear of the right edge
        sp["exit_at"] = sp["start"] + int(np.ceil((w + 14 - x0) / sp["speed"]))
        sp["gap"] = int(round(2.0 * depth / sp["speed"]))

    frames = [bg.copy() for _ in range(n)]
    truth = []
    for sp in specs:
        entry_at = sp["exit_at"] + sp["gap"]
        seen = {}
        for i in range(n):
            x = None
            if sp["start"] <= i <= sp["exit_at"]:
                x = x0 + sp["speed"] * (i - sp["start"])
            elif i >= entry_at:
                x = w + 14 - sp["speed"] * (i - entry_at)
            if x is None or not (-20 < x < w + 20):
                continue
            cv2.circle(frames[i], (int(x), int(sp["y"])), 10, sp["colour"], -1)
            cv2.circle(frames[i], (int(x), int(sp["y"])), 10, (18, 40, 60), 2)
            seen[i] = x
        outs = [i for i in seen if i <= sp["exit_at"]]
        ins = [i for i in seen if i >= entry_at]
        if outs and ins:
            truth.append(dict(exit_frame=max(outs), entry_frame=min(ins),
                              speed=sp["speed"], gap=min(ins) - max(outs)))
    return frames, truth, depth / w


# ---------------------------------------------------------------- tests

def test_detection():
    frames, truth = make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]        # locked camera, exactly
    dets = off.detect_moving(frames, Hs=Hs)

    on_screen = [i for i in range(len(frames)) if i in truth["path"]
                 and 12 < truth["path"][i][0] < w - 12]
    found = sum(1 for i in on_screen if dets[i])
    check("finds the disc while it is on screen", found >= 0.9 * len(on_screen),
          f"{found}/{len(on_screen)} frames")

    off_screen = range(truth["exit_frame"] + 4, truth["entry_frame"] - 3)
    spurious = sum(1 for i in off_screen if dets[i])
    check("detects nothing while it is away", spurious == 0,
          f"{spurious} false detections over {len(list(off_screen))} empty frames")

    i = on_screen[len(on_screen) // 2]
    tx, ty = truth["path"][i]
    err = min(np.hypot(d.cx - tx, d.cy - ty) for d in dets[i])
    check("centre lands on the disc", err < 6.0, f"{err:.1f}px from truth")


def test_tracks_and_excursion():
    frames, truth = make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    dets = off.detect_moving(frames, Hs=Hs)
    tracks = off.link_tracks(dets, frame_size=(w, h))

    check("the absence splits it into two tracks", len(tracks) == 2,
          f"{len(tracks)} tracks")

    ex = off.find_excursions(tracks, w, h)
    check("exactly one excursion", len(ex) == 1, f"{len(ex)} found")
    if not ex:
        return
    e = ex[0]
    check("left by the right edge", e.side == "R", e.side)
    check("returned through the right edge", e.entry_side == "R", str(e.entry_side))
    check("exit frame matches truth", abs(e.exit_frame - truth["exit_frame"]) <= 2,
          f"{e.exit_frame} vs {truth['exit_frame']}")
    check("entry frame matches truth", abs(e.entry_frame - truth["entry_frame"]) <= 2,
          f"{e.entry_frame} vs {truth['entry_frame']}")
    check("exit velocity points outward", e.exit_v[0] > 0, f"vx={e.exit_v[0]:.2f}")
    check("exit speed close to truth", abs(abs(e.exit_v[0]) - truth["speed"]) < 2.5,
          f"{abs(e.exit_v[0]):.2f} vs {truth['speed']}")


def test_no_leakage():
    """A predictor must not be able to see the answer it is scored against."""
    frames, truth = make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)
    if not ex:
        check("harness has an excursion to score", False)
        return

    seen = {}

    def spy(e, W, H):
        seen["fields"] = [f for f in ("entry_frame", "entry_y", "entry_side", "gap")
                          if getattr(e, f, None) is not None]
        return dict(frame=e.exit_frame + 10, side=e.side, y=e.exit_y)

    off.score(ex, w, h, predictors={"spy": spy})
    # the Excursion carries truth because the HARNESS needs it; the contract is
    # that shipped predictors do not read it. Assert the shipped ones comply.
    import inspect
    for name, fn in off.PREDICTORS.items():
        src = inspect.getsource(fn)
        leaks = [f for f in ("entry_frame", "entry_y", "entry_side", ".gap")
                 if f in src]
        check(f"{name} reads no re-entry field", not leaks, ",".join(leaks))


def test_scoring():
    frames, truth = make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)
    s = off.score(ex, w, h)

    check("persist calls no returns", s["persist"]["returns_called"] == 0)
    check("persist reports no error it did not earn",
          s["persist"]["frame_err"] is None and s["persist"]["y_err_frac"] is None)
    check("elastic calls the return", s["elastic"]["returns_called"] == len(ex))
    check("elastic names the right edge",
          s["elastic"]["side_correct"] == s["elastic"]["returns_called"])
    for name in ("elastic", "ballistic"):
        e = s[name]["frame_err"]
        check(f"{name} produced a frame error", e is not None,
              f"{e} frames" if e is not None else "none")
    return s


def test_scale_invariance():
    """Same shot, twice the resolution: the harness must not change its mind."""
    out = {}
    for w, h in ((320, 180), (640, 360)):
        frames, truth = make_excursion(w=w, h=h, speed=9.0 * w / 320)
        Hs = [np.eye(3) for _ in frames]
        ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                                 frame_size=(w, h)), w, h)
        out[w] = ex
    check("finds the excursion at both resolutions",
          len(out[320]) == len(out[640]) == 1,
          f"{len(out[320])} at 320, {len(out[640])} at 640")
    if len(out[320]) == len(out[640]) == 1:
        d = abs(out[320][0].gap - out[640][0].gap)
        check("measured gap is resolution independent", d <= 2, f"{d} frames apart")


def test_no_excursion_when_nothing_leaves():
    """A disc that stays in frame must not produce an excursion."""
    frames, _ = make_excursion(n=60, exit_at=200, gap=1, speed=1.2)
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)
    check("no excursion invented for a figure that never leaves", len(ex) == 0,
          f"{len(ex)} found")


def test_calibration():
    """A depth prior fitted on other excursions must beat the shipped guess."""
    frames, truth, true_depth = make_multi_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)
    check("finds all three excursions", len(ex) == len(truth),
          f"{len(ex)} found, {len(truth)} staged")
    if len(ex) < 2:
        return

    fitted = off.fit_depth(ex, w)
    check("fitted depth recovers the staged depth",
          abs(fitted - true_depth) < 0.06, f"{fitted:.3f} vs {true_depth:.3f}")

    fixed = off.score(ex, w, h)["elastic"]["frame_err"]
    cal = off.score_calibrated(ex, w, h)["elastic"]
    check("leave-one-out actually fitted on others",
          cal["calibrated"] == len(ex), f"{cal['calibrated']}/{len(ex)}")
    check("calibrated beats the shipped 0.22 prior",
          cal["frame_err"] < fixed, f"{cal['frame_err']:.0f} vs {fixed:.0f} frames")
    print(f"       fixed prior {fixed:.0f} frames err, "
          f"leave-one-out calibrated {cal['frame_err']:.0f}")


if __name__ == "__main__":
    print("detection")
    test_detection()
    print("tracks and excursions")
    test_tracks_and_excursion()
    print("harness integrity")
    test_no_leakage()
    print("scoring")
    s = test_scoring()
    print("invariance")
    test_scale_invariance()
    print("negative case")
    test_no_excursion_when_nothing_leaves()
    print("calibration")
    test_calibration()

    if s:
        print("\npredictors, scored against ground truth:")
        for name, v in s.items():
            if v["returns_called"]:
                print(f"  {name:10s} called {v['returns_called']}/{v['of_excursions']}  "
                      f"side {v['side_correct']}/{v['returns_called']}  "
                      f"frame err {v['frame_err']:.0f}  y err {v['y_err_frac']*100:.1f}% of height")
            else:
                print(f"  {name:10s} called no returns")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
