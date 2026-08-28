"""
triage -- which shots can be earned, decided before anything is rendered.

A full conversion of one 27-second clip takes hours. A studio deciding whether
a title is worth converting at all cannot wait hours per title, and does not
need to: the question "is there anything out there to recover" is answered by
geometry and coverage, and both are cheap. The expensive part is drawing the
pixels, and drawing them is pointless on a shot the gate is going to refuse.

So this runs the same classifier, the same backend, the same hold-out geometry
probe and the same gate as a real render -- on a window of each shot rather
than all of it -- and reports the verdict it would reach.

    triage_film(path)   -> a verdict per shot, in seconds per shot

WHY A CONSECUTIVE WINDOW AND NOT AN EVEN SAMPLE
-----------------------------------------------
Sampling 60 frames spread across a 500-frame shot would be faster still, and it
would lie. Coverage depends on how far the camera travels between the frames
you hold: spread them out and each one sees much further from the last, the
wing fills from fewer frames back, staleness collapses and effective coverage
comes back far higher than a real render will ever produce. The number would
flatter every shot.

A consecutive window has the opposite bias. It holds less footage than the full
shot does, so it has fewer donors to recover from and reports a LOWER figure
than the render will achieve. That is the direction to be wrong in -- the same
argument `polish._charge` makes about unmeasured repaints. A triage that
understates costs you a shot you could have had. One that overstates costs you
the four weeks of artist time you committed on the strength of it.

Everything returned is therefore a floor, and says so.
"""
from __future__ import annotations

import cv2


from . import backends as bk
from . import gating as g
from . import shotdetect as sd
from . import wingcoverage as wc
WINDOW = 72          # consecutive frames per shot; enough for a wing to fill
MIN_SHOT = 12        # shorter than this and there is nothing to register against

# What the verdicts mean to somebody deciding where to spend an artist.
MEANING = {
    "FULL":     "wings earned from this shot's own footage",
    "NARROW":   "wings earned, narrower than full width",
    "BORROWED": "wings earned from other footage of the same scene",
    "GEN":      "nothing recoverable; wings would have to be invented",
    "OFF":      "nothing recoverable; wings stay dark",
    "LOCKED":   "camera never moved, so nothing was ever filmed out there",
}


def _window(cap, a, b, maxw, n=WINDOW, rotate=0):
    """A consecutive run from the middle of a shot, downscaled."""
    total = b - a
    start = a + max(0, (total - n) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    out = []
    while len(out) < min(n, total):
        ok, f = cap.read()
        if not ok:
            break
        if rotate:
            from . import render as sx
            if sx.ROTATIONS.get(rotate) is not None:
                f = cv2.rotate(f, sx.ROTATIONS[rotate])
        if f.shape[1] > maxw:
            s = maxw / f.shape[1]
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def triage_shot(frames, tracker=None, thresholds=None, wing=None, probes=2):
    """
    The verdict this shot would get, without rendering it.

    Same classifier, same backend choice, same hold-out geometry probe, same
    gate. The only difference from a real render is how much footage it was
    allowed to look at, and that difference is stated rather than hidden.
    """
    from . import render as sx
    tracker = tracker or wc.Tracker()
    wing = float(wing if wing is not None else sx.WING)
    h, w = frames[0].shape[:2]
    ww = int(w * wing)

    kind, stats = wc.classify_motion(tracker, frames)
    out = dict(motion=kind, frames=len(frames), backend="none", geometry_db=0.0,
               coverage=0.0, effective=0.0, state="OFF", reasons="",
               displacement=float(stats.get("displacement") or 0.0))

    backend = bk.pick(kind)
    if backend is None:
        out.update(state="LOCKED",
                   reasons="locked off: nothing was filmed beyond the frame")
        out["meaning"] = MEANING["LOCKED"]
        return out

    out["backend"] = backend.name
    try:
        res = backend.propagate(frames, ww, tracker)
    except RuntimeError as e:
        out.update(state="OFF", reasons=f"backend refused: {e}"[:160])
        out["meaning"] = MEANING["OFF"]
        return out

    res = wc.settle_wings(res, backend.warp_between, ww)
    geom, _ = g.leave_one_out(frames, backend, tracker, probes=probes)
    mid = res[len(res) // 2]
    m = wc.wing_metrics(mid[1], mid[2], ww, w, 24.0, mid[0])
    state, _ratio, why = g.decide(m, geom, thresholds)

    out.update(geometry_db=round(float(geom), 1),
               coverage=round(float(m["coverage"]), 4),
               effective=round(float(m["effective_coverage"]), 4),
               state=state, reasons="; ".join(why))
    out["meaning"] = MEANING.get(state, state)
    return out


# Seconds per rendered frame, measured rather than modelled. Two real runs:
#
#   480px   0.9 s/frame  (laptop CPU)  ~1.2 on a 4-vCPU Cloud Run instance
#   1024px  9.0 s/frame  (laptop CPU)  799 frames in about two hours
#
# That is steeper than the 4.5x the pixel count alone implies, because a wider
# wing needs donors from further back and the settle pass grows with it. Two
# points is not a curve, so this interpolates between them and says so.
COST = {480: 0.9, 1024: 9.0}


def estimate_seconds(frames: int, maxw: int) -> float:
    """Roughly how long a render will take. Wrong by a factor on new hardware."""
    lo, hi = 480, 1024
    if maxw <= lo:
        per = COST[lo] * (maxw / lo) ** 2
    elif maxw >= hi:
        per = COST[hi] * (maxw / hi) ** 2
    else:
        t = (maxw - lo) / (hi - lo)
        per = COST[lo] + t * (COST[hi] - COST[lo])
    return frames * per


def recommend(source_width: int, total_frames: int, longest_shot: int, fps: float):
    """
    What to render this clip at, and what it will cost.

    The defaults exist to keep a first run short, and on a good clip they throw
    away most of what makes it good: 640px discards resolution the camera
    already captured, and a 200-frame cap on a 27-second take delivers under
    seven seconds of film. Both are traps precisely because they are defaults --
    somebody presses go, gets a worse result than the tool is capable of, and
    concludes that is what the tool does.

    Everything needed to avoid that is already known by the time triage has run:
    how wide the source is, how long the shots are, and what a frame costs.
    """
    native = int(max(320, min(source_width, 1920)))
    frames = int(max(1, longest_shot))
    full = estimate_seconds(total_frames, native)
    fast = estimate_seconds(total_frames, 480)
    return dict(
        maxw=native,
        frames_per_shot=frames,
        est_seconds=int(full),
        est_human=_human(full),
        fast_maxw=480,
        fast_est_human=_human(fast),
        why=(f"source is {source_width}px wide, so anything under {native} "
             f"discards resolution the camera already captured; the longest "
             f"shot is {frames} frames ({frames / max(fps, 1):.1f}s), so a "
             f"lower cap would cut it off"),
        caveat="timings are interpolated from two measured runs and will be "
               "wrong by a factor on different hardware",
    )


def _human(sec: float) -> str:
    sec = int(sec)
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"about {round(sec / 60)} min"
    return f"about {sec / 3600:.1f} hours"


def triage_film(path, maxw=480, window=WINDOW, max_shots=None, rotate=0,
                thresholds=None, verbose=True, on_shot=None):
    """
    Every shot's verdict, in seconds per shot instead of hours per film.

    Returns a dict with the per-shot verdicts and a summary a person can act
    on: how much of the film can be widened from its own footage, and how much
    would have to be invented or left dark.

    Every coverage figure is a FLOOR. Each shot is judged on a window rather
    than all of it, and a window holds fewer donor frames than the shot does --
    so a render recovers more, never less. Measured on a 27-second handheld pan:
    triage said 50.7% effective, the full render delivered 52.54%.
    """
    seg = sd.segment(str(path))
    shots = [s for s in seg["shots"] if s[1] - s[0] >= MIN_SHOT]
    if max_shots:
        shots = shots[:max_shots]

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    tracker = wc.Tracker()
    verdicts = []
    for si, (a, b) in enumerate(shots):
        # Letterbox cropping is the renderer's job, not triage's: the verdict
        # turns on whether the camera MOVED, and black bars move with it.
        fr = _window(cap, a, b, maxw, window, rotate)
        if len(fr) < MIN_SHOT:
            continue
        v = triage_shot(fr, tracker, thresholds)
        v.update(shot=si, start=int(a), shot_frames=int(b - a),
                 seconds=round((b - a) / fps, 2))
        verdicts.append(v)
        if verbose:
            print(f"  shot {si:3d}  {v['state']:8s} {v['motion']:8s} "
                  f"geom {v['geometry_db']:5.1f}dB  eff {v['effective']*100:5.1f}%  "
                  f"{v['meaning']}", flush=True)
        if on_shot is not None:
            try:
                on_shot(v)
            except Exception:                    # a watcher must not stop triage
                pass
    cap.release()

    earned = [v for v in verdicts if v["state"] in ("FULL", "NARROW", "BORROWED")]
    secs = sum(v["seconds"] for v in verdicts) or 1.0
    longest = max((v["shot_frames"] for v in verdicts), default=0)
    total = sum(v["shot_frames"] for v in verdicts)
    return dict(
        source=str(path).replace("\\", "/").rsplit("/", 1)[-1],
        fps=round(fps, 3), shots=len(verdicts),
        seconds=round(secs, 2),
        earned_shots=len(earned),
        earned_seconds=round(sum(v["seconds"] for v in earned), 2),
        earned_fraction=round(sum(v["seconds"] for v in earned) / secs, 4),
        basis=(f"a {window}-frame window per shot. A full render holds more "
               f"footage and recovers more, so these are floors, not estimates"),
        recommended=recommend(src_w, total, longest, fps),
        verdicts=verdicts,
    )


def main(argv=None):
    """The CLI. Exposed as a function so `frameflow-triage` is a real entry point."""
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description="which shots can be widened, decided before rendering")
    ap.add_argument("video")
    ap.add_argument("--maxw", type=int, default=480)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = triage_film(a.video, a.maxw, a.window, a.max_shots, a.rotate,
                      verbose=not a.json)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("")
        print(f"{rep['earned_shots']}/{rep['shots']} shots can be widened from "
              f"their own footage -- {rep['earned_seconds']:.1f}s of "
              f"{rep['seconds']:.1f}s ({rep['earned_fraction']*100:.0f}%)")
        print(f"basis: {rep['basis']}")
        r = rep.get("recommended") or {}
        if r:
            print("")
            print(f"to render it: --maxw {r['maxw']} --frames-per-shot "
                  f"{r['frames_per_shot']}   ({r['est_human']})")
            print(f"  {r['why']}")
            print(f"  faster: --maxw {r['fast_maxw']} ({r['fast_est_human']}), "
                  f"smaller and softer")
    return rep


if __name__ == "__main__":
    main()
