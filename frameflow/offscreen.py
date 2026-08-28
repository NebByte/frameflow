"""
offscreen — what a thing did while it was out of frame.

ROADMAP Tier 3.1 and 3.2, in that order and no further. This finds excursions
and scores predictions of them. **It generates nothing.** Putting an inferred
character onto a wall before you can measure whether the inference is right is
how you get a convincing wrong answer, which is worse than a dark wall.

WHY THIS IS TRACTABLE AT ALL
----------------------------
Ghost is thrown out of frame left and comes back 40 frames later. Both ends are
photographed. So off-screen motion is INTERPOLATION between two observations,
not free extrapolation -- and it is checkable: predict the return from the exit
alone, then compare against the return that was actually filmed.

That is the whole design. A predictor here never sees the re-entry it is scored
against.

WHAT IT DETECTS WITH
--------------------
No detection model, deliberately -- there are no weights on this machine and a
classical detector keeps the dependency list at opencv + numpy. Camera motion is
removed with the homography chain this repo already computes, each frame is
differenced against a median plate built from its own neighbours, and what
survives is something that moved differently from the room.

That finds bodies, vehicles and thrown objects against a background. It does not
know what any of them ARE. Identity here is a colour histogram plus size, which
is enough to re-associate one figure across a 40-frame absence inside a single
shot and nowhere near enough to re-identify across a cut. Tier 3.5's vision
model is the upgrade, and the interfaces below are shaped so it can be dropped
in without touching the harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import wingcoverage as wc
HIST_BINS = (8, 8)


# ---------------------------------------------------------------- detections

@dataclass
class Detection:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    area: float
    hist: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.cx, self.cy], np.float64)


def _hist(patch) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def background_plate(frames, Hs, i, window=12, samples=6):
    """
    What frame `i` would look like with nothing moving in it.

    Neighbours are warped into frame i's own plane -- not into frame 0's, which
    slides the scene out of the canvas on any real pan -- and the per-pixel
    median taken. Median, not mean: a body crossing frame contaminates a mean
    everywhere it passes, and a median only where it is in the majority.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    lo, hi = max(0, i - window), min(n, i + window + 1)
    idx = [j for j in np.linspace(lo, hi - 1, samples).astype(int) if j != i]

    # chain_homographies returns None for any frame it could not register, and
    # says so in its own docstring -- a hop that fails leaves that anchor
    # unusable. This assumed every frame had a matrix, so the first
    # unregisterable frame in a shot turned into `None @ ndarray` and killed the
    # whole run. A frame with no homography cannot be warped into another
    # frame's plane; it can only be left out of the plate.
    if i >= len(Hs) or Hs[i] is None:
        return frames[i].copy()
    idx = [j for j in idx if j < len(Hs) and Hs[j] is not None]
    if not idx:
        return frames[i].copy()

    Hi = np.linalg.inv(Hs[i])
    stack = []
    for j in idx:
        M = Hi @ Hs[j]                      # frame j -> frame i's plane
        stack.append(cv2.warpPerspective(frames[j], M, (w, h)))
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def detect_moving(frames, Hs=None, tracker=None, min_area=0.0008, thresh=28,
                  window=12, samples=6):
    """Per frame, the things that moved differently from the room."""
    if Hs is None:
        tracker = tracker or wc.Tracker()
        Hs = wc.chain_homographies(tracker, frames)

    h, w = frames[0].shape[:2]
    floor = int(min_area * h * w)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = []

    for i, f in enumerate(frames):
        bg = background_plate(frames, Hs, i, window, samples)
        d = cv2.absdiff(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
        m = (cv2.GaussianBlur(d, (5, 5), 0) > thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

        n, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        dets = []
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < floor:
                continue
            patch = f[y:y + bh, x:x + bw]
            if patch.size == 0:
                continue
            dets.append(Detection(i, float(cent[k][0]), float(cent[k][1]),
                                  float(bw), float(bh), float(area), _hist(patch)))
        out.append(dets)
    return out


# ---------------------------------------------------------------- tracks

@dataclass
class Track:
    id: int
    dets: list = field(default_factory=list)

    @property
    def first(self):
        return self.dets[0]

    @property
    def last(self):
        return self.dets[-1]

    def _slope(self, tail) -> np.ndarray:
        if len(tail) < 2:
            return np.zeros(2)
        dt = tail[-1].frame - tail[0].frame
        if dt <= 0:
            return np.zeros(2)
        return (tail[-1].centre - tail[0].centre) / dt

    def velocity(self, span=5) -> np.ndarray:
        """Mean per-frame velocity over the last `span` detections."""
        return self._slope(self.dets[-span:])

    def exit_velocity(self, w, span=5, pad=2.0) -> np.ndarray:
        """
        Speed measured BEFORE the object starts leaving the frame.

        Measured: taking the last few detections instead under-reported a disc
        travelling at 9.0 px/frame as 7.29. Once a body begins crossing the
        edge, only the part still inside is detected, so its centroid slows
        down while the body does not. Every predictor divides by this speed, so
        the bias lands directly on the predicted return frame -- it made the
        gap estimate 19 frames against a true 27.

        Falls back to the plain tail if every detection is already clipped.
        """
        inside = [d for d in self.dets
                  if d.cx - d.w / 2 > pad and d.cx + d.w / 2 < w - pad]
        return self._slope(inside[-span:]) if len(inside) >= 2 else self.velocity(span)

    def entry_velocity(self, span=5) -> np.ndarray:
        return self._slope(self.dets[:span])

    def mean_hist(self) -> np.ndarray:
        return np.mean([d.hist for d in self.dets], axis=0)


def link_tracks(per_frame, frame_size=None, max_gap=3, max_move=0.05, min_len=3,
                appearance=0.55):
    """
    Association against a PREDICTED position, gated on appearance.

    Deliberately conservative: it would rather cut one track in two than braid
    two people into one. A braided track invents an excursion that never
    happened, and the harness would then be scoring predictions against fiction.

    Two things earn their place here, both found by the fixture braiding:

    - matching to `last + velocity*step` rather than to `last`. Nearest-to-last
      quietly prefers whichever object is moving SLOWEST, since a stationary
      candidate is always nearer than the one this track is actually chasing.
    - `max_move` of 0.05 diag, not 0.12. On a 360x200 frame 0.12 permitted 49px
      of travel per frame between objects only 56px apart, and the tracker hopped
      between two discs whose colours correlated at 0.78 -- inside the 0.55
      appearance gate. Two of three staged excursions were lost that way.
    """
    tracks, live, next_id = [], [], 0
    if not per_frame:
        return tracks
    diag = float(np.hypot(*frame_size)) if frame_size else 1000.0

    for i, dets in enumerate(per_frame):
        unmatched = list(dets)
        for t in list(live):
            if i - t.last.frame > max_gap:
                live.remove(t)
                continue
            best, best_score = None, 1e9
            for d in unmatched:
                step = max(d.frame - t.last.frame, 1)
                expect = t.last.centre + t.velocity() * step
                move = float(np.linalg.norm(d.centre - expect)) / step
                if move > max_move * diag:
                    continue
                sim = float(cv2.compareHist(t.mean_hist().astype(np.float32),
                                            d.hist.astype(np.float32),
                                            cv2.HISTCMP_CORREL))
                if sim < appearance:
                    continue
                s = move - 40.0 * sim
                if s < best_score:
                    best, best_score = d, s
            if best is not None:
                t.dets.append(best)
                unmatched.remove(best)
        for d in unmatched:
            t = Track(next_id, [d])
            next_id += 1
            live.append(t)
            tracks.append(t)

    return [t for t in tracks if len(t.dets) >= min_len]


# ---------------------------------------------------------------- excursions

@dataclass
class Excursion:
    """One thing leaving frame and coming back. Both ends observed."""
    side: str                 # 'L' or 'R', the edge it left by
    exit_frame: int
    exit_y: float
    exit_v: np.ndarray
    entry_frame: int
    entry_y: float
    entry_side: str
    gap: int
    similarity: float


def _near_edge(x, w, margin):
    if x <= margin:
        return "L"
    if x >= w - margin:
        return "R"
    return None


def find_excursions(tracks, w, h, margin_frac=0.06, min_gap=4, max_gap=180,
                    appearance=0.5):
    """
    Pair a track that leaves by an edge with one that returns through an edge.

    A pairing needs all of: the exit sits at an edge, the exit velocity points
    OUT through that edge, the return is later, and the two look alike. The
    velocity test is what stops a figure who merely stops at the edge of frame,
    or walks behind a pillar there, from being called an excursion.
    """
    margin = margin_frac * w
    exits, entries = [], []

    for t in tracks:
        side = _near_edge(t.last.cx, w, margin)
        if side:
            v = t.velocity()
            if (side == "L" and v[0] < 0) or (side == "R" and v[0] > 0):
                exits.append(t)
        side_in = _near_edge(t.first.cx, w, margin)
        if side_in:
            v = t.entry_velocity()
            if (side_in == "L" and v[0] > 0) or (side_in == "R" and v[0] < 0):
                entries.append(t)

    used, out = set(), []
    for t in sorted(exits, key=lambda t: t.last.frame):
        side = _near_edge(t.last.cx, w, margin)
        best, best_sim = None, -1.0
        for e in entries:
            if id(e) in used or e is t:
                continue
            gap = e.first.frame - t.last.frame
            if not (min_gap <= gap <= max_gap):
                continue
            sim = float(cv2.compareHist(t.mean_hist().astype(np.float32),
                                        e.mean_hist().astype(np.float32),
                                        cv2.HISTCMP_CORREL))
            if sim < appearance:
                continue
            if sim > best_sim:
                best, best_sim = e, sim
        if best is None:
            continue
        used.add(id(best))
        out.append(Excursion(side=side, exit_frame=t.last.frame,
                             exit_y=t.last.cy, exit_v=t.exit_velocity(w),
                             entry_frame=best.first.frame, entry_y=best.first.cy,
                             entry_side=_near_edge(best.first.cx, w, margin),
                             gap=best.first.frame - t.last.frame,
                             similarity=best_sim))
    return out


# ---------------------------------------------------------------- predictors
#
# Each sees ONLY the exit state and returns a prediction, or None for "it does
# not come back". None is a real answer: most things that leave a shot never
# return, and a predictor that always calls a return is not useful even when it
# scores well on the excursions we happened to find.

def persist(ex, w, h):
    """It kept going. Predicts no return -- the null hypothesis."""
    return None


def elastic(ex, w, h, depth_frac=0.22, damping=1.0):
    """
    It went one wing-width out, turned, and came back the way it went.

    `depth_frac` is a prior about how far off-screen the action goes: one wing
    width, because that is the strip this project is trying to fill. Vertical
    motion continues at the speed it left with.
    """
    vx = abs(float(ex.exit_v[0]))
    if vx < 1e-3:
        return None
    gap = int(round(2.0 * depth_frac * w / (vx * damping)))
    return dict(frame=ex.exit_frame + gap, side=ex.side,
                y=float(ex.exit_y + ex.exit_v[1] * gap))


def ballistic(ex, w, h, depth_frac=0.22, g=None):
    """
    Elastic in x, but y follows a thrown body rather than a straight line.

    `g` in pixels per frame squared, defaulted from frame height so it is
    resolution-independent rather than tuned to one clip.
    """
    vx = abs(float(ex.exit_v[0]))
    if vx < 1e-3:
        return None
    g = (h * 0.0016) if g is None else g
    gap = int(round(2.0 * depth_frac * w / vx))
    y = ex.exit_y + ex.exit_v[1] * gap + 0.5 * g * gap * gap
    return dict(frame=ex.exit_frame + gap, side=ex.side,
                y=float(np.clip(y, 0, h - 1)))


PREDICTORS = {"persist": persist, "elastic": elastic, "ballistic": ballistic}


# ---------------------------------------------------------------- calibration

def fit_depth(excursions, w) -> float:
    """
    How far off-screen the action actually goes, in frame widths.

    An excursion of `gap` frames at speed `vx` covers gap*vx out and back, so
    the depth is half of it. Median over excursions, because one bad pairing
    should not move the prior.

    This exists because the harness caught the prior being wrong. `depth_frac`
    defaulted to 0.22 -- one wing width, chosen because that is the strip this
    project fills, which is a statement about the SCREEN and not about where
    actors go. Measured on the ground-truth fixture the true value is 0.38.

    Worse, that error was hidden: exit speed was under-measured by edge
    clipping (9.0 read as 7.29) and the two errors cancelled to a plausible
    9-frame error. Fixing the speed alone made the score worse, 9 to 12, which
    is what a compensating pair of errors looks like from the outside.
    """
    vals = [abs(float(e.exit_v[0])) * e.gap / (2.0 * w)
            for e in excursions if abs(float(e.exit_v[0])) > 1e-3]
    return float(np.median(vals)) if vals else 0.22


def score_calibrated(excursions, w, h, predictors=None):
    """
    Leave-one-out: fit the depth prior on the OTHER excursions, predict this one.

    Fitting on all of them and scoring on all of them would report how well the
    prior fits the data it came from, which is not a measurement of anything.
    With fewer than two excursions there is nothing to fit on and this falls
    back to the shipped prior, and says so in `calibrated`.
    """
    predictors = predictors or {k: v for k, v in PREDICTORS.items() if k != "persist"}
    out = {}
    for name, fn in predictors.items():
        called, side_ok, ferr, yerr, fitted = 0, 0, [], [], 0
        for k, ex in enumerate(excursions):
            others = excursions[:k] + excursions[k + 1:]
            if others:
                depth, fitted = fit_depth(others, w), fitted + 1
                p = fn(ex, w, h, depth_frac=depth)
            else:
                p = fn(ex, w, h)
            if p is None:
                continue
            called += 1
            if p["side"] == ex.entry_side:
                side_ok += 1
            ferr.append(abs(p["frame"] - ex.entry_frame))
            yerr.append(abs(p["y"] - ex.entry_y) / h)
        out[name] = dict(
            returns_called=called, of_excursions=len(excursions),
            side_correct=side_ok, calibrated=fitted,
            frame_err=float(np.median(ferr)) if ferr else None,
            y_err_frac=round(float(np.median(yerr)), 4) if yerr else None,
        )
    return out



# ---------------------------------------------------------------- the harness

def score(excursions, w, h, predictors=None):
    """
    Predict each return from its exit alone, then compare with what was filmed.

    Per predictor:
      returns_called   it said "comes back" and one did
      side_correct     of those, how many named the right edge
      frame_err        median |predicted return frame - actual|
      y_err_frac       median |predicted y - actual| over frame height

    A predictor that never calls a return scores zero called and no errors,
    which is the honest way for `persist` to look: right about most objects in
    most films, useless for this job.
    """
    predictors = predictors or PREDICTORS
    out = {}
    for name, fn in predictors.items():
        called, side_ok, ferr, yerr = 0, 0, [], []
        for ex in excursions:
            p = fn(ex, w, h)
            if p is None:
                continue
            called += 1
            if p["side"] == ex.entry_side:
                side_ok += 1
            ferr.append(abs(p["frame"] - ex.entry_frame))
            yerr.append(abs(p["y"] - ex.entry_y) / h)
        out[name] = dict(
            returns_called=called,
            of_excursions=len(excursions),
            side_correct=side_ok,
            frame_err=float(np.median(ferr)) if ferr else None,
            y_err_frac=round(float(np.median(yerr)), 4) if yerr else None,
        )
    return out


def analyse(frames, tracker=None, Hs=None):
    """One shot end to end: detections -> tracks -> excursions -> scores."""
    h, w = frames[0].shape[:2]
    dets = detect_moving(frames, Hs=Hs, tracker=tracker)
    tracks = link_tracks(dets, frame_size=(w, h))
    ex = find_excursions(tracks, w, h)
    return dict(detections=sum(len(d) for d in dets), tracks=len(tracks),
                excursions=ex, scores=score(ex, w, h),
                calibrated=score_calibrated(ex, w, h),
                depth_frac=fit_depth(ex, w) if ex else None)
