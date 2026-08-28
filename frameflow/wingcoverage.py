"""
wingcoverage — measure how much of a ScreenX-style side wall is RECOVERED REAL
FOOTAGE rather than invention, frame by frame.

The claim this tool exists to support:
    Every camera move already filmed the periphery. Cropping threw it away.
    Before you generate anything, measure how much you can simply get back.

Pipeline per shot:
    1. shot detection            (HSV histogram correlation)
    2. inter-frame homographies  (ORB + RANSAC, chained to a reference frame)
    3. motion classification     LOCKED / ROTATION / PARALLAX
    4. wing propagation          nearest-in-time source pixel wins
    5. coverage metric           % real pixels, + how stale those pixels are
    6. gating decision           wings ON only above threshold

PARALLAX shots are correctly detected and REFUSED by the mosaic backend --
those are the ones that need 3D Gaussian Splatting. Refusing them is the point;
a homography quietly applied to a parallax shot produces confident garbage.

CPU only. No GPU required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings

import cv2
import numpy as np

# ---------------------------------------------------------------- parameters

WING_RATIO = 0.75      # each wing is 0.75 x main-screen width  (~270 deg feel)
MIN_MATCHES = 25       # below this, homography is not trustworthy
RANSAC_PX = 3.0
LOCKED_PX = 1.5        # median feature displacement below this => tripod
PARALLAX_MARGIN = 0.04 # fraction of points in a 2nd motion layer => parallax
PARALLAX_DISAGREE_PX = 8.0  # ...and the two layers must map the frame this far apart
COVER_ON = 0.45        # wings on above this coverage
COVER_OFF = 0.32       # wings off below this (hysteresis)
MIN_RUN = 12           # frames; ignore wing states shorter than this


# ---------------------------------------------------------------- warping

def warp_with_mask(img, M, size, border=cv2.BORDER_REPLICATE):
    """
    Warp an image and return the mask of pixels that are ACTUALLY valid.

    The obvious spelling of this -- bilinear for the image, INTER_NEAREST for a
    255-filled mask -- is wrong, and it is where the thin dark lines down every
    recovered wall came from.

    A bilinear warp against a zero border blends the outermost source pixels
    with black. INTER_NEAREST on the mask rounds those same pixels to "valid",
    so a column at roughly half brightness gets composited into the wall and
    labelled RECOVERED. Every donor footprint contributed one such hairline;
    2.3% of wing columns in a delivered film were these.

    Two changes fix it for good:
      - the image warps with BORDER_REPLICATE, so an edge sample can only ever
        blend with more image, never with black
      - the mask warps BILINEAR and is accepted only where it comes back fully
        opaque, then erodes by one pixel

    The cost is a one-pixel ring of coverage per donor. That is the honest
    trade: those pixels were never wholly observed, and claiming them was what
    put a black line where a wall should be.
    """
    warped = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR,
                                 borderMode=border)
    h, w = img.shape[:2]
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = m >= 254
    if mask.any():
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=1).astype(bool)
    return warped, mask


def match_exposure(src, dst, where, max_gain=1.6):
    """
    Put a donor on the same exposure as the wall it is joining.

    Two frames of a handheld pan are not the same brightness -- auto-exposure
    rides the light in the room -- so a donor pasted straight in meets its
    neighbour at a step, and a step down a wall reads as a seam. Solves a single
    gain and bias over the pixels the two already share and applies it.

    `where` is the overlap: pixels this donor covers that are already filled.
    Too small an overlap means the fit is noise, so it is left alone.
    """
    n = int(where.sum())
    if n < 200:
        return src
    a = src[where].reshape(-1, 3).astype(np.float64)
    b = dst[where].reshape(-1, 3).astype(np.float64)
    # a gain and a bias are two numbers per channel; fitting them on a quarter
    # of a million pixels rather than a few thousand buys no accuracy and costs
    # real time, since this runs once per donor per frame
    if n > 4000:
        step = n // 4000
        a, b = a[::step], b[::step]
    out = src.astype(np.float32)
    for c in range(3):
        va = a[:, c].var()
        if va < 1e-3:
            continue
        gain = float(np.clip(np.cov(a[:, c], b[:, c])[0, 1] / va,
                             1.0 / max_gain, max_gain))
        bias = float(b[:, c].mean() - gain * a[:, c].mean())
        out[:, :, c] = out[:, :, c] * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- shot detect

def detect_shots(frames, thresh=0.55):
    """Cut where consecutive HSV histograms decorrelate."""
    hists = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    cuts = [0]
    for i in range(1, len(frames)):
        if cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL) < thresh:
            cuts.append(i)
    cuts.append(len(frames))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= 8]


# ---------------------------------------------------------------- geometry

class Tracker:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=7)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(g, None)

    def match(self, kp1, d1, kp2, d2):
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        raw = self.bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return np.empty((0, 2)), np.empty((0, 2))
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return p1, p2


def _second_layer(p1, p2, w, h):
    """
    Two-layer test. Fit one homography by RANSAC, then ask whether the OUTLIERS
    are themselves explained by a second, DIFFERENT homography.

    One coherent plane -> outliers are noise, no second fit.
    Real parallax      -> foreground and background are two coherent layers.

    Returns (fraction of points in layer 2, how far the two layers disagree in px).
    """
    H1, m1 = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    if H1 is None or m1 is None:
        return 0.0, 0.0, 0.0
    inl = m1.ravel().astype(bool)
    o1, o2 = p1[~inl], p2[~inl]
    if len(o1) < MIN_MATCHES:
        return 0.0, 0.0, 0.0

    H2, m2 = cv2.findHomography(o1, o2, cv2.RANSAC, RANSAC_PX)
    if H2 is None or m2 is None:
        return 0.0, 0.0, 0.0        # three values, like every other exit here
    purity = float(m2.ravel().mean())
    layer2 = purity * (len(o1) / len(p1))

    # do the two layers actually map the frame differently?
    probe = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]],
                     np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(probe, H1).reshape(-1, 2)
    b = cv2.perspectiveTransform(probe, H2).reshape(-1, 2)
    disagree = float(np.median(np.linalg.norm(a - b, axis=1)))
    return layer2, disagree, purity


def classify_motion(tracker, frames, gap=3):
    """LOCKED / ROTATION / PARALLAX."""
    n = len(frames)
    gap = min(gap, n - 1)
    h, w = frames[0].shape[:2]
    feats = {}

    def get(i):
        if i not in feats:
            feats[i] = tracker.features(frames[i])
        return feats[i]

    disps, l2s, dgs, pus = [], [], [], []
    for i in range(0, max(1, n - gap), max(1, gap // 2)):
        (k1, d1), (k2, d2) = get(i), get(i + gap)
        p1, p2 = tracker.match(k1, d1, k2, d2)
        if len(p1) < MIN_MATCHES:
            continue
        disps.append(float(np.median(np.linalg.norm(p1 - p2, axis=1))))
        l2, dg, pu = _second_layer(p1, p2, w, h)
        l2s.append(l2); dgs.append(dg); pus.append(pu)

    if not disps:
        return "LOCKED", dict(displacement=0.0, layer2=0.0, layer_disagree=0.0)

    disp = float(np.median(disps))
    stats = dict(displacement=round(disp, 2))

    # tripod first: no baseline means every other test is degenerate
    if disp < LOCKED_PX:
        stats.update(layer2=0.0, layer_disagree=0.0)
        return "LOCKED", stats

    l2 = float(np.median(l2s)) if l2s else 0.0
    dg = float(np.median(dgs)) if dgs else 0.0
    stats.update(layer2=round(l2, 3), layer_disagree=round(dg, 1),
                 layer2_purity=round(float(np.median(pus)) if pus else 0.0, 3))

    # a second coherent layer that maps the frame somewhere else = parallax
    #
    # A ratio trigger was tried here and removed. A gym pan reports layer2=0.013
    # against a 0.04 margin while its two layers disagree by 11.4px on a frame
    # that moved 4.0px, so on paper it is thinly-supported parallax being missed
    # -- and routing it to LayeredBackend did nothing for the defect it was
    # meant to fix. Measured on 80 frames of that clip, mosaic and layered both
    # left the wing/centre seam at 1.88x, while layered dropped hold-out
    # geometry from 32.4 to 29.2 dB and put a warped notch in the frame corner.
    # The threshold is not what is wrong: a wing built from ANY single warp per
    # layer places off-plane content at the wrong offset, which is why the walls
    # can repeat signage that is still on screen. Fixing that needs depth, not a
    # different classifier -- see GaussianBackend.
    if l2 >= PARALLAX_MARGIN and dg > PARALLAX_DISAGREE_PX:
        return "PARALLAX", stats
    return "ROTATION", stats


def chain_homographies(tracker, frames, anchor_stride=5):
    """
    H[j] maps frame j into frame 0's plane.

    Naive consecutive chaining composes n noisy homographies and drifts badly
    (measured: ~8 dB of avoidable error over a 90-frame pan). Instead we pick
    anchor frames every `anchor_stride`, chain only between anchors, and register
    every other frame DIRECTLY to its nearest anchor. Drift then accumulates over
    n/stride compositions instead of n, for about the same amount of matching.
    """
    n = len(frames)
    fcache = {}

    def feats(k):
        if k not in fcache:
            fcache[k] = tracker.features(frames[k])
        return fcache[k]

    def solve(src, dst):
        (ks, ds), (kd, dd) = feats(src), feats(dst)
        ps, pd = tracker.match(ks, ds, kd, dd)
        if len(ps) < MIN_MATCHES:
            return None
        H, m = cv2.findHomography(ps, pd, cv2.RANSAC, RANSAC_PX)
        if H is None or m is None or float(m.mean()) < 0.35:
            return None
        return H

    anchors = list(range(0, n, anchor_stride))
    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    # chain between anchors only
    A = {anchors[0]: np.eye(3)}
    for k in range(1, len(anchors)):
        a_prev, a_cur = anchors[k - 1], anchors[k]
        H = solve(a_cur, a_prev)
        if H is None:                       # anchor hop failed; try a wider hop
            H = solve(a_cur, anchors[max(0, k - 2)])
            base = A.get(anchors[max(0, k - 2)])
        else:
            base = A.get(a_prev)
        A[a_cur] = (base @ H) if (H is not None and base is not None) else None

    # every frame registers directly to its nearest usable anchor
    Hs = []
    for j in range(n):
        cand = sorted(anchors, key=lambda a: abs(a - j))
        Hj = None
        for a in cand[:3]:
            if A.get(a) is None:
                continue
            if a == j:
                Hj = A[a]
                break
            H = solve(j, a)
            if H is not None:
                Hj = A[a] @ H
                break
        Hs.append(Hj)
    return Hs


# ---------------------------------------------------------------- propagation

def propagate_wings(frames, Hs, wing_w, tracker=None, stick=True, stick_max=24,
                    exposure=True):
    """
    For each frame i, fill the extended canvas [-wing_w, W+wing_w] from the
    frames that saw each pixel.

    Registration quality is set by chain_homographies, which uses anchor frames
    to bound drift accumulation.

    Two rules beyond "nearest in time wins", both aimed at how the wall behaves
    OVER TIME rather than in any single frame:

    `stick`     a pixel keeps the donor that fed it last frame for as long as
                that donor still covers it and is still fresh. The plain
                nearest-in-time contest is re-run per frame, so the boundary
                between two donors' territory crawls as the camera moves and the
                whole wall re-assembles every frame -- measured at 2.6x the
                centre's frame-to-frame change, and visible as a shimmer. A
                pixel fed by one photograph for a stretch is steady.

    `exposure`  a donor is gain/bias matched to the wall it is joining before it
                composites, so two frames shot at different auto-exposure no
                longer meet at a step. See match_exposure.

    Neither invents anything: every pixel is still one photographed pixel from
    this camera, so the provenance ladder is untouched.

    Returns per frame: wing image, coverage mask, temporal-offset map (frames).
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    cw = w + 2 * wing_w
    out = []
    tracker = tracker or Tracker()
    shift = np.array([[1, 0, wing_w], [0, 1, 0], [0, 0, 1]], np.float64)
    shift_inv = np.linalg.inv(shift)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                       np.float64).reshape(-1, 1, 2)

    prev_donor, prev_i = None, None

    for i in range(n):
        canvas = np.zeros((h, cw, 3), np.uint8)
        filled = np.zeros((h, cw), bool)
        tmap = np.zeros((h, cw), np.int32)
        donor = np.full((h, cw), -1.0, np.float32)   # which frame fed each pixel

        # the frame itself occupies the centre, always real, offset 0
        canvas[:, wing_w:wing_w + w] = frames[i]
        filled[:, wing_w:wing_w + w] = True

        if Hs[i] is None:
            out.append((canvas, filled, tmap))
            prev_donor, prev_i = None, None
            continue

        Hi_inv = np.linalg.inv(Hs[i])
        cache = {}

        def place(j, want=None):
            """Composite donor j into whatever of `want` it can actually cover."""
            if j == i or Hs[j] is None:
                return
            if j not in cache:
                M = shift @ Hi_inv @ Hs[j]
                proj = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
                x0, y0 = proj.min(0)
                x1, y1 = proj.max(0)
                if x1 < 0 or x0 > cw or y1 < 0 or y0 > h:   # cheap cull
                    cache[j] = None
                else:
                    cache[j] = warp_with_mask(frames[j], M, (cw, h))
            if cache[j] is None:
                return
            warped, wmask = cache[j]
            new = wmask & ~filled
            if want is not None:
                new = new & want
            if not new.any():
                return
            if exposure:
                warped = match_exposure(warped, canvas, wmask & filled)
            canvas[new] = warped[new]
            tmap[new] = abs(j - i)
            donor[new] = j
            filled[new] = True        # in place: `filled |= new` rebinds it local

        # 1. whatever fed this wall last frame keeps it, while it is still fresh
        if stick and prev_donor is not None and Hs[prev_i] is not None:
            Mp = shift @ Hi_inv @ Hs[prev_i] @ shift_inv
            carried = cv2.warpPerspective(prev_donor, Mp, (cw, h),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=-1.0)
            ids = [int(v) for v in np.unique(carried)
                   if v >= 0 and abs(int(v) - i) <= stick_max]
            for j in sorted(ids, key=lambda j: abs(j - i)):
                place(j, want=(carried == j))

        # 2. nearest in time fills whatever is still open
        #
        # `patience` is what stops this being quadratic. Donors are visited
        # nearest-first and coverage only grows, so once a run of them in a row
        # has nothing left to add, the ones further out will not either -- they
        # see less of the wing, not more. The old spelling relied on coverage
        # crossing 99.5% to break, which a wall that tops out at 93% never does:
        # every frame then warped every other frame, for nothing.
        #
        # It is a heuristic and it has one blind spot worth naming: a camera
        # that HOLDS for longer than `patience` frames and then moves again
        # produces a run of donors that add nothing followed by donors that
        # would have, and this stops at the first run. Measured against the
        # exhaustive scan on 80 frames of a handheld gym pan the wing came out
        # 0.909 filled against 0.910, so the cost on real footage is a tenth of
        # a percent; a locked-off hold long enough to trip it is a shot the
        # gate refuses anyway.
        stall, patience = 0, 15
        for j in sorted(range(n), key=lambda j: abs(j - i)):
            if filled.mean() > 0.995 or stall >= patience:
                break
            before = int(filled.sum())
            place(j)
            stall = 0 if int(filled.sum()) > before else stall + 1

        out.append((canvas, filled, tmap))
        prev_donor, prev_i = donor, i
    return out


def settle_wings(propagated, warp_of, wing_w, k=2, min_samples=3, max_spread=18.0):
    """
    Damp the shimmer out of a recovered wall using only its own photography.

    Even with sticky donors a wall pixel is resampled from a different exposure
    of the same surface as the camera moves, and the residue reads as a fine
    boil across the wing. This aligns each frame's NEIGHBOURS into its own
    coordinates and takes the per-pixel median of the wall.

    A median of aligned samples of the same surface, all shot a fraction of a
    second apart by the same camera on the same move, is still that camera's
    photography -- closer to stacking frames than to inventing one. So the rung
    does not move, and neither does the metric: this only refines the VALUE of
    pixels that were already recovered. It never fills an empty pixel, never
    touches `filled` or `tmap`, and never crosses into the centre.

    `max_spread` is what keeps it honest on everything a single warp cannot
    align. Where the aligned samples DISAGREE -- a person walking through the
    wing, or a near layer that the background homography puts in the wrong
    place -- the median of them would be a smear, so the original is kept. Only
    pixels whose samples already agree get replaced, which is exactly the set
    where the disagreement was noise rather than content.

    `warp_of(i, t)` maps frame t into frame i's coordinates, or returns None.
    """
    n = len(propagated)
    if n == 0 or k < 1 or wing_w < 1:
        return propagated
    h, cw = propagated[0][1].shape
    w = cw - 2 * wing_w
    shift = np.array([[1, 0, wing_w], [0, 1, 0], [0, 0, 1]], np.float64)
    shift_inv = np.linalg.inv(shift)

    def bands(a):
        """The two wings side by side; the centre is not ours to touch."""
        return np.concatenate([a[:, :wing_w], a[:, cw - wing_w:]], axis=1)

    # Written back in place, behind the window rather than into a second list.
    #
    # A full stack out beside a full stack in doubles peak memory, and at 1024px
    # over 799 frames that is four gigabytes of canvases for no reason: once the
    # window has passed frame i - k - 1, nothing will read it as a neighbour
    # again, so its settled version can take its slot.
    pending = {}
    for i in range(n):
        canvas, filled, tmap = propagated[i]
        vals = [bands(canvas).astype(np.float32)]
        oks = [bands(filled)]
        for t in range(max(0, i - k), min(n, i + k + 1)):
            if t == i:
                continue
            M = warp_of(i, t)
            if M is None:
                continue
            M = shift @ M @ shift_inv
            wcv, wm = warp_with_mask(propagated[t][0], M, (cw, h))
            fm = cv2.warpPerspective(propagated[t][1].astype(np.uint8) * 255, M,
                                     (cw, h), flags=cv2.INTER_NEAREST) > 128
            vals.append(bands(wcv).astype(np.float32))
            oks.append(bands(wm & fm))

        if len(vals) < min_samples:
            pending[i] = (canvas, filled, tmap)
            _retire(propagated, pending, i - k - 1)
            continue

        stack = np.stack(vals)                     # S x h x 2ww x 3
        seen = np.stack(oks)                       # S x h x 2ww
        masked = np.where(seen[..., None], stack, np.nan)
        # The agreement test asks whether the REPLACEMENTS agree, which means
        # asking the neighbours about each other and leaving this frame's own
        # sample out of it. Including it gets the question backwards: a pixel
        # sitting on a baked-in hairline disagrees with four clean neighbours,
        # and a test that reads that as "the samples disagree" vetoes fixing
        # precisely the pixel most in need of fixing. Four samples that agree
        # with each other are a good value whatever the fifth one says.
        others = masked[1:] if len(vals) > 1 else masked
        # a pixel no sample saw is an all-NaN slice, which is the normal case
        # out at the edge of the wall rather than anything to warn about
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(masked, axis=0)
            # Median absolute deviation, not the min-to-max range: one odd
            # sample barely moves a MAD and sends a range wide open, and it is
            # genuine disagreement -- a figure crossing the wing, a near layer
            # the background warp cannot place -- that must veto a replacement,
            # not a speck.
            omed = np.nanmedian(others, axis=0)
            mad = np.nanmedian(np.abs(others - omed), axis=0)
        spread = np.nan_to_num(mad, nan=1e9).max(axis=2)
        agree = np.nan_to_num(np.stack(oks)[1:].sum(axis=0) if len(oks) > 1
                              else seen.sum(axis=0))
        take = ((seen.sum(axis=0) >= min_samples) & bands(filled)
                & (agree >= min_samples - 1) & (spread <= max_spread))

        band = bands(canvas).copy()
        band[take] = np.nan_to_num(med, nan=0.0)[take].astype(np.uint8)

        settled = canvas.copy()
        settled[:, :wing_w] = band[:, :wing_w]
        settled[:, cw - wing_w:] = band[:, wing_w:]
        settled[:, wing_w:wing_w + w] = canvas[:, wing_w:wing_w + w]   # the fence
        pending[i] = (settled, filled, tmap)
        _retire(propagated, pending, i - k - 1)

    for idx in sorted(pending):
        propagated[idx] = pending[idx]
    return propagated


def _retire(propagated, pending, idx):
    """Move a finished frame into the stack it came from, once it is safe."""
    if idx >= 0 and idx in pending:
        propagated[idx] = pending.pop(idx)


# ---------------------------------------------------------------- metrics

DETAIL_WIN_AT = 480      # the working width the 9-pixel window was tuned at


def detail_weight(canvas, floor=2.0, full=12.0, win=None):
    """
    How much INFORMATION a recovered pixel carries.

    Shot 50 of the real trailer scored 43% coverage with a wing that was almost
    entirely black. Recovering darkness is free and meaningless -- a filled pixel
    counts the same whether it carries texture or nothing. This weights each
    pixel by local standard deviation, so featureless regions stop inflating the
    metric.

    floor: local std below this carries no information at all
    full : local std at or above this is fully informative

    THE WINDOW IS A FRACTION OF THE FRAME, NOT A COUNT OF PIXELS
    -----------------------------------------------------------
    It was a fixed nine pixels, and that made the whole metric punish
    resolution. A physical texture spread over twice as many pixels varies less
    within any nine of them, so the same shot measured:

        480px  detail 0.62      960px  detail 0.42      1280px  detail 0.34

    -- identical content, scored a third lower for being sharper. That is
    backwards on its own terms, and it was not theoretical: a 1024px render of a
    clip that passed comfortably at 480px came back at 24.57% effective
    coverage against a 25% bar, was gated OFF, and delivered a film with black
    walls. The operator had done nothing but ask for more resolution.

    Nine pixels at 480 wide is 1.875% of the frame. That fraction is what was
    tuned, so that fraction is what is kept, and a window now spans the same
    piece of the WORLD at any working width.
    """
    if win is None:
        win = int(round(9.0 * canvas.shape[1] / DETAIL_WIN_AT)) | 1
    win = max(3, win)
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win))
    sq = cv2.boxFilter(g * g, -1, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return np.clip((std - floor) / (full - floor), 0.0, 1.0)


def staleness_weight(frames_back):
    """
    Empirical quality decay of a recovered pixel vs how far back its source is.

    Measured on ground-truth footage (see validate.py): recovered pixel PSNR runs
    28.0 dB at 0-3 frames back -- which IS the sub-pixel resampling ceiling, i.e.
    perfect -- then 25.3 / 22.6 / 22.5 / 21.6 / 20.7 / 16.8 dB as the source
    recedes. Registration error and edge-of-frame sampling both grow with distance.

    So raw coverage over-reports. A wing that is 90% filled from 25 frames back is
    worse than one 70% filled from 2 frames back. This weights each covered pixel
    by its expected quality, normalised to 1.0 at zero staleness.
    """
    psnr = 28.0 - 0.40 * np.asarray(frames_back, dtype=np.float64)
    return np.clip((psnr - 16.0) / (28.0 - 16.0), 0.0, 1.0)


def wing_metrics(filled, tmap, wing_w, w, fps, canvas=None):
    """Coverage and staleness for the two wings only (centre is trivially real)."""
    left_f = filled[:, :wing_w]
    right_f = filled[:, wing_w + w:]
    both = np.concatenate([left_f, right_f], 1)
    cov = float(both.mean())

    stale_px = np.concatenate([tmap[:, :wing_w][left_f], tmap[:, wing_w + w:][right_f]])
    stale = float(np.median(stale_px)) / fps if stale_px.size else 0.0

    # quality-weighted: what fraction of the wing is USABLE real footage.
    # two independent discounts -- how stale the pixel is, and whether it
    # actually carries any detail.
    if stale_px.size:
        sw = staleness_weight(stale_px)
        if canvas is not None:
            dw_full = detail_weight(canvas)
            dw = np.concatenate([dw_full[:, :wing_w][left_f],
                                 dw_full[:, wing_w + w:][right_f]])
        else:
            dw = np.ones_like(sw)
        eff = float((sw * dw).sum() / both.size)
        det = float(dw.mean())
    else:
        eff, det = 0.0, 0.0

    return dict(
        coverage=round(cov, 4),
        effective_coverage=round(eff, 4),
        mean_detail=round(det, 4),
        coverage_left=round(float(left_f.mean()), 4),
        coverage_right=round(float(right_f.mean()), 4),
        stale_seconds=round(stale, 2),
    )


def gate(coverages, on=COVER_ON, off=COVER_OFF, min_run=MIN_RUN):
    """Hysteresis + minimum run length, so wings don't strobe on and off."""
    state = False
    raw = []
    for c in coverages:
        if not state and c >= on:
            state = True
        elif state and c < off:
            state = False
        raw.append(state)

    # kill runs shorter than min_run
    out = raw[:]
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run:
            for k in range(i, j):
                out[k] = out[i - 1] if i > 0 else False
        i = j
    return out


# ---------------------------------------------------------------- driver

def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def run(video, outdir, wing_ratio=WING_RATIO, preview=True):
    os.makedirs(outdir, exist_ok=True)
    frames, fps = read_video(video)
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    h, w = frames[0].shape[:2]
    wing_w = int(w * wing_ratio)
    tracker = Tracker()

    shots = detect_shots(frames)
    rows, shot_report = [], []

    for si, (a, b) in enumerate(shots):
        seg = frames[a:b]
        kind, stats = classify_motion(tracker, seg)

        if kind in ("LOCKED", "PARALLAX"):
            # refuse: no recoverable periphery, or homography would lie about it
            for i in range(a, b):
                rows.append(dict(frame=i, shot=si, motion=kind, coverage=0.0,
                                 effective_coverage=0.0, mean_detail=0.0,
                                 coverage_left=0.0, coverage_right=0.0,
                                 stale_seconds=0.0))
            shot_report.append(dict(shot=si, start=a, end=b, motion=kind,
                                    backend="refused", mean_coverage=0.0, **stats))
            continue

        Hs = chain_homographies(tracker, seg)
        prop = propagate_wings(seg, Hs, wing_w, tracker)

        covs = []
        for k, (canvas, filled, tmap) in enumerate(prop):
            m = wing_metrics(filled, tmap, wing_w, w, fps, canvas)
            covs.append(m["effective_coverage"])
            rows.append(dict(frame=a + k, shot=si, motion=kind, **m))

        shot_report.append(dict(shot=si, start=a, end=b, motion=kind,
                                backend="mosaic",
                                mean_coverage=round(float(np.mean(covs)), 4), **stats))

        if preview:
            save_preview(prop, outdir, si, wing_w, w, covs)

    on = gate([r["effective_coverage"] for r in rows])
    for r, s in zip(rows, on):
        r["wings_on"] = int(s)

    with open(f"{outdir}/coverage.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = dict(
        video=os.path.basename(video),
        frames=len(frames), fps=round(fps, 2),
        resolution=f"{w}x{h}", wing_width_px=wing_w,
        shots=shot_report,
        wings_on_frames=int(sum(on)),
        wings_on_pct=round(100.0 * sum(on) / len(on), 1),
        mean_coverage_when_on=round(
            float(np.mean([r["coverage"] for r in rows if r["wings_on"]])), 4
        ) if any(on) else 0.0,
        mean_effective_when_on=round(
            float(np.mean([r["effective_coverage"] for r in rows if r["wings_on"]])), 4
        ) if any(on) else 0.0,
    )
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def save_preview(prop, outdir, si, wing_w, w, covs, every=15):
    """Three-panel contact sheet: dimmed wings marked where invented."""
    os.makedirs(f"{outdir}/preview", exist_ok=True)
    for k in range(0, len(prop), every):
        canvas, filled, _ = prop[k]
        vis = canvas.copy()
        # tint uncovered wing pixels magenta so gaps are unmissable
        gap = ~filled
        vis[gap] = (255, 0, 255)
        # seam lines between wall and main screen
        cv2.line(vis, (wing_w, 0), (wing_w, vis.shape[0]), (0, 255, 255), 1)
        cv2.line(vis, (wing_w + w, 0), (wing_w + w, vis.shape[0]), (0, 255, 255), 1)
        cv2.putText(vis, f"frame {k}  coverage {covs[k]*100:.1f}%", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{outdir}/preview/shot{si:02d}_f{k:04d}.png", vis)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default="out")
    ap.add_argument("--wing", type=float, default=WING_RATIO)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    s = run(args.video, args.outdir, args.wing, preview=not args.no_preview)
    print(json.dumps(s, indent=2))
