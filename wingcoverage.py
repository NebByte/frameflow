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

def propagate_wings(frames, Hs, wing_w, tracker=None):
    """
    For each frame i, fill the extended canvas [-wing_w, W+wing_w] using the
    NEAREST-IN-TIME frame that saw each pixel.

    Registration quality is set by chain_homographies, which uses anchor frames
    to bound drift accumulation.

    Returns per frame: wing image, coverage mask, temporal-offset map (frames).
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    cw = w + 2 * wing_w
    out = []
    tracker = tracker or Tracker()

    for i in range(n):
        canvas = np.zeros((h, cw, 3), np.uint8)
        filled = np.zeros((h, cw), bool)
        tmap = np.zeros((h, cw), np.int32)

        # the frame itself occupies the centre, always real, offset 0
        canvas[:, wing_w:wing_w + w] = frames[i]
        filled[:, wing_w:wing_w + w] = True

        if Hs[i] is None:
            out.append((canvas, filled, tmap))
            continue

        Hi_inv = np.linalg.inv(Hs[i])
        shift = np.array([[1, 0, wing_w], [0, 1, 0], [0, 0, 1]], np.float64)

        # visit sources nearest in time first, so early fills win
        for j in sorted(range(n), key=lambda j: abs(j - i)):
            if j == i or Hs[j] is None:
                continue
            if filled.mean() > 0.995:
                break
            M = shift @ Hi_inv @ Hs[j]

            # cheap cull: does frame j's footprint touch anything unfilled?
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
            x0, y0 = proj.min(0)
            x1, y1 = proj.max(0)
            if x1 < 0 or x0 > cw or y1 < 0 or y0 > h:
                continue

            warped = cv2.warpPerspective(frames[j], M, (cw, h))
            wmask = cv2.warpPerspective(
                np.ones((h, w), np.uint8) * 255, M, (cw, h), flags=cv2.INTER_NEAREST
            ) > 128

            new = wmask & ~filled
            if not new.any():
                continue
            canvas[new] = warped[new]
            tmap[new] = abs(j - i)
            filled |= new

        out.append((canvas, filled, tmap))
    return out


# ---------------------------------------------------------------- metrics

def detail_weight(canvas, floor=2.0, full=12.0, win=9):
    """
    How much INFORMATION a recovered pixel carries.

    Shot 50 of the real trailer scored 43% coverage with a wing that was almost
    entirely black. Recovering darkness is free and meaningless -- a filled pixel
    counts the same whether it carries texture or nothing. This weights each
    pixel by local standard deviation, so featureless regions stop inflating the
    metric.

    floor: local std below this carries no information at all
    full : local std at or above this is fully informative
    """
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
