"""
gating — turn the measurements into one per-shot decision: OFF / NARROW / FULL.

Includes the piece that was missing: a correctness check that needs NO ground
truth.

WHY IT EXISTS
-------------
On the synthetic parallax clip, LayeredBackend reported 87.8% wing coverage
against MosaicBackend's 47.9% -- and both recovered pixels scored ~12 dB against
truth. Nearly double the coverage, zero extra correctness. Coverage alone would
have shipped that as a 1.8x win.

LEAVE-ONE-OUT REPROJECTION
--------------------------
Hide frame i. Reconstruct its CENTRE -- the region we can verify, because we
have the real frame -- using only the other frames and the backend's own
geometry. Compare to the actual frame i.

If the geometry is valid, a hidden frame's centre reconstructs well. If the
backend is warping a multi-depth scene with a single plane, it reconstructs
badly. The centre is a fair proxy for the wings because both come from the same
transforms; the only difference is that the centre is checkable.

This runs on any footage, needs no panorama, no camera track, and catches a
lying backend before its pixels reach the metric.
"""
from __future__ import annotations

import cv2
import numpy as np

import wingcoverage as wc


# ---------------------------------------------------------------- self check

def band_psnr(recon, truth, valid, margin=0.18, min_px=500):
    """
    PSNR over the OUTER BAND only -- the region most like an extrapolated wing.

    Both the 2D and the 3D hold-out score through this one function. That is
    deliberate: a 3D backend and a mosaic backend are compared against the same
    20 dB threshold in `decide`, so if they computed dB by even slightly
    different means the threshold would silently mean two things.

    -> dB, or None when too little of the band was reconstructed to judge.
    """
    h, w = truth.shape[:2]
    band = int(w * margin)
    mask = np.zeros((h, w), bool)
    mask[:, :band] = True
    mask[:, -band:] = True
    mask &= valid
    if mask.sum() < min_px:
        return None
    a = recon[mask].astype(np.float64)
    b = truth[mask].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))


def leave_one_out_3d(frames, backend, tracker=None, probes=3, margin=0.18,
                     iters=None):
    """
    The hold-out for a 3D backend: fit WITHOUT frame i, re-render its pose.

    A 3D reconstruction has no 2D warp between frames -- the mapping runs
    through depth -- so `leave_one_out`'s warp-the-neighbours-in approach cannot
    express it. The check that IS available is the same idea one level up: drop
    a frame from the fit entirely, render its camera from splats that never saw
    it, and score the outer band against the real frame.

    This is stricter than the 2D version, not weaker. The 2D check tests one
    homography; this tests poses, reconstruction, and rasterisation together,
    and a frame excluded from the fit contributes nothing to its own score.

    COST. One fit per probe, which is why `iters` defaults to a third of the
    backend's and `probes` to 3. A validation fit is asking whether the geometry
    is sound, not producing final pixels.
    """
    import splat as sp

    n = len(frames)
    if n < 8:
        return 0.0, []
    h, w = frames[0].shape[:2]
    it = iters if iters is not None else max(300, getattr(backend, "iters", 3000) // 3)
    device = getattr(backend, "device", "cuda")

    poses = backend.poses_for(frames, tracker)
    if len(poses) != n:
        return 0.0, []
    if not poses.trustworthy and not getattr(backend, "allow_untrusted_poses", False):
        # Same refusal as propagate(). Scoring untrusted poses would hand the
        # gate a number it cannot act on: a high dB from a globally wrong
        # reconstruction is exactly the failure the gate exists to catch.
        return 0.0, []

    masks = sp.dynamic_mask(frames, tracker) if getattr(backend, "mask_dynamics", True) else None
    idx = np.linspace(3, n - 4, min(probes, max(1, n - 7))).astype(int)
    scores = []

    for i in idx:
        keep = [j for j in range(n) if j != i]
        sub_frames = [frames[j] for j in keep]
        sub_masks = [masks[j] for j in keep] if masks is not None else None
        sub_poses = poses.subset(keep)
        try:
            model = sp.fit_splats(sub_frames, sub_poses, sub_masks, iters=it,
                                  device=device, verbose=False, tracker=tracker)
            # render_raw, NOT render_widened: the latter pastes the real frame
            # into the centre, which would score the ground truth against itself
            recon, alpha = sp.render_raw(model, poses.viewmats[i], w, h,
                                         K=poses.Ks[i], device=device)
        except RuntimeError:
            continue
        db = band_psnr(recon, frames[i], alpha >= getattr(backend, "alpha_thresh", 0.5),
                       margin)
        if db is not None:
            scores.append(db)

    return (float(np.mean(scores)) if scores else 0.0), scores


def leave_one_out(frames, backend, tracker=None, probes=6, margin=0.18):
    """
    Reconstruct held-out frames from their NEIGHBOURS ONLY.

    Pose for frame i comes from the backend's geometry (that is allowed -- pose
    is what we are testing). Pixels come strictly from other frames. An earlier
    version fed frame i back in as its own source and scored 138 dB on every
    clip, including one where the recovered wings were 12 dB against truth.

    Returns (mean PSNR dB, per-probe list). Higher = geometry is trustworthy.
    """
    tracker = tracker or wc.Tracker()
    n = len(frames)
    if n < 8:
        return 0.0, []
    h, w = frames[0].shape[:2]
    band = int(w * margin)
    # A backend with no 2D warp scores itself its own way. Without this
    # dispatch `warps()` raises, the score is 0.0 dB, and `decide` gates every
    # such shot OFF before a single recovered pixel is looked at -- a silent
    # refusal that looks exactly like bad geometry.
    if getattr(backend, "native_holdout", False):
        return leave_one_out_3d(frames, backend, tracker,
                                probes=min(probes, 3), margin=margin)
    try:
        warp_of = backend.warps(frames, tracker)
    except NotImplementedError:
        return 0.0, []

    idx = np.linspace(3, n - 4, min(probes, max(1, n - 7))).astype(int)
    scores = []

    for i in idx:
        canvas = np.zeros((h, w, 3), np.uint8)
        filled = np.zeros((h, w), bool)
        for j in sorted(range(n), key=lambda j: abs(j - i)):
            if j == i:
                continue                      # the held-out frame never contributes
            if filled.mean() > 0.995:
                break
            for M in warp_of(i, j):           # far to near
                warped = cv2.warpPerspective(frames[j], M, (w, h))
                wm = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M,
                                         (w, h), flags=cv2.INTER_NEAREST) > 128
                new = wm & ~filled
                if not new.any():
                    continue
                canvas[new] = warped[new]
                filled |= new

        db = band_psnr(canvas, frames[i], filled, margin)
        if db is not None:
            scores.append(db)

    return (float(np.mean(scores)) if scores else 0.0), scores


# ---------------------------------------------------------------- decision

WING_OPTIONS = (0.0, 0.10, 0.25, 0.50)

THRESHOLDS = dict(
    geometry_db=20.0,     # below this the backend is not recovering real content
    eff_full=0.55,
    eff_narrow=0.25,
    detail_min=0.10,      # a wing of featureless black is not worth showing
    stale_max=1.20,       # seconds; older than this and the wing is anachronistic
)


def decide(metrics, geometry_db, thresholds=None):
    """
    metrics: dict with effective_coverage, mean_detail, stale_seconds, coverage
    geometry_db: leave_one_out score for the shot

    Returns (state, wing_ratio, reasons)
    """
    t = dict(THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    reasons = []

    if geometry_db < t["geometry_db"]:
        return "OFF", 0.0, [f"geometry unverified ({geometry_db:.1f} dB < {t['geometry_db']:.0f})"]

    if metrics["mean_detail"] < t["detail_min"]:
        return "OFF", 0.0, [f"wing carries no detail ({metrics['mean_detail']*100:.0f}%)"]

    if metrics["stale_seconds"] > t["stale_max"]:
        reasons.append(f"stale {metrics['stale_seconds']:.2f}s")

    eff = metrics["effective_coverage"]
    if eff >= t["eff_full"] and not reasons:
        return "FULL", 0.50, [f"effective {eff*100:.0f}%"]
    if eff >= t["eff_narrow"]:
        return "NARROW", 0.25, reasons + [f"effective {eff*100:.0f}%"]
    return "OFF", 0.0, reasons + [f"effective {eff*100:.0f}% below {t['eff_narrow']*100:.0f}%"]


def hysteresis(states, min_run=12):
    """Kill runs shorter than min_run so wings don't strobe between shots."""
    out = list(states)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_run and i > 0:
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out
