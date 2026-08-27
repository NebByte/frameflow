"""
crossres — merge two cuts of the same take into ONE donor pool.

Measured on Thunderbolts* (main trailer vs Big Game spot): 15 shared takes,
zero false positives. But single-frame donation gained almost nothing, because
most shared takes are framed identically (scale 0.99-1.01). Two real gains were
left on the table:

  1. TEMPORAL EXTENT. Several takes run 17-33 frames LONGER in the other cut.
     Extra frames = extra camera sweep = extra periphery. The previous donation
     used exactly one time-matched frame and ignored the rest of the take.

  2. RESOLUTION. Identical framing, but the alternate is 1920 wide against 640.
     Same pixels, 3x the detail. Propagating into a 2x canvas captures it.

This module registers EVERY frame of both cuts into a common canvas and treats
them as a single pool, nearest-in-time wins.

Composition, for alternate frame m into primary frame k:

    M = inv(HA[k]) @ HA[ia] @ inv(H) @ inv(HB[ib]) @ HB[m]

where HA/HB are each cut's own intra-shot chains, and H is the cross-cut
homography measured between primary sample frame ia and alternate frame ib.
"""
from __future__ import annotations

import cv2
import numpy as np

import wingcoverage as wc


def unified_propagate(fa, fb_small, fb_high, H, ia, ib, wing_ratio=0.15,
                      out_scale=2.0, tracker=None, prefer_highres_centre=True):
    """
    Fill primary's extended wings from BOTH cuts, at out_scale resolution.

    ALL geometry is done in primary-native coordinates. `fb_small` is the
    alternate resampled to primary scale -- H and the alternate's own chain are
    both measured on it, so they compose without a scale mismatch. `fb_high` is
    the alternate at native resolution, used only for SAMPLING, via one explicit
    scale term. Getting this wrong (chaining on high-res while H was measured on
    small) silently corrupts every donated pixel.

    Returns (canvas, filled, tmap, src) per primary frame;
    src: 0 = primary, 1 = alternate.
    """
    tracker = tracker or wc.Tracker()
    h, w = fa[0].shape[:2]
    S = float(out_scale)
    W, Hh = int(w * S), int(h * S)
    ww = int(W * wing_ratio)
    CW = W + 2 * ww

    HA = wc.chain_homographies(tracker, fa)
    HB = wc.chain_homographies(tracker, fb_small)
    if HA[ia] is None or HB[ib] is None:
        return None

    # alternate native -> alternate-at-primary-scale
    r = fb_high[0].shape[1] / fb_small[0].shape[1]
    B2S = np.array([[1 / r, 0, 0], [0, 1 / r, 0], [0, 0, 1]], np.float64)

    upS = np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]], np.float64)
    shift = np.array([[1, 0, ww], [0, 1, 0], [0, 0, 1]], np.float64)
    to_canvas = shift @ upS
    Hinv = np.linalg.inv(H)

    out = []
    n_a, n_b = len(fa), len(fb_small)

    for k in range(n_a):
        canvas = np.zeros((Hh, CW, 3), np.uint8)
        filled = np.zeros((Hh, CW), bool)
        tmap = np.zeros((Hh, CW), np.int32)
        src = np.zeros((Hh, CW), np.uint8)

        if HA[k] is None:
            canvas[:, ww:ww + W] = cv2.resize(fa[k], (W, Hh), interpolation=cv2.INTER_CUBIC)
            filled[:, ww:ww + W] = True
            out.append((canvas, filled, tmap, src))
            continue

        HAk_inv = np.linalg.inv(HA[k])
        A_from_B = HAk_inv @ HA[ia] @ Hinv          # alt-at-primary-scale -> primary k

        def blit(frame, M, cut, dist, only_empty=True):
            nonlocal canvas, filled, tmap, src
            fh, fw = frame.shape[:2]
            corners = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]],
                               np.float64).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
            if proj[:, 0].max() < 0 or proj[:, 0].min() > CW:
                return
            if proj[:, 1].max() < 0 or proj[:, 1].min() > Hh:
                return
            # cubic here, not the shared helper's bilinear: this path exists to
            # carry an alternate cut's NATIVE resolution across, and cubic is
            # what keeps that detail. The mask is the part that has to change --
            # a nearest-neighbour mask calls the warp's own darkened edge valid.
            warped = cv2.warpPerspective(frame, M, (CW, Hh), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            wm = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), M,
                                     (CW, Hh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0) >= 254
            wm = cv2.erode(wm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            new = wm & ~filled if only_empty else wm
            if not new.any():
                return
            canvas[new] = warped[new]
            tmap[new] = dist
            src[new] = cut
            filled[new] = True

        # 1. seed the centre. prefer the alternate's NATIVE resolution over an
        #    upscale of the primary -- identical framing, more real detail.
        seeded = False
        if prefer_highres_centre and r > 1.05:
            m = int(round(ib + (k - ia)))
            if 0 <= m < n_b and HB[m] is not None:
                rel = np.linalg.inv(HB[ib]) @ HB[m]
                M = to_canvas @ A_from_B @ rel @ B2S
                blit(fb_high[m], M, 1, 0)
                seeded = True
        if not seeded or not filled[:, ww:ww + W].all():
            centre = np.zeros((Hh, CW, 3), np.uint8)
            centre[:, ww:ww + W] = cv2.resize(fa[k], (W, Hh),
                                              interpolation=cv2.INTER_CUBIC)
            gap = np.zeros((Hh, CW), bool)
            gap[:, ww:ww + W] = True
            gap &= ~filled
            canvas[gap] = centre[gap]
            filled |= gap

        # 2. wings: nearest-in-time across BOTH cuts
        donors = [(abs(j - k), 0, j) for j in range(n_a) if j != k]
        donors += [(abs((m - ib) - (k - ia)), 1, m) for m in range(n_b)]
        donors.sort()
        for dist, cut, idx in donors:
            if filled.mean() > 0.995:
                break
            if cut == 0:
                if HA[idx] is None:
                    continue
                M = to_canvas @ HAk_inv @ HA[idx]
                blit(fa[idx], M, 0, dist)
            else:
                if HB[idx] is None:
                    continue
                rel = np.linalg.inv(HB[ib]) @ HB[idx]
                M = to_canvas @ A_from_B @ rel @ B2S
                blit(fb_high[idx], M, 1, dist)

        out.append((canvas, filled, tmap, src))
    return out


def wing_stats(filled, src, ww, W):
    left = slice(0, ww)
    right = slice(ww + W, None)
    f = np.concatenate([filled[:, left], filled[:, right]], 1)
    s = np.concatenate([src[:, left], src[:, right]], 1)
    cov = float(f.mean())
    from_alt = float(((s == 1) & f).sum() / max(f.sum(), 1))
    return cov, from_alt


def sharpness(img):
    """Variance of Laplacian — proxy for real detail vs upscaled mush."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
