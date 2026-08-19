"""
backends — pluggable propagation backends behind one interface.

    frames in  ->  (canvas, filled, tmap) out

Everything downstream (metrics, gating, fenced fill) is backend-agnostic, so
swapping the geometry engine changes nothing else.

    MosaicBackend    single homography per frame. Correct for rotation-dominant
                     and planar shots. REFUSES parallax -- a single warp applied
                     to a multi-depth scene produces confident garbage.

    LayeredBackend   fits K motion layers by RANSAC-and-remove, each with its own
                     homography chain, composited back-to-front. Handles the
                     PARALLAX class that MosaicBackend refuses. CPU only.
                     Approximates depth with piecewise-planar layers -- it is not
                     a 3D reconstruction, and it degrades on continuous depth
                     gradients (a receding road) where no finite set of planes
                     fits.

    GaussianBackend  true 3D. Reconstructs the static scene, renders a widened
                     frustum. Needs CUDA; raises on unavailable hardware rather
                     than silently falling back, because a silent fallback would
                     put mosaic-quality pixels behind a 3D confidence label.
"""
from __future__ import annotations

import cv2
import numpy as np

import wingcoverage as wc


# --------------------------------------------------------------- interface

class Backend:
    name = "base"
    handles = ()          # motion classes this backend accepts
    native_holdout = False   # True: score yourself, gating must not warp for you

    def accepts(self, motion: str) -> bool:
        return motion in self.handles

    def propagate(self, frames, wing_w, tracker=None):
        """-> list of (canvas HxCWx3 uint8, filled HxCW bool, tmap HxCW int32)"""
        raise NotImplementedError

    def warps(self, frames, tracker=None):
        """
        -> callable (i, j) -> [M, ...] mapping frame j into frame i's
        coordinates, ordered FAR to NEAR so later entries composite in front.

        Exposed separately from propagate() so the geometry can be validated
        independently of the pixels it produces -- see gating.leave_one_out.
        """
        raise NotImplementedError


# --------------------------------------------------------------- mosaic

class MosaicBackend(Backend):
    name = "mosaic"
    handles = ("ROTATION",)

    def warps(self, frames, tracker=None):
        tracker = tracker or wc.Tracker()
        Hs = wc.chain_homographies(tracker, frames)

        def f(i, j):
            if Hs[i] is None or Hs[j] is None:
                return []
            return [np.linalg.inv(Hs[i]) @ Hs[j]]
        return f

    def propagate(self, frames, wing_w, tracker=None):
        tracker = tracker or wc.Tracker()
        Hs = wc.chain_homographies(tracker, frames)
        return [(c, f, t) for c, f, t in wc.propagate_wings(frames, Hs, wing_w, tracker)]


# --------------------------------------------------------------- layered

def fit_layers(p1, p2, k_max=3, min_pts=25, ransac_px=3.0):
    """
    RANSAC-and-remove: fit a homography, strip its inliers, refit on the rest.
    Returns [(H, point_count, median_motion_px), ...] ordered by motion
    magnitude descending -- faster layers are nearer the camera, so they
    composite in front.
    """
    layers = []
    idx = np.arange(len(p1))
    for _ in range(k_max):
        if len(idx) < min_pts:
            break
        H, m = cv2.findHomography(p1[idx], p2[idx], cv2.RANSAC, ransac_px)
        if H is None or m is None:
            break
        inl = m.ravel().astype(bool)
        if inl.sum() < min_pts:
            break
        sel = idx[inl]
        motion = float(np.median(np.linalg.norm(p1[sel] - p2[sel], axis=1)))
        layers.append((H, int(inl.sum()), motion))
        idx = idx[~inl]
    layers.sort(key=lambda L: -L[2])
    return layers


class LayeredBackend(Backend):
    name = "layered"
    handles = ("ROTATION", "PARALLAX")

    def __init__(self, k_max=3, anchor_stride=4):
        self.k_max = k_max
        self.anchor_stride = anchor_stride

    def _pair_layers(self, tracker, fsrc, fdst):
        (k1, d1) = tracker.features(fsrc)
        (k2, d2) = tracker.features(fdst)
        p1, p2 = tracker.match(k1, d1, k2, d2)
        if len(p1) < 25:
            return []
        return fit_layers(p1, p2, self.k_max)

    def _chains(self, frames, tracker):
        n = len(frames)
        chains = []
        base = self._pair_layers(tracker, frames[min(1, n - 1)], frames[0])
        n_layers = max(1, len(base))
        for _ in range(n_layers):
            chains.append([np.eye(3)] + [None] * (n - 1))
        for j in range(1, n):
            L = self._pair_layers(tracker, frames[j], frames[j - 1])
            for l in range(n_layers):
                Hl = L[l][0] if l < len(L) else (L[0][0] if L else None)
                prev = next((chains[l][t] for t in range(j - 1, -1, -1)
                             if chains[l][t] is not None), np.eye(3))
                chains[l][j] = prev @ Hl if Hl is not None else None
        return chains, n_layers

    def warps(self, frames, tracker=None):
        tracker = tracker or wc.Tracker()
        chains, n_layers = self._chains(frames, tracker)

        def f(i, j):
            out = []
            for l in range(n_layers - 1, -1, -1):
                Hi, Hj = chains[l][i], chains[l][j]
                if Hi is None or Hj is None:
                    continue
                out.append(np.linalg.inv(Hi) @ Hj)
            return out
        return f

    def propagate(self, frames, wing_w, tracker=None):
        tracker = tracker or wc.Tracker()
        n = len(frames)
        h, w = frames[0].shape[:2]
        cw = w + 2 * wing_w
        shift = np.array([[1, 0, wing_w], [0, 1, 0], [0, 0, 1]], np.float64)

        # per-layer chains, each anchored to frame 0
        chains = []          # chains[l][j] : frame j -> frame 0 for layer l
        base = self._pair_layers(tracker, frames[min(1, n - 1)], frames[0])
        n_layers = max(1, len(base))
        for _ in range(n_layers):
            chains.append([np.eye(3)] + [None] * (n - 1))

        for j in range(1, n):
            L = self._pair_layers(tracker, frames[j], frames[j - 1])
            for l in range(n_layers):
                Hl = L[l][0] if l < len(L) else (L[0][0] if L else None)
                prev = next((chains[l][t] for t in range(j - 1, -1, -1)
                             if chains[l][t] is not None), np.eye(3))
                chains[l][j] = prev @ Hl if Hl is not None else None

        out = []
        for i in range(n):
            canvas = np.zeros((h, cw, 3), np.uint8)
            filled = np.zeros((h, cw), bool)
            tmap = np.zeros((h, cw), np.int32)
            canvas[:, wing_w:wing_w + w] = frames[i]
            filled[:, wing_w:wing_w + w] = True

            for j in sorted(range(n), key=lambda j: abs(j - i)):
                if j == i or filled.mean() > 0.995:
                    continue
                # back-to-front: later layers are further away, drawn first,
                # so nearer layers overwrite them where they overlap
                for l in range(n_layers - 1, -1, -1):
                    Hi, Hj = chains[l][i], chains[l][j]
                    if Hi is None or Hj is None:
                        continue
                    M = shift @ np.linalg.inv(Hi) @ Hj
                    warped = cv2.warpPerspective(frames[j], M, (cw, h))
                    wm = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M,
                                             (cw, h), flags=cv2.INTER_NEAREST) > 128
                    new = wm & ~filled
                    if not new.any():
                        continue
                    canvas[new] = warped[new]
                    tmap[new] = abs(j - i)
                    filled |= new
            out.append((canvas, filled, tmap))
        return out


# --------------------------------------------------------------- gaussian

class GaussianBackend(Backend):
    """
    3D Gaussian Splatting. Reconstruct the shot's static scene, then render each
    frame's camera with widened intrinsics -- the wings fall out of the render,
    and the rasteriser's alpha IS the coverage mask, with no separate bookkeeping.

    Pipeline:
      1. mask dynamics (video segmentation) -> static-only frames
      2. poses: SfM on static regions, or a MASt3R-class dense matcher when
         baselines are wide and SfM degenerates
      3. fit splats on the static background
      4. per frame, widen fx/cx for the target wing ratio and rasterise
      5. alpha -> filled ; per-splat first-observation frame -> tmap

    Requires CUDA. Raises rather than falling back: a silent downgrade to mosaic
    would label homography pixels as 3D-recovered and corrupt the metric that is
    the whole point of this tool.
    """
    name = "gaussian"
    handles = ("ROTATION", "PARALLAX")
    native_holdout = True     # gating.leave_one_out_3d, not a homography

    def __init__(self, iters=3000, mask_dynamics=True, colmap_dir=None,
                 allow_untrusted_poses=False, alpha_thresh=0.5, device="cuda",
                 poses=None):
        self.iters = iters
        self.mask_dynamics = mask_dynamics
        self.colmap_dir = colmap_dir
        # Explicit poses, when the caller already has them: a synthetic scene
        # with known truth, or a reconstruction solved elsewhere. Without this
        # the backend can only re-derive poses from the frames, which for a
        # ground-truth test means throwing the ground truth away.
        self.poses = poses
        self.allow_untrusted_poses = allow_untrusted_poses
        self.alpha_thresh = alpha_thresh
        self.device = device
        self.last_poses = None          # for callers that want to report on it

    @staticmethod
    def available():
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def poses_for(self, frames, tracker=None):
        """
        COLMAP if a reconstruction was supplied, essential-matrix chain if not.

        The fallback is allowed to RUN because a smoke test is useful; it is not
        allowed to produce pixels unless you pass allow_untrusted_poses=True and
        accept what that means. See splat.PoseSet.trustworthy.
        """
        import splat as sp
        if self.poses is not None:
            return self.poses
        if self.colmap_dir is not None:
            return sp.poses_from_colmap(self.colmap_dir)
        return sp.poses_from_essential(frames, tracker=tracker)

    def propagate(self, frames, wing_w, tracker=None):
        import splat as sp

        if not self.available():
            raise RuntimeError(
                "GaussianBackend requires CUDA. No GPU detected. "
                "Use LayeredBackend for a CPU parallax approximation, or run "
                "this backend on a GPU host -- see remote.py."
            )

        poses = self.poses_for(frames, tracker)
        self.last_poses = poses
        if not poses.trustworthy and not self.allow_untrusted_poses:
            raise RuntimeError(
                f"refusing to render from {poses.source} poses "
                f"(inlier_ratio={poses.inlier_ratio:.2f}). These translations "
                f"have no recovered scale, so the reconstruction can be "
                f"internally consistent and globally wrong -- and every pixel it "
                f"produces would be labelled RETRIEVED. Run COLMAP and pass "
                f"colmap_dir=, or pass allow_untrusted_poses=True if you are "
                f"looking at the output rather than measuring it."
            )

        masks = sp.dynamic_mask(frames, tracker) if self.mask_dynamics else None
        model = sp.fit_splats(frames, poses, masks, iters=self.iters,
                              device=self.device, tracker=tracker)
        return sp.render_widened(model, poses, frames, wing_w,
                                 alpha_thresh=self.alpha_thresh,
                                 device=self.device)

    def warps(self, frames, tracker=None):
        """
        Not available in 3D, and that is not an oversight.

        `gating.leave_one_out` validates geometry by warping a held-out frame in
        and scoring it, which presumes a 2D warp exists between any two frames.
        For a 3D reconstruction it does not -- the mapping runs through depth,
        and collapsing it to a homography would validate a different model from
        the one producing the pixels. A 3D-native hold-out (re-render a held-out
        pose from splats fitted without it) is the right check and belongs in
        gating.py, not here.
        """
        raise NotImplementedError(
            "GaussianBackend has no 2D warp. Score it by re-rendering a held-out "
            "pose from splats fitted without that frame; see the docstring.")


# --------------------------------------------------------------- dispatch

def pick(motion, prefer_3d=False):
    """Choose the best backend that accepts this motion class."""
    if motion == "LOCKED":
        return None                       # nothing recoverable, by definition
    if prefer_3d and GaussianBackend.available():
        return GaussianBackend()
    if motion == "PARALLAX":
        return LayeredBackend()
    return MosaicBackend()
