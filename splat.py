"""
splat -- the 3D path: poses, splats, and a widened-frustum render.

WHY THIS MODULE EXISTS SEPARATELY FROM backends.py
--------------------------------------------------
`GaussianBackend` is one class with one job, but most of what it does is
arithmetic that does not need a GPU: choosing intrinsics, converting a
rasteriser's alpha into the (filled, tmap) pair the rest of the toolkit expects,
deciding whether a set of poses is trustworthy enough to use at all. Only the
fit and the rasterise need CUDA.

Keeping them apart means the parts that can be tested on any machine ARE tested
on any machine. Everything above `fit_splats` in this file runs on CPU and is
covered by `test_splat.py`; only `fit_splats` and `render_widened` need a GPU.

THE WIDENING, WHICH IS THE WHOLE TRICK
--------------------------------------
To recover wings, render the same camera with a wider sensor. Hold fx and fy
fixed and shift the principal point:

    fx' = fx        cx' = cx + wing_w        width' = width + 2*wing_w

Because fx is unchanged, angular resolution per pixel is unchanged, so the
centre columns [wing_w, wing_w+w) of the widened render reproject EXACTLY where
the original frame's pixels are. That is not a nicety -- the fence forbids any
rung from touching the primary region, so a render whose centre drifted by even
a pixel would be unusable. `centre_matches_original()` checks the property
directly and is the first thing to run when a render looks wrong.

Scaling fx instead (a "zoom out") would also widen the field of view, and would
be wrong: it shrinks the primary region and resamples pixels the fence protects.

WHAT THIS CANNOT DO
-------------------
Poses. Monocular pose estimation from an essential matrix recovers translation
only up to scale, and chaining pairs accumulates scale drift that no amount of
splat fitting repairs. COLMAP is the correct tool and `poses_from_colmap` is the
supported path. `poses_from_essential` exists as a fallback that reports its own
untrustworthiness rather than pretending -- see `PoseSet.trustworthy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# A focal prior, not a measurement. Roughly a 28mm-equivalent lens, defensible
# for feature photography and wrong for anything shot long.
FOCAL_PRIOR = 1.2


# --------------------------------------------------------------- intrinsics

def default_K(w: int, h: int, focal_mult: float = FOCAL_PRIOR) -> np.ndarray:
    """A pinhole K from image size alone. A prior; override when you know."""
    f = focal_mult * max(w, h)
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], np.float64)


def widen_intrinsics(K, w: int, wing_w: int):
    """
    -> (K_wide, width_wide). fx/fy untouched, cx shifted by exactly wing_w.

    See the module docstring: preserving fx is what keeps the centre of the
    widened render pixel-aligned with the original frame.
    """
    if wing_w < 0:
        raise ValueError("wing_w must be non-negative")
    Kw = np.asarray(K, np.float64).copy()
    Kw[0, 2] = Kw[0, 2] + wing_w
    return Kw, w + 2 * wing_w


def centre_matches_original(K, K_wide, wing_w: int, tol: float = 1e-9) -> bool:
    """
    Does a point at pixel (u, v) in the original land at (u + wing_w, v) in the
    widened render? If this is False the fence rejects every frame, so it is
    worth asserting before spending GPU minutes.
    """
    K, K_wide = np.asarray(K, np.float64), np.asarray(K_wide, np.float64)
    if abs(K_wide[0, 0] - K[0, 0]) > tol or abs(K_wide[1, 1] - K[1, 1]) > tol:
        return False
    if abs(K_wide[1, 2] - K[1, 2]) > tol:
        return False
    return abs((K_wide[0, 2] - K[0, 2]) - wing_w) <= tol


# --------------------------------------------------------------- poses

@dataclass
class PoseSet:
    """
    World-to-camera 4x4 matrices, one per view, plus an honest account of where
    they came from and whether they can be believed.

    Intrinsics are PER VIEW. A single K would be right for one shot and wrong
    for the case this whole tier exists to serve: bridging two setups of one
    location, which by definition were shot on different lenses. `Ks` accepts a
    single [3,3] and broadcasts it, so single-camera callers are unaffected.
    """
    viewmats: np.ndarray                 # [N, 4, 4] float64, world-to-camera
    Ks: np.ndarray                       # [N, 3, 3] float64
    source: str                          # "colmap" | "essential" | "given"
    inlier_ratio: float = 0.0
    scale_is_metric: bool = False
    note: str = ""

    def __post_init__(self):
        self.viewmats = np.asarray(self.viewmats, np.float64).reshape(-1, 4, 4)
        Ks = np.asarray(self.Ks, np.float64)
        if Ks.ndim == 2:
            Ks = np.tile(Ks, (len(self.viewmats), 1, 1))
        if len(Ks) != len(self.viewmats):
            raise ValueError(
                f"{len(Ks)} intrinsics for {len(self.viewmats)} views")
        self.Ks = Ks

    def __len__(self):
        return len(self.viewmats)

    @property
    def K(self) -> np.ndarray:
        """The first view's intrinsics. Convenience for single-camera callers."""
        return self.Ks[0]

    def subset(self, idx) -> "PoseSet":
        """Same poses, fewer views. Used by the leave-one-out check."""
        idx = list(idx)
        return PoseSet(self.viewmats[idx], self.Ks[idx], self.source,
                       self.inlier_ratio, self.scale_is_metric, self.note)

    def single_camera(self) -> bool:
        return bool(np.allclose(self.Ks, self.Ks[0]))

    @property
    def trustworthy(self) -> bool:
        """
        Whether these poses may produce RETRIEVED pixels.

        Essential-matrix chains are refused outright. They are useful for a
        smoke test and they are not evidence: unrecovered per-pair scale means
        the reconstruction can be internally consistent and globally wrong, and
        a wing filled from a globally wrong reconstruction is invention wearing
        a photography label. That is the exact failure `provenance.py` exists to
        prevent, so the check lives here and not in a reviewer's judgement.
        """
        return self.source == "colmap" and self.inlier_ratio >= 0.5


def quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    """COLMAP stores wxyz. Returns the world-to-camera rotation."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], np.float64)


def poses_from_colmap(sparse_dir) -> PoseSet:
    """
    Read a COLMAP sparse reconstruction (text format: cameras.txt, images.txt).

    This is the supported path. Run `colmap automatic_reconstructor` on the
    shot's static frames, then point this at `sparse/0`.
    """
    d = Path(sparse_dir)
    cams_f, imgs_f = d / "cameras.txt", d / "images.txt"
    if not cams_f.exists() or not imgs_f.exists():
        raise FileNotFoundError(
            f"expected cameras.txt and images.txt in {d}. For the binary format: "
            f"colmap model_converter --input_path {d} --output_path {d} "
            f"--output_type TXT")

    # ALL cameras, not just the first. Two setups of one location are two
    # lenses, and reading only camera 1 would render the wide setup's frames
    # through the close-up's focal length -- a wrong reconstruction that still
    # looks like a reconstruction.
    cams = {}
    for line in cams_f.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cam_id, model, params = p[0], p[1], [float(x) for x in p[4:]]
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            raise ValueError(f"unhandled COLMAP camera model {model}")
        cams[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    if not cams:
        raise ValueError("no camera found in cameras.txt")

    # Identify image lines STRUCTURALLY rather than by position. The obvious
    # reading -- "every other line is POINTS2D" -- breaks the moment an image
    # has no registered points, because that POINTS2D line is empty and any
    # blank-line filter silently shifts the alternation by one. An image line is
    # exactly 10 tokens (ID qw qx qy qz tx ty tz CAMERA_ID NAME); a POINTS2D
    # line is triples, so its token count is a multiple of 3 and never 10.
    rows = []
    for line in imgs_f.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) != 10 or not p[0].isdigit() or not p[8].isdigit():
            continue
        qw, qx, qy, qz = (float(x) for x in p[1:5])
        tx, ty, tz = (float(x) for x in p[5:8])
        rows.append((p[-1], quat_to_R(qw, qx, qy, qz), np.array([tx, ty, tz]), p[8]))
    if not rows:
        raise ValueError(f"no image rows parsed from {imgs_f}")

    rows.sort(key=lambda r: r[0])                # filename order == view order
    mats = np.tile(np.eye(4), (len(rows), 1, 1))
    Ks = np.zeros((len(rows), 3, 3))
    for i, (_, R, t, cam_id) in enumerate(rows):
        mats[i, :3, :3] = R
        mats[i, :3, 3] = t
        if cam_id not in cams:
            raise ValueError(f"image references camera {cam_id}, not in cameras.txt")
        Ks[i] = cams[cam_id]
    return PoseSet(mats, Ks, "colmap", inlier_ratio=1.0, scale_is_metric=False,
                   note=f"{len(rows)} images, {len(cams)} camera(s)")


def poses_from_essential(frames, K=None, tracker=None) -> PoseSet:
    """
    Chained two-view poses. A SMOKE TEST, not a reconstruction.

    Each pair's translation is a unit direction with no recoverable scale, so
    this walks the camera by an arbitrary fixed step. The rotations are usually
    close to right; the translations are not to be believed. `source` is
    "essential", which makes `trustworthy` False, which stops these poses ever
    reaching the metric.
    """
    import wingcoverage as wc
    tracker = tracker or wc.Tracker()
    h, w = frames[0].shape[:2]
    K = default_K(w, h) if K is None else np.asarray(K, np.float64)

    mats = [np.eye(4)]
    inliers = []
    feats = [tracker.features(f) for f in frames]
    for i in range(1, len(frames)):
        (kp1, d1), (kp2, d2) = feats[i - 1], feats[i]
        p1, p2 = tracker.match(kp1, d1, kp2, d2)
        step = np.eye(4)
        if len(p1) >= 8:
            E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
            if E is not None and E.shape == (3, 3):
                n_in, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
                step[:3, :3] = R
                step[:3, 3] = t.ravel() * 0.1      # arbitrary; see docstring
                inliers.append(n_in / max(len(p1), 1))
        mats.append(step @ mats[-1])

    return PoseSet(np.array(mats), K, "essential",
                   inlier_ratio=float(np.mean(inliers)) if inliers else 0.0,
                   scale_is_metric=False,
                   note="unit-scale translations; rotations only are meaningful")


# --------------------------------------------------------------- dynamics

def dynamic_mask(frames, tracker=None, thresh=18.0):
    """
    Per-frame boolean mask, True where a pixel is NOT static background.

    Fits a homography to the neighbouring frame, warps it in, and calls large
    residuals dynamic. Crude next to a video segmentation model, and enough to
    keep moving actors out of a static reconstruction, which is all this is for.
    Anything left in becomes a smear of Gaussians at the wrong depth.
    """
    import wingcoverage as wc
    tracker = tracker or wc.Tracker()
    n = len(frames)
    out = []
    for i in range(n):
        j = i + 1 if i + 1 < n else i - 1
        if j < 0:
            out.append(np.zeros(frames[i].shape[:2], bool))
            continue
        (kp1, d1), (kp2, d2) = tracker.features(frames[i]), tracker.features(frames[j])
        p1, p2 = tracker.match(kp1, d1, kp2, d2)
        if len(p1) < 8:
            out.append(np.zeros(frames[i].shape[:2], bool))
            continue
        H, _ = cv2.findHomography(p2, p1, cv2.RANSAC, 3.0)
        if H is None:
            out.append(np.zeros(frames[i].shape[:2], bool))
            continue
        h, w = frames[i].shape[:2]
        warped = cv2.warpPerspective(frames[j], H, (w, h))
        diff = cv2.absdiff(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        m = cv2.GaussianBlur(diff, (0, 0), 2.0) > thresh
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8)).astype(bool)
        out.append(m)
    return out


# --------------------------------------------------------------- alpha -> masks

def alpha_to_masks(alphas, first_obs, frame_index: int, wing_w: int, w: int,
                   alpha_thresh: float = 0.5):
    """
    Turn one widened render into the (filled, tmap) pair the toolkit expects.

    `alphas`    [H, W_wide] float, the rasteriser's accumulated opacity
    `first_obs` [H, W_wide] float, alpha-weighted mean of each splat's
                first-observation frame index

    A NOTE ON tmap SEMANTICS, because it differs from the mosaic backend.
    `propagate_wings` fills each pixel from the NEAREST-IN-TIME frame that saw
    it, so its tmap is a true minimum offset. A rasteriser blends every
    contributing Gaussian, so what comes back is an alpha-weighted MEAN
    first-observation. The mean is >= the minimum, so this backend reports
    staleness that is conservative rather than flattering, which is the correct
    direction for a number that gates confidence. It is not the same quantity
    and should not be pooled with mosaic tmaps in a single statistic.
    """
    alphas = np.asarray(alphas, np.float64)
    first_obs = np.asarray(first_obs, np.float64)
    if alphas.shape != first_obs.shape:
        raise ValueError(f"alpha {alphas.shape} and first_obs {first_obs.shape} differ")

    filled = alphas >= alpha_thresh
    tmap = np.zeros(alphas.shape, np.int32)
    tmap[filled] = np.abs(frame_index - first_obs[filled]).astype(np.int32)

    # the primary region is the frame itself: always filled, always offset 0.
    filled[:, wing_w:wing_w + w] = True
    tmap[:, wing_w:wing_w + w] = 0
    return filled, tmap


def paste_primary(rendered, frame, wing_w: int):
    """
    Put the ORIGINAL frame back into the centre of a widened render.

    The rasterised centre is a reconstruction of the frame, not the frame. Close
    is not good enough: `PRIMARY` means these exact photons, and the fence
    compares the protected region byte-for-byte. So the render supplies wings
    only, and the centre is the untouched original.
    """
    h, w = frame.shape[:2]
    out = np.asarray(rendered).copy()
    if out.shape[0] != h or out.shape[1] != w + 2 * wing_w:
        raise ValueError(
            f"render is {out.shape[:2]}, expected {(h, w + 2 * wing_w)} for a "
            f"{w}px frame with {wing_w}px wings")
    out[:, wing_w:wing_w + w] = frame
    return out


# --------------------------------------------------------------- splat model

@dataclass
class SplatModel:
    """
    A fitted set of 3D Gaussians plus, per splat, the frame that first saw it.

    `first_obs` is what makes the staleness map free. It is not bookkeeping
    bolted on afterwards -- each splat is triangulated from a feature track, and
    the earliest frame in that track is a real observation time.
    """
    means: object                 # torch [N, 3]
    quats: object                 # torch [N, 4] wxyz
    scales: object                # torch [N, 3] (log-space during fitting)
    opacities: object             # torch [N]    (logit-space during fitting)
    colors: object                # torch [N, 3] rgb in 0..1
    first_obs: object             # torch [N]    frame index
    K: np.ndarray
    note: str = ""

    def __len__(self):
        return int(self.means.shape[0])


def seed_points(frames, poses: PoseSet, masks=None, tracker=None,
                min_matches=12):
    """
    Triangulate consecutive-pair feature matches into a seed point cloud.

    -> (points [M, 3], colours [M, 3] in 0..1, first_obs [M] frame index)

    Seeding this way rather than randomly is what gives `first_obs` a real
    meaning: a point exists because two specific frames saw it, and the earlier
    of those is a genuine observation time. A random init would need visibility
    bookkeeping bolted on afterwards to recover the same information, less well.
    """
    import wingcoverage as wc
    tracker = tracker or wc.Tracker()
    P = [poses.Ks[i] @ poses.viewmats[i][:3, :4] for i in range(len(poses))]

    pts, cols, obs = [], [], []
    feats = [tracker.features(f) for f in frames]
    for i in range(len(frames) - 1):
        (kp1, d1), (kp2, d2) = feats[i], feats[i + 1]
        p1, p2 = tracker.match(kp1, d1, kp2, d2)
        if len(p1) < min_matches:
            continue
        if masks is not None:
            keep = np.array([not masks[i][int(min(max(y, 0), masks[i].shape[0] - 1)),
                                          int(min(max(x, 0), masks[i].shape[1] - 1))]
                             for x, y in p1])
            p1, p2 = p1[keep], p2[keep]
            if len(p1) < min_matches:
                continue

        X = cv2.triangulatePoints(P[i], P[i + 1], p1.T, p2.T)
        w_ = X[3]
        ok = np.abs(w_) > 1e-8
        X = (X[:3, ok] / w_[ok]).T
        if len(X) == 0:
            continue

        # cheirality: keep points in front of BOTH cameras. Triangulation
        # happily returns points behind the camera and they fit as well as any
        # other until you render from a new viewpoint and the scene turns inside
        # out.
        for idx, cam in ((i, poses.viewmats[i]), (i + 1, poses.viewmats[i + 1])):
            Z = (cam[:3, :3] @ X.T).T + cam[:3, 3]
            front = Z[:, 2] > 1e-6
            X = X[front]
            p1 = p1[ok][front] if idx == i else p1
            if len(X) == 0:
                break
        if len(X) == 0:
            continue

        h, w = frames[i].shape[:2]
        uv = np.clip(p1[:len(X)].astype(int), [0, 0], [w - 1, h - 1])
        c = frames[i][uv[:, 1], uv[:, 0]][:, ::-1] / 255.0    # BGR -> RGB
        pts.append(X)
        cols.append(c)
        obs.append(np.full(len(X), i, np.float64))

    if not pts:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,)))
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(obs)


# --------------------------------------------------------------- GPU: fit

def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError("splat fitting needs PyTorch; pip install torch") from e


def _gsplat():
    try:
        from gsplat import rasterization
        return rasterization
    except ImportError as e:
        raise RuntimeError(
            "splat fitting needs gsplat; pip install gsplat "
            "(needs a CUDA toolchain -- see remote.py for the GPU-host path)"
        ) from e


def fit_splats(frames, poses: PoseSet, masks=None, iters: int = 3000,
               lr: float = 0.01, device: str = "cuda", verbose: bool = True,
               tracker=None) -> SplatModel:
    """
    Fit Gaussians to the STATIC background of one shot. Needs CUDA.

    Deliberately does NOT densify. Standard 3DGS splits and clones Gaussians on
    a gradient criterion, which sharpens the fit and destroys the provenance of
    `first_obs`: a split child has no observation time of its own, and inventing
    one for it would put a made-up staleness value behind a real-looking number.
    Fixed splat count costs sharpness, which is recoverable, and keeps the
    observation times exact, which is not. Densify later only if you also carry
    first_obs through the split.
    """
    torch = _torch()
    rasterization = _gsplat()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("fit_splats: device='cuda' but no CUDA device is visible")

    pts, cols, obs = seed_points(frames, poses, masks, tracker)
    if len(pts) < 64:
        raise RuntimeError(
            f"only {len(pts)} points triangulated; too few to fit. Usually means "
            f"the poses are wrong or the shot has no parallax at all")

    h, w = frames[0].shape[:2]
    dev = torch.device(device)
    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=dev)

    # scale prior: a fraction of the mean nearest-neighbour distance, so splats
    # start roughly filling the space between their neighbours rather than at an
    # arbitrary absolute size that depends on the reconstruction's unknown scale.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :64, :], axis=-1) if len(pts) > 64 \
        else np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d[:, :d.shape[1]], np.inf)
    nn = np.median(np.min(d, axis=1))
    nn = float(nn) if np.isfinite(nn) and nn > 0 else 0.01

    means = t(pts).requires_grad_(True)
    scales = t(np.log(np.full((len(pts), 3), nn * 0.5))).requires_grad_(True)
    quats = t(np.tile([1.0, 0.0, 0.0, 0.0], (len(pts), 1))).requires_grad_(True)
    opac = t(np.full(len(pts), 2.0)).requires_grad_(True)          # logit(~0.88)
    colors = t(np.clip(cols, 1e-4, 1 - 1e-4)).requires_grad_(True)
    first_obs = t(obs)

    viewmats = t(poses.viewmats)
    Ks_all = t(poses.Ks)                       # [N, 3, 3], one per view
    gt = t(np.stack([f[:, :, ::-1] / 255.0 for f in frames]))       # BGR -> RGB
    if masks is not None:
        keep = t(np.stack([~m for m in masks]).astype(np.float32)).unsqueeze(-1)
    else:
        keep = None

    opt = torch.optim.Adam([
        {"params": [means], "lr": lr * 0.1},
        {"params": [scales], "lr": lr * 0.5},
        {"params": [quats], "lr": lr * 0.1},
        {"params": [opac], "lr": lr * 5.0},
        {"params": [colors], "lr": lr * 2.5},
    ])

    n = len(frames)
    for it in range(iters):
        i = it % n
        rgb, _alpha, _meta = rasterization(
            means=means,
            quats=torch.nn.functional.normalize(quats, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opac),
            colors=torch.sigmoid(colors),
            viewmats=viewmats[i:i + 1],
            Ks=Ks_all[i:i + 1], width=w, height=h,
            render_mode="RGB", packed=True,
        )
        pred = rgb[0]
        target = gt[i]
        if keep is not None:
            pred, target = pred * keep[i], target * keep[i]
        loss = torch.abs(pred - target).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"    splat fit {it + 1}/{iters}  L1 {loss.item():.4f}")

    return SplatModel(
        means=means.detach(),
        quats=torch.nn.functional.normalize(quats, dim=-1).detach(),
        scales=torch.exp(scales).detach(),
        opacities=torch.sigmoid(opac).detach(),
        colors=torch.sigmoid(colors).detach(),
        first_obs=first_obs,
        K=poses.K,          # representative only; render uses the pose's own

        note=f"{len(pts)} splats, {iters} iters, no densification",
    )


# --------------------------------------------------------------- GPU: render

def render_widened(model: SplatModel, poses: PoseSet, frames, wing_w: int,
                   alpha_thresh: float = 0.5, device: str = "cuda"):
    """
    Render every frame's camera through a widened frustum.

    -> list of (canvas HxCWx3 uint8, filled HxCW bool, tmap HxCW int32),
       which is exactly `Backend.propagate`'s contract.

    Two channels ride along with RGB so the coverage and staleness maps come
    back from the same rasterisation rather than a second pass:

        render_alphas       -> filled       (the alpha IS the coverage mask)
        colour channel 3    -> first_obs    (alpha-weighted; un-premultiplied
                                             here by dividing through by alpha)
    """
    torch = _torch()
    rasterization = _gsplat()
    h, w = frames[0].shape[:2]
    wide = []
    for i in range(len(poses)):
        Kw, cw = widen_intrinsics(poses.Ks[i], w, wing_w)
        if not centre_matches_original(poses.Ks[i], Kw, wing_w):
            raise RuntimeError(f"widened intrinsics broke centre alignment at view {i}")
        wide.append(Kw)
    wide = np.stack(wide)

    dev = torch.device(device)
    Ks_all = torch.tensor(wide, dtype=torch.float32, device=dev)
    viewmats = torch.tensor(poses.viewmats, dtype=torch.float32, device=dev)

    # [N, 4]: rgb + first-observation frame index as a fourth channel
    fo = model.first_obs.to(dev).reshape(-1, 1).float()
    colors4 = torch.cat([model.colors.to(dev), fo], dim=-1)

    out = []
    for i in range(len(frames)):
        canvas, a, first_obs_map = _render_one(
            rasterization, model, colors4, viewmats[i:i + 1], Ks_all[i:i + 1],
            cw, h, dev)
        canvas = paste_primary(canvas, frames[i], wing_w)
        filled, tmap = alpha_to_masks(a, first_obs_map, i, wing_w, w, alpha_thresh)
        out.append((canvas, filled, tmap))
    return out


def _render_one(rasterization, model, colors4, viewmat, Ks, width, height, dev):
    """One rasterisation -> (bgr uint8, alpha float, first_obs float). No paste."""
    import torch
    with torch.no_grad():
        rgba, alpha, _meta = rasterization(
            means=model.means.to(dev), quats=model.quats.to(dev),
            scales=model.scales.to(dev), opacities=model.opacities.to(dev),
            colors=colors4, viewmats=viewmat, Ks=Ks,
            width=width, height=height, render_mode="RGB", packed=True,
        )
    img = rgba[0, ..., :3].clamp(0, 1).cpu().numpy()
    fo_premult = rgba[0, ..., 3].cpu().numpy()
    a = alpha[0, ..., 0].cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        first_obs_map = np.where(a > 1e-6, fo_premult / np.maximum(a, 1e-6), 0.0)
    return (img[:, :, ::-1] * 255).astype(np.uint8), a, first_obs_map


def render_raw(model: SplatModel, viewmat, width: int, height: int, K=None,
               device: str = "cuda"):
    """
    Render ONE pose with no primary paste and no widening.

    This exists for the leave-one-out check and the distinction is the whole
    point of it. `render_widened` puts the real frame back into the centre,
    because PRIMARY must mean those exact photons. Scoring a reconstruction that
    has had the ground truth pasted into it measures nothing -- it is the same
    mistake `gating`'s docstring records as having once scored 138 dB on a clip
    whose wings were actually 12 dB.

    -> (bgr uint8 [H, W, 3], alpha float [H, W])
    """
    torch = _torch()
    rasterization = _gsplat()
    dev = torch.device(device)
    K = model.K if K is None else K
    Ks = torch.tensor(np.asarray(K, np.float64), dtype=torch.float32,
                      device=dev).unsqueeze(0)
    vm = torch.tensor(np.asarray(viewmat, np.float64).reshape(1, 4, 4),
                      dtype=torch.float32, device=dev)
    fo = model.first_obs.to(dev).reshape(-1, 1).float()
    colors4 = torch.cat([model.colors.to(dev), fo], dim=-1)
    bgr, a, _fo = _render_one(rasterization, model, colors4, vm, Ks,
                              width, height, dev)
    return bgr, a
