"""
verify_gpu -- prove the CUDA half of Tier 2 actually works.

Run on a GPU host:  python tools/verify_gpu.py

`test_splat.py` and `test_tier2.py` cover everything that runs on CPU. Two
functions cannot be covered there -- `fit_splats` and `render_widened` -- and
they are the two that produce pixels. This builds a synthetic scene whose 3D
structure is KNOWN, runs the real CUDA path over it, and checks the results
against that truth rather than against plausibility.

WHY A SYNTHETIC SCENE AND NOT A CLIP
------------------------------------
On real footage there is no ground truth for the wings -- that is the entire
problem this toolkit exists for. Here the scene is constructed, so its own
renderer can be asked for frame i's WIDENED view directly: the correct contents
of the wings, at the exact intrinsics the backend is supposed to produce. A
failure is unambiguous -- either the reconstruction put the geometry where it
belongs or it did not.

WHAT EACH CHECK CATCHES
-----------------------
  centre alignment   a widening bug -- the fence would reject every frame
  wing coverage      a fit that collapsed, or an alpha threshold that is wrong
  wing accuracy      splats in the wrong PLACE, which coverage alone will not show
  held-out render    memorisation: a model that only reproduces its own inputs
  tmap sanity        first_obs riding along the wrong colour channel
  provenance         a render that somehow moved the protected primary region
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import sys

import cv2
import numpy as np

from frameflow import gating as g
from frameflow import splat as sp
PASS, FAIL, SKIP = [], [], []
RESULTS = []
RESULT_FILE = "verify_gpu_result.json"


def write_result(iters, path=RESULT_FILE):
    """
    Leave evidence on disk, because stdout does not survive a Colab runtime.

    This run is the only thing standing between "Tier 2 is written" and "Tier 2
    works", and it happens on a machine that is not the one holding the repo.
    Printing the numbers and nothing else means the next person -- or the same
    person a week later -- cannot tell which of those two states they are in.
    Download this file next to the notebook and the answer travels with it.
    """
    import datetime
    import json
    import platform

    env = dict(python=platform.python_version(), platform=platform.platform())
    try:
        import torch
        env.update(torch=torch.__version__, cuda=bool(torch.cuda.is_available()),
                   device=(torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None))
    except ImportError:
        env["torch"] = None
    try:
        import gsplat
        env["gsplat"] = getattr(gsplat, "__version__", "unknown")
    except ImportError:
        env["gsplat"] = None

    payload = dict(
        when=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        iters=iters, environment=env,
        passed=len(PASS), failed=len(FAIL), skipped=len(SKIP),
        ok=not FAIL and bool(PASS), checks=RESULTS,
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {path} -- keep it, it is the proof this ran")
    return payload


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    RESULTS.append(dict(name=name, passed=bool(cond), detail=str(detail)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return cond


# ------------------------------------------------------------------ scene

def layered_scene(n_frames=16, w=192, h=108, seed=0):
    """
    Two textured planes at different depths, photographed by a tracking camera.

    -> (frames, poses, render) where `render(i, K, width)` re-renders view i's
       ground truth at ANY intrinsics -- which is what makes the wings scorable:
       ask it for the widened view and you have the correct answer.

    TEXTURE CHOICE IS NOT COSMETIC. An earlier version of this scene painted
    coloured discs from a point cloud. It looked right and had real parallax,
    and ORB matched exactly ZERO features across frames -- repeated similar
    blobs fail the ratio test -- so `seed_points` triangulated nothing and
    `fit_splats` would have raised on the GPU for a reason with nothing to do
    with the code being tested. Band-limited noise is distinctive per-pixel and
    matches densely, which is what the seeder needs.

    Two planes rather than one because one plane IS a homography, and a
    homography is what this backend exists to beat. Measured on this scene: the
    far plane's centre pixel travels 92 px across the sweep and the near plane's
    travels 230 px, and the best single homography fitted between two frames
    leaves a mean residual of 40 grey levels across the frame. That is real
    multi-depth structure, which is what makes fitting it a meaningful test.

    One caveat, so nobody reads too much into it: `wc.classify_motion` labels
    this scene ROTATION, not PARALLAX. ORB only matches densely on a narrow band
    where both planes stay visible, so the two-layer test never sees a second
    homography. That is a fact about feature matching on synthetic noise, not
    about the scene -- but it does mean this scene is NOT a test of the motion
    classifier, only of the reconstruction.
    """
    rng = np.random.default_rng(seed)
    K = sp.default_K(w, h)
    pad = 3

    def texture(scale):
        t = rng.integers(0, 255, (h * pad, w * pad, 3), dtype=np.uint8)
        return cv2.GaussianBlur(t, (0, 0), scale)

    far, near = texture(1.1), texture(1.4)
    # the near plane covers a vertical slab, so it occludes without hiding all
    near_alpha = np.zeros((h * pad, w * pad), np.uint8)
    x0 = int(w * pad * 0.42)
    near_alpha[:, x0:x0 + int(w * pad * 0.20)] = 255

    Z_far, Z_near = 8.0, 3.2
    off = np.array([[1, 0, w], [0, 1, h], [0, 0, 1]], np.float64)

    def render(i, Kr=None, width=None):
        Kr = K if Kr is None else np.asarray(Kr, np.float64)
        width = w if width is None else int(width)
        tx = -1.6 + 3.2 * (i / max(n_frames - 1, 1))
        out = np.zeros((h, width, 3), np.uint8)
        for tex, Z, alpha in ((far, Z_far, None), (near, Z_near, near_alpha)):
            Hn = np.eye(3)
            Hn[0, 2] = -tx / Z              # (R - t n^T / d), R = I, n = [0,0,1]
            M = Kr @ Hn @ np.linalg.inv(Kr) @ Kr @ np.linalg.inv(K) @ np.linalg.inv(off)
            warped = cv2.warpPerspective(tex, M, (width, h))
            if alpha is None:
                out = warped
            else:
                a = cv2.warpPerspective(alpha, M, (width, h),
                                        flags=cv2.INTER_NEAREST) > 128
                out[a] = warped[a]
        return out

    frames = [render(i) for i in range(n_frames)]
    viewmats = []
    for i in range(n_frames):
        tx = -1.6 + 3.2 * (i / max(n_frames - 1, 1))
        V = np.eye(4)
        V[:3, 3] = [-tx, 0.0, 0.0]
        viewmats.append(V)
    poses = sp.PoseSet(np.array(viewmats), K, "colmap", inlier_ratio=1.0,
                       note="synthetic ground truth")
    return frames, poses, render


def wing_truth(render, poses, i, wing_w, w):
    """Frame i's WIDENED view, rendered from the scene's own ground truth."""
    K_wide, cw = sp.widen_intrinsics(poses.Ks[i], w, wing_w)
    return render(i, K_wide, cw)


# ------------------------------------------------------------------ checks

def main(iters=2000, n_frames=16, wing_w=None):
    print("verify_gpu -- the CUDA half of Tier 2\n")

    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "none"
    except ImportError:
        cuda, dev = False, "torch not installed"
    print(f"  CUDA: {cuda}  ({dev})")
    if not cuda:
        print("\n  No GPU here. This script must run on the GPU host.\n"
              "  Everything it checks is untested until it does.\n"
              "    python remote.py bootstrap <host>\n")
        return 2
    try:
        import gsplat  # noqa: F401
    except ImportError:
        print("\n  gsplat is not installed: pip install gsplat\n")
        return 2

    frames, poses, render = layered_scene(n_frames=n_frames)
    h, w = frames[0].shape[:2]
    ww = wing_w if wing_w is not None else int(w * 0.22)
    print(f"  scene: {n_frames} frames {w}x{h}, two textured planes, "
          f"wing {ww}px, {iters} iters\n")

    # -- 1. intrinsics, before spending GPU minutes
    print("intrinsics")
    K_wide, cw = sp.widen_intrinsics(poses.K, w, ww)
    check("centre stays aligned under widening",
          sp.centre_matches_original(poses.K, K_wide, ww))

    # -- 2. the fit
    print("\nfit_splats")
    masks = None
    model = sp.fit_splats(frames, poses, masks, iters=iters, verbose=True)
    check("produced splats", len(model) > 500, f"{len(model)}")
    check("first_obs spans the shot",
          float(model.first_obs.max()) > float(model.first_obs.min()),
          f"{float(model.first_obs.min()):.0f}..{float(model.first_obs.max()):.0f}")
    check("opacities are probabilities",
          0.0 <= float(model.opacities.min()) and float(model.opacities.max()) <= 1.0)
    check("no NaNs in the means", bool(np.isfinite(model.means.cpu().numpy()).all()))

    # -- 3. the widened render
    print("\nrender_widened")
    out = sp.render_widened(model, poses, frames, ww)
    check("one result per frame", len(out) == len(frames), f"{len(out)}")
    canvas, filled, tmap = out[len(out) // 2]
    check("canvas is the widened size", canvas.shape == (h, w + 2 * ww, 3),
          f"{canvas.shape}")
    check("filled is bool, tmap is int32",
          filled.dtype == bool and tmap.dtype == np.int32)

    mid = len(out) // 2
    left = filled[:, :ww].mean()
    right = filled[:, ww + w:].mean()
    check("left wing has coverage", left > 0.30, f"{left * 100:.1f}%")
    check("right wing has coverage", right > 0.30, f"{right * 100:.1f}%")

    # -- 4. the primary region is untouched, byte for byte
    print("\nfence")
    centre = canvas[:, ww:ww + w]
    check("primary region is the original frame, byte for byte",
          np.array_equal(centre, frames[mid]))
    check("primary region is marked filled", filled[:, ww:ww + w].all())
    check("primary region has zero staleness", (tmap[:, ww:ww + w] == 0).all())

    # -- 5. are the wings in the RIGHT PLACE? coverage alone cannot say.
    print("\nwing accuracy against truth")
    truth = wing_truth(render, poses, mid, ww, w)
    wing_mask = np.zeros(filled.shape, bool)
    wing_mask[:, :ww] = True
    wing_mask[:, ww + w:] = True
    valid = wing_mask & filled
    if valid.sum() > 500:
        mse = float(np.mean((canvas[valid].astype(np.float64)
                             - truth[valid].astype(np.float64)) ** 2))
        db = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
        check("recovered wings match the truth render", db > 15.0, f"{db:.1f} dB")
    else:
        SKIP.append("wing accuracy")
        print("  skip  wing accuracy -- too little coverage to score")

    # -- 6. staleness is plausible
    print("\nstaleness")
    wt = tmap[valid] if valid.any() else np.array([0])
    check("staleness is bounded by the shot length",
          int(wt.max()) <= len(frames), f"max {int(wt.max())} of {len(frames)}")
    check("wings are staler than the centre", float(wt.mean()) > 0.0,
          f"mean {float(wt.mean()):.1f} frames")

    # -- 7. the hold-out: does it generalise, or has it memorised?
    print("\nleave_one_out_3d (the gate's own check)")
    from frameflow import backends as bk
    backend = bk.GaussianBackend(iters=max(400, iters // 4))
    geom, scores = g.leave_one_out_3d(frames, backend, probes=2)
    check("held-out frames score above the gate's threshold", geom >= 20.0,
          f"{geom:.1f} dB from {len(scores)} probe(s)")
    m = dict(effective_coverage=0.62, mean_detail=0.5, stale_seconds=0.2,
             coverage=0.8)
    state = g.decide(m, geom)[0]
    check("a good 3D shot is no longer gated OFF", state != "OFF", state)

    # -- 8. the whole backend, end to end
    print("\nGaussianBackend.propagate")
    res = backend.propagate(frames, ww)
    check("propagate returns the Backend contract", len(res) == len(frames))
    cov = float(np.mean([r[1].mean() for r in res]))
    check("mean coverage is non-trivial", cov > 0.5, f"{cov * 100:.1f}%")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        return 1
    print("\nThe CUDA path works on known truth. Real footage is the next "
          "question, and a different one.")
    return 0


if __name__ == "__main__":
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    try:
        code = main(iters=it)
    finally:
        # a crashed run is exactly when the partial numbers are worth keeping;
        # writing only on success lost every check the first time this fell
        # over inside propagate()
        write_result(it)
    raise SystemExit(code)
