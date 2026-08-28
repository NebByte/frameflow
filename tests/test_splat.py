"""
test_splat -- everything in the 3D path that does not need a GPU.

Run: python test_splat.py

The point of splitting `splat.py` out of `backends.py` was that most of the 3D
path is arithmetic. This is the file that cashes that in. It builds a synthetic
scene with a KNOWN depth and known camera motion, so triangulation can be
checked against truth rather than against plausibility.

What is NOT covered here, and must be run on a GPU host before anyone believes a
number: `fit_splats` and `render_widened`. See remote.py.
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import tempfile
from pathlib import Path

import cv2
import numpy as np

from frameflow import splat as sp
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ scene

def planar_scene(n_frames=6, w=320, h=180, Z=4.0, step=0.06, seed=0):
    """
    A textured plane at known depth Z, viewed by a camera translating sideways.

    Pure translation past a plane, so the exact inter-view homography is known:
        H = K (R - t n^T / d) K^-1,  R = I, n = [0,0,1], d = Z
    which gives frames whose true 3D structure is a plane at depth Z. Anything
    that triangulates these frames should recover Z.
    """
    rng = np.random.default_rng(seed)
    K = sp.default_K(w, h)
    big = rng.integers(0, 255, (h * 3, w * 3, 3), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 1.2)

    frames, viewmats = [], []
    for i in range(n_frames):
        t = np.array([step * i, 0.0, 0.0])
        # world->camera for a camera that has MOVED by +t is a translation of -t
        V = np.eye(4)
        V[:3, 3] = -t
        viewmats.append(V)

        Hn = np.eye(3)
        Hn[0, 2] = -t[0] / Z            # (R - t n^T / d) with R=I, n=[0,0,1]
        H = K @ Hn @ np.linalg.inv(K)
        # sample from the centre of the oversized texture
        off = np.array([[1, 0, w], [0, 1, h], [0, 0, 1]], np.float64)
        frames.append(cv2.warpPerspective(big, H @ np.linalg.inv(off), (w, h),
                                          flags=cv2.INTER_LINEAR))

    poses = sp.PoseSet(np.array(viewmats), K, "given", inlier_ratio=1.0,
                       note="synthetic")
    return frames, poses, Z


# ------------------------------------------------------------------ tests

def test_intrinsics():
    print("\nintrinsics")
    w, h, wing = 320, 180, 70
    K = sp.default_K(w, h)
    Kw, cw = sp.widen_intrinsics(K, w, wing)

    check("width grows by 2*wing_w", cw == w + 2 * wing, f"{cw}")
    check("fx unchanged", Kw[0, 0] == K[0, 0])
    check("fy unchanged", Kw[1, 1] == K[1, 1])
    check("cy unchanged", Kw[1, 2] == K[1, 2])
    check("cx shifted by wing_w", Kw[0, 2] - K[0, 2] == wing)
    check("centre_matches_original agrees", sp.centre_matches_original(K, Kw, wing))
    check("and rejects a wrong wing_w", not sp.centre_matches_original(K, Kw, wing - 1))

    # the property that actually matters: a 3D point lands wing_w to the right
    rng = np.random.default_rng(1)
    P = np.column_stack([rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200),
                         rng.uniform(2, 8, 200)])
    uv = (K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    uvw = (Kw @ P.T).T
    uvw = uvw[:, :2] / uvw[:, 2:3]
    dx = uvw[:, 0] - uv[:, 0]
    dy = uvw[:, 1] - uv[:, 1]
    check("projection shifts by exactly wing_w in x",
          np.allclose(dx, wing), f"max dev {np.abs(dx - wing).max():.2e}")
    check("projection does not move in y",
          np.allclose(dy, 0), f"max dev {np.abs(dy).max():.2e}")

    try:
        sp.widen_intrinsics(K, w, -1)
        check("negative wing_w rejected", False)
    except ValueError:
        check("negative wing_w rejected", True)


def test_alpha_to_masks():
    print("\nalpha -> (filled, tmap)")
    h, w, wing = 20, 40, 10
    cw = w + 2 * wing
    alpha = np.zeros((h, cw))
    alpha[:, :wing] = 0.9                 # left wing covered
    alpha[:, wing + w:] = 0.2             # right wing below threshold
    first = np.full((h, cw), 3.0)

    filled, tmap = sp.alpha_to_masks(alpha, first, frame_index=10,
                                     wing_w=wing, w=w, alpha_thresh=0.5)
    check("alpha above threshold -> filled", filled[:, :wing].all())
    check("alpha below threshold -> empty", not filled[:, wing + w:].any())
    check("primary region always filled", filled[:, wing:wing + w].all())
    check("tmap is |frame - first_obs|", (tmap[:, :wing] == 7).all(),
          f"got {tmap[0, 0]}")
    check("primary region has zero staleness", (tmap[:, wing:wing + w] == 0).all())
    check("unfilled region has zero staleness", (tmap[:, wing + w:] == 0).all())

    try:
        sp.alpha_to_masks(alpha, first[:, :-1], 0, wing, w)
        check("mismatched shapes rejected", False)
    except ValueError:
        check("mismatched shapes rejected", True)


def test_paste_primary():
    print("\npaste_primary")
    h, w, wing = 30, 50, 12
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    rendered = rng.integers(0, 255, (h, w + 2 * wing, 3), dtype=np.uint8)
    out = sp.paste_primary(rendered, frame, wing)

    check("centre is byte-identical to the frame",
          np.array_equal(out[:, wing:wing + w], frame))
    check("left wing untouched by the paste",
          np.array_equal(out[:, :wing], rendered[:, :wing]))
    check("right wing untouched by the paste",
          np.array_equal(out[:, wing + w:], rendered[:, wing + w:]))
    check("input not mutated", not np.array_equal(rendered[:, wing:wing + w], frame))

    try:
        sp.paste_primary(rendered[:, :-3], frame, wing)
        check("wrong render size rejected", False)
    except ValueError:
        check("wrong render size rejected", True)


def test_colmap_roundtrip():
    print("\nCOLMAP reader")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "cameras.txt").write_text(
            "# camera list\n1 PINHOLE 320 180 400.0 400.0 160.0 90.0\n")
        # identity, then a 90-degree yaw, then a pure translation
        (d / "images.txt").write_text(
            "# image list\n"
            "1 1 0 0 0 0 0 0 1 f000.png\n"
            "\n"
            "2 0.7071068 0 0.7071068 0 0 0 0 1 f001.png\n"
            "\n"
            "3 1 0 0 0 1.5 -2.0 3.0 1 f002.png\n"
            "\n")
        ps = sp.poses_from_colmap(d)

    check("read all three images", len(ps) == 3, f"{len(ps)}")
    check("fx parsed", ps.K[0, 0] == 400.0)
    check("cx parsed", ps.K[0, 2] == 160.0)
    check("first pose is identity rotation", np.allclose(ps.viewmats[0][:3, :3], np.eye(3)))
    R = ps.viewmats[1][:3, :3]
    check("quaternion decodes to a rotation matrix",
          np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0))
    check("90-degree yaw decoded", np.allclose(R, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-6))
    check("translation parsed", np.allclose(ps.viewmats[2][:3, 3], [1.5, -2.0, 3.0]))
    check("colmap poses are trustworthy", ps.trustworthy)

    with tempfile.TemporaryDirectory() as d:
        try:
            sp.poses_from_colmap(Path(d))
            check("missing files raise", False)
        except FileNotFoundError:
            check("missing files raise", True)


def test_pose_trust():
    print("\npose trust gate")
    V = np.tile(np.eye(4), (3, 1, 1))
    K = sp.default_K(320, 180)
    check("essential chain is never trustworthy",
          not sp.PoseSet(V, K, "essential", inlier_ratio=0.99).trustworthy)
    check("low-inlier colmap is not trustworthy",
          not sp.PoseSet(V, K, "colmap", inlier_ratio=0.2).trustworthy)
    check("good colmap is trustworthy",
          sp.PoseSet(V, K, "colmap", inlier_ratio=0.9).trustworthy)


def test_seed_points():
    print("\nseed_points (triangulation against known depth)")
    frames, poses, Z = planar_scene()
    pts, cols, obs = sp.seed_points(frames, poses)

    check("triangulated a usable number of points", len(pts) > 200, f"{len(pts)}")
    if len(pts) == 0:
        return
    med = float(np.median(pts[:, 2]))
    check("median depth recovers the true plane",
          abs(med - Z) < 0.25 * Z, f"got {med:.3f}, truth {Z}")
    check("all points are in front of the camera", (pts[:, 2] > 0).all(),
          f"min Z {pts[:, 2].min():.3f}")
    check("colours are in 0..1", cols.min() >= 0 and cols.max() <= 1.0)
    check("first_obs is a valid frame index",
          obs.min() >= 0 and obs.max() <= len(frames) - 1)
    check("first_obs spans more than one frame", len(np.unique(obs)) > 1,
          f"{len(np.unique(obs))} distinct")
    check("shapes agree", len(pts) == len(cols) == len(obs))


def test_essential_poses():
    print("\nposes_from_essential")
    frames, truth, Z = planar_scene()
    ps = sp.poses_from_essential(frames)

    check("one pose per frame", len(ps) == len(frames), f"{len(ps)}")
    check("labelled as essential", ps.source == "essential")
    check("declares itself untrustworthy", not ps.trustworthy)
    check("declares scale non-metric", not ps.scale_is_metric)
    R = ps.viewmats[-1][:3, :3]
    check("rotations stay near identity for a pure translation",
          np.allclose(R, np.eye(3), atol=0.15),
          f"max dev {np.abs(R - np.eye(3)).max():.3f}")


def test_dynamic_mask():
    print("\ndynamic_mask")
    frames, poses, Z = planar_scene(n_frames=4)
    moving = [f.copy() for f in frames]
    for i, f in enumerate(moving):
        cv2.rectangle(f, (30 + i * 25, 60), (70 + i * 25, 120), (0, 0, 255), -1)

    static_masks = sp.dynamic_mask(frames)
    moving_masks = sp.dynamic_mask(moving)
    s = float(np.mean([m.mean() for m in static_masks]))
    d = float(np.mean([m.mean() for m in moving_masks]))
    check("static scene flags little as dynamic", s < 0.10, f"{s:.3f}")
    check("moving object raises the dynamic fraction", d > s, f"{d:.3f} vs {s:.3f}")
    check("mask shape matches the frame",
          moving_masks[0].shape == moving[0].shape[:2])


def test_backend_refuses():
    print("\nGaussianBackend refusal path")
    from frameflow import backends as bk
    b = bk.GaussianBackend()
    frames, poses, Z = planar_scene(n_frames=3)
    try:
        b.propagate(frames, 40)
        check("raises without CUDA", False)
    except RuntimeError as e:
        check("raises without CUDA", "CUDA" in str(e), str(e)[:60])
    except NotImplementedError:
        check("raises without CUDA", False, "still the old stub")

    try:
        b.warps(frames)
        check("warps() refuses rather than faking a homography", False)
    except NotImplementedError:
        check("warps() refuses rather than faking a homography", True)

    check("pick() never returns Gaussian without a GPU",
          not isinstance(bk.pick("PARALLAX", prefer_3d=True), bk.GaussianBackend))


if __name__ == "__main__":
    print("splat.py -- CPU-testable surface")
    test_intrinsics()
    test_alpha_to_masks()
    test_paste_primary()
    test_colmap_roundtrip()
    test_pose_trust()
    test_seed_points()
    test_essential_poses()
    test_dynamic_mask()
    test_backend_refuses()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
