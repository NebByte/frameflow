"""
Build synthetic test footage with exact ground truth.

We paint a wide panoramic scene, then fly a narrow 16:9 camera window across it.
The pipeline only ever sees the narrow crops. Because we know the panorama, we
know exactly what SHOULD appear on the side walls -> we can score recovery.

Two clips:
  pan_flat   : pure horizontal pan, single depth plane   (homography is exact)
  dolly_par  : lateral move with two depth layers        (parallax; homography must fail)
"""
import cv2
import numpy as np

rng = np.random.default_rng(7)

PANO_W, PANO_H = 3600, 900
FRAME_W, FRAME_H = 640, 360      # narrow "main screen" the pipeline receives
N_FRAMES = 90


def paint_layer(w, h, n_shapes, palette, seed):
    """Procedural city-ish layer: lots of corners so ORB has something to bite."""
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    # sky gradient
    for y in range(h):
        t = y / h
        img[y, :] = (np.array(palette[0]) * (1 - t) + np.array(palette[1]) * t).astype(np.uint8)
    # fine grain so flat regions still carry texture
    grain = r.normal(0, 6, (h, w, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    for _ in range(n_shapes):
        bw = int(r.integers(40, 160))
        bh = int(r.integers(80, h - 60))
        x = int(r.integers(0, w - bw))
        y = h - bh
        c = tuple(int(v) for v in r.integers(25, 200, 3))
        cv2.rectangle(img, (x, y), (x + bw, h), c, -1)
        cv2.rectangle(img, (x, y), (x + bw, h), (10, 10, 10), 2)
        # windows -> dense corner features
        for wy in range(y + 12, h - 12, 26):
            for wx in range(x + 10, x + bw - 14, 22):
                if r.random() < 0.65:
                    lit = tuple(int(v) for v in r.integers(120, 255, 3))
                    cv2.rectangle(img, (wx, wy), (wx + 11, wy + 14), lit, -1)
    return img


def make_pan_flat(path_dir):
    """Pure pan across one plane. Ground truth = a straight crop of the panorama."""
    pano = paint_layer(PANO_W, PANO_H, 90, [(20, 18, 40), (90, 70, 60)], 1)
    cv2.imwrite(f"{path_dir}/pano_flat.png", pano)

    y0 = (PANO_H - FRAME_H) // 2
    # camera x positions: smooth ease-in/out pan
    t = np.linspace(0, 1, N_FRAMES)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    x_start, x_end = 200, PANO_W - FRAME_W - 200
    xs = (x_start + ease * (x_end - x_start)).astype(int)

    frames = [pano[y0:y0 + FRAME_H, x:x + FRAME_W].copy() for x in xs]
    write(f"{path_dir}/pan_flat.mp4", frames)
    np.save(f"{path_dir}/pan_flat_cam.npy", np.stack([xs, np.full_like(xs, y0)], 1))
    return pano, xs, y0


def make_dolly_parallax(path_dir):
    """Two depth layers sliding at different rates -> genuine parallax."""
    back = paint_layer(PANO_W, PANO_H, 70, [(15, 15, 35), (70, 60, 55)], 2)
    fore = paint_layer(PANO_W, PANO_H, 45, [(0, 0, 0), (0, 0, 0)], 3)
    fmask = (fore.sum(2) > 40).astype(np.uint8)

    y0 = (PANO_H - FRAME_H) // 2
    t = np.linspace(0, 1, N_FRAMES)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    xb = (300 + ease * 1400).astype(int)          # background drifts slowly
    xf = (300 + ease * 2600).astype(int)          # foreground drifts fast

    frames = []
    for i in range(N_FRAMES):
        b = back[y0:y0 + FRAME_H, xb[i]:xb[i] + FRAME_W].copy()
        f = fore[y0:y0 + FRAME_H, xf[i]:xf[i] + FRAME_W]
        m = fmask[y0:y0 + FRAME_H, xf[i]:xf[i] + FRAME_W].astype(bool)
        b[m] = f[m]
        frames.append(b)
    write(f"{path_dir}/dolly_parallax.mp4", frames)


def make_locked_off(path_dir):
    """Tripod, no motion at all. Coverage must report ~zero."""
    pano = paint_layer(PANO_W, PANO_H, 90, [(20, 18, 40), (90, 70, 60)], 4)
    y0 = (PANO_H - FRAME_H) // 2
    x = 1200
    base = pano[y0:y0 + FRAME_H, x:x + FRAME_W]
    frames = []
    for _ in range(N_FRAMES):
        n = rng.normal(0, 2.0, base.shape)
        frames.append(np.clip(base.astype(np.float64) + n, 0, 255).astype(np.uint8))
    write(f"{path_dir}/locked_off.mp4", frames)


def write(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    print("wrote", path, len(frames), "frames")


if __name__ == "__main__":
    import argparse
    import os
    # `media/` beside this file, not an absolute path from whichever machine
    # this was first written on -- it was "/home/claude/test", which on Windows
    # silently created C:\home\claude\test and reported success.
    ap = argparse.ArgumentParser(description="synthetic clips with known truth")
    ap.add_argument("-o", "--outdir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "media"))
    d = ap.parse_args().outdir
    os.makedirs(d, exist_ok=True)
    make_pan_flat(d)
    make_dolly_parallax(d)
    make_locked_off(d)
