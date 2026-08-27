"""
Ground-truth validation.

The synthetic pan was cut from a panorama we still have, and we know the exact
camera x for every frame. So for each frame we know precisely what the side
walls SHOULD contain. This scores recovered pixels against that truth.

Coverage says how much came back. This says whether it came back CORRECT.
"""
import os

import cv2
import numpy as np

import wingcoverage as wc

# `media/` beside this file. These were absolute paths from the machine this was
# first written on, so on any other machine the script reported a missing file
# -- or, on Windows, silently looked in C:\home\claude\test. Run
# `python make_test_clip.py` to produce all three.
MEDIA = os.environ.get(
    "SCREENX_MEDIA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "media"))
PANO = os.path.join(MEDIA, "pano_flat.png")
CAM = os.path.join(MEDIA, "pan_flat_cam.npy")
VIDEO = os.path.join(MEDIA, "pan_flat.mp4")


def main():
    pano = cv2.imread(PANO)
    cam = np.load(CAM)
    frames, fps = wc.read_video(VIDEO)
    h, w = frames[0].shape[:2]
    wing_w = int(w * wc.WING_RATIO)

    tr = wc.Tracker()
    Hs = wc.chain_homographies(tr, frames)
    prop = wc.propagate_wings(frames, Hs, wing_w, tr)

    psnrs, maes, covs, stales = [], [], [], []
    worst = (1e9, -1)

    for i, (canvas, filled, tmap) in enumerate(prop):
        x, y = int(cam[i][0]), int(cam[i][1])
        # ground truth for the full extended strip
        gx0, gx1 = x - wing_w, x + w + wing_w
        truth = np.zeros_like(canvas)
        valid = np.zeros(canvas.shape[:2], bool)
        s0, s1 = max(0, gx0), min(pano.shape[1], gx1)
        if s1 > s0:
            truth[:, s0 - gx0:s1 - gx0] = pano[y:y + h, s0:s1]
            valid[:, s0 - gx0:s1 - gx0] = True

        # score wings only, where we both recovered something and truth exists
        wingmask = np.ones(canvas.shape[:2], bool)
        wingmask[:, wing_w:wing_w + w] = False
        m = filled & valid & wingmask
        if m.sum() < 500:
            continue

        a = canvas[m].astype(np.float64)
        b = truth[m].astype(np.float64)
        mse = float(np.mean((a - b) ** 2))
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
        mae = float(np.mean(np.abs(a - b)))

        psnrs.append(psnr)
        maes.append(mae)
        covs.append(float((filled & wingmask).sum() / wingmask.sum()))
        st = tmap[m]
        stales.append(float(np.median(st)) / fps)
        if psnr < worst[0]:
            worst = (psnr, i)

    print(f"frames scored          : {len(psnrs)}")
    print(f"mean wing coverage     : {np.mean(covs)*100:.1f}%")
    print(f"mean PSNR (recovered)  : {np.mean(psnrs):.1f} dB")
    print(f"median PSNR            : {np.median(psnrs):.1f} dB")
    print(f"worst frame            : {worst[0]:.1f} dB at frame {worst[1]}")
    print(f"mean abs error         : {np.mean(maes):.2f} / 255")
    print(f"median pixel staleness : {np.median(stales):.2f} s")
    print()
    print("reference: >40 dB is visually identical, 30-40 dB is very good,")
    print("           <20 dB would mean the propagation is misregistered.")

    # a visual for the demo
    i = len(prop) // 2
    canvas, filled, tmap = prop[i]
    vis = canvas.copy()
    vis[~filled] = (255, 0, 255)
    stale_vis = np.clip(tmap / max(tmap.max(), 1) * 255, 0, 255).astype(np.uint8)
    stale_vis = cv2.applyColorMap(stale_vis, cv2.COLORMAP_INFERNO)
    stale_vis[~filled] = (0, 0, 0)
    cv2.line(vis, (wing_w, 0), (wing_w, h), (0, 255, 255), 1)
    cv2.line(vis, (wing_w + w, 0), (wing_w + w, h), (0, 255, 255), 1)
    cv2.putText(vis, "RECOVERED  (magenta = never filmed)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(stale_vis, "STALENESS  (bright = pixel is from further back in time)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([vis, stale_vis])
    cv2.imwrite("/mnt/user-data/outputs/wing_recovery_example.png", out)
    print("\nwrote wing_recovery_example.png")


if __name__ == "__main__":
    main()
