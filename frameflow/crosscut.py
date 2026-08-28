"""
crosscut — find the SAME TAKE across two different cuts of a film, and donate
periphery from whichever cut framed it wider.

Why this and not a general "context engine": a homography can only bridge
near-identical viewpoints. Measured on the real trailer, only 1 of 400 candidate
shot pairs from DIFFERENT setups verified geometrically. Retrieval was never the
bottleneck -- registration was.

But the same TAKE appearing in a second trailer or TV spot is not a different
viewpoint. It is the same camera, same moment, differently cropped and graded.
That registers trivially, and if the other cut framed wider, it donates real
periphery the primary cut never had.

GROUND TRUTH HARNESS
--------------------
We synthesise the second cut from the same source, so correspondence is known:
  PRIMARY   : centre-cropped to 78% (the "tight" trailer we want to extend),
              grade A, low bitrate
  ALTERNATE : full frame (wider), grade B, different bitrate, different shot
              subset, shuffled order, trimmed durations
Shots present in both are the shared takes. Shots unique to one cut are
distractors that MUST NOT match. That gives precision and recall.

And because PRIMARY was cropped from frames we still hold, every donated wing
pixel can be scored against the true original.
"""
from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------------ grading

def grade(img, gain, gamma, tint):
    """Crude but realistic colour-grade difference between two cuts."""
    f = img.astype(np.float32) / 255.0
    f = np.clip(f * np.array(gain, np.float32), 0, 1)
    f = np.power(f, gamma)
    f = np.clip(f + np.array(tint, np.float32) / 255.0, 0, 1)
    return (f * 255).astype(np.uint8)


def recode(frames, q):
    """Round-trip through JPEG to simulate a different distribution encode."""
    out = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else f)
    return out


def build_cuts(frames, shots, rng, crop=0.78, n_shared=18, n_only=8):
    """Return (primary, alternate, truth) where truth maps primary->alternate."""
    usable = [s for s in shots if s[1] - s[0] >= 20]
    rng.shuffle(usable)
    shared = usable[:n_shared]
    p_only = usable[n_shared:n_shared + n_only]
    a_only = usable[n_shared + n_only:n_shared + 2 * n_only]

    h, w = frames[shots[0][0]].shape[:2]
    cw, ch = int(w * crop), int(h * crop)
    ox, oy = (w - cw) // 2, (h - ch) // 2

    primary, alternate, truth = [], [], {}

    def tight(seg):
        # centre-crop then scale back up: the "tighter framing" of this cut
        return [cv2.resize(f[oy:oy + ch, ox:ox + cw], (w, h),
                           interpolation=cv2.INTER_CUBIC) for f in seg]

    for a, b in shared + p_only:
        seg = frames[a:b]
        # each cut trims the take differently
        s0 = rng.integers(0, max(1, (b - a) // 6))
        seg = seg[s0:s0 + max(16, (b - a) - int(s0) - int(rng.integers(0, 4)))]
        pid = len(primary)
        primary.append(dict(src=(a, b),
                            frames=recode(tight([grade(f, (1.05, 1.0, 0.92), 0.95, (2, 0, -3))
                                                 for f in seg]), 62),
                            full=[f.copy() for f in seg]))
        if (a, b) in shared:
            truth[pid] = (a, b)

    for a, b in shared + a_only:
        seg = frames[a:b]
        s0 = rng.integers(0, max(1, (b - a) // 6))
        seg = seg[s0:s0 + max(16, (b - a) - int(s0) - int(rng.integers(0, 4)))]
        alternate.append(dict(src=(a, b),
                              frames=recode([grade(f, (0.94, 1.02, 1.08), 1.06, (-3, 1, 4))
                                             for f in seg], 78)))

    order = rng.permutation(len(alternate))
    alternate = [alternate[i] for i in order]
    lookup = {tuple(alt["src"]): i for i, alt in enumerate(alternate)}
    truth = {p: lookup[src] for p, src in truth.items() if src in lookup}
    return primary, alternate, truth


# ------------------------------------------------------------------ matching

class TakeMatcher:
    """Appearance shortlist, then geometric verification. Geometry decides."""

    def __init__(self, n_features=1500, min_inliers=45, samples=3):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self.min_inliers = min_inliers
        self.samples = samples

    def _prep(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(g)          # grade differences flattened here

    def descriptors(self, seg):
        idx = np.linspace(0, len(seg) - 1, self.samples).astype(int)
        out = []
        for i in idx:
            k, d = self.orb.detectAndCompute(self._prep(seg[i]), None)
            out.append((i, k, d))
        return out

    @staticmethod
    def embed(seg):
        """Grade-invariant appearance signature: normalised tiny gradient image."""
        f = seg[len(seg) // 2]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        m = cv2.resize(np.sqrt(gx * gx + gy * gy), (16, 12))
        v = m.ravel()
        return (v - v.mean()) / (v.std() + 1e-6)

    def verify(self, dA, dB):
        """Best homography between any sampled pair. Returns (inliers, H, ia, ib)."""
        best = (0, None, 0, 0)
        for ia, k1, d1 in dA:
            if d1 is None or len(d1) < 2:
                continue
            for ib, k2, d2 in dB:
                if d2 is None or len(d2) < 2:
                    continue
                raw = self.bf.knnMatch(d1, d2, k=2)
                good = [m for m, n in (p for p in raw if len(p) == 2)
                        if m.distance < 0.75 * n.distance]
                if len(good) < 20:
                    continue
                p1 = np.float32([k1[m.queryIdx].pt for m in good])
                p2 = np.float32([k2[m.trainIdx].pt for m in good])
                H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
                if H is None or mask is None:
                    continue
                n = int(mask.sum())
                # a same-take pair differs by roughly a similarity transform;
                # reject wild projective solutions
                if abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
                    continue
                if n > best[0]:
                    best = (n, H, ia, ib)
        return best

    def match(self, primary, alternate, shortlist=6):
        embA = [self.embed(p["frames"]) for p in primary]
        embB = [self.embed(a["frames"]) for a in alternate]
        descA = [self.descriptors(p["frames"]) for p in primary]
        descB = [self.descriptors(a["frames"]) for a in alternate]

        results = {}
        for i in range(len(primary)):
            sims = [(float(np.dot(embA[i], embB[j]) / len(embA[i])), j)
                    for j in range(len(alternate))]
            sims.sort(reverse=True)
            best = (0, None, 0, 0, -1)
            for _, j in sims[:shortlist]:
                n, H, ia, ib = self.verify(descA[i], descB[j])
                if n > best[0]:
                    best = (n, H, ia, ib, j)
            results[i] = dict(inliers=best[0], H=best[1], ia=best[2],
                              ib=best[3], alt=best[4],
                              accepted=best[0] >= self.min_inliers)
        return results
