"""
Robust shot detection + letterbox handling for real graded footage.

The fixed-threshold HSV histogram method in wingcoverage.detect_shots was
calibrated on synthetic clips and finds ZERO cuts in a real trailer: modern
grading pushes everything to a narrow dark teal/orange band, so histograms
barely move across a cut.

Replacement:
  - crop letterbox bars first (they dominate any global statistic)
  - score each frame pair with luma MAD + edge-histogram distance
  - threshold ADAPTIVELY: a cut is a peak far above the LOCAL median score,
    so the detector adapts to a calm dialogue scene and a chaotic action
    scene within the same film
"""
import cv2
import numpy as np


def detect_letterbox(path, samples=40, tol=12):
    """Find constant black bars by sampling frames across the whole clip."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows_max, cols_max = None, None
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * n / samples))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = g.max(1)
        c = g.max(0)
        rows_max = r if rows_max is None else np.maximum(rows_max, r)
        cols_max = c if cols_max is None else np.maximum(cols_max, c)
    cap.release()
    if rows_max is None:
        return None

    def span(v):
        nz = np.where(v > tol)[0]
        return (int(nz[0]), int(nz[-1] + 1)) if len(nz) else (0, len(v))

    y0, y1 = span(rows_max)
    x0, x1 = span(cols_max)
    return x0, y0, x1, y1


def frame_scores(path, crop=None, scale=0.25):
    """Per-frame-pair dissimilarity: luma MAD + edge histogram distance."""
    cap = cv2.VideoCapture(path)
    prev_g, prev_e = None, None
    scores = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if crop:
            x0, y0, x1, y1 = crop
            f = f[y0:y1, x0:x1]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        e = cv2.Canny(g, 50, 150)
        if prev_g is None:
            scores.append(0.0)
        else:
            mad = float(np.mean(np.abs(g.astype(np.int16) - prev_g.astype(np.int16))))
            # edge overlap: cuts destroy edge correspondence
            inter = float(np.logical_and(e > 0, prev_e > 0).sum())
            union = float(np.logical_or(e > 0, prev_e > 0).sum()) + 1e-6
            edge = 1.0 - inter / union
            scores.append(mad / 255.0 * 100.0 + edge * 40.0)
        prev_g, prev_e = g, e
    cap.release()
    return np.array(scores)


def cuts_from_scores(scores, win=61, k=4.0, floor=8.0, min_gap=6):
    """
    Adaptive: a cut is a score far above the LOCAL median.
    `win` is the rolling window; `k` how many local-MADs above median.
    """
    n = len(scores)
    pad = win // 2
    p = np.pad(scores, pad, mode="edge")
    med = np.array([np.median(p[i:i + win]) for i in range(n)])
    dev = np.array([np.median(np.abs(p[i:i + win] - med[i])) for i in range(n)]) + 1e-6
    z = (scores - med) / dev

    cand = np.where((z > k) & (scores > floor))[0]
    cuts, last = [], -min_gap
    for c in cand:
        if c - last >= min_gap:
            cuts.append(int(c))
            last = c
    return cuts, z


def survives_correspondence(path, crop, cut, min_ratio=0.10, max_side=320):
    """
    True if `cut` still looks like a cut once the pixels are compared.

    Matches ORB features across the boundary. A real cut leaves almost no
    correspondence; a fast pan or a whip still matches its own room. Returns
    True when the evidence is unavailable -- a detector that silently drops
    cuts it could not check would be worse than one that keeps a few false ones.
    """
    cap = cv2.VideoCapture(str(path))
    x0, y0, x1, y1 = crop
    got = []
    for idx in (cut - 1, cut):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, f = cap.read()
        if not ok:
            cap.release()
            return True
        f = f[y0:y1, x0:x1]
        if f.shape[1] > max_side:
            sc = max_side / f.shape[1]
            f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        got.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()

    orb = cv2.ORB_create(600)
    ka, da = orb.detectAndCompute(got[0], None)
    kb, db = orb.detectAndCompute(got[1], None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return True

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    ratio = len(good) / float(min(len(ka), len(kb)))
    return ratio < min_ratio


def segment(path, min_len=16, verify=True, verbose=False):
    crop = detect_letterbox(path)
    scores = frame_scores(path, crop)
    cuts, z = cuts_from_scores(scores)
    if verify:
        kept = [c for c in cuts if survives_correspondence(path, crop, c)]
        dropped = len(cuts) - len(kept)
        if dropped and verbose:
            print(f"  {dropped} candidate cut(s) dropped: the pixels still "
                  f"correspond across them", flush=True)
        cuts = kept
    bounds = [0] + cuts + [len(scores)]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] - bounds[i] >= min_len]
    return dict(crop=crop, n_frames=len(scores), cuts=cuts, shots=shots,
                scores=scores)


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    r = segment(path)
    x0, y0, x1, y1 = r["crop"]
    print(f"letterbox crop : x{x0}-{x1}  y{y0}-{y1}   (active {x1-x0}x{y1-y0})")
    print(f"frames         : {r['n_frames']}")
    print(f"cuts detected  : {len(r['cuts'])}")
    lens = np.array([b - a for a, b in r["shots"]])
    if len(lens):
        print(f"shots >= 16f   : {len(r['shots'])}")
        print(f"shot length    : median {np.median(lens):.0f}f "
              f"({np.median(lens)/23.976:.2f}s)  max {lens.max()}f "
              f"({lens.max()/23.976:.2f}s)")
        top = sorted(r["shots"], key=lambda s: -(s[1] - s[0]))[:10]
        print("longest shots  :", [(a, b, f"{(b-a)/23.976:.1f}s") for a, b in top])
