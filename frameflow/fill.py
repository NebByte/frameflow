"""
fill — generative completion that CANNOT overwrite recovered pixels.

The whole project rests on one number: what fraction of the wing is real. The
moment a generator is allowed to write into the recovered region, that number is
unverifiable and the contribution evaporates. So the fence is structural, not a
convention:

  * every pixel carries provenance -- PRIMARY / RECOVERED / GENERATED
  * the generator is handed a mask of ONLY the empty region
  * its output is composited through that mask alone
  * a post-condition asserts the recovered pixels are bit-identical, and raises
    if not

CONFIDENCE CONDITIONING
-----------------------
The generator receives a per-pixel confidence map built from the same weights
the metric uses -- staleness and detail. Pixels adjacent to fresh, detailed
recovered content are high confidence and should be extended faithfully; pixels
deep in a hole far from anything real are low confidence and should be treated
as free invention. A generative model conditions on that map alongside the
recovered canvas as geometric anchor.

WHAT SHIPS HERE
---------------
Two CPU generators that invent nothing a model would: OpenCV inpainting, and a
mirror-and-blur built for wing-sized holes. Both are honest about knowing very
little out there. The only generative path in the project is Gemini, via
`frameflow.gemini` -- the fence is identical whichever writes the pixels, and
it is the fence, not the generator, that makes the real-footage number mean
something.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import wingcoverage as wc
from .provenance import GENERATED, PRIMARY, RECOVERED  # noqa: F401  (re-exported)


class FenceViolation(RuntimeError):
    pass


# ---------------------------------------------------------------- generators

class InpaintGenerator:
    """CPU stand-in. Structure-propagating, not semantic. Good enough to test."""
    name = "inpaint"

    def __call__(self, canvas, hole, confidence):
        r = int(max(3, min(12, 0.01 * canvas.shape[1])))
        return cv2.inpaint(canvas, hole.astype(np.uint8) * 255, r, cv2.INPAINT_TELEA)


class MirrorGenerator:
    """
    CPU stand-in built for wings specifically, where the hole is one huge block
    off the side of frame rather than a scattering of small gaps.

    Structure-propagating inpainting is the wrong tool at that size: TELEA has
    no signal to march from once it is a few dozen pixels past the edge, and
    returns a smear. This reflects the outer strip outward instead and blurs it
    progressively with distance, so the wall keeps the frame's colour and rough
    layout and gives up detail as it goes -- which is honest about how little is
    known out there, and is roughly what the eye expects from peripheral vision.

    Still GENERATED. It invents every pixel it places; it just does not pretend
    to know more than it does.
    """
    name = "mirror"

    def __call__(self, canvas, hole, confidence):
        h, w = canvas.shape[:2]
        out = canvas.copy()
        cols = np.where(~hole.all(0))[0]
        if not len(cols):
            return out
        x0, x1 = int(cols[0]), int(cols[-1]) + 1      # the known centre block
        for side, (lo, hi) in (("L", (0, x0)), ("R", (x1, w))):
            span = hi - lo
            if span <= 0:
                continue
            if side == "L":
                strip = canvas[:, x0:x0 + span][:, ::-1]
            else:
                strip = canvas[:, max(x0, x1 - span):x1][:, ::-1]
            if strip.shape[1] < span:                 # centre narrower than wing
                strip = np.pad(strip, ((0, 0), (span - strip.shape[1], 0), (0, 0)),
                               mode="edge")
            band = strip[:, -span:] if side == "L" else strip[:, :span]
            # blur and dim harder the further from real pixels we get
            steps = max(1, span // 24)
            piece = max(1, span // steps)
            built = band.copy()
            for s in range(steps):
                a, b = s * piece, min(span, (s + 1) * piece)
                if b <= a:
                    continue
                far = (s + 1) / steps
                if side == "L":
                    a, b = span - b, span - a         # left wing runs outward leftward
                    far = 1.0 - (a / max(span - 1, 1))
                k = int(max(1, min(41, round(far * 34)))) | 1
                seg = cv2.GaussianBlur(band[:, a:b], (k, k), 0)
                built[:, a:b] = (seg.astype(np.float32) * (1.0 - 0.35 * far)).astype(np.uint8)
            out[:, lo:hi] = built
        return out


# ---------------------------------------------------------------- confidence

def confidence_map(canvas, filled, tmap, fps=24.0, dilate=41):
    """
    Per-pixel confidence for the generator, in [0, 1].

    Built from the same weights the metric uses, then blurred outward: a hole
    pixel inherits confidence from how good the real content around it is.
    """
    conf = np.zeros(filled.shape, np.float32)
    if filled.any():
        sw = wc.staleness_weight(tmap.astype(np.float64))
        dw = wc.detail_weight(canvas)
        conf[filled] = (sw * dw)[filled]
    k = dilate | 1
    spread = cv2.GaussianBlur(conf, (k, k), 0)
    support = cv2.GaussianBlur(filled.astype(np.float32), (k, k), 0)
    out = np.where(support > 1e-3, spread / np.maximum(support, 1e-3), 0.0)
    out[filled] = conf[filled]
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------- the fence

def fenced_fill(canvas, filled, tmap, wing_w, generator=None, fps=24.0):
    """
    Complete the empty region without touching a single recovered pixel.

    Returns (out, provenance, confidence).
      provenance: PRIMARY / RECOVERED / GENERATED per pixel
      confidence: what the generator was conditioned on
    """
    generator = generator or InpaintGenerator()
    h, cw = filled.shape
    w = cw - 2 * wing_w

    prov = np.full(filled.shape, GENERATED, np.uint8)
    prov[filled] = RECOVERED
    prov[:, wing_w:wing_w + w] = np.where(filled[:, wing_w:wing_w + w],
                                          PRIMARY, GENERATED)

    conf = confidence_map(canvas, filled, tmap, fps)
    hole = ~filled

    before = canvas[filled].copy()          # for the post-condition

    if hole.any():
        produced = generator(canvas, hole, conf)
        out = canvas.copy()
        out[hole] = produced[hole]          # composite through the hole ONLY
    else:
        out = canvas.copy()

    if not np.array_equal(out[filled], before):
        raise FenceViolation(
            "generator modified recovered pixels -- coverage metric would be "
            "invalid; refusing to return this frame"
        )

    return out, prov, conf


def provenance_summary(prov, wing_w, w):
    """Fraction of the WING that is real vs invented. The number that ships."""
    left = prov[:, :wing_w]
    right = prov[:, wing_w + w:]
    wing = np.concatenate([left, right], 1)
    total = wing.size
    return dict(
        real=round(float((wing == RECOVERED).sum() / total), 4),
        generated=round(float((wing == GENERATED).sum() / total), 4),
    )
