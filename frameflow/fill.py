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
as free invention. That map is what a diffusion model conditions on, alongside
the recovered canvas as geometric anchor.

The default generator here is OpenCV inpainting -- a stand-in that runs on CPU
and lets the fence be tested end to end. Swap `DiffusionGenerator` in on a GPU
host; the fence is unchanged.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

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


class HostedGenerator:
    """
    Call a hosted video-outpainting endpoint. No GPU on this machine required.

    Deliberately thin. Endpoints disagree about everything -- field names, how
    the mask is encoded, whether frames go up singly or as a batch -- so the
    request shape is supplied by the caller rather than guessed at here:

        HostedGenerator(url=..., token=...,
                        encode=lambda canvas, hole, conf: {...json body...},
                        decode=lambda payload: rgb_array)

    Reads FRAMEFLOW_ENDPOINT and FRAMEFLOW_TOKEN from the environment when not
    passed. Compositing and the fence stay on this side: the endpoint returns a
    full frame and never decides which pixels are kept.

    UNTESTED against a live service -- written from the endpoint contract, not
    from a round trip. Expect to adjust `encode`/`decode` for your provider.
    """
    name = "hosted"

    def __init__(self, url=None, token=None, encode=None, decode=None, timeout=120):
        # the pre-rename names still work, so an existing shell profile does
        # not silently stop supplying an endpoint
        self.url = (url or os.environ.get("FRAMEFLOW_ENDPOINT", "")
                    or os.environ.get("SCREENX_ENDPOINT", ""))
        self.token = (token or os.environ.get("FRAMEFLOW_TOKEN", "")
                      or os.environ.get("SCREENX_TOKEN", ""))
        self.encode = encode
        self.decode = decode
        self.timeout = timeout

    def _default_encode(self, canvas, hole, confidence):
        ok_i, buf_i = cv2.imencode(".png", canvas)
        ok_m, buf_m = cv2.imencode(".png", hole.astype(np.uint8) * 255)
        if not (ok_i and ok_m):
            raise RuntimeError("could not encode the frame for upload")
        return dict(image=base64.b64encode(buf_i).decode(),
                    mask=base64.b64encode(buf_m).decode(),
                    prompt="extend the scene sideways, same place and moment")

    @staticmethod
    def _default_decode(payload):
        blob = payload.get("image") or payload.get("output")
        if isinstance(blob, list):
            blob = blob[0]
        if not blob:
            raise RuntimeError("endpoint returned no image")
        raw = base64.b64decode(blob)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("endpoint returned something that is not an image")
        return img

    def __call__(self, canvas, hole, confidence):
        if not self.url:
            raise RuntimeError(
                "HostedGenerator needs an endpoint. Set FRAMEFLOW_ENDPOINT (and "
                "FRAMEFLOW_TOKEN), or pass url=. Use MirrorGenerator to stay local."
            )
        body = (self.encode or self._default_encode)(canvas, hole, confidence)
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read())
        out = (self.decode or self._default_decode)(payload)
        if out.shape[:2] != canvas.shape[:2]:
            out = cv2.resize(out, (canvas.shape[1], canvas.shape[0]))
        return out


class DiffusionGenerator:
    """
    Video-diffusion outpainting, conditioned on the recovered canvas.

    The recovered pixels are the geometric anchor: the model receives correct
    parallax and structure and only has to invent texture, which is what keeps
    generation from free-running. Feed `confidence` as an extra conditioning
    channel and run the model at low strength where confidence is high.

    Requires CUDA.

    STRENGTH IS DRIVEN BY CONFIDENCE, WHICH IS THE POINT
    ----------------------------------------------------
    A single denoising strength across the whole hole treats a pixel one column
    from recovered geometry the same as one in the middle of a void. `confidence`
    already measures that difference -- it is the metric's own staleness and
    detail weights, blurred outward from the real content. Feeding it in as a
    per-pixel strength means the model barely touches well-anchored pixels and
    is free where there is nothing to anchor to.

    The output is still GENERATED everywhere. Conditioning on real geometry makes
    invention better, not real, and `GenerateTool` labels it accordingly.
    """
    name = "diffusion"

    #: Model id. SD2 inpainting is the default because it is the smallest thing
    #: that accepts a mask and runs in 8 GB; anything with the same
    #: (prompt, image, mask_image, strength) call signature drops in.
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-inpainting"

    def __init__(self, model=None, prompt=None, steps=25, guidance=7.0,
                 max_side=768, seed=0, device="cuda", pipe=None):
        self.model = model or self.DEFAULT_MODEL
        self.prompt = prompt or ("continuation of the scene, same lighting, "
                                 "same lens, photographic, no new subjects")
        self.steps = steps
        self.guidance = guidance
        self.max_side = max_side
        self.seed = seed
        self.device = device
        self._pipe = pipe                # inject a fake in tests

    @staticmethod
    def available():
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def pipe(self):
        if self._pipe is None:
            try:
                import torch
                from diffusers import StableDiffusionInpaintPipeline
            except ImportError as e:
                raise RuntimeError(
                    "DiffusionGenerator needs diffusers; pip install diffusers "
                    "transformers accelerate") from e
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model, torch_dtype=torch.float16).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def bands(self, confidence, hole, n=3):
        """
        Split the hole into confidence bands, each denoised at its own strength.

        Per-pixel strength is not something a diffusion pipeline accepts, so the
        hole is quantised into a few bands and the model is run once per band,
        strongest last. Three bands is not a tuned number -- it is the fewest
        that distinguishes "just outside the recovered region", "somewhere in
        between", and "nothing to go on".
        """
        conf = np.clip(np.asarray(confidence, np.float32), 0.0, 1.0)
        out = []
        edges = np.linspace(0.0, 1.0, n + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = hole & (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
            if band.any():
                # high confidence -> low strength. A well-anchored pixel should
                # be nudged, not reimagined.
                strength = float(np.clip(1.0 - (lo + hi) / 2.0, 0.2, 1.0))
                out.append((band, strength))
        return sorted(out, key=lambda t: t[1])

    def __call__(self, canvas, hole, confidence):
        if self._pipe is None and not self.available():
            raise RuntimeError(
                "DiffusionGenerator requires CUDA. Use InpaintGenerator on CPU, "
                "MirrorGenerator for a plausible wing, or run this on a GPU host "
                "-- see remote.py."
            )
        from PIL import Image

        h, w = canvas.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        sw, sh = _mult8(int(w * scale)), _mult8(int(h * scale))

        work = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)
        pipe = self.pipe()
        gen = None
        try:
            import torch
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
        except Exception:
            pass

        for band, strength in self.bands(confidence, hole):
            m = cv2.resize(band.astype(np.uint8) * 255, (sw, sh),
                           interpolation=cv2.INTER_NEAREST)
            if not m.any():
                continue
            res = pipe(prompt=self.prompt,
                       image=Image.fromarray(work[:, :, ::-1]),
                       mask_image=Image.fromarray(m),
                       strength=strength,
                       num_inference_steps=self.steps,
                       guidance_scale=self.guidance,
                       generator=gen)
            got = np.asarray(res.images[0])[:, :, ::-1]
            if got.shape[:2] != (sh, sw):
                got = cv2.resize(got, (sw, sh))
            # keep each band's result only inside that band, so a later, freer
            # pass cannot overwrite a better-anchored earlier one
            work[m > 0] = got[m > 0]

        out = cv2.resize(work, (w, h), interpolation=cv2.INTER_CUBIC)
        return out


def _mult8(x, lo=64):
    """Diffusion UNets need dimensions divisible by 8."""
    return max(lo, int(round(x / 8.0)) * 8)


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
