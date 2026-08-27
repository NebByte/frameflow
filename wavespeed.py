"""
wavespeed — the video outpainting half, and the shot-level contract it needs.

`select_provider` picks this for the generator slot because it is the only entry
in `agent.REGISTRY` that is both hosted and `conditions_on_known`: it accepts the
recovered canvas as an anchor instead of re-imagining the middle. Everything
higher-scoring in that table (Kling, Runway, Luma) is unanchored, which makes it
better at video and useless here.

WHY THIS NEEDED A NEW CONTRACT
------------------------------
`fill`'s generators take one frame:  (canvas, hole, confidence) -> frame.
That is right for `MirrorGenerator`, which is deterministic per frame, and it is
wrong for anything temporal. Call a diffusion model once per frame and each call
is independent: the invented wall is re-imagined 24 times a second and crawls.
A side wall that flickers in peripheral vision is worse than one left dark.

So a generator may also offer:

    generate_shot(frames, wing_w, prompt) -> list of widened frames

One submission, one temporally-coherent result, and the caller still composites
every frame through its own hole -- the fence does not move. `screenx_render`
prefers this path when a generator has it and falls back to per-frame otherwise.

WHAT IS AND IS NOT VERIFIED
---------------------------
Everything except the wire: submission shape, polling, timeout, failure paths,
frame encode/decode round trip and the fence are covered by `test_wavespeed.py`
through an injected transport. The endpoint path, the field names and the exact
polling semantics are written from the platform's documented async pattern --
submit a job, poll it, fetch the output -- and **must be checked against the
current API before the first real call**. They are constructor arguments, not
constants, for exactly that reason.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DEFAULT_BASE = "https://api.wavespeed.ai/api/v3"
DEFAULT_MODEL = "wavespeed-ai/video-outpainter"
TIMEOUT = 60
POLL_EVERY = 3.0
POLL_LIMIT = 600.0


def _request(url, body=None, headers=None, timeout=TIMEOUT):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        # The body of a rejection is the only thing that says WHY. Discarding it
        # turns a service that explained itself into "HTTP Error 400: Bad
        # Request", which is indistinguishable from a wrong key, an oversized
        # clip and an unsupported aspect ratio -- and each has a different fix.
        try:
            detail = (e.read() or b"")[:400].decode("utf-8", "replace").strip()
        except OSError:
            detail = ""
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason}: {detail}" if detail else str(e.reason),
            e.headers, None) from None
    try:
        return json.loads(raw)
    except ValueError:
        return {"_bytes": raw}


class WaveSpeedOutpainter:
    """
    Hosted video outpainting. Anchored, temporal, no GPU on your side.

    The prompt comes from `reasoning.Plan`, so what this is asked to draw is the
    plan's measured elements first and its asserted ones after -- not a bare
    instruction to invent something cinematic.
    """
    name = "wavespeed"

    def __init__(self, token=None, base=DEFAULT_BASE, model=DEFAULT_MODEL,
                 transport=_request, poll_every=POLL_EVERY, poll_limit=POLL_LIMIT,
                 fps=24.0, sleep=time.sleep, seed=-1):
        self.token = token or os.environ.get("WAVESPEED_API_KEY", "") or \
            os.environ.get("SCREENX_TOKEN", "")
        self.base = base.rstrip("/")
        self.model = model
        self.transport = transport
        self.poll_every = poll_every
        self.poll_limit = poll_limit
        self.fps = fps
        self.sleep = sleep
        self.seed = seed          # -1: let the service choose
        self.prompt = "extend the scene sideways, same place and moment"

    # -- headers

    def headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # -- codec

    @staticmethod
    def encode_clip(frames, fps=24.0) -> str:
        """Frames -> base64 mp4. One upload instead of N."""
        h, w = frames[0].shape[:2]
        path = Path(tempfile.mkdtemp()) / "clip.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (w, h))
        try:
            for f in frames:
                vw.write(f)
        finally:
            vw.release()
        blob = base64.b64encode(path.read_bytes()).decode()
        try:
            path.unlink()
            path.parent.rmdir()
        except OSError:
            pass
        return blob

    @staticmethod
    def decode_clip(raw: bytes, expect=None):
        """Returned mp4 -> frames. Short results are padded by holding the last."""
        path = Path(tempfile.mkdtemp()) / "out.mp4"
        path.write_bytes(raw)
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        try:
            path.unlink()
            path.parent.rmdir()
        except OSError:
            pass
        if expect and frames and len(frames) < expect:
            frames += [frames[-1]] * (expect - len(frames))
        return frames[:expect] if expect else frames

    # -- the job

    # The API expands to one of a fixed set of aspect ratios. It takes no
    # per-side expansion amounts, and its request schema is
    # additionalProperties:false, so sending them is a rejected request rather
    # than an ignored field.
    ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9,
               "9:16": 9 / 16, "3:2": 1.5, "2:3": 2 / 3,
               "21:9": 21 / 9, "9:21": 9 / 21}

    @classmethod
    def aspect_for(cls, w, h, wing_w):
        """
        The enum ratio closest to the canvas we actually want.

        Never narrower than the source: a ratio below it would crop the picture
        instead of extending it, and the caller would composite the real centre
        back over a frame that no longer lines up with it.
        """
        want = (w + 2 * wing_w) / float(h)
        src = w / float(h)
        wide = {k: v for k, v in cls.ASPECTS.items() if v >= src}
        pool = wide or cls.ASPECTS
        return min(pool, key=lambda k: abs(pool[k] - want))

    def submit(self, frames, wing_w, prompt) -> str:
        h, w = frames[0].shape[:2]
        body = {
            "video": f"data:video/mp4;base64,{self.encode_clip(frames, self.fps)}",
            "prompt": prompt or "",
            "aspect_ratio": self.aspect_for(w, h, wing_w),
            "seed": int(self.seed),
        }
        payload = self.transport(f"{self.base}/{self.model}", body, self.headers())
        job = (payload.get("data") or payload).get("id") or payload.get("id")
        if not job:
            raise RuntimeError(f"no job id in submission response: {list(payload)[:6]}")
        return str(job)

    def poll(self, job_id):
        """Wait for the job, then hand back the raw output bytes."""
        waited = 0.0
        url = f"{self.base}/predictions/{job_id}/result"
        while waited <= self.poll_limit:
            payload = self.transport(url, None, self.headers())
            data = payload.get("data") or payload
            status = str(data.get("status", "")).lower()
            if status in ("completed", "succeeded", "success"):
                return self.fetch_output(data)
            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"job {job_id} {status}: {data.get('error')}")
            self.sleep(self.poll_every)
            waited += self.poll_every
        raise TimeoutError(f"job {job_id} still {status or 'pending'} after "
                           f"{self.poll_limit:.0f}s")

    def fetch_output(self, data) -> bytes:
        outs = data.get("outputs") or data.get("output") or []
        if isinstance(outs, str):
            outs = [outs]
        if not outs:
            raise RuntimeError("job completed with no output")
        first = outs[0]
        if isinstance(first, dict):
            first = first.get("url") or first.get("video") or ""
        if isinstance(first, str) and first.startswith("data:"):
            return base64.b64decode(first.split(",", 1)[1])
        if isinstance(first, str) and first.startswith("http"):
            got = self.transport(first, None, self.headers())
            if isinstance(got, dict) and "_bytes" in got:
                return got["_bytes"]
            raise RuntimeError("output url did not return bytes")
        raise RuntimeError(f"unrecognised output entry: {type(first).__name__}")

    # -- the shot-level contract

    def generate_shot(self, frames, wing_w, prompt=None):
        """
        One submission for the whole shot. Returns widened frames, uncomposited.

        Callers still composite through their own hole mask -- this never decides
        which pixels are kept, so a model that returns a re-imagined centre
        cannot corrupt the metric, only waste the call.
        """
        if not self.token:
            raise RuntimeError(
                "WaveSpeedOutpainter needs a credential. Set WAVESPEED_API_KEY, "
                "or use MirrorGenerator to stay local.")
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        job = self.submit(frames, wing_w, prompt or self.prompt)
        raw = self.poll(job)
        out = self.decode_clip(raw, expect=len(frames))
        if not out:
            raise RuntimeError("job returned a clip with no frames")
        return [self.fit_to_canvas(f, w, h, wing_w) for f in out]

    @staticmethod
    def fit_to_canvas(f, w, h, wing_w):
        """
        Height-match, then centre-crop or pad to the wing canvas.

        The model expands to an aspect ratio, not to our wing width, so what
        comes back is rarely exactly w + 2*wing_w. Squashing it to fit would
        rescale the picture horizontally, and the centre would no longer line up
        with the real frame the caller composites back on top -- the wings would
        stop continuing the shot, which is the one thing they have to do.
        Matching height preserves scale; the surplus is trimmed symmetrically.
        """
        target_w = int(w + 2 * wing_w)
        if f is None or f.size == 0 or f.ndim != 3:
            raise ValueError(f"the job returned an unusable frame: "
                             f"{None if f is None else getattr(f, 'shape', '?')}")
        fh, fw = f.shape[:2]
        if fh <= 0 or fw <= 0:
            raise ValueError(f"the job returned a degenerate frame {fw}x{fh}")
        if fh != h:
            # A returned frame far from our aspect makes this scale wildly, and
            # cv2 answers an out-of-range size with "Unknown C++ exception",
            # which says nothing about which of a dozen calls raised it.
            scaled_w = int(round(fw * h / float(fh)))
            if not 0 < scaled_w <= 1 << 15:
                raise ValueError(
                    f"the job returned {fw}x{fh}, which height-matches to "
                    f"{scaled_w}px wide against a {target_w}px canvas")
            f = cv2.resize(f, (scaled_w, h))
            fh, fw = f.shape[:2]
        if fw == target_w:
            return f
        if fw > target_w:
            x = (fw - target_w) // 2
            return f[:, x:x + target_w]
        out = np.zeros((h, target_w, 3), f.dtype)
        left = (target_w - fw) // 2
        out[:, left:left + fw] = f
        return out

    # -- per-frame fallback, so it still satisfies the old contract

    def __call__(self, canvas, hole, confidence):
        """
        Single frame. Works, and is the wrong way to use this.

        Kept so the generator is a drop-in wherever `fill`'s contract is
        expected, but every call is an independent job: slow, expensive, and
        temporally incoherent. `generate_shot` is the path that exists.
        """
        h, cw = canvas.shape[:2]
        cols = np.where(~hole.all(0))[0]
        if not len(cols):
            return canvas.copy()
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        centre = canvas[:, x0:x1]
        out = self.generate_shot([centre], x0, self.prompt)
        return out[0] if out else canvas.copy()
