"""
gemini — Google's models behind this repo's two model-shaped holes.

    ApiReasoner.call     "look at this frame and tell me what is off to the side"
    HostedGenerator      "draw it"

The first is a clean fit. The second is not, and the mismatch is worth stating
plainly before anyone wires it and wonders why the fence keeps firing.

WHY A VIDEO MODEL DOES NOT DROP INTO THE GENERATOR SLOT
-------------------------------------------------------
`fill.fenced_fill` hands a generator a canvas and a MASK, takes back a full
frame, and composites only through the hole -- then asserts the protected centre
came back bit-identical. That contract wants *masked outpainting*: extend this
exact frame sideways, do not touch the middle.

Text- or image-to-video models are not that. They synthesise a new clip from a
prompt and maybe a starting frame. Nothing in the interface accepts "keep these
pixels and fill only around them", so the centre comes back re-imagined,
per-frame drift is unconstrained, and the wall no longer lines up with the
screen it is next to. You can still composite the result through our own mask --
the fence will hold, because the fence is on this side -- but what lands in the
wings will not be a continuation of that frame. It will be a different room that
rhymes.

So: `GeminiVision` below is real work and worth doing today. Image editing per
frame (`GeminiImageEdit`) is the closest thing to masked outpainting Google
offers, and it comes with a temporal-consistency problem this project cares
about more than most, because a flickering side wall is worse than a dark one.
Both are marked for what they are.

CREDENTIALS
-----------
Two kinds work, and they are not interchangeable:

    AIza...     an API key. Long-lived. Header: x-goog-api-key.
    ya29. / AQ. an OAuth access token. Expires in about an hour.
                Header: Authorization: Bearer.

This picks by prefix and lets you override. If calls start failing after an
hour, that is the token expiring, not the code.

UNTESTED AGAINST THE LIVE SERVICE. There is no outbound network on the machine
this was written on. Everything except the HTTP call itself is covered by
`test_gemini.py` through an injected transport; the request shape is written
from the documented contract and the model id is a parameter, because Google's
lineup moves faster than any string hardcoded here.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"
VERTEX_HOST = "aiplatform.googleapis.com"
DEFAULT_VISION_MODEL = "gemini-2.5-flash"
TIMEOUT = 60
DEFAULT_LOCATION = "global"
_ADC = {}


def adc_token(now=None, ttl=1800.0):
    """
    A bearer token minted from Application Default Credentials.

    This exists because the credential a user actually has to hand is an `AQ.`
    OAuth token, which dies inside an hour -- long enough to paste into a chat
    and already dead by the time a render reaches its second shot. gcloud
    reissues one on demand, so nothing is pasted and nothing expires mid-run.
    Cached briefly: a subprocess per frame would cost more than the request.
    """
    now = time.time() if now is None else now
    if _ADC.get("expires", 0) > now:
        return _ADC["token"]
    try:
        out = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True, text=True, timeout=90, shell=(os.name == "nt"))
        token = (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if token:
        _ADC.update(token=token, expires=now + ttl)
    return token


def adc_project():
    """The project ADC bills against. Vertex needs it in the URL."""
    env = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(["gcloud", "config", "get-value", "project"],
                             capture_output=True, text=True, timeout=60,
                             shell=(os.name == "nt"))
        val = (out.stdout or "").strip()
        return "" if val in ("", "(unset)") else val
    except (OSError, subprocess.SubprocessError):
        return ""



def auth_headers(token: str, style: str = "auto") -> dict:
    """API key or bearer token, chosen by how the credential looks."""
    if style == "auto":
        style = "key" if token.startswith("AIza") else "bearer"
    if style == "key":
        return {"x-goog-api-key": token}
    return {"Authorization": f"Bearer {token}"}


def _post(url, body, headers, timeout=TIMEOUT):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class GeminiVision:
    """
    The vision half. Fits `ApiReasoner.call(prompt, image) -> list[str]`.

    Returns short claims about what is immediately off-frame. They come back as
    `asserted`, never `measured`: the model is another party making claims about
    a place it was not present at, and it is held to the same standard as a
    script page. `ApiReasoner` already tells it which elements are observations
    it must not contradict.

    A dead endpoint returns nothing rather than raising, because a missing rung
    should cost the planner one option and not the render.
    """

    # The shape the answer must take, appended to whatever the caller asks.
    #
    # A default rather than a constant, because it used to be stapled onto every
    # prompt unconditionally: a caller asking anything else -- polish asks what
    # is WRONG with a wall, not what stands beside it -- had its question
    # silently replaced by this one and got a confident answer to a question it
    # had not asked. Pass "" when the prompt states its own shape.
    OFF_FRAME = ("\n\nAnswer as a JSON array of at most {n} short phrases, each "
                 "naming ONE thing that would be immediately off the left or "
                 "right edge of this frame, in the same place and moment. Say "
                 "which side. No prose, no preamble.")

    def __init__(self, token=None, model=DEFAULT_VISION_MODEL, style="auto",
                 max_claims=4, transport=_post, project=None,
                 location=DEFAULT_LOCATION, answer_shape=None):
        # FRAMEFLOW_TOKEN is the current name; SCREENX_TOKEN is still read
        # so a shell profile written before the rename keeps working.
        self.token = (token
                      or os.environ.get("GEMINI_API_KEY", "")
                      or os.environ.get("FRAMEFLOW_TOKEN", "")
                      or os.environ.get("SCREENX_TOKEN", ""))
        self.model = model
        self.style = style
        self.max_claims = max_claims
        self.transport = transport      # injectable, so the logic is testable
        self.answer_shape = (self.OFF_FRAME if answer_shape is None
                             else answer_shape)

        # Vertex is the keyless route: it accepts the cloud-platform scope this
        # machine already holds, so no credential is pasted, stored, or rotated.
        # Only consulted when no explicit key was given.
        self.location = location
        self.project = project if project is not None else (
            "" if self.token else adc_project())

    def endpoint(self) -> str:
        """Vertex when running on ADC, the public API when given a key."""
        if self.project and not self.token:
            host = (VERTEX_HOST if self.location == "global"
                    else self.location + "-" + VERTEX_HOST)
            return (f"https://{host}/v1/projects/{self.project}/locations/"
                    f"{self.location}/publishers/google/models/"
                    f"{self.model}:generateContent")
        return f"{BASE}/models/{self.model}:generateContent"

    def credential(self) -> str:
        """An explicit key wins; otherwise mint one from ADC."""
        return self.token or (adc_token() if self.project else "")


    # -- request shaping

    def body(self, prompt: str, image=None) -> dict:
        parts = [{"text": prompt + (
            self.answer_shape.format(n=self.max_claims)
            if self.answer_shape else "")}]
        if image is not None:
            parts.append({"inline_data": {"mime_type": "image/png",
                                          "data": self.encode_image(image)}})
        return {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.4,
                                     "responseMimeType": "application/json"}}

    @staticmethod
    def encode_image(image) -> str:
        """A numpy BGR frame, or bytes already encoded."""
        if isinstance(image, (bytes, bytearray)):
            return base64.b64encode(bytes(image)).decode()
        import cv2
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("could not encode the frame")
        return base64.b64encode(buf).decode()

    # -- response reading

    @staticmethod
    def claims_from(payload) -> list:
        """
        Pull the phrases out, whatever shape they arrive in.

        `responseMimeType: application/json` usually yields a bare array, but a
        model that ignores it and writes prose should degrade to one claim
        rather than to a stack trace.
        """
        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return []
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except ValueError:
            # A model asked for JSON sometimes writes a Python list instead --
            # single quotes, which json refuses. Degrading that to one claim
            # lumps every phrase into a single string, and callers split on the
            # phrases: polish drops the sides that read "acceptable" and keeps
            # the rest, which cannot work on one blob. literal_eval reads a
            # literal and nothing else, so it evaluates no expression.
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return [text[:120]]
            if not isinstance(data, (list, dict)):
                return [text[:120]]
        if isinstance(data, list):
            return [str(x).strip()[:120] for x in data if str(x).strip()]
        if isinstance(data, dict):
            for k in ("claims", "items", "elements", "off_frame"):
                if isinstance(data.get(k), list):
                    return [str(x).strip()[:120] for x in data[k] if str(x).strip()]
        return [text[:120]]

    # -- the ApiReasoner contract

    def __call__(self, prompt: str, image=None) -> list:
        token = self.credential()
        if not token:
            return []
        url = self.endpoint()
        try:
            payload = self.transport(url, self.body(prompt, image),
                                     auth_headers(token, self.style))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, TimeoutError):
            return []
        return self.claims_from(payload)[:self.max_claims]


class GeminiImageEdit:
    """
    Per-frame image editing, the closest thing here to masked outpainting.

    Fits `fill`'s generator contract -- canvas in, full frame out, our side
    composites through the hole -- so the fence still guarantees the centre. What
    it does NOT guarantee is that frame 41's invented wall resembles frame 40's.
    Each call is independent, so the wings can crawl, and a side wall that
    flickers in peripheral vision is worse than one left dark.

    Before using this on anything longer than a few seconds, measure the
    flicker: mean absolute difference between consecutive generated wings.
    `MirrorGenerator` is temporally stable by construction and is the honest
    baseline to beat.

    UNTESTED against the live service.
    """
    name = "gemini-edit"

    def __init__(self, token=None, model="gemini-2.5-flash-image", style="auto",
                 transport=_post, project=None, location=DEFAULT_LOCATION):
        # FRAMEFLOW_TOKEN is the current name; SCREENX_TOKEN is still read
        # so a shell profile written before the rename keeps working.
        self.token = (token
                      or os.environ.get("GEMINI_API_KEY", "")
                      or os.environ.get("FRAMEFLOW_TOKEN", "")
                      or os.environ.get("SCREENX_TOKEN", ""))
        # same keyless route as GeminiVision: Vertex accepts the cloud-platform
        # scope this machine already holds, so nothing is pasted or rotated
        self.location = location
        self.project = project if project is not None else (
            "" if self.token else adc_project())
        self.model = model
        self.style = style
        self.transport = transport
        self.prompt = "extend the scene sideways, same place and moment"

    def __call__(self, canvas, hole, confidence):
        token = self.credential()
        if not token:
            raise RuntimeError(
                "GeminiImageEdit needs a credential. Set GEMINI_API_KEY, or use "
                "MirrorGenerator to stay local.")
        import cv2
        import numpy as np

        body = {"contents": [{"role": "user", "parts": [
            {"text": self.prompt + " Keep the central region exactly as given; "
                                   "fill only the blank margins."},
            {"inline_data": {"mime_type": "image/png",
                             "data": GeminiVision.encode_image(canvas)}},
        ]}]}
        url = self.endpoint()
        payload = self.transport(url, body, auth_headers(token, self.style))

        for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            blob = (part.get("inline_data") or part.get("inlineData") or {}).get("data")
            if not blob:
                continue
            raw = base64.b64decode(blob)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != canvas.shape[:2]:
                img = cv2.resize(img, (canvas.shape[1], canvas.shape[0]))
            return img
        raise RuntimeError("Gemini returned no image for this frame")

    def endpoint(self) -> str:
        """Vertex when running on ADC, the public API when handed a key."""
        if self.project and not self.token:
            host = (VERTEX_HOST if self.location == "global"
                    else self.location + "-" + VERTEX_HOST)
            return (f"https://{host}/v1/projects/{self.project}/locations/"
                    f"{self.location}/publishers/google/models/"
                    f"{self.model}:generateContent")
        return f"{BASE}/models/{self.model}:generateContent"

    def credential(self) -> str:
        return self.token or (adc_token() if self.project else "")

def reasoner(token=None, model=DEFAULT_VISION_MODEL, depth_frac=0.22):
    """An ApiReasoner already wired to Gemini's vision endpoint."""
    from . import reasoning as rz
    return rz.ApiReasoner(call=GeminiVision(token, model), depth_frac=depth_frac)
