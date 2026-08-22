"""
fetchers — where external reference material comes from.

`ExternalReferenceTool` has always taken a `fetcher(ctx) -> [Asset]` and nobody
ever supplied one, so the rung was dead. These are the suppliers.

Every asset carries its licence or it does not get used. That is not decoration:
the output is a derivative of someone's film, and an unlicensed plate composited
into a side wall makes the whole render unshippable no matter how good it looks.
`SourcePolicy` enforces it; these fetchers only have to record the truth.

    LocalLibraryFetcher   a folder you control, with a licences.json manifest.
                          Production stills, set photography, location plates.
    OpenverseFetcher      openly-licensed images from the Openverse API, licence
                          string taken from the API response, not assumed.

WHAT PICKS THE SEARCH TERMS
---------------------------
Both take a `query_fn(ctx) -> str`. The default is deliberately weak -- it can
only describe colour and time of day, because nothing in this pipeline yet knows
what is IN the frame. That hook is where a vision model belongs: give it the
shot and let it say "rain-soaked Manhattan street at night, low angle", and this
rung starts returning material that belongs in the scene instead of material
that merely matches its palette.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import agent as ag

USER_AGENT = "screenx-toolkit/1.0 (wing coverage research)"
TIMEOUT = 12


# ---------------------------------------------------------------- queries

def describe(ctx) -> str:
    """
    Fallback query from pixels alone: brightness and dominant hue.

    This is the honest ceiling without a vision model. It cannot tell you the
    scene contains a fire escape; it can tell you the scene is dark and teal.
    """
    frame = ctx.canvas[:, ctx.wing_w:-ctx.wing_w] if ctx.wing_w else ctx.canvas
    hsv = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2HSV)
    v = float(hsv[..., 2].mean()) / 255.0
    hue = float(np.median(hsv[..., 0])) * 2.0
    light = "night" if v < 0.25 else ("dim" if v < 0.45 else "daylight")
    band = [(15, "warm"), (45, "golden"), (75, "green"), (150, "teal"),
            (210, "blue"), (280, "violet"), (330, "magenta"), (361, "warm")]
    tone = next(name for edge, name in band if hue < edge)
    return f"{light} {tone} exterior background plate"


# ---------------------------------------------------------------- local

class LocalLibraryFetcher:
    """
    A directory of material you already have rights to.

    Expects `licences.json` beside the images:

        {"skyline_night.jpg": {"licence": "owned", "source": "2nd unit plate"},
         "alley.png":         {"licence": "licensed", "source": "Shutterstock 123",
                               "url": "https://..."}}

    A file with no manifest entry gets licence=None and the policy refuses it.
    That is the intended behaviour, not an error to work around.
    """
    name = "local_library"

    def __init__(self, directory, query_fn=describe, limit=4):
        self.dir = Path(directory)
        self.query_fn = query_fn
        self.limit = limit

    def manifest(self) -> dict:
        path = self.dir / "licences.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {}

    def __call__(self, ctx):
        if not self.dir.is_dir():
            return []
        man = self.manifest()
        out = []
        for p in sorted(self.dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            entry = man.get(p.name, {})
            out.append(ag.Asset(pixels=img, source=entry.get("source", p.name),
                                licence=entry.get("licence"), url=entry.get("url")))
            if len(out) >= self.limit:
                break
        return out


# ---------------------------------------------------------------- online

class OpenverseFetcher:
    """
    Openly-licensed images, fetched at run time.

    The licence is read from the API response and passed through verbatim. If a
    result carries a licence this project does not accept, SourcePolicy drops it
    -- the fetcher never decides admissibility itself.

    Network failures return an empty list rather than raising: a missing rung
    should cost the planner one option, not the render.
    """
    name = "openverse"
    ENDPOINT = "https://api.openverse.org/v1/images/"

    def __init__(self, query_fn=describe, limit=3, licences=("cc0", "pdm", "by")):
        self.query_fn = query_fn
        self.limit = limit
        self.licences = licences

    def _get(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()

    def search(self, query: str):
        q = urllib.parse.urlencode(dict(q=query, page_size=self.limit,
                                        license=",".join(self.licences)))
        try:
            payload = json.loads(self._get(f"{self.ENDPOINT}?{q}"))
        except (urllib.error.URLError, OSError, ValueError):
            return []
        return payload.get("results", []) or []

    @staticmethod
    def widen(query: str):
        """
        The asked-for phrase, then progressively broader ones.

        Adjectives come first out of `describe` ("dim warm exterior background
        plate"), and they are the part that narrows a search to nothing, so they
        are the part dropped first. Always ends on the head noun, which is what
        the caller actually wanted a picture of.
        """
        words = [w for w in (query or "").split() if w]
        seen, out = set(), []
        for start in range(len(words)):
            phrase = " ".join(words[start:])
            if len(phrase) > 2 and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
        return out or ["background plate"]

    def __call__(self, ctx):
        out = []
        hits, used = [], ""
        for phrase in self.widen(self.query_fn(ctx)):
            hits = self.search(phrase)
            if hits:
                used = phrase
                break
        for hit in hits:
            url = hit.get("url")
            if not url:
                continue
            try:
                raw = self._get(url)
            except (urllib.error.URLError, OSError):
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # licence comes from the response; never assumed
            lic = (hit.get("license") or "").lower()
            lic = {"cc0": "cc0", "pdm": "public-domain", "by": "cc-by"}.get(lic, lic)
            out.append(ag.Asset(pixels=img,
                                source=f"openverse[{used}]:{hit.get('title', '?')[:44]}",
                                licence=lic or None,
                                url=hit.get("foreign_landing_url") or url))
        return out


def default_fetcher(library=None, online=False, query_fn=describe):
    """Compose what is available. Returns None if there is nothing to ask."""
    parts = []
    if library:
        parts.append(LocalLibraryFetcher(library, query_fn))
    if online:
        parts.append(OpenverseFetcher(query_fn))
    if not parts:
        return None

    def fetch(ctx):
        assets = []
        for p in parts:
            assets.extend(p(ctx) or [])
        return assets
    return fetch
