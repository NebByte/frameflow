"""
context — everything anyone claims about what was out there.

Two features that look different and are the same thing:

    "take any file and use it as context"
    "let a person pause a shot and say what needs to be there"

A subtitle track, a script page, a set photo, and a human typing "fire escape,
camera left" are all ASSERTIONS about a place, from sources that may well know,
with no photons behind any of them. They enter the same store, bind to shots the
same way, drive the same generator, and are labelled the same: `DIRECTED`. The
one thing they must never do is move the coverage number, and the ladder makes
that structural rather than a promise.

WHAT BINDING MEANS, AND WHY SUBTITLES ARE WORTH MORE THAN A SCRIPT
------------------------------------------------------------------
Context is only useful if you can tell WHICH shot it describes. Formats differ
enormously in how much they tell you:

    .srt / .vtt   timecoded. Binds to a shot exactly, for free, no model.
    screenplay    ordered scenes, no timecodes. Available to every shot, and
                  honest about being unbound.
    stills/plates one image, no time at all, unless a sidecar says otherwise.
    a human note  bound to the exact shot the person was looking at -- the
                  tightest binding of the lot, which is what makes the
                  human-in-the-loop worth building.

A subtitle file is therefore the cheapest useful vision model in this repo:
zero API calls, exact timing, and it tells you the scene has a helicopter in it
because someone shouts about the helicopter.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not call a model, and it does not generate. It turns files and notes
into a prompt plus a provenance label, and hands both to whatever generator is
configured. `fill.HostedGenerator` is the API path; this is what gives it
something worth saying.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".fountain", ".fdx"}
SUB_EXT = {".srt", ".vtt"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DATA_EXT = {".json"}

SCENE_HEADING = re.compile(r"^\s*(INT\.|EXT\.|INT/EXT|I/E)[^\n]*", re.I | re.M)
_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


# ---------------------------------------------------------------- items

@dataclass
class ContextItem:
    """One assertion about the footage, and how tightly it is bound to it."""
    kind: str                       # dialogue | scene | note | plate | data
    text: str = ""
    source: str = ""
    licence: str | None = None
    t_start: float | None = None    # seconds; None means unbound in time
    t_end: float | None = None
    shot: int | None = None         # set for human notes
    author: str = ""
    path: str = ""

    @property
    def bound(self) -> bool:
        return self.shot is not None or self.t_start is not None

    def covers(self, t0: float, t1: float) -> bool:
        if self.t_start is None:
            return False
        return not (self.t_end is not None and self.t_end < t0 or self.t_start > t1)


# ---------------------------------------------------------------- loaders

def _secs(m, g):
    return (int(m.group(g)) * 3600 + int(m.group(g + 1)) * 60 + int(m.group(g + 2))
            + int(m.group(g + 3).ljust(3, "0")) / 1000.0)


def load_subtitles(path: Path):
    """Timecoded dialogue. The only common format that binds itself to shots."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    out, blocks = [], re.split(r"\n\s*\n", raw)
    for b in blocks:
        m = _SRT_TIME.search(b)
        if not m:
            continue
        lines = [ln for ln in b.splitlines() if not _SRT_TIME.search(ln)]
        text = " ".join(ln.strip() for ln in lines
                        if ln.strip() and not ln.strip().isdigit()
                        and not ln.strip().upper().startswith("WEBVTT"))
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            out.append(ContextItem("dialogue", text, source=path.name,
                                   t_start=_secs(m, 1), t_end=_secs(m, 5),
                                   path=str(path)))
    return out


def load_screenplay(path: Path):
    """
    Scene headings and their bodies. Ordered, but carrying no timecodes.

    Deliberately left unbound rather than guessed at. Mapping scene k onto shot
    k is wrong on any film ever cut, and a confident wrong binding is worse than
    an honest unbound one -- it would put the warehouse on the wall of the
    kitchen while the report says a person asked for it.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    heads = list(SCENE_HEADING.finditer(raw))
    if not heads:
        return [ContextItem("scene", raw.strip()[:4000], source=path.name,
                            path=str(path))]
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        body = raw[m.start():end].strip()
        out.append(ContextItem("scene", body[:2000], source=f"{path.name}:{m.group(0).strip()}",
                               path=str(path)))
    return out


def load_text(path: Path):
    return [ContextItem("note", path.read_text(encoding="utf-8", errors="replace").strip()[:4000],
                        source=path.name, path=str(path))]


def load_data(path: Path):
    """A json list of item dicts, or anything else recorded as one blob."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if isinstance(payload, list):
        out = []
        for d in payload:
            if isinstance(d, dict):
                fields = {k: v for k, v in d.items()
                          if k in ContextItem.__dataclass_fields__}
                fields.setdefault("kind", "data")
                fields.setdefault("source", path.name)
                out.append(ContextItem(**fields))
        return out
    return [ContextItem("data", json.dumps(payload)[:2000], source=path.name,
                        path=str(path))]


def load_image(path: Path, manifest=None):
    """A reference plate. Licence comes from licences.json or it has none."""
    entry = (manifest or {}).get(path.name, {})
    return [ContextItem("plate", entry.get("description", ""), source=path.name,
                        licence=entry.get("licence"), path=str(path))]


def load_any(path):
    """
    Take a file, any file. Unreadable ones are recorded, not swallowed.

    A format this cannot parse still gets an item saying so, because "we ignored
    six of your eight files" is something you want on the report rather than in
    the silence.
    """
    path = Path(path)
    if not path.is_file():
        return []
    ext = path.suffix.lower()
    try:
        if ext in SUB_EXT:
            return load_subtitles(path)
        if ext in TEXT_EXT:
            return (load_screenplay(path) if SCENE_HEADING.search(
                path.read_text(encoding="utf-8", errors="replace")) else load_text(path))
        if ext in DATA_EXT:
            return load_data(path)
        if ext in IMAGE_EXT:
            man = {}
            lic = path.parent / "licences.json"
            if lic.exists():
                try:
                    man = json.loads(lic.read_text())
                except ValueError:
                    man = {}
            return load_image(path, man)
    except OSError as exc:
        return [ContextItem("unreadable", f"{exc}", source=path.name, path=str(path))]
    return [ContextItem("unsupported", f"no loader for {ext or 'this file'}",
                        source=path.name, path=str(path))]


# ---------------------------------------------------------------- the bundle

@dataclass
class ContextBundle:
    items: list = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths):
        b = cls()
        for p in paths:
            p = Path(p)
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.name != "licences.json":
                        b.items.extend(load_any(f))
            else:
                b.items.extend(load_any(p))
        return b

    def add(self, item: ContextItem):
        self.items.append(item)
        return item

    def for_shot(self, shot: int, start_frame: int, n_frames: int, fps: float):
        """
        What applies to this shot: notes pinned to it, dialogue overlapping it,
        then everything unbound. Ordered by how tightly it is bound, because
        that is the order a prompt should spend its words in.
        """
        t0, t1 = start_frame / max(fps, 1e-6), (start_frame + n_frames) / max(fps, 1e-6)
        pinned = [i for i in self.items if i.shot == shot]
        timed = [i for i in self.items if i.shot is None and i.covers(t0, t1)]
        loose = [i for i in self.items
                 if not i.bound and i.kind not in ("unsupported", "unreadable")]
        return pinned + timed + loose

    def summary(self):
        kinds = {}
        for i in self.items:
            kinds[i.kind] = kinds.get(i.kind, 0) + 1
        return dict(items=len(self.items), kinds=kinds,
                    bound=sum(1 for i in self.items if i.bound),
                    unusable=sum(1 for i in self.items
                                 if i.kind in ("unsupported", "unreadable")))


# ---------------------------------------------------------------- directions

class DirectionStore:
    """
    Human-in-the-loop notes, on disk so they survive the run that made them.

    Pausing on a shot and typing what belongs there is the tightest binding any
    context gets -- the person is looking at the frame while they say it. It is
    still an assertion, so it lands on the same rung as a script page, and the
    report says how much of the wall is there because somebody asked for it.
    """

    FILE = "directions.json"

    def __init__(self, outdir):
        self.path = Path(outdir) / self.FILE
        self.items = []
        if self.path.exists():
            try:
                self.items = [ContextItem(**d) for d in json.loads(self.path.read_text())]
            except (ValueError, TypeError, OSError):
                self.items = []

    def add(self, shot: int, text: str, author: str = "", source: str = "direction"):
        item = ContextItem("note", text.strip(), source=source, shot=int(shot),
                           author=author)
        self.items.append(item)
        self.save()
        return item

    def for_shot(self, shot: int):
        return [i for i in self.items if i.shot == shot]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(i) for i in self.items], indent=2))


# ---------------------------------------------------------------- the prompt

def build_prompt(items, fallback="", limit=420):
    """
    Turn what is bound to this shot into one instruction for a video model.

    Pinned human notes go first and are never truncated away: if a person
    stopped to say what belongs there, that is the most specific information
    anyone has about this wall, and burying it under a page of screenplay is how
    you end up ignoring the only source that was actually looking at the frame.
    """
    notes = [i for i in items if i.kind == "note"]
    dialogue = [i for i in items if i.kind == "dialogue"]
    scenes = [i for i in items if i.kind == "scene"]
    plates = [i for i in items if i.kind == "plate" and i.text]

    parts = []
    if notes:
        parts.append("; ".join(n.text for n in notes))
    if scenes:
        head = scenes[0].text.splitlines()[0].strip()
        parts.append(head.lower())
    if dialogue:
        said = " ".join(d.text for d in dialogue)[:160]
        parts.append(f"spoken here: {said}")
    if plates:
        parts.append("matching: " + ", ".join(p.text for p in plates[:2]))
    if not parts and fallback:
        parts.append(fallback)

    prompt = "extend the scene sideways, same place and moment"
    body = " | ".join(p for p in parts if p)
    if body:
        prompt = f"{prompt}. {body}"
    return prompt[:limit]


def provenance_for(items):
    """
    DIRECTED when something asserted what belongs here, GENERATED otherwise.

    Deliberately not "DIRECTED whenever a context file was loaded anywhere".
    Unbound screenplay text that happens to be in the bundle did not describe
    THIS wall, and letting it relabel the pixels would turn the rung into a
    statement about the run rather than about the shot.
    """
    from . import provenance as P
    return P.DIRECTED if any(i.bound and i.text for i in items) else P.GENERATED
