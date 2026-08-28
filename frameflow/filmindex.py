"""
filmindex — what else this film contains, offered to the agent as sources.

THE BUG THIS FIXES
------------------
`Director.run()` accepts `corpus_finder`, `scene_finder` and `fetcher`, and
`ResourceScout` probes all three. The render pipeline passed none of them. Every
probe returned an empty list, the planner's option set collapsed to `generate`
alone, and the DONATED and RETRIEVED rungs had never once fired in a real run.
The ladder was there; nothing was ever put on it.

This builds the missing corpus from the film being processed:

    sample     3 frames per shot, small -- what TakeMatcher samples anyway
    signature  grade-invariant gradient embedding of each shot
    group      shots whose signatures agree become one `scene`, i.e. a guess
               that they are the same location photographed more than once
    serve      corpus_finder / scene_finder closures the scout can call

WHAT THIS CAN AND CANNOT REACH
------------------------------
Grouping by appearance is a shortlist, not a verification. Every candidate it
offers still has to survive geometric verification inside the tool, and the
measured reality is unkind: same TAKE across two cuts registers trivially, while
two different SETUPS of one location verified in 1 of 400 attempts by
homography. So `scene_finder` will mostly hand back candidates that
`SameLocationTool` then refuses until the 3D backend exists to bridge them.

That is the honest state: the wiring is now real, and it exposes that the
remaining gap is registration, not retrieval -- which is what crosscut.py said
all along.
"""
from __future__ import annotations

import numpy as np

from . import crosscut as cc
class FilmIndex:
    """Every shot of one film, grouped by how much they look alike."""

    # Measured on a 78-shot trailer: pairwise signature similarity ran mean
    # 0.026, p95 0.195, max 0.316 across 30 shots. An earlier 0.55 default
    # grouped precisely nothing and made the rung look dead a second time.
    SCENE_THRESHOLD = 0.30
    SHORTLIST = 6

    def __init__(self, samples=3, max_side=320, scene_threshold=SCENE_THRESHOLD,
                 shortlist=SHORTLIST):
        self.samples = samples
        self.max_side = max_side
        self.scene_threshold = scene_threshold
        self.shortlist = shortlist
        self.shots: list[dict] = []
        self._scenes: dict[str, list[int]] | None = None
        self._forced: dict[int, str] = {}

    # -- building

    def add(self, shot_id: int, frames: list, film: str = "primary",
            scene: str | None = None, sfm_frames: list | None = None) -> None:
        """
        Keep a few small frames per shot. Whole films do not fit in memory.

        `scene` overrides appearance grouping. Measured on a staged wide plus
        close-up of one wall: the signature put them in SEPARATE scenes, because
        a tight close-up and a wide of the same surface do not look alike by
        gradient statistics. That is the cross-setup problem itself, and no
        threshold fixes it -- lowering it enough to join these two joins
        everything else as well.

        So it stops being inferred. If you filmed two setups of one wall you
        know that, and saying so is an assertion about the SHOOT, not about the
        pixels: it decides only what is worth attempting to reconstruct, and
        every candidate still has to survive geometric verification. A wrong
        claim here costs a failed reconstruction, not a false measurement.
        """
        if len(frames) < 2:
            return
        import cv2
        idx = np.linspace(0, len(frames) - 1, self.samples).astype(int)
        keep = []
        for i in idx:
            f = frames[int(i)]
            if f.shape[1] > self.max_side:
                s = self.max_side / f.shape[1]
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            keep.append(f)
        self.shots.append(dict(shot=shot_id, frames=keep, film=film,
                               sfm_frames=sfm_frames,
                               signature=cc.TakeMatcher.embed(keep)))
        if scene:
            self._forced[len(self.shots) - 1] = scene
        self._scenes = None

    def frames_for_sfm(self, i: int) -> list:
        """
        Frames fit to reconstruct from, which are NOT the ones stored above.

        `add` keeps three thumbnails at 320px because that is all appearance
        matching needs and whole films do not fit in memory. Structure from
        motion needs something else entirely, and for a long time it did not get
        it: `build_film` read the same three thumbnails, so COLMAP was handed a
        six-image contact sheet at 320x240 and asked to bridge two camera setups.
        It registered nothing -- 0 usable, 0 partial -- and the backend then fell
        back to essential poses and refused to render, which read like a pose bug
        and was really a starvation bug.

        So a caller that intends to reconstruct passes `sfm_frames`: more frames,
        at working resolution. Opt-in, because holding them for every shot of a
        feature is hundreds of megabytes.
        """
        sh = self.shots[i]
        return sh.get("sfm_frames") or sh["frames"]

    def add_film(self, name: str, shot_frames, id_base: int = 0) -> int:
        """
        Index a SECOND cut of the same picture -- a TV spot, a teaser, an
        international trailer.

        This is what `DONATED` was always waiting for. Inside one film a take
        appears once, so `SameTakeTool` had nothing to find and the rung never
        fired. Across two cuts the same take appears twice, differently cropped
        and graded, which registers trivially -- and if the other cut framed
        wider, it donates real periphery this one never had.

        Costs no GPU. `TakeMatcher` is ORB and a homography.
        """
        added = 0
        for i, frames in enumerate(shot_frames):
            if len(frames) < 2:
                continue
            before = len(self.shots)
            # id_base keeps this cut's shot ids clear of the primary film's.
            # Without it the second cut's shot 0 is the first cut's shot 0, and
            # anything keyed on (shot, frame) -- sfm.write_images, the manifest
            # agent.py unpacks -- silently reads one film's frame through the
            # other's entry. That exact collision already cost a GPU session
            # once, on the --also path.
            self.add(id_base + i, frames, film=name)
            added += len(self.shots) - before
        return added

    # -- grouping

    def scenes(self) -> dict[str, list[int]]:
        """
        Single-link grouping on the appearance signature.

        Deliberately loose: this only decides what is worth *attempting* to
        register. A false grouping costs one failed verification; a missed one
        costs a source the agent never sees.
        """
        if self._scenes is not None:
            return self._scenes
        groups: list[list[int]] = []
        named: dict[str, list[int]] = {}
        for i, s in enumerate(self.shots):
            if i in self._forced:                 # declared, not guessed
                named.setdefault(self._forced[i], []).append(i)
                continue
            placed = False
            for g in groups:
                if any(self._similar(s["signature"], self.shots[j]["signature"])
                       for j in g):
                    g.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
        self._scenes = dict(named)
        self._scenes.update({f"scene{n:03d}": g for n, g in enumerate(groups)})
        return self._scenes

    def _similar(self, a, b) -> bool:
        return float(np.dot(a, b) / len(a)) >= self.scene_threshold

    def scene_of(self, shot_id: int, film: str = "primary") -> str | None:
        """
        Which scene a shot belongs to.

        Takes the film as well as the shot id: once a second clip is indexed the
        ids collide, and matching on the number alone would hand a shot of the
        wide the scene of the close-up that happens to share its index.
        """
        for name, members in self.scenes().items():
            for i in members:
                sh = self.shots[i]
                if sh["shot"] == shot_id and sh.get("film", "primary") == film:
                    return name
        return None

    # -- serving the scout

    def _ranked(self, exclude_shot: int, limit: int, film="primary",
                cross_film_first=True):
        """
        Closest shots by appearance, best first.

        Shots from ANOTHER cut are preferred when scores are close, because
        those are the ones that can actually donate: the same take from a second
        cut registers, while a different shot of the same film is a different
        setup and will not.
        """
        me = next((s for s in self.shots
                   if s["shot"] == exclude_shot and s.get("film", "primary") == film), None)
        if me is None:
            return []
        scored = []
        for s in self.shots:
            if s["shot"] == exclude_shot and s.get("film", "primary") == film:
                continue
            sim = float(np.dot(me["signature"], s["signature"]) / len(me["signature"]))
            other = s.get("film", "primary") != film
            scored.append(((1 if (other and cross_film_first) else 0, sim), s))
        scored.sort(key=lambda t: (-t[0][0], -t[0][1]))
        return [s for _, s in scored[:limit]]

    def corpus_finder(self, exclude_shot: int):
        """
        Other shots that might be the same TAKE.

        Shortlisted by signature rather than offered whole: geometric
        verification is the expensive half, and handing the tool all N shots
        makes the film cost N^2 ORB comparisons for candidates the embedding
        already knows are hopeless. Same shortlist-then-verify shape
        TakeMatcher.match uses.

        Within a single film this rarely fires -- a take usually appears once.
        It pays when the corpus spans two cuts of the same picture, the case
        crosscut.py measured at 100% precision.
        """
        def find(ctx):
            return [dict(frames=s["frames"], shot=s["shot"],
                         film=s.get("film", "primary"))
                    for s in self._ranked(exclude_shot, self.shortlist)]
        return find

    def scene_finder(self, exclude_shot: int):
        """Other setups grouped to the same location."""
        def find(scene_id):
            members = self.scenes().get(scene_id, [])
            return [dict(frames=self.shots[i]["frames"], shot=self.shots[i]["shot"])
                    for i in members if self.shots[i]["shot"] != exclude_shot]
        return find

    def summary(self) -> dict:
        sc = self.scenes()
        sizes = sorted((len(v) for v in sc.values()), reverse=True)
        films = {}
        for sh in self.shots:
            f = sh.get("film", "primary")
            films[f] = films.get(f, 0) + 1
        return dict(shots=len(self.shots), scenes=len(sc),
                    largest_scene=sizes[0] if sizes else 0,
                    multi_setup_scenes=sum(1 for n in sizes if n > 1),
                    films=films)
