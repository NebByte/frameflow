"""
agent — an agent that fills wings, cheapest-and-truest first.

THE LADDER
----------
Every pixel it places is labelled with where it came from, and the labels are
ordered by how much you can trust them:

    PRIMARY    the frame itself
    RECOVERED  propagated from elsewhere in THIS shot        (same camera)
    DONATED    same take, another cut, geometrically verified (same camera)
    RETRIEVED  another setup of the same location, 3D-verified(same set)
    REFERENCED licensed external material, UNVERIFIED        (somewhere else)
    GENERATED  invented

The agent always tries the ladder top-down and only descends when a rung comes
back empty. That ordering is the product. A system that reaches for a diffusion
model before checking whether the pixels were already filmed produces a prettier
demo and a worthless metric.

WHY PROVENANCE HAS SIX LEVELS AND NOT TWO
-----------------------------------------
RETRIEVED is real photography, but of a different setup -- different lens,
lighting, and a different moment in time. It is not the same as RECOVERED and
must not be counted as such. Collapsing them is how "we recovered 80%" quietly
becomes a lie.

REFERENCED is the same argument one rung further down. External material is
real photons too, but nothing checks that it depicts this location, so it sits
outside PHOTOGRAPHIC and outside the headline number. See provenance.py for the
measurement that forced the split, and for the promotion path once a wide-
baseline backend can verify these assets.

TOOLS
-----
The agent has a registry it can call. `find_in_film` is the one that matters and
it is real: a feature covers each location from several setups, so the master
wide usually contains the periphery of every tighter shot in the scene. External
tools exist for reference material and are gated on licence -- an asset with no
recorded rights is refused by default, because the output here is a derivative
of someone's film.

GENERATION
----------
Providers are pluggable and declare capabilities. Selection is automatic: the
combination this project needs -- large spatial extrapolation over long
sequences -- is exactly the one the literature flags as unsolved, so the agent
chunks per shot and prefers providers that accept a known-region condition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import fill as fence_mod
import wingcoverage as wc

# ------------------------------------------------------------------ provenance
# defined in provenance.py so the fence and the ladder cannot disagree; see the
# note there about the value 2 meaning two different things.
from provenance import (DONATED, GENERATED, NOT_THIS_PLACE,  # noqa: F401
                        PHOTOGRAPHIC, PRIMARY, PROV_NAMES, REAL_LEVELS,
                        RECOVERED, REFERENCED, RETRIEVED)


# ------------------------------------------------------------------ licensing

@dataclass
class Asset:
    """External material with its rights recorded. No licence, no use."""
    pixels: np.ndarray
    source: str
    licence: Optional[str] = None
    url: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.licence is not None


@dataclass
class SourcePolicy:
    """
    Default-deny on unlicensed external material.

    The deliverable is a derivative of a copyrighted film. Anything the agent
    pulls from outside the production needs recorded rights, or the whole output
    is unshippable regardless of how good it looks.
    """
    allow_unlicensed: bool = False
    allowed_licences: tuple = ("owned", "licensed", "cc0", "cc-by", "public-domain")

    def admit(self, asset: Asset) -> bool:
        if asset.usable and asset.licence in self.allowed_licences:
            return True
        return self.allow_unlicensed


# ------------------------------------------------------------------ tools

@dataclass
class ToolResult:
    pixels: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    provenance: int
    note: str = ""
    confidence: float = 0.0


class Tool:
    name = "tool"
    provenance = GENERATED

    def applicable(self, ctx) -> bool:
        return True

    def run(self, ctx) -> ToolResult:
        raise NotImplementedError


class SameShotTool(Tool):
    """Rung 1. Already done by the backend; present so the ladder is explicit."""
    name = "same_shot"
    provenance = RECOVERED

    def run(self, ctx):
        return ToolResult(None, None, RECOVERED, "handled by backend propagate()")


class SameTakeTool(Tool):
    """
    Rung 2. Another cut of this same take, geometrically verified.
    Wraps the crosscut matcher; measured 100% precision, 0 false positives.
    """
    name = "same_take"
    provenance = DONATED

    def __init__(self, corpus=None, matcher=None):
        self.corpus = corpus or []
        self.matcher = matcher

    def applicable(self, ctx):
        return bool(self.corpus) and self.matcher is not None

    def run(self, ctx):
        try:
            import crosscut as cc
        except Exception:
            return ToolResult(None, None, DONATED, "crosscut unavailable")
        best = None
        for cand in self.corpus:
            dA = self.matcher.descriptors(ctx.frames)
            dB = self.matcher.descriptors(cand["frames"])
            n, H, ia, ib = self.matcher.verify(dA, dB)
            if n >= 45 and (best is None or n > best[0]):
                best = (n, H, ia, ib, cand)
        if best is None:
            return ToolResult(None, None, DONATED, "no shared take found")
        n, H, ia, ib, cand = best
        scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        if abs(scale - 1.0) < 0.02:
            return ToolResult(None, None, DONATED,
                              f"shared take found ({n} inl) but framing identical "
                              f"(scale {scale:.3f}) -- no periphery to gain")
        return ToolResult(cand["frames"][min(ib, len(cand['frames']) - 1)], None,
                          DONATED, f"donor scale {scale:.3f}, {n} inliers",
                          confidence=min(1.0, n / 300))


class SameLocationTool(Tool):
    """
    Rung 3. A DIFFERENT setup of the same location elsewhere in the film.

    This is the rung that matters for a feature. A locked-off close-up has no
    recoverable periphery of its own, but the scene's master wide photographed
    that same wall an hour earlier. Wide baseline, so it needs the 3D backend --
    a homography cannot bridge two setups (measured: 1 verified pair in 400).
    """
    name = "same_location"
    provenance = RETRIEVED

    def __init__(self, scene_index=None, backend=None, scene_models=None,
                 max_setup_frames=24):
        self.scene_index = scene_index or {}
        self.backend = backend
        # scene_id -> sfm.SceneModel. One reconstruction spanning every setup of
        # that location, with a manifest saying which (shot, frame) each view is.
        # Without one the poses come from an essential chain, which the backend
        # refuses to render from; see splat.PoseSet.trustworthy.
        self.scene_models = scene_models or {}
        self.max_setup_frames = max_setup_frames
        self._cache = {}

    def applicable(self, ctx):
        return bool(self.scene_index) and ctx.scene_id in self.scene_index

    def _frame_index(self, ctx):
        """
        Which of the shot's frames is this canvas showing?

        The Director hands each frame's canvas to the tool but passes the whole
        shot as `frames`, so the tool is not told which one it is looking at.
        The centre of the canvas IS the frame, byte for byte -- the fence
        guarantees it -- so the index is recoverable by comparison rather than
        by threading a new argument through five call sites.
        """
        h, cw = ctx.canvas.shape[:2]
        ww = ctx.wing_w
        w = cw - 2 * ww
        centre = ctx.canvas[:, ww:ww + w]
        for i, f in enumerate(ctx.frames):
            if f.shape[:2] == centre.shape[:2] and np.array_equal(f, centre):
                return i
        return None

    def _scene_frames(self, ctx, model):
        """
        Every view of the scene, in the reconstruction's own order.

        The reconstruction is the authority on order, not the setup list:
        COLMAP's images.txt is in registration order and `poses_from_colmap`
        re-sorts by filename, so lining frames up any other way would associate
        each pose with the wrong picture -- a failure that renders something
        coherent and completely wrong.

        `model.views`, not `model.manifest`. The manifest is every frame SENT
        to COLMAP; the poses are only the ones it REGISTERED. At 100% they are
        the same list, which is why this read correctly for as long as nothing
        ever failed to register -- but the gate admits anything from
        MIN_REGISTERED up, so at 80-99% the manifest is longer than the pose
        set and every frame after the first dropout pairs with the wrong
        camera. That is precisely the coherent-and-wrong render this docstring
        warns about.
        """
        by_shot = {}
        if ctx.shot_id is not None:
            by_shot[int(ctx.shot_id)] = list(ctx.frames)
        for st in (self.scene_index.get(ctx.scene_id) or []):
            if isinstance(st, dict) and "shot" in st:
                by_shot[int(st["shot"])] = list(st["frames"])

        frames = []
        for shot, fi in model.views:
            src = by_shot.get(int(shot))
            if src is None or int(fi) >= len(src):
                raise RuntimeError(
                    f"the reconstruction names (shot {shot}, frame {fi}) but that "
                    f"frame is not available here. It was built over different "
                    f"footage than this run is showing.")
            frames.append(src[int(fi)])
        return frames

    def _model_for(self, ctx):
        """Fit once per location, not once per frame. -> (model, poses, scene)."""
        import splat as sp
        key = ctx.scene_id
        if key in self._cache:
            return self._cache[key]

        scene = self.scene_models.get(ctx.scene_id)
        if scene is None:
            raise RuntimeError(
                f"no reconstruction for {ctx.scene_id!r}. Bridging two setups is "
                f"exactly the case a homography fails at -- 1 verified pair in "
                f"400 -- so this rung needs real poses. Build them with "
                f"sfm.build_film(index, out_dir).")
        if not scene.usable:
            raise RuntimeError(
                f"{ctx.scene_id}: only {scene.registered_fraction * 100:.0f}% of "
                f"views registered. Rendering the rest would cover the easy "
                f"setups while the report read as though it covered the scene.")

        poses = scene.poses()
        frames = self._scene_frames(ctx, scene)
        if len(poses) != len(frames):
            raise RuntimeError(
                f"{len(poses)} poses for {len(frames)} manifest views in "
                f"{ctx.scene_id}; the reconstruction and the manifest disagree")

        bk = self.backend
        masks = sp.dynamic_mask(frames) if getattr(bk, "mask_dynamics", True) else None
        model = sp.fit_splats(frames, poses, masks,
                              iters=getattr(bk, "iters", 3000),
                              device=getattr(bk, "device", "cuda"))
        self._cache[key] = (model, poses, scene)
        return model, poses, scene

    def run(self, ctx):
        if self.backend is None:
            return ToolResult(None, None, RETRIEVED,
                              "needs GaussianBackend (wide baseline); none supplied")
        if ctx.shot_id is None:
            return ToolResult(None, None, RETRIEVED,
                              "no shot_id on the context; cannot locate this "
                              "shot's views in the scene reconstruction")

        i = self._frame_index(ctx)
        if i is None:
            return ToolResult(None, None, RETRIEVED,
                              "could not locate this canvas among the shot's frames")
        try:
            import splat as sp
            model, poses, scene = self._model_for(ctx)
            view = scene.index_of(ctx.shot_id, i)
            if view is None:
                return ToolResult(
                    None, None, RETRIEVED,
                    f"frame {i} of shot {ctx.shot_id} is not in the "
                    f"reconstruction (sampled out, or failed to register)")
            rendered = sp.render_widened(model, poses.subset([view]),
                                         [ctx.frames[i]], ctx.wing_w,
                                         alpha_thresh=getattr(self.backend,
                                                              "alpha_thresh", 0.5),
                                         device=getattr(self.backend, "device", "cuda"))
        except (RuntimeError, NotImplementedError, ImportError, FileNotFoundError) as e:
            return ToolResult(None, None, RETRIEVED, str(e)[:180])

        canvas, filled, _tmap = rendered[0]
        # hand back only what the reconstruction actually covered. The fence
        # intersects this with the hole, so the primary region is safe either
        # way, but returning an all-True mask would claim the misses too.
        return ToolResult(canvas, filled, RETRIEVED,
                          f"{len(model)} splats over {len(poses)} views of "
                          f"{ctx.scene_id}; covered {filled.mean() * 100:.1f}%",
                          confidence=float(min(1.0, scene.registered_fraction)))


class ExternalReferenceTool(Tool):
    """
    Rung 4. Licensed material from outside the production.

    This is the first rung that is NOT photographic, and the gap between it and
    RETRIEVED is the widest on the ladder. A licensed photograph of "a food
    court" is a camera pointed at *a* place; it is not a camera pointed at
    *this* place, and nothing here can establish that it is. So the pixels are
    labelled REFERENCED, which lives in NOT_THIS_PLACE, and they cannot move the
    headline number however good the match looks.

    An earlier build labelled the same material RETRIEVED and a flat colour
    plate duly pushed the real fraction from 3.36% to 5.34%. That is the failure
    this rung's position exists to prevent.

    The licence check is default-deny: the deliverable is a derivative of a
    copyrighted film, so material with no recorded rights is refused rather than
    used and flagged.
    """
    name = "external"
    provenance = REFERENCED

    def __init__(self, fetcher=None, policy=None):
        self.fetcher = fetcher
        self.policy = policy or SourcePolicy()

    def applicable(self, ctx) -> bool:
        return self.fetcher is not None

    def run(self, ctx) -> ToolResult:
        if self.fetcher is None:
            return ToolResult(None, None, REFERENCED, "no fetcher supplied")
        try:
            assets = list(self.fetcher(ctx) or [])
        except Exception as e:                    # a dead host is not a crash
            return ToolResult(None, None, REFERENCED,
                              f"fetch failed: {type(e).__name__}")
        if not assets:
            return ToolResult(None, None, REFERENCED, "nothing found")

        admitted = [a for a in assets if self.policy.admit(a)]
        refused = len(assets) - len(admitted)
        if not admitted:
            return ToolResult(None, None, REFERENCED,
                              f"{len(assets)} found, all refused on licence")

        # the largest usable plate: upscaling a thumbnail across a wing looks
        # exactly like the smear this rung is supposed to beat
        best = max(admitted, key=lambda a: getattr(a.pixels, "size", 0))
        if getattr(best.pixels, "size", 0) == 0:
            return ToolResult(None, None, REFERENCED, "asset had no pixels")

        h, w = ctx.canvas.shape[:2]
        px = cv2.resize(best.pixels, (w, h), interpolation=cv2.INTER_AREA)
        if px.ndim == 2:
            px = cv2.cvtColor(px, cv2.COLOR_GRAY2BGR)
        note = f"{best.source} ({best.licence})"
        if refused:
            note += f", {refused} refused on licence"
        # confidence stays low on purpose: this is the right place, asserted, by
        # somebody who was not there
        return ToolResult(px, None, REFERENCED, note, confidence=0.25)


class GenerateTool(Tool):
    """Rung 5. Last resort. Always writes GENERATED."""
    name = "generate"
    provenance = GENERATED

    def __init__(self, provider=None):
        self.provider = provider

    def run(self, ctx):
        prov = self.provider or fence_mod.InpaintGenerator()
        px = prov(ctx.canvas, ctx.hole, ctx.confidence)
        return ToolResult(px, None, GENERATED, f"provider={getattr(prov,'name','?')}")


# ------------------------------------------------------------------ providers

@dataclass
class Capability:
    name: str
    max_side: int
    temporal: bool          # models a sequence, not independent frames
    conditions_on_known: bool   # accepts the recovered region as anchor
    licence: str
    hosted: bool
    note: str = ""


REGISTRY = [
    # research methods, self-hosted
    Capability("unboxed", 1024, True, True, "research", False,
               "3DGS static reconstruction + diffusion; best geometric consistency"),
    Capability("hl-outpaint", 2048, True, True, "research", False,
               "targets long-range AND large extrapolation -- the combination "
               "the rest of the literature leaves unresolved"),
    Capability("m3ddm", 512, True, True, "research", False,
               "masked 3D diffusion, coarse-to-fine; good on long sequences"),
    Capability("motia", 720, True, True, "research", False,
               "per-case test-time adaptation; slow, strong on source-specific texture"),
    Capability("vace", 1024, True, True, "research", False,
               "large-scale diffusion training; strong generative quality"),
    # open weights
    Capability("hunyuanvideo-1.5", 1080, True, False, "apache-2.0", False,
               "DiT + 3D causal VAE; open weights, self-hostable"),
    Capability("wan-2.7", 1080, True, False, "open", False, "open weights"),
    # hosted
    Capability("wavespeed-outpainter", 1080, True, True, "commercial", True,
               "purpose-built video outpainting endpoint, direction-wise"),
    Capability("kling-3.0", 2160, True, False, "commercial", True,
               "native 4K/60fps, 15s clips"),
    Capability("runway-gen-4.5", 1080, True, False, "commercial", True,
               "strongest control surface: motion brushes, scene consistency"),
    Capability("luma-ray3", 1080, True, False, "commercial", True, "native HDR"),
]


def select_provider(target_side, need_anchor=True, allow_hosted=True,
                    allow_commercial=True):
    """
    Pick a provider. Anchoring is weighted highest: conditioning on the recovered
    canvas is what stops generation free-running, and it is worth more here than
    raw fidelity because the wings are peripheral vision on a side wall.
    """
    def ok(c):
        if need_anchor and not c.conditions_on_known:
            return False
        if not allow_hosted and c.hosted:
            return False
        if not allow_commercial and c.licence == "commercial":
            return False
        return True

    cands = [c for c in REGISTRY if ok(c)]
    if not cands:
        return None
    cands.sort(key=lambda c: (c.max_side >= target_side, c.temporal, c.max_side),
               reverse=True)
    return cands[0]


# ------------------------------------------------------------------ the agent

@dataclass
class Context:
    canvas: np.ndarray
    filled: np.ndarray
    tmap: np.ndarray
    wing_w: int
    frames: list = field(default_factory=list)
    scene_id: Optional[str] = None
    hole: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    # which shot these frames are. Needed to find this shot's own views inside a
    # scene reconstruction that spans several setups -- the manifest is keyed on
    # (shot, frame) and "the asking shot" is not otherwise recoverable.
    shot_id: Optional[int] = None


class WingAgent:
    """
    Runs the ladder, records provenance, and composites everything through the
    fence so no rung below RECOVERED can ever overwrite a rung above it.
    """

    def __init__(self, tools=None, policy=None, provider=None):
        self.policy = policy or SourcePolicy()
        self.tools = tools or [
            SameShotTool(),
            SameTakeTool(),
            SameLocationTool(),
            ExternalReferenceTool(policy=self.policy),
            GenerateTool(provider),
        ]

    def fill(self, canvas, filled, tmap, wing_w, frames=None, scene_id=None,
             fps=24.0, shot_id=None):
        ctx = Context(canvas.copy(), filled.copy(), tmap, wing_w,
                      frames or [], scene_id, shot_id=shot_id)
        prov = np.full(filled.shape, GENERATED, np.uint8)
        h, cw = filled.shape
        w = cw - 2 * wing_w
        prov[filled] = RECOVERED
        centre = np.zeros(filled.shape, bool)
        centre[:, wing_w:wing_w + w] = True
        prov[centre & filled] = PRIMARY

        protected = filled.copy()
        log = []

        for tool in self.tools:
            ctx.hole = ~ctx.filled
            if not ctx.hole.any():
                log.append((tool.name, "skipped: nothing left to fill"))
                break
            if not tool.applicable(ctx):
                log.append((tool.name, "not applicable"))
                continue
            ctx.confidence = fence_mod.confidence_map(ctx.canvas, ctx.filled,
                                                      ctx.tmap, fps)

            # snapshot BEFORE the tool runs. Snapshotting after lets a tool that
            # mutates the canvas in place bake its damage in before the check.
            before = ctx.canvas[protected].copy()

            # tools see a READ-ONLY view: in-place mutation fails at numpy level
            # rather than being caught after the fact.
            ro = ctx.canvas.view()
            ro.flags.writeable = False
            guarded = Context(ro, ctx.filled, ctx.tmap, ctx.wing_w,
                              ctx.frames, ctx.scene_id, ctx.hole, ctx.confidence,
                              ctx.shot_id)
            try:
                res = tool.run(guarded)
            except ValueError as e:
                if "read-only" in str(e).lower():
                    raise fence_mod.FenceViolation(
                        f"{tool.name} attempted to write the canvas in place") from e
                raise
            if res.pixels is None:
                log.append((tool.name, res.note or "no result"))
                continue

            px = res.pixels
            if px.shape[:2] != ctx.canvas.shape[:2]:
                px = cv2.resize(px, (ctx.canvas.shape[1], ctx.canvas.shape[0]))
            new = ctx.hole.copy()
            if res.mask is not None:
                new &= res.mask
            if not new.any():
                log.append((tool.name, "produced nothing usable"))
                continue

            ctx.canvas[new] = px[new]
            prov[new] = res.provenance
            ctx.filled |= new
            if not np.array_equal(ctx.canvas[protected], before):
                raise fence_mod.FenceViolation(
                    f"{tool.name} wrote into protected pixels")
            log.append((tool.name, f"filled {new.mean()*100:.1f}% :: {res.note}"))

        return ctx.canvas, prov, log

    @staticmethod
    def report(prov, wing_w, w):
        left = prov[:, :wing_w]
        right = prov[:, wing_w + w:]
        wing = np.concatenate([left, right], 1)
        n = wing.size
        out = {PROV_NAMES[k]: round(float((wing == k).sum() / n), 4)
               for k in PROV_NAMES}
        out["real_same_camera"] = round(
            float(np.isin(wing, REAL_LEVELS).sum() / n), 4)
        out["photographic"] = round(
            float(np.isin(wing, PHOTOGRAPHIC).sum() / n), 4)
        # generated + referenced. What a budget on invention has to cover, or
        # `max_generated` is satisfiable with a wing full of unverified stock.
        out["not_this_place"] = round(
            float(np.isin(wing, NOT_THIS_PLACE).sum() / n), 4)
        return out
