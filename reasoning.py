"""
reasoning — work out what belongs on the wall before asking anyone to draw it.

`context.build_prompt` concatenates whatever was bound to a shot and hands it
over. That is a courier, not a mind. This is the step in between: assemble the
evidence, work out what follows from it, and emit a PLAN that says what should
be out there, where, when, and -- the part that matters here -- on what basis.

WHY A PLAN AND NOT A PROMPT
---------------------------
A prompt is unreviewable. Once "burning skyline, camera left" goes into a model
you cannot ask which part of the output was evidence and which was the model
being helpful. A plan is a list of elements, each carrying its own support, so
the render can label pixels by WHY they exist and a person can read the
reasoning back before a single frame is generated.

THREE KINDS OF SUPPORT, AND THEY ARE NOT EQUAL
----------------------------------------------
    measured   derived from this footage. A figure exited right at 9.1 px/frame
               on frame 40, so on frame 50 it is ~91px into the right wing.
               Arithmetic over observations, not a guess.
    asserted   a script, a subtitle, or a person said so. May well be true.
               No photons behind it.
    inferred   nothing supports it; the wall needs to be something. Texture
               continuation, and honest about being filler.

CAUSAL vs INTERPOLATING, WHICH IS THE WHOLE TRICK
-------------------------------------------------
For SCORING a predictor, only the exit may be used -- otherwise the harness is
grading a model that already saw the answer, and `test_offscreen.py` enforces
that. For RENDERING, both ends are legitimately available: the film shows the
figure leaving AND coming back, so its off-screen path is an interpolation
between two observations rather than an extrapolation from one. That is why
this problem is tractable at all, and the two modes must never be confused --
hence `mode` on the plan, and a refusal to score an interpolating plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import context as cx
import provenance as P


@dataclass
class Element:
    """One thing the wall should contain, and why."""
    kind: str                 # figure | detail | continuation
    support: str              # measured | asserted | inferred
    side: str = ""            # 'L' | 'R' | '' for both
    text: str = ""
    because: str = ""
    frames: tuple = ()        # (first, last) within the shot
    depth_frac: float = 0.0   # 0 at the screen edge, 1 at the outer wing edge
    y_frac: float = 0.5

    def describe(self) -> str:
        where = {"L": "on the left wall", "R": "on the right wall"}.get(
            self.side, "on both walls")
        if self.kind == "figure":
            if self.depth_frac > 1.0:
                far = "further out than the wall reaches for most of it"
            elif self.depth_frac < 0.4:
                far = "just past the frame edge"
            else:
                far = "well out"
            return (f"{where}, {self.text or 'a figure'} {far}, "
                    f"{int(self.y_frac * 100)}% down the frame")
        return f"{where}, {self.text}"


@dataclass
class Brief:
    """Everything known and claimed about one shot, kept separate on purpose."""
    shot: int
    n_frames: int
    wing_w: int
    frame_w: int
    frame_h: int
    excursions: list = field(default_factory=list)
    context_items: list = field(default_factory=list)
    recovered_frac: float = 0.0
    motion: str = ""

    def summary(self) -> dict:
        return dict(shot=self.shot, measured=len(self.excursions),
                    asserted=sum(1 for i in self.context_items if i.bound and i.text),
                    recovered=round(self.recovered_frac, 3), motion=self.motion)


@dataclass
class Plan:
    shot: int
    mode: str                       # 'causal' | 'interpolating'
    elements: list = field(default_factory=list)

    def by_support(self, s):
        return [e for e in self.elements if e.support == s]

    def label(self) -> int:
        """
        The provenance these pixels earn.

        Measured elements are still invented pixels -- nothing photographed that
        wall -- but they are constrained at both ends by observation, which is a
        stronger claim than a script page. They deserve their own rung when
        something actually renders them; until Tier 3.4 exists, inventing a rung
        for content nothing produces would be labelling a hypothetical. They ride
        with DIRECTED, which is already outside PHOTOGRAPHIC, and the plan keeps
        the distinction visible in the meantime.
        """
        if any(e.support in ("measured", "asserted") for e in self.elements):
            return P.DIRECTED
        return P.GENERATED

    def prompt(self, limit=420) -> str:
        """
        Strongest support first. A model given ten equal clauses weights them
        by wording; given the measured facts first, it treats them as the frame
        to hang the rest on.
        """
        parts = []
        for s in ("measured", "asserted", "inferred"):
            for e in self.by_support(s):
                parts.append(e.describe())
        head = "extend the scene sideways, same place and moment"
        body = "; ".join(p for p in parts if p)
        return (f"{head}. {body}" if body else head)[:limit]

    def explain(self) -> str:
        """The reasoning, in the order it was arrived at. For a human to read."""
        lines = [f"shot {self.shot} ({self.mode})"]
        for e in self.elements:
            lines.append(f"  [{e.support:8s}] {e.describe()}"
                         + (f"  <- {e.because}" if e.because else ""))
        if not self.elements:
            lines.append("  nothing known and nothing claimed")
        return "\n".join(lines)


# ---------------------------------------------------------------- the reasoner

class LocalReasoner:
    """
    Reasons from arithmetic and the context store. No model, no network.

    It is not clever and does not need to be: the useful inferences here are
    geometric. Where is the figure that left. How far out. Which wall. Those
    follow from measurements this repo already produces, and a language model
    asked the same question would have to be TOLD them anyway.

    `ApiReasoner` is the subclass that adds judgement it cannot have -- what a
    fire escape looks like, whether a crowd would still be there.
    """

    def __init__(self, depth_frac=0.22):
        self.depth_frac = depth_frac      # calibrate with offscreen.fit_depth

    def plan(self, brief: Brief, mode="interpolating") -> Plan:
        p = Plan(shot=brief.shot, mode=mode)
        p.elements.extend(self._from_excursions(brief, mode))
        p.elements.extend(self._from_context(brief))
        if not p.elements:
            p.elements.append(Element(
                "continuation", "inferred", "",
                "continue the texture and light of the frame edge",
                because="nothing was measured and nothing was claimed"))
        return p

    def _from_excursions(self, brief: Brief, mode):
        """
        Where the thing that left actually is, frame by frame.

        Interpolating mode uses the observed return, so the figure is placed on
        a path that reaches the edge again exactly when the film says it did.
        Causal mode may only use the exit, and predicts the turn from the depth
        prior -- worse, and the only honest choice when the return has not been
        observed yet.
        """
        out = []
        mid_frames = brief.n_frames
        for ex in brief.excursions:
            vx = abs(float(ex.exit_v[0]))
            if vx < 1e-3:
                continue
            if mode == "interpolating" and ex.entry_frame > ex.exit_frame:
                gap = ex.entry_frame - ex.exit_frame
                out_depth = vx * gap / 2.0            # observed turnaround
                because = (f"left at {vx:.1f}px/frame on f{ex.exit_frame}, "
                           f"returned f{ex.entry_frame}: {gap} frames out")
            else:
                out_depth = self.depth_frac * brief.frame_w
                gap = int(round(2.0 * out_depth / vx))
                because = (f"left at {vx:.1f}px/frame on f{ex.exit_frame}; "
                           f"return not used, depth prior {self.depth_frac:.2f}")

            deepest = out_depth / max(brief.wing_w, 1)
            first = min(ex.exit_frame + 1, mid_frames)
            last = min(ex.exit_frame + gap, mid_frames)
            out.append(Element(
                kind="figure", support="measured", side=ex.side,
                text="the figure that left frame",
                because=because, frames=(first, last),
                depth_frac=float(np.clip(deepest, 0.0, 1.5)),
                y_frac=float(np.clip(ex.exit_y / max(brief.frame_h, 1), 0, 1))))
        return out

    def _from_context(self, brief: Brief):
        out = []
        for i in brief.context_items:
            if not (i.bound and i.text):
                continue
            side = ""
            low = i.text.lower()
            if "left" in low:
                side = "L"
            elif "right" in low:
                side = "R"
            kind = "detail"
            who = i.author or (i.source if i.source != i.kind else "")
            out.append(Element(kind, "asserted", side, i.text.strip()[:120],
                               because=f"{i.kind} from {who}" if who else i.kind))
        return out


class ApiReasoner(LocalReasoner):
    """
    Adds a model's judgement on top of the arithmetic.

    The local plan goes UP with the frame, not instead of it: the model is asked
    to add what it can see and what it knows about the world, and explicitly not
    to contradict the measured elements, because those are observations and it
    is guessing. Whatever comes back is merged at support='asserted' -- a model
    is another party making claims, held to the same standard as a script page.

    UNTESTED against a live service. `call` is left abstract on purpose: the
    request shape differs per provider and guessing at one produces code that
    looks finished and has never run.
    """

    def __init__(self, call=None, depth_frac=0.22):
        super().__init__(depth_frac)
        self.call = call            # call(prompt, image) -> list[str]

    def plan(self, brief: Brief, mode="interpolating", image=None) -> Plan:
        p = super().plan(brief, mode)
        if self.call is None:
            return p
        measured = [e.describe() for e in p.by_support("measured")]
        ask = ("Describe what is immediately off-frame to the left and right, "
               "in this same place and moment. These are observed and must not "
               "be contradicted: " + ("; ".join(measured) or "none") + ". "
               "Context: " + cx.build_prompt(brief.context_items))
        try:
            claims = self.call(ask, image) or []
        except Exception as exc:                       # a dead API is not a crash
            p.elements.append(Element("detail", "inferred", "",
                                      "continue the texture of the frame edge",
                                      because=f"model unavailable: {exc}"))
            return p
        for c in claims:
            c = str(c).strip()
            if c:
                p.elements.append(Element("detail", "asserted", "", c[:120],
                                          because="vision model"))
        return p


def brief_for(shot_record, n_frames, wing_w, frame_w, frame_h,
              excursions=(), context_items=(), recovered_frac=0.0) -> Brief:
    return Brief(shot=int(shot_record.get("shot", 0)), n_frames=n_frames,
                 wing_w=wing_w, frame_w=frame_w, frame_h=frame_h,
                 excursions=list(excursions), context_items=list(context_items),
                 recovered_frac=recovered_frac,
                 motion=str(shot_record.get("motion", "")))
