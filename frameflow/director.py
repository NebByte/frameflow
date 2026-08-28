"""
director — goal-driven wing completion.

The ladder in agent.py is a fixed order. This replaces it with a planner that
decides what to do next from the state it is actually in.

    goal      declarative targets + a budget
    scout     discovers sources at RUNTIME (other cuts, same-location setups,
              licensed asset providers) instead of being handed a corpus
    planner   picks the action with the best expected gain per unit cost,
              executes, measures the ACTUAL gain, updates its estimate, re-plans
    stop      target met, budget spent, or nothing left with positive expected
              yield -- and it reports which of the three happened

WHY EXPECTED-YIELD AND NOT A LADDER
-----------------------------------
Measured on real footage: `same_take` fires on 15 of 75 shots and, when the
framing scale is within 2% of 1.0, yields nothing at all -- 10 of those 15. A
fixed ladder pays that cost every shot forever. A planner that updates on
observed yield stops calling it within a few shots and spends the budget on
rungs that are actually paying.

WHAT PLANNING CANNOT DO
-----------------------
It searches harder over sources that exist. It does not create photons. On a
locked-off shot with no other setup of that location anywhere in the corpus,
every branch terminates in GENERATED, and the honest outcome is a reported
shortfall rather than a filled wing. The planner is built to say so.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import agent as ag
from . import fill as fence_mod
# ------------------------------------------------------------------- goals

@dataclass
class Goal:
    """Declarative targets. The planner works toward these, not a script."""
    real_same_camera: float = 0.60      # PRIMARY+RECOVERED+DONATED fraction of wing
    photographic: float = 0.85          # ...plus RETRIEVED
    max_generated: float = 0.15         # ...of anything not of this place
    min_confidence_for_generation: float = 0.05

    # budget
    max_actions: int = 12
    max_seconds: float = 120.0
    max_cost_units: float = 10.0        # abstract; API calls cost more than local

    def satisfied(self, report) -> bool:
        # `not_this_place` is generated + referenced. Checking `generated`
        # alone would let a wing of 85% unverified licensed stock pass a 15%
        # invention budget, which is the exact failure the REFERENCED rung was
        # created to make visible.
        return (report["real_same_camera"] >= self.real_same_camera
                and report["photographic"] >= self.photographic
                and report["not_this_place"] <= self.max_generated)

    def gaps(self, report) -> dict:
        return dict(
            real=max(0.0, self.real_same_camera - report["real_same_camera"]),
            photographic=max(0.0, self.photographic - report["photographic"]),
            generated=max(0.0, report["not_this_place"] - self.max_generated),
        )


# ------------------------------------------------------------------- actions

@dataclass
class Action:
    name: str
    tool: ag.Tool
    cost: float = 1.0
    provenance: int = ag.GENERATED
    prior_yield: float = 0.25           # expected fraction of remaining hole filled
    observations: list = field(default_factory=list)

    @property
    def expected_yield(self) -> float:
        """Beta-ish online estimate: prior, then dominated by observation."""
        if not self.observations:
            return self.prior_yield
        obs = float(np.mean(self.observations))
        n = len(self.observations)
        w = n / (n + 2.0)
        return (1 - w) * self.prior_yield + w * obs

    def score(self, gaps) -> float:
        """
        Expected gain per unit cost, weighted by which gap this action can close.
        An action producing GENERATED cannot close the `real` gap at all, so it
        scores zero while that gap is what dominates.
        """
        if self.provenance in ag.REAL_LEVELS:
            relevance = gaps["real"] + gaps["photographic"]
        elif self.provenance == ag.RETRIEVED:
            relevance = gaps["photographic"]
        else:
            # REFERENCED and GENERATED both close nothing. They are ranked
            # against each other in the fallback phase, not here.
            relevance = 0.0 if (gaps["real"] > 0 or gaps["photographic"] > 0) else 1.0
        return (self.expected_yield * relevance) / max(self.cost, 1e-6)

    def observe(self, filled_fraction):
        self.observations.append(float(filled_fraction))


# ------------------------------------------------------------------- scouting

class ResourceScout:
    """
    Finds sources at runtime rather than receiving them.

    Each probe returns Actions to add to the planner's option set. Probes are
    cheap and are themselves budgeted -- scouting that never yields an action
    gets deprioritised the same way a failing action does.
    """

    def __init__(self, policy=None):
        self.policy = policy or ag.SourcePolicy()
        self.probe_stats = {}

    def probe_corpus(self, ctx, corpus_finder: Optional[Callable] = None):
        """Look for other cuts containing this take."""
        if corpus_finder is None:
            return []
        found = corpus_finder(ctx) or []
        if not found:
            return []
        from . import crosscut as cc
        return [Action("same_take", ag.SameTakeTool(found, cc.TakeMatcher()),
                       cost=2.0, provenance=ag.DONATED, prior_yield=0.30)]

    def probe_scene(self, ctx, scene_finder: Optional[Callable] = None,
                    backend=None, scene_models=None):
        """Look for other setups of this location in the film itself."""
        if scene_finder is None or ctx.scene_id is None:
            return []
        setups = scene_finder(ctx.scene_id) or []
        if not setups:
            return []
        return [Action("same_location",
                       ag.SameLocationTool({ctx.scene_id: setups}, backend,
                                           scene_models=scene_models),
                       cost=4.0, provenance=ag.RETRIEVED, prior_yield=0.55)]

    def probe_external(self, ctx, fetcher: Optional[Callable] = None):
        """Licensed external material. Default-deny is enforced in the tool."""
        if fetcher is None:
            return []
        return [Action("external_reference",
                       ag.ExternalReferenceTool(fetcher, self.policy),
                       cost=6.0, provenance=ag.REFERENCED, prior_yield=0.20)]

    def scout(self, ctx, corpus_finder=None, scene_finder=None, fetcher=None,
              backend=None, scene_models=None):
        acts = []
        acts += self.probe_corpus(ctx, corpus_finder)
        acts += self.probe_scene(ctx, scene_finder, backend, scene_models)
        acts += self.probe_external(ctx, fetcher)
        return acts


# ------------------------------------------------------------------- director

@dataclass
class Step:
    action: str
    gain: float
    cost: float
    note: str
    report_after: dict


class Director:
    """
    Plan -> act -> measure -> re-plan, against a Goal and a budget.
    Everything still composites through the fence.
    """

    def __init__(self, goal=None, scout=None, provider=None, policy=None):
        self.goal = goal or Goal()
        self.scout = scout or ResourceScout(policy)
        self.provider = provider
        self.policy = policy or ag.SourcePolicy()

    def _base_actions(self):
        return [
            Action("generate", ag.GenerateTool(self.provider),
                   cost=3.0, provenance=ag.GENERATED, prior_yield=0.95),
        ]

    def run(self, canvas, filled, tmap, wing_w, frames=None, scene_id=None,
            fps=24.0, corpus_finder=None, scene_finder=None, fetcher=None,
            backend=None, verbose=True, shot_id=None, scene_models=None,
            allow_provenance=None):

        ctx = ag.Context(canvas.copy(), filled.copy(), tmap, wing_w,
                         frames or [], scene_id, shot_id=shot_id)
        h, cw = filled.shape
        w = cw - 2 * wing_w

        prov = np.full(filled.shape, ag.GENERATED, np.uint8)
        prov[filled] = ag.RECOVERED
        centre = np.zeros(filled.shape, bool)
        centre[:, wing_w:wing_w + w] = True
        prov[centre & filled] = ag.PRIMARY
        protected = filled.copy()

        actions = self._base_actions()
        actions += self.scout.scout(ctx, corpus_finder, scene_finder, fetcher,
                                    backend, scene_models)
        if allow_provenance is not None:
            # A caller may restrict the planner to rungs whose evidence does not
            # come from this shot. A gate-refused shot uses this: its own
            # recovery is what the gate rejected, but a take verified against
            # another cut was never the gate's business. Note this filters out
            # `generate` too -- borrowing is not an excuse to invent.
            actions = [a for a in actions if a.provenance in allow_provenance]

        trace, spent, t0 = [], 0.0, time.time()
        report = ag.WingAgent.report(prov, wing_w, w)
        stop = "target met"

        for step in range(self.goal.max_actions):
            if self.goal.satisfied(report):
                stop = "target met"
                break
            if spent >= self.goal.max_cost_units:
                stop = "budget exhausted"
                break
            if time.time() - t0 > self.goal.max_seconds:
                stop = "time exhausted"
                break
            if not ctx.filled.any() or (~ctx.filled).sum() == 0:
                stop = "nothing left to fill"
                break

            gaps = self.goal.gaps(report)

            # PHASE 1: photography OF THIS PLACE -- the only thing that can
            # close a gap. Membership is PHOTOGRAPHIC, not `!= GENERATED`:
            # external reference material is real photons but unverified, so it
            # belongs in the fallback with generation, not here.
            real_acts = [a for a in actions
                         if a.provenance not in ag.NOT_THIS_PLACE]
            ranked = sorted(real_acts, key=lambda a: -a.score(gaps))
            ranked = [a for a in ranked if a.score(gaps) > 1e-6]

            # PHASE 2: only once no real source has positive expected yield do
            # we fill the rest. Ordering this last is the point of the system;
            # scoring it zero outright was a bug -- it meant the wing was never
            # completed at all and the shortfall was reported against an empty
            # canvas rather than a finished one.
            #
            # Within the fallback, rung order still decides: a licensed plate
            # is preferred over an invented one. Neither counts toward the
            # number, so that preference is about which wing looks better, not
            # about which is more true -- which is exactly the kind of choice
            # it is safe to make once the metric can no longer be moved by it.
            phase = "real"
            if not ranked:
                fill_acts = [a for a in actions
                             if a.provenance in ag.NOT_THIS_PLACE]
                ranked = sorted(
                    fill_acts,
                    key=lambda a: (a.provenance,
                                   -(a.expected_yield / max(a.cost, 1e-6))))
                ranked = [a for a in ranked if a.expected_yield > 1e-6]
                phase = "fallback-fill"

            if not ranked:
                stop = "no action with positive expected yield"
                break

            act = ranked[0]
            ctx.hole = ~ctx.filled
            hole_before = int(ctx.hole.sum())
            ctx.confidence = fence_mod.confidence_map(ctx.canvas, ctx.filled,
                                                      ctx.tmap, fps)

            before = ctx.canvas[protected].copy()
            ro = ctx.canvas.view()
            ro.flags.writeable = False
            guarded = ag.Context(ro, ctx.filled, ctx.tmap, ctx.wing_w,
                                 ctx.frames, ctx.scene_id, ctx.hole,
                                 ctx.confidence, ctx.shot_id)
            try:
                res = act.tool.run(guarded)
            except ValueError as e:
                if "read-only" in str(e).lower():
                    raise fence_mod.FenceViolation(
                        f"{act.name} wrote the canvas in place") from e
                raise

            spent += act.cost
            gain = 0.0

            if res.pixels is not None:
                px = res.pixels
                if px.shape[:2] != ctx.canvas.shape[:2]:
                    import cv2
                    px = cv2.resize(px, (ctx.canvas.shape[1], ctx.canvas.shape[0]))
                new = ctx.hole.copy()
                if res.mask is not None:
                    new &= res.mask
                if new.any():
                    ctx.canvas[new] = px[new]
                    prov[new] = res.provenance
                    ctx.filled |= new
                    gain = float(new.sum()) / max(hole_before, 1)

            if not np.array_equal(ctx.canvas[protected], before):
                raise fence_mod.FenceViolation(
                    f"{act.name} modified protected pixels")

            act.observe(gain)
            report = ag.WingAgent.report(prov, wing_w, w)
            trace.append(Step(f"{act.name}[{phase}]", round(gain, 4), act.cost,
                              res.note, dict(report)))
            if verbose:
                print(f"  [{step}] {act.name:20s} gain {gain*100:5.1f}%  "
                      f"cost {act.cost:.1f}  real {report['real_same_camera']*100:5.1f}% "
                      f"photo {report['photographic']*100:5.1f}%  :: {res.note}")

            # a STRUCTURAL no-op is dropped at once -- retrying cannot change
            # a scale term or a missing backend. Only stochastic misses need a
            # second observation.
            structural = any(k in (res.note or "").lower() for k in
                             ("framing identical", "needs gaussianbackend",
                              "no recorded licence", "unavailable", "none supplied"))
            if gain < 0.01 and structural:
                actions = [a for a in actions if a is not act]
                if verbose:
                    print(f"       dropped {act.name}: structural no-op")
            elif len(act.observations) >= 2 and max(act.observations) < 0.01:
                actions = [a for a in actions if a is not act]
                if verbose:
                    print(f"       dropped {act.name}: no yield in "
                          f"{len(act.observations)} attempts")

        gaps = self.goal.gaps(report)
        return dict(
            canvas=ctx.canvas, provenance=prov, report=report,
            stop_reason=stop, cost_spent=spent,
            seconds=round(time.time() - t0, 2),
            shortfall={k: round(v, 4) for k, v in gaps.items() if v > 0},
            trace=trace,
        )
