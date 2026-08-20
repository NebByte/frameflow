# Roadmap

What is built, what is stubbed, and what each remaining step costs. Every number
here was measured on this machine, not estimated. Where something is unverified,
it says so.

The one invariant across all tiers: **no step may make the coverage number less
true.** A tier that fills more wall while making "94% real" mean less has moved
backwards. Each tier below states how it protects that.

---

## Where things stand

| | |
|---|---|
| Runs on CPU today | shot detection, the four metrics, two backends, the fence, the six-rung ladder, the planner, the licence gate, mirror/inpaint generation, all of `splat.py` except the fit and the render |
| Needs a GPU, path verified on one | `GaussianBackend`, `SameLocationTool`, `DiffusionGenerator` |
| Needs credentials | `HostedGenerator` (no GPU) |
| Never yet fired on real footage | `DONATED`, `RETRIEVED` |
| Fires, but outside the headline number | `REFERENCED` |

Measured on a 78-shot, 2:00 trailer (`videoplayback.mp4`, 640×360 letterboxed to
2.36:1):

- 78 shots, median length **30 frames (1.2 s)**
- **47 of 78 LOCKED** — a static camera has no periphery to recover, by definition
- 5 shots cleared on measurement; mean effective coverage **4.59%**, real wing **3.36%**
- geometry, where it ran at all: 31 shots scored, median **27.6 dB**, range 19.5–40.0
- wall depth **0.00 m** — scope aspect, vertical constraint binds (`walls.py:75`)
- per-frame cost **0.2–0.9 s** at 480 px, varying with machine load

The shape of that result matters: registration is *not* the bottleneck. The
geometry engine scores well wherever it gets to run. The bottleneck is that
fast-cut, locked-off footage never photographs the periphery in the first place.

---

## Tier 0 — done

- `demo.py` — one command, synthetic footage with ground truth, report in the browser
- `serve.py` — drop a clip in the browser, LAN-shareable, queue, licence-free
- `MirrorGenerator` + `--wings-on-dark` — light the walls on shots the gate refuses
- `screenx_render.py:73` `synth_wings()` — invents a wing without touching the metric

**Verified:** turning wings on for 73 of 78 shots left `mean_real_wing` at 3.36%,
bit-identical to the run with dark walls.

## Tier 1 — done

| # | change | file |
|---|---|---|
| 3 | one provenance enum, imported by both fence and ladder | `provenance.py` |
| 1 | film index, so the scout has something to find | `filmindex.py` |
| 2 | external material suppliers behind the licence gate | `fetchers.py` |
| 4 | hosted outpainting adapter, no GPU | `fill.py:112` |

**The bug that mattered.** `Director.run()` accepted `corpus_finder`,
`scene_finder` and `fetcher`; the render pipeline passed none of them. Every
probe returned `[]`, the option set collapsed to `generate`, and two rungs of a
five-rung ladder had never once been attempted. The planner now reports
`external_reference[real], same_take[real]` — it tries photography before it
invents.

**Calibration, measured not guessed.** A first scene threshold of 0.55 grouped
nothing: actual pairwise signature similarity ran mean 0.026, p95 0.195, max
0.316. Default is 0.30, and the film now groups into 61 scenes with 9 holding
more than one setup. Candidates are shortlisted to the closest 6 rather than
offered whole, which turns an N² ORB cost back into N×6.

**Untested:** `OpenverseFetcher` (no network from the build shell) and
`HostedGenerator` (no credentials). Both are written from the endpoint contract,
not from a round trip. Expect to adjust `encode`/`decode` per provider.

---

## Decided — the `REFERENCED` rung

`ExternalReferenceTool` composited any *licensed* image and labelled it
`RETRIEVED`, which counted as photographic, with **no check that the asset
depicts this location**. With a dummy library wired in, `mean_real_wing` moved
3.36% → 5.34% on the strength of a flat colour plate.

Resolved by option (2): a sixth rung, `REFERENCED`, below `RETRIEVED` and
**outside `PHOTOGRAPHIC`**. Real photons, wrong place. `GENERATED` moved 4 → 5.

| file | change |
|---|---|
| `provenance.py` | `REFERENCED = 4`, `GENERATED = 5`, and `NOT_THIS_PLACE = (REFERENCED, GENERATED)` |
| `agent.py` | tool writes `REFERENCED`; `report()` gains `referenced` and `not_this_place` |
| `director.py` | phase 1 is now `provenance not in NOT_THIS_PLACE`, was `!= GENERATED` |
| `director.py` | `Goal.satisfied` budgets `not_this_place`, not `generated` |
| `walls.py` | reveal tints `NOT_THIS_PLACE` |

**Verified.** A licensed plate filling 87.5% of a wing leaves `photographic` at
0.125, equal to `real_same_camera` — the plate moves the report's shape and not
its headline. On 30 shots of the trailer, `mean_real_wing` is **3.26% with the
dummy library wired in and 3.26% with it removed** — identical, which is the
property that matters. `mean_effective` (0.049) and `wings_on` (2) match too, so
the library changed what the planner tried and nothing about what it claims.

**Two things this decision bought that (1) and (3) would not have.**

- **A promotion path.** This is not a rejection of verification. When Tier 2.1
  lands, an asset that registers to the shot is promoted `REFERENCED` →
  `RETRIEVED` and earns its way into the number. Option (1) was unexecutable
  today — a production still is another setup of the location, so verifying it
  is the same cross-setup problem measured at 1 in 400.
- **The tool stays alive.** Option (3) would have made it dead code: in the
  fallback phase the planner ranks on `expected_yield / cost`, where generate
  scores 0.95/3.0 against external's 0.20/6.0 and wins every time, forever. The
  fallback now orders by rung first, so a licensed plate is preferred over an
  invented one. That preference is about which wing *looks* better, not which is
  more true — the kind of choice it is safe to make once the metric can no
  longer be moved by it.

**One bug this surfaced.** `walls.py:214` read `provenance >= 4`, the one place
in the repo that hardcoded a rung value. Under the old numbering that meant
GENERATED; inserting a rung silently changed its meaning. It now reads
`NOT_THIS_PLACE`, which is the behaviour you want anyway — a reviewer asking
"what am I showing that was not filmed here" needs unverified stock lit magenta
alongside invention.

---

## Tier 2 — done

**Goal:** make `RETRIEVED` fire. A locked-off close-up has no periphery of its
own, but the scene's master wide photographed that same wall an hour earlier.

| step | file | state |
|---|---|---|
| 2.1 | `splat.py`, `backends.py` | `GaussianBackend` fits gsplat Gaussians, renders a widened frustum |
| 2.2 | `agent.py` | `SameLocationTool` renders this shot's pose from a whole-location reconstruction |
| 2.3 | `fill.py` | `DiffusionGenerator` outpaints, strength driven per-band by `confidence` |
| 2.4 | `remote.py` | job-file protocol so any of it runs on a rented GPU |
| 2.5 | `gating.py` | `leave_one_out_3d` — the hold-out a 3D backend can actually take |
| 2.6 | `sfm.py` | builds the COLMAP reconstructions the whole tier depends on |

**Verified.** `verify_gpu.py` was run on a CUDA host and the path worked —
reported by the author, 2026-08-19. The specific numbers are **not recorded
here**, because at the time the script printed to stdout and wrote nothing, and
a Colab runtime takes its stdout with it when it recycles.

That gap is now closed: the script writes `verify_gpu_result.json` beside itself
— every check, its number, the GPU that produced it, the gsplat and torch
versions — and `colab_verify.ipynb` downloads it. **Re-run it once and commit
that file**, so this section can carry measurements like every other section
instead of an assurance.

```
python verify_gpu.py          # checks the CUDA path against known truth
```

Free Colab is enough for it — `colab_verify.ipynb` is the notebook. Set the
runtime to T4, upload the folder as a zip, run four cells. No SSH is involved,
which sidesteps the free-tier restriction entirely: the toolkit is an ordinary
script and never needs a shell. The slow step is `pip install gsplat`, which
compiles CUDA kernels and has to match whatever torch Colab shipped that day.

Colab is fine for *verification* and poor for a *film*: runtimes recycle, the
filesystem is ephemeral, and COLMAP over 78 shots is hours. Rent a real host for
that — `remote.py` packages the work as job files precisely so a recycled
runtime costs one shot instead of the run.

98 CPU assertions pass here (`test_splat.py` 52, `test_tier2.py` 46), including
triangulation scored against a synthetic scene of known depth — recovered 3.84
against a true 4.0. `fit_splats` and `render_widened` have never executed **on this machine**,
which has no GPU; they have been run once on a CUDA host. `verify_gpu.py` is the one command that closes
that gap, and it checks placement, not just coverage: it re-renders the scene's
own ground truth at the widened intrinsics and scores the recovered wings
against it, because coverage alone cannot tell you the splats are in the right
place.

### 2.5, and why nothing worked without it

`gating.leave_one_out` reconstructs a held-out frame by warping its neighbours
in. `GaussianBackend.warps()` raises — correctly, because a 3D mapping runs
through depth and collapsing it to a homography would validate a different model
from the one making the pixels. But `leave_one_out` caught that exception and
returned **0.0 dB**, and `decide` gates anything under 20 dB to OFF.

Measured, before the fix: identical frames scored **30.9 dB → FULL** on mosaic
and **0.0 dB → OFF** on Gaussian. Every 3D shot was gated off before a single
recovered pixel was looked at, and the failure was indistinguishable from bad
geometry. `leave_one_out_3d` fits without frame *i*, re-renders its pose from
splats that never saw it, and scores the same outer band through the same
`band_psnr` — one formula, so the 20 dB threshold keeps one meaning.

It renders through `render_raw`, not `render_widened`, and the distinction is
the point: the latter pastes the real frame back into the centre, and scoring a
reconstruction with the ground truth pasted into it is how this repo once
recorded 138 dB on a clip whose wings were really 12 dB.

### 2.6, one reconstruction per scene

`sfm.build_film` solves each multi-setup location ONCE, over every setup
together, and records a `manifest.json` of `(shot, frame)` per view. Solving
per-shot instead would re-derive the same location repeatedly in incomparable
coordinate frames. Filenames are `s{shot:04d}_f{frame:04d}.png` so filename order
— which is what `poses_from_colmap` sorts by — equals `(shot, frame)` order by
construction rather than by hope.

Intrinsics are now **per view**. Two setups of one location are two lenses, and
reading only COLMAP's first camera would render the wide setup through the
close-up's focal length: a wrong reconstruction that still looks like one.

### Four refusals, each guarding the metric

- **Untrusted poses cannot render.** `poses_from_essential` recovers translation
  only up to scale, so a chain of them can be internally consistent and globally
  wrong. `PoseSet.trustworthy` is COLMAP-only; the backend raises rather than
  labelling those pixels `RETRIEVED`.
- **Partial reconstructions are refused.** `SceneModel.usable` needs 80% of views
  registered. The frames that fail to register are exactly the geometrically
  ambiguous ones, so rendering the rest covers the easy setups while the report
  reads as though it covered the scene.
- **No densification.** A split Gaussian has no observation time of its own, and
  inventing one puts a made-up staleness behind a real-looking number. Sharpness
  is recoverable; exact `first_obs` is not.
- **`warps()` raises** rather than faking a homography. See 2.5.

**The alpha channel really is free coverage.** `render_widened` rides `first_obs`
as a fourth colour channel, so one rasterisation returns RGB, the coverage mask,
and the staleness map. Caveat recorded in `alpha_to_masks`: a rasteriser blends
every contributing Gaussian, so this tmap is an alpha-weighted MEAN
first-observation where the mosaic backend's is a true minimum. Conservative, but
a different quantity — do not pool them in one statistic.

**The widening is exact.** fx held, cx shifted by exactly `wing_w`. Verified to
5.7e-14 px over 200 random points, which matters because the fence compares the
protected centre byte-for-byte.

**Hardware.** 24 GB is the practical floor; 48–80 GB for 1080p. **Rent before
buying** — a shot is an independent unit of work, which is what `remote.py`
packages. On Colab specifically: SSH is disallowed on the free tier and permitted
on paid plans, but the job-file protocol needs no shell at all, which is the
better Colab path either way. See `remote.py`'s docstring.

**How you would know it worked on real footage:** `wings_on` rises on LOCKED
shots sharing a scene with a wider setup, and those pixels come back
`RETRIEVED`. If `mean_real_wing` moves without `retrieved` appearing, something
is mislabelled.

---

## Tier 3 — the vision model

**Goal:** wings that know what is out there, not just what colour it is.

The worked example: Sentry fights Ghost. Ghost exits frame left, fights off
camera, is thrown back into frame 40 frames later. The wall should show what she
plausibly did out there.

**The property that makes this tractable.** You know *both ends*. Her exit state
(position, velocity, pose) and her re-entry state are both photographed. Off-screen
motion is therefore **interpolation between two observations**, not free
extrapolation — far more constrained, and *checkable*: hold out the re-entry,
predict it from the exit alone, score the prediction. Everything else in this
toolkit earns trust by being verifiable. This can too, and it should be built
that way from the first commit.

| step | work | state |
|---|---|---|
| 3.1 | entity tracking + re-identification across the frame boundary | **done** — `offscreen.py` |
| 3.2 | **verification harness** — hold out re-entry, score prediction | **done** — `offscreen.score`, `score_calibrated` |
| 3.3 | off-screen state estimation between the two observations | **done** — `reasoning.LocalReasoner`, scored by 3.2 |
| 3.4 | conditioned rendering onto the wing, anchored to the recovered plate | **wired** — `wavespeed.py`, shot-level; the wire is untested |
| 3.5 | context in, prompt out, driven by a model | **done offline** — `context.py`, `reasoning.py`; the API call itself is untested |
| 3.6 | any file as context, and a human in the loop | **done** — `context.py`, `DirectionStore`, `DIRECTED` |

### 3.1 and 3.2 — done, 27 assertions

`offscreen.py` finds excursions and scores predictions of them. **It generates
nothing**, which is the order the section below insisted on and worth keeping.

Camera motion comes out through the homography chain this repo already computes;
each frame is differenced against a median plate built from its own neighbours;
what survives moved differently from the room. No detection model and no weights
— identity is a colour histogram plus size, enough to re-associate one figure
across a 40-frame absence *inside one shot* and nowhere near enough across a cut.

The fixture in `test_offscreen.py` stages excursions with exact truth, so every
claim is checked against a number. `test_no_leakage` reads the source of every
shipped predictor and fails if it references a re-entry field — a harness that
quietly hands over the answer would report excellent accuracy forever.

**Four things the harness caught, which is the entire reason it was built first.**

- **Edge clipping under-reports exit speed.** A disc travelling 9.0 px/frame
  measured 7.29: once a body starts crossing the edge only the part still inside
  is detected, so its centroid slows while the body does not. Every predictor
  divides by that speed, so the bias lands straight on the predicted return frame.
  Fixed by measuring speed before any clipping (`Track.exit_velocity`) — 9.05.
- **Two errors were cancelling.** Fixing the speed alone made the score *worse*,
  9 frames to 12. The speed bias had been hiding a wrong depth prior. This is
  what compensating errors look like from outside, and it is why a single
  aggregate score is not enough on its own.
- **The depth prior was a statement about the wrong thing.** `depth_frac` = 0.22,
  one wing width, because that is the strip this project fills — a fact about the
  SCREEN, not about how far actors go. Measured on ground truth: 0.33–0.37.
  `fit_depth` now estimates it from observed excursions, and `score_calibrated`
  fits it leave-one-out so it never scores on data it was fitted to. **Median
  return-frame error: 14 frames on the shipped prior, 1 frame calibrated.**
- **Nearest-to-last association braids tracks.** It quietly prefers whichever
  candidate moves SLOWEST, since a stationary object is always nearer than the
  one this track is chasing. With `max_move` at 0.12 diag — 49 px/frame on a
  360×200 frame, between objects 56 px apart, colours correlating at 0.78 inside
  a 0.55 gate — two of three staged excursions were lost. Now matched against
  `last + velocity*step` with `max_move` at 0.05.

### 3.3, 3.5, 3.6 — context, and reasoning over it

Two features that arrived as separate asks and turned out to be one:

> take any file and use it as context
> let a person pause a shot and say what needs to be there

A script page saying "fire escape, camera left" and a person typing the same
thing are both assertions about a place, from sources that may well know, with
no photons behind them. Same store, same binding, same generator, and one new
rung: **`DIRECTED`**, above `GENERATED` because something constrained the pixels,
below `REFERENCED` because that at least involved a camera, and outside
`PHOTOGRAPHIC` because a claim is not evidence. Verified: a wing 100% directed
reports 0% real and 0% photographic.

**Binding is what separates a useful context file from a decorative one.**

| | binds to a shot |
|---|---|
| `.srt` / `.vtt` | exactly, by timecode, free, no model |
| screenplay | not at all — ordered scenes, no timecodes |
| stills / plates | only via a sidecar |
| a human note | to the exact shot the person was looking at |

A subtitle track is the cheapest useful vision model in this repo. It tells you
there is a helicopter in the scene because somebody shouts about the helicopter.

**Two refusals.** Unbound screenplay does not make pixels `DIRECTED` — text that
never described *this* wall would turn the rung into a statement about the run
rather than the shot. And scene *k* is not mapped onto shot *k*: that is wrong on
any film ever cut, and a confident wrong binding puts the warehouse on the
kitchen wall while the report says a human asked for it.

**`reasoning.py` is the thinking step**, and it emits a plan rather than a
prompt, because a prompt is unreviewable — once the text is in the model you
cannot ask which part of the output was evidence. Every element carries its
support:

- **measured** — arithmetic over this footage. *"left at 9.0 px/frame on f27,
  returned f55: 28 frames out."* Not a guess.
- **asserted** — a script, a subtitle, a person, or a vision model. A model is
  just another party making claims and is held to the same standard.
- **inferred** — nothing supports it; the wall needs to be something.

**Causal vs interpolating is the whole trick.** Scoring a predictor may use only
the exit, or the harness is grading a model that already saw the answer.
Rendering may use both ends, because the film shows the figure leaving *and*
coming back — its path is an interpolation between two observations. The two
modes are separate and provably differ: on the fixture, 1.01 wing-widths of
predicted depth against 1.50 measured.

**And the finding that falls out of it.** On the ground-truth fixture the figure
travels **1.50 wing-widths** past the frame edge. The wall is one wing-width
deep. So for most of its absence the thing you now know the position of is
*beyond the projectable wall entirely*, and no amount of generation should put it
there. Knowing where it went does not mean there is anywhere to show it — the
same wall-depth ceiling that `walls.py:75` describes, arrived at from the other
direction.

**What it cannot do yet.** Re-identification across a cut, anything about what an
object *is*, and any excursion where the returning thing looks different from the
thing that left (costume change, transformation, a different actor). Those are
3.5's vision model, and the interfaces are shaped so it drops in without touching
the harness.

3.5 is separable and cheap — it upgrades the external rung on its own, without
any of 3.1–3.4. The hook already exists and its current default is deliberately
honest about being weak.

**Order matters here.** 3.1 and 3.2 are built; 3.3 and 3.4 come after. A correctly
inferred character placed onto a geometrically wrong wall is worse than a dark
wall, because it looks convincing. Only run this on shots whose wings are already
sound.

**A new provenance question.** Interpolated off-screen content is invented pixels
constrained by observed evidence at both ends. That is not the same as free
generation, and it is not photography. If it earns a rung, that rung sits between
`RETRIEVED` and `GENERATED` and stays outside `PHOTOGRAPHIC`. Do not let it into
the real fraction on the grounds that it was "well constrained".

### 3.4 — the generator slot, and the contract it needed

`select_provider` picks **wavespeed-outpainter**: the only entry in
`agent.REGISTRY` that is both hosted and `conditions_on_known`. Everything
scoring higher in that table -- Kling, Runway, Luma -- is unanchored, which makes
it better at video and useless here. Capability and fitness are inversely
correlated in this slot, which is why the selector weights anchoring above
fidelity.

**A structural gap had to close first.** `fill`'s generators take one frame:
`(canvas, hole, confidence) -> frame`. That is right for `MirrorGenerator`, which
is deterministic, and wrong for anything temporal -- call a diffusion model once
per frame and each call is independent, so the invented wall is re-imagined 24
times a second and crawls. A flickering side wall in peripheral vision is worse
than a dark one.

So a generator may now also offer:

    generate_shot(frames, wing_w, prompt) -> widened frames

One submission, one coherent result. `process_shot` prefers it when present and
falls back per frame otherwise. **The caller still composites every frame
through its own hole**, so a model that returns a re-imagined centre cannot
corrupt the metric -- only waste the call. There is a test that hands the
pipeline a generator which paints the entire canvas and asserts the filmed
centre comes back byte-identical.

**Both API halves are wired, neither wire is tested.** There is no outbound
network on the build machine. `gemini.py` (vision) and `wavespeed.py` (video) are
covered by 57 assertions through injected transports -- auth, request shape,
polling through pending states, failure, timeout, mp4 round trip, the fence --
and the endpoint paths and field names are constructor arguments rather than
constants because they are the part written from a documented pattern instead of
a round trip. Check them against the current API before the first real call.

**Provider split:** Gemini serves the vision half (`ApiReasoner.call`), WaveSpeed
the generator half. A vision model's claims arrive as `asserted`, never
`measured` -- it is another party making claims about a place it was not present
at, held to the same standard as a script page.

---

## What will not work, so nobody spends a month on it

- **Cross-setup homography.** 1 of 400. Not a tuning problem. Needs 3D.
- **Same-take matching inside one film.** A take usually appears once. `crosscut.py` is for a corpus spanning two *cuts* of a picture — trailer plus TV spot — where it measured 100% precision, 0 false positives.
- **More coverage as a goal.** `LayeredBackend` reported 87.8% against mosaic's 47.9% and both scored ~12 dB against truth. Nearly double the coverage, zero extra correctness.
- **Planning your way out of a locked-off shot.** `director.py:25`: it searches harder over sources that exist; it does not create photons. On a shot with no other setup anywhere in the corpus, every branch terminates in `GENERATED`. The Marvel title card stays invented at every tier.
- **Scope footage.** At 2.36:1 the vertical constraint puts `z_near` past the viewer, and wall depth is 0.00 m regardless of coverage. Open-matte or IMAX 1.90 differ from scope in *height* — the axis that binds. Format choice is a bigger lever here than any backend.
