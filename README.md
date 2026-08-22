# wingcoverage

**Measure how much of a ScreenX-style side wall is recovered real footage — before generating anything.**

Every camera move already filmed the periphery; cropping threw it away. This
tool gets it back and, more importantly, reports *how much* it got back and
*how good* those pixels are. Existing outpainting work generates the sides and
hopes. This says when not to.

## Run

```
python3 make_test_clip.py                  # synthetic clips with ground truth
python3 wingcoverage.py clip.mp4 -o out/   # coverage.csv + summary.json + previews
python3 validate.py                        # score recovered pixels vs truth
```

CPU only. OpenCV + numpy.

## What it does per shot

1. **Shot detection** — HSV histogram correlation.
2. **Motion classification** — `LOCKED` / `ROTATION` / `PARALLAX`, via a two-layer
   homography test: fit one H by RANSAC, then ask whether the *outliers* are
   themselves a second coherent H that maps the frame somewhere different.
   One plane → outliers are noise. Parallax → two real layers.
3. **Registration** — anchor frames every 5, direct match to nearest anchor.
   Bounds drift to n/5 compositions instead of n.
4. **Propagation** — for each frame, fill the extended canvas from the
   *nearest-in-time* frame that saw each pixel.
5. **Metrics** — coverage, effective (quality-weighted) coverage, staleness.
6. **Gating** — hysteresis + minimum run length, so wings don't strobe.

`PARALLAX` and `LOCKED` shots are **refused**, not approximated. A homography
quietly applied to a parallax shot produces confident garbage. Those shots are
where the 3DGS backend goes.

## Measured results (synthetic, exact ground truth)

| clip | classified | backend | wings on |
|---|---|---|---|
| pure pan, one plane | ROTATION | mosaic | 85.6% |
| lateral dolly, two depth layers | PARALLAX | refused | 0% |
| tripod, no motion | LOCKED | refused | 0% |

Registration accuracy: **0.097 px median** error vs known camera track.

Recovered-pixel accuracy vs staleness — this is the finding:

| source is N frames back | PSNR |
|---|---|
| 0–3 | **28.0 dB** |
| 4–7 | 25.3 dB |
| 8–11 | 22.6 dB |
| 12–15 | 22.5 dB |
| 16–19 | 21.6 dB |
| 20–23 | 20.7 dB |
| 24–27 | 16.8 dB |

The sub-pixel resampling ceiling for a propagated pixel is **28.1 dB** (measured
independently), so fresh pixels come back essentially perfect and *all*
degradation is a function of temporal distance.

**That is why raw coverage is the wrong number.** A wing 90% filled from 25
frames back is worse than one 70% filled from 2 frames back. `effective_coverage`
weights every pixel by its expected quality. On the pan clip it drops the
headline from 80.8% raw to 59.6% effective — the honest figure.

## Known limits

- Thresholds (`PARALLAX_MARGIN`, `COVER_ON`) are calibrated on synthetic clips.
  Recalibrate on real footage before trusting them.
- No lens distortion model. Anamorphic or wide-angle sources need undistortion first.
- Moving objects are propagated as-is, so wings can show an anachronistic actor.
  Mask dynamics before the wings are used for real.
- No 3DGS backend yet — that's the slot where `PARALLAX` shots get handled.

## Next

`propagate_wings()` is the seam. A 3DGS backend implements the same signature —
frames in, canvas + coverage mask + staleness map out — and the metrics, gating,
and validation harness all work unchanged.

---

# Update: real footage, and the cross-cut engine

## What real footage broke

Tested on a 2-minute theatrical trailer (640x360, 23.976fps, 2.36:1 scope).

**Shot detection failed completely** — zero cuts found in two minutes. Modern
grading compresses everything into a narrow dark band, so HSV histograms barely
move across a cut. `shotdetect.py` replaces it: letterbox crop, then luma-MAD +
edge-overlap scored against an *adaptive local* threshold. Result: 93 cuts,
75 usable shots, median 1.29s.

**75% of shots get refused.** 34 LOCKED, 22 PARALLAX, 19 ROTATION. Verified by
hand — the LOCKED ones are genuinely locked cameras with movement inside frame,
not matcher failures (600-1400 matches each).

**Self-propagation recovers almost nothing.** At wing=0.75, effective coverage
**2.0%** (vs 59.6% synthetic). The arithmetic: a shot must accumulate
wing-width of lateral camera displacement. Median shot = 31 frames at ~0.7px/frame
= 22px, against a 480px wing.

| wing/side | coverage | effective |
|---|---|---|
| 0.06 (38px) | 26.3% | 18.7% |
| 0.10 (64px) | 18.2% | 12.7% |
| 0.75 (480px) | 2.8% | 2.0% |

## Detail weighting

Shot 50 scored 43.1% coverage with a wing that was almost entirely black.
Recovering darkness is free and meaningless. `detail_weight()` discounts pixels
by local standard deviation:

| shot | coverage | detail | effective |
|---|---|---|---|
| 60 bright tunnel | 68.6% | 19.8% | 13.1% |
| 50 dark rain | 43.1% | **2.6%** | **0.8%** |
| 27 curtains | 33.4% | 45.7% | 12.6% |

Shot 27 has lower coverage than shot 50 but 15x the effective score. That
ordering matches what the eye sees.

## Cross-shot donation does NOT work

Tested every shot pair within the trailer: **1 of 400 candidates** verified
geometrically. A homography cannot bridge two different setups of the same
location. Retrieval was never the bottleneck — registration is.

## Cross-CUT donation does work (`crosscut.py`)

The same *take* in a second trailer or TV spot is not a different viewpoint —
it is the same camera, differently cropped and graded. Validated with a
synthetic alternate cut (tighter primary, wider alternate, different grade,
different bitrate, shuffled order, trimmed durations, plus distractor shots
present in only one cut):

- **same-take detection: precision 100%, recall 83%**
- inlier separation: shared takes median **308**, distractors max **0**
- **wing coverage 17.7% -> 90.6%** (5.1x)
- **donated pixel accuracy 28.4 dB** mean vs the true originals

The strongest case is a locked-off close-up: self-propagation recovers **0%**,
donation recovers **100% at 33.0 dB**. Shots the geometry backend correctly
refuses are exactly the ones another cut rescues.

Architecture note: appearance embedding proposes candidates, geometry decides.
The proposer is allowed to suggest, never to assert a pixel — which is what
keeps the coverage number honest.

## Caveats on the cross-cut result

- The alternate cut is synthesised from the same source, so registration is
  perfect by construction. Real trailers share takes but may differ in VFX
  version, speed ramps, stabilisation, and grade far more than simulated here.
- Recall 83%: the 3 misses are near-black shots where ORB finds nothing even
  after CLAHE.
- One shot scored 17.7 dB — uncompensated grade mismatch. Colour transfer
  before donation would fix it.
- Gains are bounded by how much wider the other cut actually framed. That
  margin was built in here; in the wild it may be zero.

Next: a real second trailer for the same title.

---

# Update 2: real second cut (Thunderbolts* main trailer vs Big Game spot)

## Same-take detection holds on real footage

- **15 shared takes** found across 75 trailer-1 shots
- best match **605 inliers**; alignment PSNR mean **24.0 dB**, max **35.8 dB**
- **negative control**: 22x22 shots across two *unrelated* films -> **0 accepted,
  max inliers 0**. Not "below threshold" — literally zero geometric agreement.

Precision now verified on real material, not just synthetic distractors.

## Single-frame donation mostly does NOT pay

Wing coverage 23.7% -> 25.6%; only 3 of 15 pairs gained. Most shared takes come
back at **scale 0.99-1.01** — both cuts used identical framing, so there is no
periphery to donate.

**The homography's scale term is a free predictor.** Anything within +/-2% is a
wasted donation attempt. Check it before doing any work.

Where framing did differ, the mechanism worked exactly as designed:
`A@2403, scale 0.901` (alt framed 10% wider) went **8.8% -> 28.5%**.

## Unified propagation (`crossres.py`) — what actually pays

Merging BOTH cuts into one donor pool, nearest-in-time across the union, at 2x
output scale:

| take | dur gain | scale | wings | detail |
|---|---|---|---|---|
| A@1543 | +33f | 0.994 | 43.7% -> 46.1% | 17 -> 87 |
| A@467 | +27f | 1.129 | 35.6% -> 35.6% | 4 -> 10 |
| A@2207 | +17f | 1.000 | 25.2% -> 25.1% | 8 -> 8 |
| A@2403 | +3f | 0.901 | **8.8% -> 36.1%** | 42 -> 119 |
| A@1141 | 0f | 0.996 | 6.5% -> 6.6% | 21 -> 78 |

- **wing coverage 24.0% -> 29.9%**
- **detail 3.00x** (variance of Laplacian) — exactly the 1920/640 resolution
  ratio between the two cuts, which is the confirmation the transforms are right

So the honest split: cross-cut matching buys you **modest coverage** and
**large resolution**. It is a super-resolution result wearing a coverage
project's clothes.

## Geometry note (cost me two wrong runs)

All composition must happen in ONE coordinate space. Chaining the alternate's
intra-shot homographies on high-res frames while measuring the cross-cut `H` on
downscaled ones silently corrupts every donated pixel — coverage went *down* and
sharpness stayed flat. `crossres.unified_propagate` now does all geometry at
primary-native scale and applies exactly one explicit scale term (`B2S`) when
sampling the high-res source.

For alternate frame m into primary frame k:

    M = to_canvas @ inv(HA[k]) @ HA[ia] @ inv(H) @ inv(HB[ib]) @ HB[m] @ B2S

---

# Update 3: backends, gating, fenced fill

## 1. Pluggable backends (`backends.py`)

One interface, `frames -> (canvas, filled, tmap)`. Everything downstream is
backend-agnostic.

| backend | handles | status |
|---|---|---|
| `MosaicBackend` | ROTATION | shipping |
| `LayeredBackend` | ROTATION, PARALLAX | **experimental — see below** |
| `GaussianBackend` | ROTATION, PARALLAX | needs CUDA; raises, never silently falls back |

`GaussianBackend` refuses rather than degrading to mosaic on purpose. A silent
fallback would put homography pixels behind a 3D confidence label and corrupt
the only number this tool produces.

### LayeredBackend: a negative result

Fits K motion layers by RANSAC-and-remove, composites back-to-front. On the
synthetic parallax clip, against exact ground truth:

| backend | wing coverage | recovered-pixel PSNR |
|---|---|---|
| mosaic | 47.9% | 11.9 dB |
| layered | **87.8%** | **11.7 dB** |

It nearly doubles coverage and recovers *nothing more correctly*. Coverage alone
would have shipped this as a 1.8x win. Piecewise planes are not depth.

## 2. Leave-one-out geometry check (`gating.leave_one_out`)

Which forced the missing piece: **a correctness check that needs no ground truth.**

Hide frame i. Reconstruct it from its neighbours only, using the backend's own
transforms — pose may come from frame i, pixels may not. Score the outer band,
the region most like an extrapolated wing.

| clip | backend | self-check | verdict | (truth said) |
|---|---|---|---|---|
| pan_flat | mosaic | **34.1 dB** | TRUSTED | 24-28 dB, good |
| pan_flat | layered | 33.7 dB | TRUSTED | good |
| dolly_parallax | mosaic | **17.2 dB** | REJECTED | 11.9 dB, bad |
| dolly_parallax | layered | **15.6 dB** | REJECTED | 11.7 dB, bad |

It separates trustworthy geometry from lying geometry with no panorama and no
camera track, and correctly ranks layered *below* mosaic on parallax.

First version scored 138 dB on everything — it was feeding the held-out frame
back in as its own source. The leak is documented in the docstring.

## 3. Gating (`gating.decide`)

Coverage + staleness + detail + geometry -> `OFF` / `NARROW` / `FULL` and a wing
ratio, with hysteresis so wings don't strobe between shots.

```
pan_flat        ROTATION  mosaic   geom 34.1dB  eff 47.6%  -> NARROW (0.25)
dolly_parallax  PARALLAX  layered  geom 15.6dB  eff 52.8%  -> OFF
locked_off      LOCKED    none                             -> OFF
```

The parallax clip had the *highest* effective coverage of the three — 52.8% —
and is still switched off, because the geometry could not be verified. That
override is the entire point of the system.

## 4. Fenced generative fill (`fill.py`)

Generated pixels **cannot** overwrite recovered ones. Structural, not by
convention:

- every pixel carries provenance: `PRIMARY` / `RECOVERED` / `GENERATED`
  (the fence's three; the agent's ladder in `provenance.py` refines this to six)
- the generator receives a mask of only the empty region
- output is composited through that mask alone
- a post-condition asserts recovered pixels are bit-identical, else `FenceViolation`

Verified, including against a rogue generator that returns zeros everywhere:

```
recovered pixels bit-identical after generation : True
wing provenance                                 : 50.8% real / 49.2% generated
confidence in holes                             : mean 0.036, max 0.937
rogue generator blocked                         : True
```

The confidence map is built from the same staleness and detail weights the
metric uses, then spread outward — a hole pixel inherits confidence from the
quality of real content around it. High near fresh detailed recovery, ~zero deep
in a hole. That is what a diffusion model conditions on, with the recovered
canvas as geometric anchor so it invents texture rather than structure.

`InpaintGenerator` (OpenCV, CPU) is the stand-in that lets the fence be tested
end to end. `DiffusionGenerator` is the GPU slot; the fence is unchanged.

## What still needs hardware

`GaussianBackend` and `DiffusionGenerator` are the two GPU slots. Everything
else — classification, propagation, all four metrics, the self-check, gating,
and the fence — runs on CPU and is verified above.

---

# Update 4: agentic fill (`agent.py`)

## The ladder

Generation is the *last* rung, not the first. Every pixel is labelled with where
it came from, ordered by trust:

| provenance | source | camera |
|---|---|---|
| `PRIMARY` | the frame itself | — |
| `RECOVERED` | elsewhere in this shot | same |
| `DONATED` | same take, another cut, geometrically verified | same |
| `RETRIEVED` | another setup of the same location, 3D-verified | same set |
| `REFERENCED` | licensed external material, **unverified** | somewhere else |
| `GENERATED` | invented | none |

Six rungs, not two, on purpose. `RETRIEVED` is real photography but of a
different setup — different lens, lighting, moment. Folding it into `RECOVERED`
is how "we recovered 80%" quietly becomes a lie. The report exposes both
`real_same_camera` and `photographic` so the distinction survives to the pitch.

`REFERENCED` is that argument one rung further down, and it is drawn where the
*evidence* stops rather than where the photons do. External stills and location
plates are real photographs, but nothing checks that they depict this location,
so they sit **outside `photographic`** and outside the headline number. Measured:
a flat colour plate from a dummy library moved `mean_real_wing` 3.36% → 5.34%
while telling you nothing new about the wall. It now moves it 0.00%.

The rung is a holding position, not a verdict. When `GaussianBackend` lands, an
asset that registers to the shot is promoted `REFERENCED` → `RETRIEVED` and earns
its way in.

`SameLocationTool` is the rung that matters for a feature: a locked-off close-up
has no periphery of its own, but the scene's master wide photographed that wall
an hour earlier. Wide baseline, so it requires `GaussianBackend` — a homography
cannot bridge two setups (measured: 1 verified pair in 400).

## Provider selection

Capabilities are declared, selection is automatic, and **anchoring outranks
fidelity** — conditioning on the recovered canvas is what stops generation
free-running, and matters more than raw quality for peripheral vision on a side
wall.

```
target 1024px, needs anchor        -> hl-outpaint   (research, 2048px)
target 2048px, anchor, self-hosted -> hl-outpaint   (research, 2048px)
target  720px, no anchor, hosted   -> kling-3.0     (commercial, 2160px)
```

Registry covers self-hosted research methods (Unboxed, HL-OutPaint, M3DDM,
MOTIA, VACE), open weights (HunyuanVideo 1.5 Apache-2.0, Wan 2.7) and hosted
endpoints (WaveSpeed video outpainter, Kling 3.0, Runway Gen-4.5, Luma Ray3).
HL-OutPaint wins the anchored cases because it targets long-range *and* large
spatial extrapolation — the combination the rest of the literature leaves
unresolved, which is exactly this problem.

## Licence gate

External material is default-deny. The output is a derivative of a copyrighted
film; anything pulled from outside the production needs recorded rights or it is
refused.

```
licensed asset admitted  : True
unlicensed asset admitted: False
```

## The fence, hardened

Two independent mechanisms:

1. **Structural.** Tool output is composited through `hole ∧ tool_mask`. A tool
   *cannot address* a protected pixel, so that path needs no runtime check.
2. **Read-only view.** Tools receive the canvas with `writeable = False`, so
   in-place mutation fails at the numpy level and is re-raised as
   `FenceViolation`.

The second exists because the first missed a case. The original check
snapshotted protected pixels *after* calling the tool, so a tool that trampled
the canvas in place had already baked in its damage before the comparison ran.
Caught by an adversarial test, fixed by snapshotting first and handing out a
read-only view.

```
composition through hole-mask: structural, unaddressable
recovered pixels untouched   : True
in-place mutation            : blocked by FenceViolation
```

## Verified end to end

```
same_shot          handled by backend propagate()
same_take          not applicable (no corpus)
same_location      not applicable (no scene index / no GPU)
external_reference not applicable (no fetcher)
generate           filled 27.9% :: provider=inpaint

wing: 50.8% recovered / 49.2% generated
```

The ladder descends correctly, records why each rung was skipped, and the
generated half never touches the recovered half.

---

# Update 5: goal-driven director (`director.py`)

Replaces the fixed ladder with a planner. Declare targets and a budget; it
discovers resources at runtime, picks actions by expected gain per cost,
measures actual gain, updates its estimate, and re-plans.

## Two-phase selection

Phase 1 considers only actions that yield real photography. Generation is Phase 2
and runs **only** once no real source has positive expected yield.

The first version scored generation at zero whenever a real-pixel gap existed —
which meant it never generated at all, spent nothing, and reported a shortfall
against an empty canvas. Ordering generation last is the point; refusing to ever
reach it is a bug.

## Structural vs stochastic failure

An action that fails for a *structural* reason is dropped immediately — a scale
term of 1.000 or a missing GPU backend will not change on retry. Only stochastic
misses get a second observation before being dropped. This is the earlier
finding turned into control flow: `same_take` fired on 15 of 75 real shots and
yielded nothing on 10 of them, all because framing scale was within 2% of 1.0.

## Verified behaviour

**A — no resources.** Falls through to generation, fills the wing, reports the
gap honestly: real 51.1% against a 60% target, generated 33.9% against a 15% cap.

**B — scout finds a wider cut.** One action, cost 2.0, target met:
```
same_take  gain 100%  donor scale 0.783, 164 inliers
real_same_camera 100%   stop: target met
```
It never generated and never touched the other rungs — found the cheap real
source and stopped.

**C — impossible target (99%).** Generates, reports shortfall, terminates. Does
not loop.

**D — corpus exists but framing is identical.** The real-world case:
```
same_take           0.0%  scale 1.000 -- no periphery to gain   -> dropped
same_location       0.0%  needs GaussianBackend                 -> dropped
external_reference  100%  studio location plate (licensed); 1 refused
final: photographic 100%, real_same_camera 51.1%, shortfall real 8.9%
```
Two dead ends identified and abandoned in one attempt each, one unlicensed asset
refused, target partially met, and the residual gap named rather than hidden.

## What planning cannot do

It searches harder over sources that exist. It does not create photons. On a
locked-off shot with no other setup of that location anywhere in the corpus,
every branch terminates in generation, and the honest output is a reported
shortfall rather than a filled wing. The director is built to say which of
`target met` / `budget exhausted` / `no positive expected yield` ended the run.

---

# Update 6: theatre geometry (`walls.py`)

Renders three projector feeds — left wall, main screen, right wall — from the
recovered canvas.

## Why a wide flat image is wrong

The canvas is a single pinhole image, and a pinhole cannot span 270 degrees:
horizontal extent goes as tan(theta) and blows up at 90. Displaying a wide strip
across side walls is what makes naive outpainting demos read as "stretched
panorama" instead of "theatre".

Each wall gets its own projection for a seated viewer. The source canvas is a
plane, each wall is a plane, the viewer is one centre of projection — so each
wall is an **exact 3x3 homography**. No spherical intermediate, no approximation.

## The finding: wing width stops mattering at ~0.22

| wing ratio | wall depth | binding constraint |
|---|---|---|
| 0.10 | 2.00 m | horizontal |
| 0.15 | 2.77 m | horizontal |
| 0.25 | **2.86 m** | **vertical** |
| 0.50 | **2.86 m** | **vertical** |
| 0.75 | **2.86 m** | **vertical** |

(screen 14m, viewer 12m back, 16:9 source)

Two constraints govern how far back the wings reach:

    horizontal:  z_h = D * (w/2) / (w/2 + wing_w)
    vertical:    z_v = D * (w/h) * (Hm/Wm)          <- independent of wing_w

Propagation widens the canvas but never makes it **taller**. So past a wing
ratio of roughly 0.22 the vertical term dominates and extra horizontal
periphery buys nothing — visible as black wedges above and below the wall image
before the fix.

**Everything measured at wing 0.75 was chasing an unusable target.** The right
operating point is ~0.20–0.25, which is also where real footage performs far
better: the earlier sweep gave 18.7% effective coverage at 0.06 and 12.7% at
0.10, versus 2.0% at 0.75.

**And this reverses an earlier call.** I had dismissed format variants
(open-matte, IMAX 1.90) on the grounds that they differ from scope in height,
"the axis you don't care about". Height is the axis that binds. An open-matte
master is the single most valuable donor in the corpus, because vertical
periphery is what unlocks wall depth.

## Output

`render()` produces the three panels with seam feathering and the slight
dimming ScreenX uses on side walls. `contact_sheet()` lays them out as a master
strip. `mark_generated=True` tints invented pixels for the review pass, off for
a screening.

---

# Update 7: re-measured at the geometrically correct wing width

Same real footage (Thunderbolts* Big Game spot, 1080p, letterbox-cropped to
1920x808), measured at wing 0.22 — the width `walls.py` says is actually
projectable — against the 0.75 used everywhere above.

| wing ratio | coverage | effective |
|---|---|---|
| 0.75 | 7.87% | 2.23% |
| **0.22** | **21.54%** | **6.15%** |

**2.8x improvement.** I predicted 5–10x from the earlier sweep; that sweep ran
on the 640x360 first trailer, and this is different footage. The prediction was
wrong and the measured number is 2.8x.

## Geometry self-check

median **24.3 dB**, **13 of 18** ROTATION shots clear the 20 dB bar.

So this is not 6% of garbage — it is 6% of verified real pixels, with five shots
correctly flagged as untrustworthy and excluded.

## Classification

```
LOCKED    49
PARALLAX  20
ROTATION  18
```

**79% of shots refused**, worse than the first trailer's 75%. A Big Game spot is
cut harder for impact, so it holds more static frames.

## The headline number

At the geometrically correct wing width, on real trailer footage:

> **~6% of the ScreenX side walls was ever actually filmed.**

ScreenX runs those walls for 60–100 minutes of a feature. The rest is authored.
Nobody has previously put a figure on it.

## Where the remaining headroom is

Not in wing width — that is now solved and capped by vertical FOV at ~0.22. It
is in the **69 refused shots**:

- 20 PARALLAX -> `GaussianBackend` (needs CUDA)
- 49 LOCKED -> `SameLocationTool` (needs a feature film; a trailer has no second
  setup of the same location to retrieve from)

Both require material a trailer cannot provide. The next measurement needs
scenes from a feature.

---

# Update 8: continuous-motion footage — the number the method was built for

Handheld pan across a room, 18.9s, 1080x1920 portrait (iPhone), 30fps.

| | trailer (Big Game) | continuous pan |
|---|---|---|
| effective coverage | 6.15% | **47.1%** |
| real wing fraction | — | **92.2%** |
| wings on | ~6 of 87 shots | **2 of 2** |

**7.7x the trailer**, for the obvious reason: the camera moves continuously
instead of cutting every 1.3 seconds.

## The self-check earned its place

Both shots classified PARALLAX and ran on `LayeredBackend` — the backend shown
untrustworthy on the synthetic parallax clip (15.6 dB, REJECTED). Here:

```
shot 0  PARALLAX  NARROW  geom 26.1 dB  eff 41.9%
shot 1  PARALLAX  NARROW  geom 20.6 dB  eff 52.3%
```

Both pass. Real handheld interior parallax is mild enough that piecewise planes
genuinely fit; the synthetic clip's foreground moving at 2x the background was
not. **A blanket ban on `LayeredBackend` would have discarded both of these
shots.** Per-shot geometric verification is what distinguishes "this backend is
wrong here" from "this backend is wrong".

## Binding constraint flipped

```
z_horizontal 8.33   z_vertical 2.89   binding: horizontal
wall depth 3.67 m (31% of room)   vs 2.86 m (24%) on 16:9
```

Portrait source has vertical FOV to spare, so horizontal binds again and wall
depth rises. Independent confirmation that **height is the scarce axis** — and a
further argument for open-matte masters as the most valuable donor.

## Renderer (`screenx_render.py`)

Full pipeline to a watchable three-panel MP4:

```
classify -> backend -> propagate -> self-check -> gate
-> director fills remainder -> theatre wall projection -> three panels
```

The file plays as two things: first half SCREENING, second half REVEAL with
invented pixels lit magenta. Gating drives wing state over time, so wings open
and close through the runtime the way ScreenX does — decided by measurement
rather than by hand.

```
python3 screenx_render.py clip.mp4 --maxw 640
```

---

# Update 9: panel aspect from physical geometry

Second continuous-pan clip (handheld, 21s, 1080x1920 portrait):

```
shot 0  ROTATION  geom 18.7 dB  eff 24.5%  -> OFF   (below the 20 dB bar)
shot 1  ROTATION  geom 27.0 dB  eff 40.0%  -> NARROW
mean effective 32.3%   wall depth 3.67 m (31% of room, horizontal-limited)
```

## The bug that made the first room render look rotated

Not source orientation — OpenCV applies the rotation metadata correctly. The
panels were hardcoded 420x236 landscape, but a side wall is
**(wall depth) x (screen height)** = 3.7m x 6m, which is TALLER than it is wide.
Warping that into a landscape box squashed the image until it read as rotated.

`walls.auto_panels()` now derives panel pixel dimensions from the physical wall
shape, and the centre panel from the screen aspect.

## A wrong fix worth recording

First attempt set the theatre screen aspect to match the source — producing a
portrait cinema screen and **zero wall depth**. That is actually correct
behaviour for an incoherent input: if the frame exactly fills the screen there is
no vertical margin, and the walls get nothing. It follows directly from

    z_v = D * (w/h) * (Hm/Wm)   ->   depth 0 when h/w == Hm/Wm

which is another way of stating the open-matte argument: **wall depth comes from
image that exists outside the displayed frame.**

## Note on portrait sources

Portrait works for measurement but a ScreenX screen is landscape. For a true
demo, shoot the same slow pan holding the camera sideways.

---

# Update 10: the demo clip

Landscape handheld pan across a café, 5.9s, 1920x1080.

```
shot 0   ROTATION   FULL   geom 26.9 dB   effective 63.7%
mean real wing 91.5%       walls 2.86 m (24% of room, vertical-limited)
```

**The first FULL gating decision in the project.** Both side walls carry the
café continuing in correct perspective, 91.5% of it genuinely filmed.

## The spread that is the pitch

| source | effective coverage |
|---|---|
| Thunderbolts* Big Game trailer | 6.15% |
| continuous handheld pan (café) | **63.7%** |

**10x**, and the only difference is whether the camera moves continuously or
cuts every 1.3 seconds.

## Rotation

This clip carried **no rotation metadata whatsoever** — recorded with the handset
physically inverted, so nothing in the file could tell the decoder which way was
up. Added an explicit flag rather than guessing:

```
python3 screenx_render.py clip.mov --maxw 640 --rotate 180
```

---

# Update 11: length pays, and the truncation trap

24s landscape handheld walk through an apartment, 1920x1080.

| frames/shot | effective | real wing | geometry |
|---|---|---|---|
| 80 (truncated) | 38.7% | 77.2% | 30.0 / 24.1 dB |
| **200 (full clip)** | **48.7%** | **94.4%** | **30.4 / 31.3 dB** |

**94.4% real is the highest real-pixel fraction measured in this project**, and the
geometry self-check scores went UP with more frames (31.3 dB, best recorded).
More donors means fresher source pixels and better-conditioned registration.

## The trap

`--frames-per-shot` defaults truncate long takes, silently discarding the single
property that makes them valuable. The first run on this clip cut 24s down to
2.7s per shot and lost 10 points of effective coverage. **For real runs use
`--frames-per-shot 200` or higher.**

## Coverage vs information

Effective coverage is 48.7% here against the cafe's 63.7%, despite far more
camera travel. The apartment is a cramped interior with large flat white walls,
which the detail weighting correctly discounts. High coverage, lower information
content — the metric working as designed.

## Clip comparison

| footage | effective | real wing | note |
|---|---|---|---|
| cafe, 6s landscape | 63.7% | 91.5% | public space, depth, FULL verdict |
| **cafe, same clip, 2026-08-22 build** | **54.3%** | **98.7%** | NARROW; highest real-wing fraction measured, both wings |
| apartment, 24s landscape | 48.7% | **94.4%** | long take, flat walls discount detail |
| room, 21s portrait | 32.3% | 30.6% | one shot gated OFF at 18.7 dB |
| Thunderbolts* trailer | 6.15% | — | 87 shots, median 1.3s |
