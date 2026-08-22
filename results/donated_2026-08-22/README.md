# DONATED: wired, reachable, and blocked by the gate -- 2026-08-22

Two cuts of one picture: `videoplayback.mp4` (120.1s) and
`videoplayback (1).mp4` (129.6s).

    python screenx_render.py videoplayback.mp4 \
        --other-cut "videoplayback (1).mp4" -o out --maxw 480 --sources

## First: the rung had no caller at all

`FilmIndex.add_film` -- written, documented, tested -- had **zero callers**
outside its own definition. DONATED was unreachable from the command line
whatever footage existed, the same dead-rung pattern Tier 1 fixed for the scout.
Now wired as `--other-cut`, deliberately distinct from `--also`: `--also` means
same LOCATION, different setup (RETRIEVED); `--other-cut` means same TAKE,
different edit (DONATED).

`add_film` also carried the shot-id collision that cost a GPU session earlier
the same night -- it numbered the second film's shots 0,1,2 straight over the
primary's. It now takes an `id_base`; the second cut lands at 1,000,000+.

## The donor exists

`TakeMatcher` over 16 shots of each cut:

```
A13 -> B03   52 inliers   scale 0.866   DIFFERENT FRAMING
1 of 16 shots matched a take; 0 identical framing, 1 reframed
```

One shared take, and cut B frames it wider (scale 0.866), so it has real
periphery to donate. That is precisely what the rung wants.

## Why it still did not fire

**Shot 13 is LOCKED, so it is gated OFF before any planner runs.** A locked shot
takes the `backend is None` path and returns immediately; the Director never
sees it. The one shot in this film with a donor is the one shot that can never
ask for it.

Stated generally: **DONATED can only reach shots that already cleared the gate**
-- shots that already have real wings and need donation least. The shots that
most need another cut's pixels are exactly the ones the architecture excludes.
This was flagged in the project's first session ("on OFF shots the director never
runs at all") and is now measured rather than asserted.

## The reporting fix that caught it

The run's action list read `generate[fallback-fill],same_take[real]`, which looks
like the rung firing. It was not. A planner Step is logged even at zero gain, so
"the tool ran" and "the tool landed pixels" read identically. `action_gain` now
records how much each action actually landed:

```
"same_take[real]":         0.0     <- ran, contributed nothing
"generate[fallback-fill]": 2600.0
```

Without that number this run would have been written up as a success.
Pinned by `test_action_gain_reporting`.

## Open design question

Whether a gate-rejected shot may accept DONATED pixels is a real decision, not an
oversight. The argument for the current behaviour: a rejected shot's own recovery
is untrustworthy. The argument against: donated pixels come from a DIFFERENT cut
and do not depend on this shot's geometry at all -- the take is verified by 52
homography inliers, independently of whether this shot's own parallax is
recoverable. Changing it would let RETRIEVED and DONATED serve locked-off shots,
which is the case the whole ladder was built for.

## Root cause, found after the gate was opened

Letting refused shots borrow (see `BORROWABLE`) was necessary but not
sufficient: shot 13 still borrowed nothing. Three probes, each ruling one thing
out:

1. Shot 13's shortlist offers 6 cutB candidates, none verifying -> suspect the
   appearance ranking.
2. Exhaustive verification against **all 57** indexed shots: still none.
   So it is not the shortlist.
3. Same pair, two frame samplings:

```
uncropped, 3 frames spread over the shot :  52 inliers vs cutB shot 3
pipeline frames (8, as indexed)          :   0 inliers, no match at all
```

Both films receive the identical crop `(0, 45, 640, 316)`, so cropping cannot
misalign them relative to each other. The difference is **which frames get
sampled**.

`load_shot` seeks to the shot's first frame and reads N *consecutive* frames.
The index's "8 samples per shot" are therefore the first 8 frames -- about a
third of a second -- not a spread across the take. Two cuts of one picture
almost never enter a take at the same instant: one trims a few frames off the
head, and the comparison is then between two different moments of the same
shot. ORB has nothing to match.

`sfm.write_images` already learned this lesson and says so in its own comment --
"even sampling, not the first N: the first N of a moving shot is a fraction of
its baseline". The index never got the same treatment.

**The fix is to spread the index's samples across the shot**, the way
`write_images` already does. Untested at time of writing: the change is one
`np.linspace` in the sampling path, and it should be measured against the
52-inlier pair above, which is a known-good target.
