"""
screenx_render — the full pipeline to a watchable three-panel video.

Produces ONE file that plays as two things:

    first half   SCREENING   — the 270 degree experience, wings filled
    second half  REVEAL      — the same frames, invented pixels lit magenta

That switch is the demo. Video generation is not surprising to anyone any more.
A system that shows you, mid-playback, exactly how much of what you just enjoyed
was fabricated — and can put a number on it — is.

Per shot:
    classify -> pick backend -> propagate -> self-check -> gate
    -> director fills the remainder (real sources first, generation last)
    -> project onto theatre wall geometry
    -> write three panels

Gating drives wing state over time, so wings open and close through the film the
way ScreenX does, except decided by measurement rather than by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

import agent as ag
import backends as bk
import context as cx
import director as dr
import fetchers as ft
import fill as fl
import offscreen as ofs
import reasoning as rz
import filmindex as fx
import gating as g
import shotdetect as sd
import walls as wl
import wingcoverage as wc

WING = 0.22          # the geometrically projectable width; see walls.wall_extent
SAME_PLACE = "declared_location"   # scene id for clips the user says share a wall
SETUP_ID_BASE = 10000              # keeps --also shot ids clear of the primary's
CUT_ID_BASE = 1000000              # and another cut's clear of both
# Rungs a gate-refused shot may still accept: both are verified against
# footage OTHER than this shot. RECOVERED is deliberately absent -- that is
# precisely the geometry the gate rejected, and readmitting it would buy a
# render by discarding the refusal.
BORROWABLE = (ag.DONATED, ag.RETRIEVED)
SFM_FRAMES = 40                    # frames per setup handed to COLMAP


ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_shot(cap, crop, a, b, maxw, cap_frames=None, rotate=0, spread=False):
    """
    rotate: explicit degrees. Some phone clips carry NO rotation metadata at all
    (recorded with the handset physically inverted), so nothing in the file can
    tell the decoder which way is up. It has to be passed in.
    """
    x0, y0, x1, y1 = crop
    n = (b - a) if cap_frames is None else min(b - a, cap_frames)
    # spread: for INDEXING, not rendering. Reading n consecutive frames from the
    # shot's head samples about a third of a second of it, and two cuts of one
    # picture almost never enter a take at the same instant -- one trims a few
    # frames off the head, so the comparison is between different moments and
    # ORB has nothing to match. Measured: the same pair scores 52 inliers spread
    # and 0 consecutive. sfm.write_images already learned this ("even sampling,
    # not the first N"); the index never did.
    picks = (np.linspace(a, b - 1, n).astype(int) if spread and n < (b - a)
             else None)
    if picks is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = []
    for step in range(n):
        if picks is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(picks[step]))
        ok, f = cap.read()
        if not ok:
            break
        f = f[y0:y1, x0:x1]
        if ROTATIONS.get(rotate) is not None:
            f = cv2.rotate(f, ROTATIONS[rotate])
        if f.shape[1] > maxw:
            s = maxw / f.shape[1]
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def _plan_for(context, frames, ww):
    """
    Work out what belongs on this wall before anything draws it.

    Returns (plan, label). The plan carries each element's support, so the
    prompt leads with what was measured from the footage rather than with
    whatever a script happened to say, and the label records that a human or a
    document was involved at all.
    """
    ctx = context or {}
    items = ctx.get("items") or []
    excursions = ctx.get("excursions") or []
    if not items and not excursions:
        return None, ag.GENERATED

    h, w = frames[0].shape[:2]
    brief = rz.brief_for(dict(shot=ctx.get("shot", 0), motion=ctx.get("motion", "")),
                         n_frames=len(frames), wing_w=ww, frame_w=w, frame_h=h,
                         excursions=excursions, context_items=items)
    reasoner = ctx.get("reasoner") or rz.LocalReasoner()
    plan = reasoner.plan(brief)
    return plan, plan.label()


def _apply_context(generator, context, frames=None, ww=0):
    """
    Hand the generator the plan, and say what that makes the pixels.

    A note pinned to THIS shot, or a figure measured leaving it, earns DIRECTED;
    a bundle full of unbound screenplay does not, and the label is what stops
    "we loaded a script" from quietly becoming "a human specified this wall".
    """
    items = (context or {}).get("items") or []
    plan = None
    if frames:
        plan, label = _plan_for(context, frames, ww)
    else:
        label = cx.provenance_for(items) if items else ag.GENERATED

    prompt = plan.prompt() if plan is not None else (
        cx.build_prompt(items) if items else None)
    if prompt:
        try:
            generator.prompt = prompt
        except AttributeError:
            pass
    if plan is not None and isinstance(context, dict):
        context["plan"] = plan
    return label


def _shot_generate(generator, frames, ww, prompt):
    """
    One submission for the whole shot when the generator is temporal.

    Every returned frame is still composited through its own hole here, so the
    fence holds whatever the model sends back.
    """
    widened = generator.generate_shot(frames, ww, prompt)
    out = []
    for src, wide in zip(frames, widened):
        h, w = src.shape[:2]
        canvas = np.zeros((h, w + 2 * ww, 3), np.uint8)
        canvas[:, ww:ww + w] = src
        known = np.zeros(canvas.shape[:2], bool)
        known[:, ww:ww + w] = True
        before = canvas[known].copy()
        canvas[~known] = wide[~known]
        if not np.array_equal(canvas[known], before):
            raise fl.FenceViolation("shot generator modified the filmed centre")
        prov = np.full(known.shape, ag.GENERATED, np.uint8)
        prov[:, ww:ww + w] = ag.PRIMARY
        out.append((canvas, prov))
    return out


def synth_wings(frame, ww, generator, label=None):
    """
    Invent both wings for one frame. Nothing real is available or claimed.

    Used where the measurement says the wings cannot be recovered: a locked-off
    camera never saw past its own frame edge, and a shot gated OFF has recovered
    pixels the self-check refused to trust. In both cases every wing pixel here
    is marked GENERATED, so the coverage number is unaffected and the reveal
    lights the whole wing magenta.

    The gated-OFF case deliberately discards the propagated pixels rather than
    keeping them: they failed the geometry check, and re-labelling them
    RECOVERED would put pixels the gate rejected back into the "real" fraction.
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + 2 * ww, 3), np.uint8)
    canvas[:, ww:ww + w] = frame

    known = np.zeros((h, w + 2 * ww), bool)
    known[:, ww:ww + w] = True
    hole = ~known

    conf = fl.confidence_map(canvas, known, np.zeros(known.shape, np.float64))
    produced = generator(canvas, hole, conf)
    before = canvas[known].copy()
    canvas[hole] = produced[hole]                  # composite through the hole only
    if not np.array_equal(canvas[known], before):
        raise fl.FenceViolation("generator modified the frame it was given")

    prov = np.full(known.shape, ag.GENERATED if label is None else label, np.uint8)
    prov[:, ww:ww + w] = ag.PRIMARY
    return canvas, prov


def safe_generate(fn, *args, **kw):
    """
    Run a generator call, or report why it could not.

    Returns (frames_or_None, note). Network failures, quotas and malformed
    replies are all the same class of event here: the wall does not get filled
    and the shot says so.
    """
    try:
        return fn(*args, **kw), ""
    except Exception as e:                  # a hosted service fails many ways
        return None, f"{type(e).__name__}: {e}"[:160]


def to_h264(path, fps=24.0):
    """
    Convert an OpenCV-written mp4 to H.264 in place.

    cv2 offers mp4v, which nothing in a browser will play, and the avc1 fourcc
    is usually unavailable in the wheels people actually install. Transcoding
    afterwards is the reliable route and costs seconds against a render's
    minutes. Returns True if the file is now H.264.
    """
    src = Path(path)
    if not src.exists() or shutil.which("ffmpeg") is None:
        return False
    tmp = src.with_suffix(".h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            src.unlink(missing_ok=True)
            tmp.rename(src)
            return True
        tmp.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
    return False


def write_shot_segment(dest, shot_id, pairs, wing_w, theatre, fps=24.0):
    """
    One shot's projector feeds, written the moment that shot is done.

    Complete files, not a stream: an interrupted mp4 has no trailer and will not
    open, so appending to one long file would lose the lot anyway.
    """
    import walls as wl
    dest = Path(dest) / "shots"
    dest.mkdir(parents=True, exist_ok=True)
    if not pairs:
        return
    height_px = int(pairs[0][0].shape[0])
    first = wl.render(pairs[0][0], wing_w, theatre, height_px=height_px)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers, sizes = {}, {}
    for side in ("left", "centre", "right"):
        h, w = first[side].shape[:2]
        sizes[side] = (w, h)
        writers[side] = cv2.VideoWriter(
            str(dest / f"shot_{shot_id:03d}_{side}.mp4"), fourcc, fps, (w, h))
    mh, mw = pairs[0][0].shape[:2]
    master = cv2.VideoWriter(str(dest / f"shot_{shot_id:03d}_master.mp4"),
                             fourcc, fps, (mw, mh))
    for canvas, _prov in pairs:
        panels = wl.render(canvas, wing_w, theatre, height_px=height_px)
        for side in ("left", "centre", "right"):
            f = panels[side]
            if (f.shape[1], f.shape[0]) != sizes[side]:
                f = cv2.resize(f, sizes[side])
            writers[side].write(f)
        master.write(canvas if canvas.shape[:2] == (mh, mw)
                     else cv2.resize(canvas, (mw, mh)))
    for w_ in writers.values():
        w_.release()
    master.release()
    for side in ("left", "centre", "right"):
        to_h264(dest / f"shot_{shot_id:03d}_{side}.mp4", fps)
    to_h264(dest / f"shot_{shot_id:03d}_master.mp4", fps)

    # the provenance map beside the pixels. Without it a later pass can see what
    # a wall looks like and not what it is, so anything that repaints one could
    # only either refuse or quietly overwrite photographed pixels while the
    # report kept counting them as filmed.
    try:
        np.savez_compressed(dest / f"shot_{shot_id:03d}_prov.npz",
                            prov=np.stack([p for _c, p in pairs]))
    except (OSError, ValueError) as e:
        print(f"  provenance for shot {shot_id} not saved: {e}", flush=True)


def checkpoint(outdir, records, source, partial=True):
    """
    The verdicts so far, on disk.

    Rewritten after every shot. `partial` says plainly that this is a run in
    progress: a summary that looks finished but is missing shots is worse than
    no summary, because every number in it is an average over the wrong
    denominator.
    """
    payload = dict(source=os.path.basename(source), partial=partial,
                   shots=len(records), per_shot=records)
    try:
        path = os.path.join(outdir, "screenx_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
    except OSError:
        pass          # a checkpoint that cannot be written must not end the run


def write_deliverable(dest, rendered, wing_w, theatre, fps=24.0):
    """
    The three projector feeds, plus the widened master. Clean.

    This is the output of the tool, as distinct from the report about it: a
    ScreenX presentation is three synchronised feeds, so they are written as
    three files rather than a contact sheet somebody has to cut up.

    Deliberately unmarked. `mark_generated` tints anything not filmed at this
    location, which is the right thing for the review view and the wrong thing
    for a screening -- an audience is not shown a QC overlay. The report says
    what was invented; the deliverable just plays.
    """
    import walls as wl
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not rendered:
        return {}

    # projected, not previewed: the feeds are rendered at the source's own
    # height, so a 1920px master delivers 1080-tall walls rather than 300
    height_px = int(rendered[0][0].shape[0])
    first = wl.render(rendered[0][0], wing_w, theatre, height_px=height_px)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    feeds, writers = {}, {}
    for side in ("left", "centre", "right"):
        h, w = first[side].shape[:2]
        path = dest / f"{side}.mp4"
        writers[side] = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        feeds[side] = dict(path=str(path), width=w, height=h)

    mh, mw = rendered[0][0].shape[:2]
    master = cv2.VideoWriter(str(dest / "master_widened.mp4"), fourcc, fps, (mw, mh))
    feeds["master_widened"] = dict(path=str(dest / "master_widened.mp4"),
                                   width=mw, height=mh)

    for canvas, _prov in rendered:
        panels = wl.render(canvas, wing_w, theatre, height_px=height_px)
        for side in ("left", "centre", "right"):
            frame = panels[side]
            want = (feeds[side]["width"], feeds[side]["height"])
            if (frame.shape[1], frame.shape[0]) != want:
                frame = cv2.resize(frame, want)
            writers[side].write(frame)
        master.write(canvas if canvas.shape[:2] == (mh, mw)
                     else cv2.resize(canvas, (mw, mh)))

    for w_ in writers.values():
        w_.release()
    master.release()

    playable = all(to_h264(Path(v["path"]), fps) for v in feeds.values())
    if not playable:
        print("  note: ffmpeg not available, so the feeds stay mp4v -- which no "
              "browser will play. Install ffmpeg for a deliverable that opens.",
              flush=True)
    print(f"  delivered {len(rendered)} frames to {dest}", flush=True)
    return feeds


def _theatre_from(a):
    """A Theatre only when something was actually asked for."""
    import walls as wl
    fields = dict(screen_width=a.screen_width, screen_height=a.screen_height,
                  viewer_distance=a.viewer_distance, wing_dim=a.wing_dim)
    given = {k: v for k, v in fields.items() if v is not None}
    return wl.Theatre(**given) if given else None


def _thresholds_from(a):
    """
    The gate, as overrides only.

    gating.decide has always accepted a thresholds dict and nothing ever passed
    one, so the bar was a constant in practice. Loosening it is a real decision
    -- the refusals are what the coverage number means -- so overrides are
    explicit per run and recorded in the summary rather than edited into the
    module.
    """
    named = dict(geometry_db=a.gate_geometry, eff_full=a.gate_full,
                 eff_narrow=a.gate_narrow, detail_min=a.gate_detail,
                 stale_max=a.gate_stale)
    out = {k: v for k, v in named.items() if v is not None}
    return out or None


def _emit_shot(rec):
    """One JSON line per finished shot, for a caller driving this as a
    subprocess. Prefixed so it cannot be confused with the human-readable
    progress on the same stream."""
    print("@@SHOT " + json.dumps(rec, default=str), flush=True)


def action_gains(shot_log):
    """
    How much wall each planner action actually landed, per action.

    The action list alone says what was TRIED. A tool can run, contribute
    nothing, and still appear in it -- `same_take[real]` reads identically
    whether a second cut donated a third of the wing or nothing at all. On a
    project whose whole claim is knowing where each pixel came from, "it ran" is
    not the interesting fact; "it landed 29% of the hole" is.
    """
    gains = {}
    for line in shot_log:
        parts = line.split(" ")
        if len(parts) < 2:
            continue
        try:
            pct = float(parts[1].rstrip("%"))
        except ValueError:
            continue
        gains[parts[0]] = round(gains.get(parts[0], 0.0) + pct, 1)
    return gains


def align_to_poses(model, backend, frames, shot_id):
    """
    Hand the backend the frames its poses actually describe.

    `poses_from_colmap` returns one pose per REGISTERED view of the whole
    scene. That is not this shot's frame list: `build_scene` submits at most
    `max_frames_per_setup` of them, and a multi-setup scene carries every other
    setup's views in the same reconstruction. `seed_points` pairs frames[i]
    with pose i, so handing over the raw shot triangulates one frame through
    another frame's camera -- the same defect as reading `manifest` where
    `registered` was meant, one layer down, and it surfaces as an IndexError
    only when the two lists happen to differ in length rather than in content.

    `views` is the authority: (shot, frame) per pose, in pose order.
    """
    if shot_id is None:
        return frames
    keep = [j for j, (sh, f) in enumerate(model.views)
            if int(sh) == int(shot_id) and 0 <= int(f) < len(frames)]
    if not keep:
        return frames
    backend.poses = model.poses().subset(keep)
    return [frames[int(model.views[j][1])] for j in keep]


def process_shot(frames, tracker, fill_holes=True, dark_generator=None,
                 sources=None, prefer_3d=False, context=None,
                 thresholds=None):
    """-> list of (canvas, provenance) plus the shot's decision record."""
    kind, stats = wc.classify_motion(tracker, frames)
    backend = bk.pick(kind, prefer_3d=prefer_3d)

    # pick() builds a bare GaussianBackend, so colmap_dir was always None and
    # every 3D shot fell back to essential poses and refused. Attach the
    # reconstruction build_film solved for this shot's scene -- but only a
    # usable one. build_film returns partial models too (that is how it reports
    # "1 partial"), and handing the backend a 48%-registered solve would buy a
    # render by discarding the refusal that makes the number mean anything.
    # Not usable: leave colmap_dir None and let the backend refuse, loudly.
    src0 = sources or {}
    model = (src0.get("scene_models") or {}).get(src0.get("scene_id"))
    if (model is not None and model.usable
            and getattr(backend, "colmap_dir", "n/a") is None):
        backend.colmap_dir = model.sparse_dir
        frames = align_to_poses(model, backend, frames, src0.get("shot_id"))
    h, w = frames[0].shape[:2]
    ww = int(w * WING)
    shot_log = []

    def borrow(reason):
        """
        Offer a refused shot the rungs whose evidence is not its own.

        The gate's verdict is about THIS shot's recovery and nothing else. A
        take matched against another cut is verified by homography inliers
        computed on footage this shot's geometry played no part in, so the
        refusal says nothing about those pixels. Withholding them meant DONATED
        and RETRIEVED could only ever reach shots that had already cleared the
        gate -- shots with real wings already, needing donation least -- while
        the locked-off close-up the whole ladder was built for was the one case
        structurally excluded from them.

        Returns (pairs, record) or None to fall through to dark/invented wings.
        """
        src = sources or {}
        if not (src and fill_holes):
            return None
        out = []
        for i, f in enumerate(frames):
            canvas = np.zeros((h, w + 2 * ww, 3), np.uint8)
            canvas[:, ww:ww + w] = f
            filled = np.zeros((h, w + 2 * ww), bool)
            filled[:, ww:ww + w] = True          # the centre only: no recovery
            tmap = np.zeros((h, w + 2 * ww), np.int32)
            r = dr.Director(goal=dr.Goal(max_actions=2)).run(
                canvas, filled, tmap, ww, frames=frames,
                scene_id=src.get("scene_id"),
                corpus_finder=src.get("corpus_finder"),
                scene_finder=src.get("scene_finder"),
                fetcher=src.get("fetcher"), verbose=False,
                shot_id=src.get("shot_id"),
                scene_models=src.get("scene_models"),
                backend=None, allow_provenance=BORROWABLE)
            for st in r.get("trace", ()):
                shot_log.append(f"{st.action} {st.gain*100:+.1f}% :: {st.note}")
            if i == 0 and not any(st.gain > 0 for st in r.get("trace", ())):
                return None      # nothing to borrow: do not pay for the rest
            out.append((r["canvas"], r["provenance"]))
        if not out:
            return None
        mid = out[len(out) // 2][1]
        wings = np.hstack([mid[:, :ww], mid[:, -ww:]])
        got = float(np.isin(wings, ag.PHOTOGRAPHIC).mean())
        stats["actions"] = ",".join(sorted(set(x.split(" ")[0] for x in shot_log)))
        stats["action_gain"] = action_gains(shot_log)
        return out, dict(motion=kind, backend="borrowed", geometry=0.0,
                         state="BORROWED", coverage=round(got, 4),
                         effective=round(got, 4),
                         reasons=f"{reason}; wings borrowed from other footage",
                         **stats)

    def unrecoverable(reason, backend_name="none"):
        """
        Wings with nothing real behind them: dark, or invented if asked.

        Two different shots land here -- a locked-off camera that never saw
        past its own frame edge, and a moving one whose poses the backend
        refused. Both mean the same thing to the metric, and both must be
        RECORDED rather than raised: a refusal is a verdict about this shot,
        not grounds for abandoning the film.
        """
        out = []
        failure = ""
        if dark_generator is not None:
            label = _apply_context(dark_generator, context, frames, ww)
            if hasattr(dark_generator, "generate_shot"):
                got, failure = safe_generate(
                    _shot_generate, dark_generator, frames, ww,
                    getattr(dark_generator, "prompt", None))
                out = [(c, np.where(p == ag.GENERATED, label, p).astype(np.uint8))
                       for c, p in (got or [])]
            else:
                for f in frames:
                    got, failure = safe_generate(synth_wings, f, ww,
                                                 dark_generator, label)
                    if got is None:
                        out = []
                        break
                    out.append(got)
            if failure:
                print(f"  generator failed, wings stay dark: {failure}", flush=True)
                reason = (reason + "; " if reason else "") + f"generator failed ({failure})"
        if dark_generator is None or not out:
            blank = np.zeros((h, w + 2 * ww, 3), np.uint8)
            for f in frames:
                c = blank.copy()
                c[:, ww:ww + w] = f
                p = np.full((h, w + 2 * ww), ag.GENERATED, np.uint8)
                p[:, ww:ww + w] = ag.PRIMARY
                out.append((c, p))
        return out, dict(motion=kind, backend=backend_name, geometry=0.0,
                         state="GEN" if (dark_generator is not None
                                         and not failure) else "OFF",
                         coverage=0.0, effective=0.0,
                         reasons=reason, **stats)

    if backend is None:
        # locked off: nothing THIS shot filmed is recoverable -- but another cut
        # may have framed the same take wider, which is the case the rung exists
        # for and the one it could never reach.
        got = borrow("locked off")
        if got:
            return got
        return unrecoverable("wings invented; nothing was filmed out there"
                             if dark_generator is not None else "")

    try:
        res = backend.propagate(frames, ww, tracker)
    except RuntimeError as e:
        # The backend refuses poses it cannot vouch for -- COLMAP failed on this
        # scene, or registered too little of it, so all that is left is an
        # essential-matrix chain with no recovered scale. That refusal used to
        # escape run() and kill the whole film: every other shot's verdict and
        # the summary itself went with it, which is the opposite of what a gate
        # is for. Record it as this shot's verdict and carry on.
        pz = getattr(backend, "last_poses", None)
        why = (f"no trustworthy poses ({pz.source}, inlier {pz.inlier_ratio:.2f})"
               if pz is not None else f"backend refused: {e}"[:120])
        print(f"  shot refused: {why}", flush=True)
        return unrecoverable(why, backend_name=backend.name)

    # Steady the wall before anything measures or fills it.
    #
    # Propagation assembles each frame's wing on its own, so the wall
    # re-assembles every frame and the result boils -- measured at 2.6x the
    # centre's frame-to-frame change on a clip whose walls were 99% real
    # photography. settle_wings medians each frame against its neighbours,
    # aligned, using nothing but pixels this camera already shot. It refines
    # values only: `filled` and `tmap` come through untouched, so coverage,
    # staleness and every provenance rung below are computed on exactly the
    # same evidence they were before.
    res = wc.settle_wings(res, backend.warp_between, ww)

    geom, _ = g.leave_one_out(frames, backend, tracker, probes=3)

    mid = res[len(res) // 2]
    m = wc.wing_metrics(mid[1], mid[2], ww, w, 24.0, mid[0])
    state, ratio, why = g.decide(m, geom, thresholds)

    if state == "OFF":
        # same argument as the locked-off case: the gate rejected this shot's
        # own recovery, which is no verdict on another cut's pixels
        got = borrow("; ".join(why) if why else "gated off")
        if got:
            return got

    # A temporal generator gets the whole shot in one submission. The loop
    # below hands the per-frame fallback one frame at a time, which for a video
    # model is not merely slower -- it is invalid: WaveSpeed rejected every such
    # job with "RIFE requires at least 2 frames, only found 1". The locked-off
    # path already did this; the gated-OFF path never did, so the shot-level
    # generator had never once run through the pipeline.
    if (state == "OFF" and dark_generator is not None
            and hasattr(dark_generator, "generate_shot")):
        label = _apply_context(dark_generator, context, frames, ww)
        got, failure = safe_generate(_shot_generate, dark_generator, frames, ww,
                                     getattr(dark_generator, "prompt", None))
        if got:
            pairs = [(c, np.where(p == ag.GENERATED, label, p).astype(np.uint8))
                     for c, p in got]
            return pairs, dict(motion=kind, backend=backend.name, geometry=round(geom, 1),
                               state="GEN", coverage=round(m["coverage"], 4),
                               effective=round(m["effective_coverage"], 4),
                               reasons="; ".join(why + ["wings invented"]), **stats)
        print(f"  shot-level generator failed, wings stay dark: {failure}", flush=True)
        why = why + [f"generator failed ({failure})"]

    out = []
    gen_failed = False
    # Released as it is consumed. Both stacks alive at once is two full copies
    # of every canvas in the shot -- four gigabytes on a 799-frame 1024px clip,
    # on a machine with less than that free -- and nothing reads a propagated
    # frame again after the planner has turned it into an output frame.
    for _i in range(len(res)):
        canvas, filled, tmap = res[_i]
        res[_i] = None
        if state == "OFF":
            if dark_generator is not None:
                got, failure = safe_generate(
                    synth_wings, canvas[:, ww:ww + w], ww, dark_generator,
                    _apply_context(dark_generator, context, frames, ww))
                if got is not None:
                    out.append(got)
                    continue
                note = f"generator failed ({failure})"
                if note not in why:          # once per shot, not once per frame
                    print(f"  generator failed, wings stay dark: {failure}",
                          flush=True)
                    why = why + [note]
                gen_failed = True
            c = np.zeros_like(canvas)
            c[:, ww:ww + w] = canvas[:, ww:ww + w]
            p = np.full(filled.shape, ag.GENERATED, np.uint8)
            p[:, ww:ww + w] = ag.PRIMARY
            out.append((c, p))
            continue
        if fill_holes:
            src = sources or {}
            # with sources in hand the planner can try the real rungs first;
            # without them its option set is `generate` and nothing else
            goal = (dr.Goal(max_actions=4) if src else
                    dr.Goal(real_same_camera=0.0, photographic=0.0,
                            max_generated=1.0, max_actions=3))
            d = dr.Director(goal=goal)
            r = d.run(canvas, filled, tmap, ww, frames=frames,
                      scene_id=src.get("scene_id"),
                      corpus_finder=src.get("corpus_finder"),
                      scene_finder=src.get("scene_finder"),
                      fetcher=src.get("fetcher"), verbose=False,
                      shot_id=src.get("shot_id"),
                      scene_models=src.get("scene_models"),
                      backend=backend if prefer_3d else None)
            out.append((r["canvas"], r["provenance"]))
            if src:
                for s in r.get("trace", ()):
                    shot_log.append(f"{s.action} {s.gain*100:+.1f}% :: {s.note}")
        else:
            p = np.full(filled.shape, ag.GENERATED, np.uint8)
            p[filled] = ag.RECOVERED
            centre = np.zeros(filled.shape, bool)
            centre[:, ww:ww + w] = True
            p[centre & filled] = ag.PRIMARY
            out.append((canvas, p))

    if state == "OFF" and dark_generator is not None and not gen_failed:
        state = "GEN"
        why = why + ["wings invented"]
    # a shot whose generator failed keeps its OFF verdict: claiming GEN would
    # report invented wings that were never drawn

    if shot_log:                       # what the planner actually tried
        seen = sorted(set(s.split(" ")[0] for s in shot_log))
        stats["actions"] = ",".join(seen)
        stats["action_gain"] = action_gains(shot_log)

    return out, dict(motion=kind, backend=backend.name, geometry=round(geom, 1),
                     state=state, coverage=m["coverage"],
                     effective=m["effective_coverage"], reasons="; ".join(why),
                     **stats)


def _wavespeed():
    import wavespeed
    return wavespeed.WaveSpeedOutpainter()


def _gemini_edit():
    import gemini
    return gemini.GeminiImageEdit()


GENERATORS = {"mirror": fl.MirrorGenerator, "inpaint": fl.InpaintGenerator,
              "diffusion": fl.DiffusionGenerator, "hosted": fl.HostedGenerator,
              "wavespeed": _wavespeed, "gemini-edit": _gemini_edit}


def run(path, outdir="jobs/cli", maxw=480, max_shots=None,
        frames_per_shot=None, theatre=None, rotate=0, wings_on_dark=None,
        use_sources=False, library=None, online=False, sfm_dir=None,
        prefer_3d=False, context_paths=None, reason=False, vision=False,
        also=None, other_cuts=None, on_shot=None, wing=None,
        thresholds=None, deliver=None):
    os.makedirs(outdir, exist_ok=True)
    dark_generator = GENERATORS[wings_on_dark]() if wings_on_dark else None

    # anything anyone asserted about this footage: scripts, subtitles, stills,
    # plus notes a person pinned to a shot on a previous pass
    bundle = cx.ContextBundle.from_paths(context_paths or [])
    directions = cx.DirectionStore(outdir)
    bundle.items.extend(directions.items)
    if bundle.items:
        cs = bundle.summary()
        print(f"  context: {cs['items']} item(s) {cs['kinds']}, "
              f"{cs['bound']} bound to time or shot, {cs['unusable']} unusable",
              flush=True)
    seg = sd.segment(path)
    shots = [s for s in seg["shots"] if s[1] - s[0] >= 12]
    if max_shots:
        shots = shots[:max_shots]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    tracker = wc.Tracker()

    theatre = theatre or wl.Theatre()
    if wing:
        # WING is read by process_shot, the metrics and the wall projection, so
        # threading it through every call site would touch more than it is
        # worth. Set once per run, and say so rather than hiding it.
        global WING
        WING = float(wing)

    # a shot only reports once it is finished; say up front how many there are
    budget = sum(min(b - a, frames_per_shot or (b - a)) for a, b in shots)
    print(f"  {len(shots)} shot(s), {budget} frames to process", flush=True)

    # pass one: index the whole film before filling any of it, so an early shot
    # can borrow from a later one. Cheap -- 3 small frames per shot.
    index = None
    fetcher = None
    if use_sources:
        index = fx.FilmIndex()
        declared = SAME_PLACE if also else None
        # reconstruction needs frames, not thumbnails: 3 samples at 320px is
        # enough to MATCH a shot and nowhere near enough to SOLVE it
        n_sample = SFM_FRAMES if sfm_dir else 8
        for si, (a, b) in enumerate(shots):
            fr = load_shot(cap, seg["crop"], a, b, maxw, n_sample, rotate,
                           spread=True)
            if len(fr) >= 2:
                index.add(si, fr, scene=declared,
                          sfm_frames=fr if sfm_dir else None)
        # other clips of the SAME LOCATION -- a master wide beside a locked-off
        # close-up. Without this the index holds one setup per scene, build_film
        # skips every one of them, and RETRIEVED cannot fire however good the
        # reconstruction is.
        # you filmed these as setups of one wall, so they are declared one
        # scene rather than guessed into one -- see FilmIndex.add
        for extra_i, extra in enumerate(also or []):
            name = os.path.splitext(os.path.basename(extra))[0]
            eseg = sd.segment(extra)
            eshots = [t for t in eseg["shots"] if t[1] - t[0] >= 12]
            ecap = cv2.VideoCapture(extra)
            n_added = 0
            for ei, (ea, eb) in enumerate(eshots):
                efr = load_shot(ecap, eseg["crop"], ea, eb, maxw, n_sample,
                                rotate, spread=True)
                if len(efr) >= 2:
                    # shot ids must stay globally unique. sfm.write_images names
                    # views s{shot}_f{frame} with no film component, so clipA's
                    # shot 0 silently overwrote clipB's -- two setups collapsed
                    # into three images of one, COLMAP produced an empty sparse
                    # dir, and the backend fell back to essential poses and
                    # refused. Offsetting here fixes it without changing the
                    # manifest schema, which agent.py unpacks as (shot, frame).
                    index.add(SETUP_ID_BASE * (extra_i + 1) + ei, efr,
                              film=name, scene=SAME_PLACE,
                              sfm_frames=efr if sfm_dir else None)
                    n_added += 1
            ecap.release()
            print(f"  + {n_added} shot(s) from {name} as another setup", flush=True)

        # a SECOND CUT of the same picture -- a TV spot beside the trailer.
        # Not the same location (that is --also) but the same TAKE, which
        # TakeMatcher finds with ORB and a homography, no GPU. Inside one film a
        # take appears once, so SameTakeTool had nothing to match against and
        # DONATED could never fire however good the footage was. Across two cuts
        # it appears twice, differently cropped and graded -- and wherever the
        # other cut framed wider, it donates real periphery this one never had.
        for cut_i, cut in enumerate(other_cuts or []):
            name = os.path.splitext(os.path.basename(cut))[0]
            cseg = sd.segment(cut)
            cshots = [t for t in cseg["shots"] if t[1] - t[0] >= 12]
            ccap = cv2.VideoCapture(cut)
            per_shot_frames = []
            for (ca, cb) in cshots:
                per_shot_frames.append(
                    load_shot(ccap, cseg["crop"], ca, cb, maxw, n_sample, rotate,
                              spread=True))
            ccap.release()
            n_added = index.add_film(name, per_shot_frames,
                                     id_base=CUT_ID_BASE * (cut_i + 1))
            print(f"  + {n_added} shot(s) from {name} as another cut", flush=True)

        fetcher = ft.default_fetcher(library=library, online=online)
        s = index.summary()
        print(f"  indexed {s['shots']} shots into {s['scenes']} scene(s); "
              f"{s['multi_setup_scenes']} have more than one setup", flush=True)

    # pass 1b: reconstruct each multi-setup location. This is the expensive
    # step and the one that makes RETRIEVED possible at all -- a homography
    # cannot bridge two setups, so without these poses the same_location rung
    # can only refuse.
    scene_models = None
    if sfm_dir and index is not None:
        import sfm
        # min_setups=1 under --prefer-3d. The default of 2 skips a lone shot on
        # the grounds that GaussianBackend can find its own poses -- but what it
        # finds is essential poses, which it then refuses to render from. So a
        # single moving shot could never use the 3D path at all: build_film
        # declined to solve it and the backend declined to guess.
        scene_models = sfm.build_film(index, sfm_dir,
                                      min_setups=1 if prefer_3d else 2)

    reasoner = None
    if reason or vision:
        if vision:
            import gemini
            reasoner = gemini.reasoner()
            print("  reasoning with Gemini vision", flush=True)
        else:
            reasoner = rz.LocalReasoner()

    # the splat renderer needs one pose per frame, and COLMAP only solved the
    # frames handed to it, so in 3D mode we render precisely those
    per_shot = SFM_FRAMES if (prefer_3d and sfm_dir) else frames_per_shot

    rendered, records, wing_px = [], [], 0
    for si, (a, b) in enumerate(shots):
        fr = load_shot(cap, seg["crop"], a, b, maxw, per_shot, rotate)
        if len(fr) < 10:
            continue
        wing_px = int(fr[0].shape[1] * WING)   # recorded, not re-derived later
        sources = None
        if index is not None:
            sources = dict(scene_id=index.scene_of(si, "primary"),
                           corpus_finder=index.corpus_finder(si),
                           scene_finder=index.scene_finder(si),
                           fetcher=fetcher, scene_models=scene_models,
                           shot_id=si)
        shot_context = None
        if bundle.items or reason:
            shot_context = dict(items=bundle.for_shot(si, a, len(fr), fps),
                                shot=si, reasoner=reasoner)
            if reason:
                # what actually left frame here -- the only evidence in the
                # plan that is not somebody's say-so
                fh, fw = fr[0].shape[:2]
                tracks = ofs.link_tracks(ofs.detect_moving(fr), frame_size=(fw, fh))
                shot_context["excursions"] = ofs.find_excursions(tracks, fw, fh)
        pairs, rec = process_shot(fr, tracker, dark_generator=dark_generator,
                                  context=shot_context,
                                  sources=sources, prefer_3d=prefer_3d,
                                  thresholds=thresholds)
        if shot_context and shot_context.get("plan") is not None:
            rec["reasoning"] = shot_context["plan"].explain()
        rec.update(shot=si, start=a, frames=len(pairs))
        records.append(rec)
        rendered.extend(pairs)
        # a shot is the unit of work, so a shot is what reaches disk
        checkpoint(outdir, records, path, partial=True)
        if deliver:
            try:
                # the source's own rate, not 24. These segments are what
                # polish.rebuild re-cuts the film from, so a hardcoded 24 here
                # re-timed a 30fps clip the moment anyone ran the finishing
                # pass -- 500 frames stretched from 16.7s to 20.8s.
                write_shot_segment(deliver, si, pairs, wing_px, theatre, fps=fps)
            except Exception as e:
                print(f"  segment for shot {si} not written: "
                      f"{type(e).__name__}: {e}", flush=True)
        print(f"  shot {si:3d} {rec['motion']:9s} {rec['state']:6s} "
              f"geom {rec['geometry']:5.1f}dB eff {rec['effective']*100:5.1f}%",
              flush=True)
        if on_shot is not None:
            # the record itself, not a line of text to be regex-scraped back
            # apart. A caller watching a render wants action_gain and the
            # reasons string, and neither survives the print above.
            try:
                on_shot(rec)
            except Exception as e:          # a watcher must not kill a render
                print(f"  (on_shot failed: {type(e).__name__}: {e})", flush=True)
    cap.release()

    if not rendered:
        raise SystemExit("no shots rendered")

    h, cw = rendered[0][0].shape[:2]
    ww = int(round((cw * WING) / (1 + 2 * WING)))
    probe = wl.render(rendered[0][0], ww, theatre)
    sheet0 = wl.contact_sheet(probe)
    H, W = sheet0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(f"{outdir}/screenx_demo.mp4", fourcc, fps, (W, H))

    n = len(rendered)
    for i, (canvas, prov) in enumerate(rendered):
        reveal = i >= n // 2
        panels = wl.render(canvas, ww, theatre, provenance=prov,
                           mark_generated=reveal)
        sheet = wl.contact_sheet(panels)
        if sheet.shape[:2] != (H, W):
            sheet = cv2.resize(sheet, (W, H))
        tag = "REVEAL: magenta = not filmed here" if reveal else "SCREENING"
        cv2.putText(sheet, tag, (W - 330, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255) if reveal else (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(sheet)
    vw.release()
    to_h264(os.path.join(outdir, "screenx_demo.mp4"), fps)

    if deliver:
        write_deliverable(Path(deliver), rendered, ww, theatre, fps)

    # BOTH wings. This counted only the left one, while WingAgent.report has
    # always used both -- so the headline number and the per-shot report were
    # measuring different things, and an asymmetric shot (a pan, where the
    # trailing wing recovers and the leading one cannot) read as whichever side
    # happened to be left.
    real = float(np.mean([
        np.isin(np.hstack([p[:, :ww], p[:, -ww:]]), ag.PHOTOGRAPHIC).mean()
        for _, p in rendered]))

    # Which rungs actually put pixels on a wall. WingAgent.report has always
    # produced this and nothing ever called it at the end of a run, so a claim
    # like "DONATED fired" lived in the pixels and never in the summary.
    def rungs(pairs):
        if not pairs:
            return {}
        w0 = pairs[0][0].shape[1] - 2 * ww
        acc = {}
        for _, prov in pairs:
            for k, v in ag.WingAgent.report(prov, ww, w0).items():
                acc[k] = acc.get(k, 0.0) + v
        return {k: round(v / len(pairs), 4) for k, v in acc.items()}

    at = 0
    for rec in records:
        n_shot = int(rec.get("frames") or 0)
        rec["provenance"] = rungs(rendered[at:at + n_shot])
        at += n_shot
    overall = rungs(rendered)
    fired = sorted(k for k in ag.PROV_NAMES.values()
                   if overall.get(k, 0.0) > 0.0005)
    summary = dict(
        source=os.path.basename(path), shots=len(records), frames=n,
        wing_ratio=WING,
        # The rate the film was shot at, and the wing width in pixels.
        #
        # Both were missing, and both had to be guessed by anything that re-cut
        # the deliverable later. polish.rebuild guessed 24fps and silently
        # slowed a 30fps film by a quarter; polish.repair re-derived wing_w from
        # the ratio with different rounding than the renderer used, which puts
        # the fence a column off and quietly breaks "the centre never changes".
        # Neither is a guess worth making when the renderer knows the answer.
        fps=round(float(fps), 6), wing_w=wing_px,
        # The delivered canvas. Recorded for the same reason as the two above:
        # a row in the ledger that cannot say what size the film was cannot be
        # compared with one rendered at a different working width, and this
        # project has already been bitten once by a measurement that silently
        # meant something different at 1024px than it did at 480.
        width=int(rendered[0][0].shape[1]) if rendered else 0,
        height=int(rendered[0][0].shape[0]) if rendered else 0,
        # what the verdicts were judged against. A summary that does not say
        # which bar it used cannot be compared with one that used another.
        gate={**g.THRESHOLDS, **(thresholds or {})},
        # the auditorium the feeds were projected onto. Recorded for the same
        # reason the gate is: without it a later pass -- polish rebuilding the
        # deliverable, or anyone reproducing the run -- has to assume the
        # defaults, and silently delivers different geometry from the one the
        # operator asked for.
        theatre={k: getattr(theatre, k) for k in
                 ("screen_width", "screen_height", "viewer_distance",
                  "panel_width", "panel_height", "wing_dim", "feather_px")},
        wings_on=sum(1 for r in records if r["state"] not in ("OFF", "GEN")),
        wings_generated=sum(1 for r in records if r["state"] == "GEN"),
        mean_effective=round(float(np.mean([r["effective"] for r in records])), 4),
        mean_real_wing=round(real, 4),
        provenance=overall,
        rungs_fired=fired,
        extent=probe["extent"],
        per_shot=records,
    )
    with open(f"{outdir}/screenx_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default="jobs/cli")
    ap.add_argument("--maxw", type=int, default=480)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--frames-per-shot", type=int, default=None)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="explicit rotation; some clips carry no metadata")
    ap.add_argument("--wings-on-dark", nargs="?", const="mirror", default=None,
                    choices=list(GENERATORS),
                    help="invent wings on shots the measurement gates OFF; "
                         "every pixel is marked GENERATED and counts as invented")
    ap.add_argument("--sources", action="store_true",
                    help="index the film and let the planner try the DONATED and "
                         "RETRIEVED rungs before it generates anything")
    ap.add_argument("--library", default=None,
                    help="folder of licensed reference material (needs licences.json)")
    ap.add_argument("--sfm", default=None,
                    help="directory for COLMAP scene reconstructions; enables "
                         "the same_location rung (needs colmap on PATH)")
    ap.add_argument("--prefer-3d", action="store_true",
                    help="use GaussianBackend where available (needs CUDA)")
    ap.add_argument("--other-cut", nargs="*", default=None, metavar="CLIP",
                    help="another CUT of the same picture -- a TV spot, a "
                         "teaser. Required for DONATED: one take has to appear "
                         "twice before a wider framing of it can donate "
                         "periphery")
    ap.add_argument("--also", nargs="*", default=None, metavar="CLIP",
                    help="other clips of the SAME LOCATION -- the master wide "
                         "beside this close-up. Required for RETRIEVED: a scene "
                         "needs two setups before it can be reconstructed")
    ap.add_argument("--reason", action="store_true",
                    help="find what left frame in each shot and reason about where "
                         "it went; slower, and the only evidence that is not hearsay")
    ap.add_argument("--vision", action="store_true",
                    help="add a Gemini vision model's claims to the reasoning "
                         "(needs GEMINI_API_KEY); its output is asserted, never measured")
    ap.add_argument("--context", nargs="*", default=None, metavar="PATH",
                    help="files or folders of context: subtitles, script, stills, "
                         "notes. Anything bound to a shot makes its wings DIRECTED "
                         "rather than GENERATED, and neither counts as filmed")
    ap.add_argument("--deliver", nargs="?", const="deliverable", default=None,
                    metavar="DIR",
                    help="write the extended film itself: left.mp4, centre.mp4, "
                         "right.mp4 and the widened master, clean and unmarked. "
                         "The demo video beside it is the review copy, with the "
                         "reveal pass and the magenta tint on anything that was "
                         "not filmed at this location")
    ap.add_argument("--wing", type=float, default=None,
                    help="wing width as a fraction of the frame (default 0.22, "
                         "the geometrically projectable width)")
    ap.add_argument("--screen-width", type=float, default=None,
                    help="auditorium screen width in metres (default 14)")
    ap.add_argument("--screen-height", type=float, default=None, help="default 6")
    ap.add_argument("--viewer-distance", type=float, default=None,
                    help="viewer to screen plane, metres (default 12)")
    ap.add_argument("--wing-dim", type=float, default=None,
                    help="side walls run darker, as ScreenX does (default 0.82)")
    ap.add_argument("--gate-geometry", type=float, default=None,
                    help="dB below which the backend is not trusted (default 20)")
    ap.add_argument("--gate-full", type=float, default=None,
                    help="effective coverage for FULL (default 0.55)")
    ap.add_argument("--gate-narrow", type=float, default=None,
                    help="effective coverage for NARROW (default 0.25)")
    ap.add_argument("--gate-detail", type=float, default=None,
                    help="minimum wing detail (default 0.10)")
    ap.add_argument("--gate-stale", type=float, default=None,
                    help="seconds before a wing is anachronistic (default 1.20)")
    ap.add_argument("--progress-json", action="store_true",
                    help="emit one JSON line per finished shot on stdout, "
                         "prefixed @@SHOT, for a caller driving this as a "
                         "subprocess")
    ap.add_argument("--online", action="store_true",
                    help="also query Openverse for openly-licensed material")
    a = ap.parse_args()
    s = run(a.video, a.outdir, a.maxw, a.max_shots, a.frames_per_shot,
            rotate=a.rotate, wings_on_dark=a.wings_on_dark,
            use_sources=(a.sources or a.online or bool(a.library)
                         or bool(a.sfm) or bool(a.also)
                         or bool(a.other_cut)),
            sfm_dir=a.sfm, prefer_3d=a.prefer_3d, context_paths=a.context,
            reason=a.reason, vision=a.vision, also=a.also,
            library=a.library, online=a.online,
            other_cuts=a.other_cut,
            on_shot=_emit_shot if a.progress_json else None,
            wing=a.wing, theatre=_theatre_from(a),
            thresholds=_thresholds_from(a),
            deliver=(os.path.join(a.outdir, a.deliver)
                     if a.deliver and not os.path.isabs(a.deliver)
                     else a.deliver))
    print(json.dumps({k: v for k, v in s.items() if k != "per_shot"},
                     indent=2, default=float))
