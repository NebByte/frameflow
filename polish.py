"""
polish -- look at the finished walls, find what is wrong, and redo only those.

A render decides what CAN be earned. It never asks whether the result looks
good, because nothing in the measurement can see: coverage, staleness and
detail are statistics over pixels, and none of them notices that a wall is a
smear of the same column repeated forty times. A person spots that in a second.
So does a vision model.

    inspect(job)   sample the finished wings, ask a model what is wrong
    repair(job)    regenerate the shots it flagged, with the finding as the brief

THE RULE THAT SHAPES ALL OF IT
-----------------------------
Any shot may be polished, including one whose wings were photographed -- a
streaked recovered wall is a real defect and a maker should be able to fix it.
What may NOT happen is a repaint that keeps counting as photography.

So every pixel the model changes is relabelled: GENERATED, or DIRECTED where a
person's note drove the change. `mean_real_wing` falls by exactly the repainted
fraction, and the polish report states the before and after side by side. The
film gets better; the number stays true; the cost is visible.

Refusing would have been the easier answer and the worse one. A tool that will
not let you fix an ugly wall is not much of a tool -- the job is to make the
trade explicit, not to forbid it.

WHAT THE MODEL'S OPINION IS WORTH
---------------------------------
Nothing, evidentially. Its findings are `asserted` -- the same rung a script
page sits on -- and they can only ever cause a regeneration of already-invented
pixels. No finding can move a number.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

# Shots whose wings are real. Polishing one is allowed and costs something
# measurable, which is why the states are named rather than inlined.
PHOTOGRAPHED_STATES = ("FULL", "NARROW", "BORROWED")

# A repaint counts as a change when it moves a pixel by more than this. Low
# enough to catch a re-drawn wall, high enough to ignore recompression: a
# re-encode alone shifts pixels by one or two levels, and calling that a repaint
# would relabel photographed pixels for no reason.
REPAINT_THRESHOLD = 12

PROMPT = (
    "This is one frame of a 270-degree cinema conversion. The middle is the "
    "original photographed frame. The left and right thirds are side walls that "
    "software produced. Look ONLY at the left and right thirds.\n"
    "List defects you can actually see, worst first. Use these words where they "
    "fit: streaking, smearing, repeated texture, stretched pixels, visible seam, "
    "colour mismatch, implausible content, empty. If a wall looks acceptable, "
    "say 'acceptable' for that side.\n"
    "Answer as a JSON array of short strings, at most 4, each naming the side."
)


def wing_strip(canvas, wing_w, side="both"):
    """The walls alone, with a sliver of centre for context on the seam."""
    h, w = canvas.shape[:2]
    ctx = max(8, wing_w // 4)
    if side == "left":
        return canvas[:, :wing_w + ctx]
    if side == "right":
        return canvas[:, w - wing_w - ctx:]
    return canvas


def sample_frames(video_path, n=2):
    """A couple of frames from a finished shot: first third and last third."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total <= 0:
        cap.release()
        return out
    for idx in np.linspace(total * 0.25, total * 0.75, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def burst(video_path, n=40, at=0.4):
    """A run of CONSECUTIVE frames. Jitter is invisible in spread samples."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(total - n, total * at))))
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
    cap.release()
    return out


def streak_score(canvas, wing_w):
    """
    A measured companion to the model's opinion.

    Streaking is the failure propagation actually produces -- the last column
    with any content gets stretched outward -- and it has a signature a model
    does not need to be trusted for: columns in the wing that are near-identical
    to their neighbour. Reported alongside the model so a reader can see whether
    the two agree.
    """
    h, w = canvas.shape[:2]
    if wing_w < 4:
        return 0.0
    flat = 0
    for band in (canvas[:, :wing_w], canvas[:, w - wing_w:]):
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.int16)
        d = np.abs(np.diff(g, axis=1)).mean(axis=0)
        flat += int((d < 1.5).sum())
    return round(flat / float(2 * wing_w), 4)


def hairline_score(canvas, wing_w):
    """
    Fraction of wing columns carrying a thin dark line.

    The defect this names was a bug in the renderer's warp, not a limit of
    recovery, and it is measured here for the same reason `streak_score` is:
    so a reader can see whether it is still there rather than take anyone's
    word for it. A hairline is a column markedly darker than BOTH its
    neighbours -- a wall genuinely has dark columns, but not ones that are dark
    only relative to what sits either side.
    """
    h, w = canvas.shape[:2]
    if wing_w < 4:
        return 0.0
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hits = 0
    for band in (g[:, :wing_w], g[:, w - wing_w:]):
        c = band.mean(axis=0)
        mid, left, right = c[1:-1], c[:-2], c[2:]
        hits += int(((mid < left - 8) & (mid < right - 8)).sum())
    return round(hits / float(2 * wing_w), 4)


def jitter_ratio(frames, wing_w):
    """
    How much more the walls boil than the picture they flank.

    The centre is one continuous take, so its frame-to-frame change is the
    scene's own motion -- the baseline any wall should be compared against.
    A wall assembled independently every frame moves more than that, and the
    excess is the shimmer. 1.0 means the wall is as steady as the film.
    """
    if len(frames) < 2 or wing_w < 1:
        return None
    cw = frames[0].shape[1]
    dw, dc = [], []
    for a, b in zip(frames, frames[1:]):
        d = np.abs(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
                   - cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32))
        dw.append(float(np.concatenate([d[:, :wing_w].ravel(),
                                        d[:, cw - wing_w:].ravel()]).mean()))
        dc.append(float(d[:, wing_w:cw - wing_w].mean()))
    c = float(np.mean(dc))
    return round(float(np.mean(dw)) / c, 3) if c > 1e-6 else None


def seam_step(canvas, wing_w):
    """
    Whether the wall actually JOINS the picture, or is cut against it.

    Nothing in the render measured this, which is how a shot scored 32.4 dB of
    geometry while its walls repeated signage that was still on screen in the
    centre: leave-one-out validates one frame warped against another, and never
    asks whether the wing lines up with the frame it is extending. The last
    wing column and the first centre column are adjacent samples of one
    surface, so on a continuous wall their difference is an ordinary
    column-to-column step. 1.0 is continuous; higher is a visible cut.
    """
    h, w = canvas.shape[:2]
    if wing_w < 2 or w - 2 * wing_w < 60:
        return None
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=1))
    inside = float(d[:, wing_w + 20:w - wing_w - 20].mean())
    if inside < 1e-6:
        return None
    edge = float((d[:, wing_w - 1].mean() + d[:, w - wing_w - 1].mean()) / 2.0)
    return round(edge / inside, 3)


def defect_mask(canvas, wing_w, prov=None):
    """
    Where the wall is actually broken -- so a repaint can be aimed at THAT.

    The first version of `repair` kept the centre and took everything else from
    the generator. On a wing that was 99.07% photographed it redrew 62% of the
    wall to fix artifacts measuring 1.5% of columns, and handed back a
    hard-edged rectangle carrying a duplicated gym logo. The trade was never
    worth making and did not need to be made: the broken pixels are findable.

    Three kinds, all of them things a generator can only improve:
      dead      nothing was ever put there
      hairline  a thin dark line, whatever put it there
      invented  the provenance map already calls it not-photographed

    Everything else is photography and is left alone. The centre is excluded
    outright -- the fence does not move for this.
    """
    h, cw = canvas.shape[:2]
    bad = canvas.max(axis=2) <= 6
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for x0, x1 in ((0, wing_w), (cw - wing_w, cw)):
        if x1 - x0 < 3:
            continue
        band = g[:, x0:x1]
        mid, left, right = band[:, 1:-1], band[:, :-2], band[:, 2:]
        bad[:, x0 + 1:x1 - 1] |= (mid < left - 8) & (mid < right - 8)
    if prov is not None and prov.shape == bad.shape:
        import agent as ag
        bad |= ~np.isin(prov, ag.PHOTOGRAPHIC)
    bad[:, wing_w:cw - wing_w] = False
    return bad


def feather(mask, grow=3, blur=11):
    """
    Soften a repaint's edge so it does not read as a pasted rectangle.

    A binary composite puts a hard vertical step down both sides of whatever
    was redrawn, and because a shot's defects sit in much the same place all
    the way through, that step stands in the same column for the entire shot --
    which is a worse artifact than the one being fixed.
    """
    m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((grow, grow), np.uint8))
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (blur | 1, blur | 1), 0)
    return np.clip(a, 0.0, 1.0)


def dims(summary, canvas_w=None):
    """
    The wing width and frame rate this job was rendered with.

    Both are recorded by the renderer now. Both used to be re-derived here, and
    both were re-derived wrongly: the wing width with different rounding than
    the renderer used, which walks the fence off by a column, and the frame
    rate as a flat 24.0, which re-timed a 30fps film by a quarter the moment
    anyone pressed the finishing button. Read what was written; fall back only
    for jobs rendered before it was.
    """
    fps = float(summary.get("fps") or 0.0) or 24.0
    ww = int(summary.get("wing_w") or 0)
    if ww <= 0 and canvas_w:
        ratio = float(summary.get("wing_ratio") or 0.22)
        ww = int(round(canvas_w * ratio / (1 + 2 * ratio)))
    return ww, fps


def inspect(job_dir, vision=None, per_shot=2, verbose=True):
    """
    Ask a model what is wrong with each finished shot's walls.

    Returns the report and writes it beside the job. Shots whose wings are
    photographed are inspected too -- knowing a recovered wall looks poor is
    useful -- but they are marked as not repairable.
    """
    job = Path(job_dir)
    summary_path = job / "screenx_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no screenx_summary.json in {job}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if vision is None:
        import gemini
        # PROMPT states its own answer shape, and it has to: the adapter's
        # default asks what stands just off-frame, which is a different question
        # from what is wrong with the wall. Left in place it overrode this one
        # and the model dutifully answered it -- the first run of this pass came
        # back listing "child's arm, chair, table" as though those were defects.
        vision = gemini.GeminiVision(answer_shape="")

    shots_dir = job / "deliverable" / "shots"
    findings = []
    for rec in summary.get("per_shot", []):
        si = int(rec.get("shot", 0))
        seg = shots_dir / f"shot_{si:03d}_master.mp4"
        prov = (rec.get("provenance") or {})
        photographed = float(prov.get("photographic") or 0.0)
        entry = dict(shot=si, state=rec.get("state"), segment=str(seg),
                     repairable=True, photographed=round(photographed, 4),
                     claims=[], streak=None, note="")
        if rec.get("state") in PHOTOGRAPHED_STATES:
            entry["note"] = (f"wings are {photographed * 100:.0f}% photographed; "
                             f"repainting relabels whatever it changes, so the "
                             f"real-wing figure will fall")
        if not seg.exists():
            entry["note"] = entry["note"] or "no segment written for this shot"
            findings.append(entry)
            continue

        frames = sample_frames(seg, per_shot)
        if not frames:
            entry["note"] = entry["note"] or "segment had no readable frames"
            findings.append(entry)
            continue

        canvas = frames[0]
        wing_w, _fps = dims(summary, canvas.shape[1])
        entry["streak"] = streak_score(canvas, wing_w)
        # Measured, and measured because none of it was.
        #
        # The gate scored this shot 32.4 dB and called the walls good while
        # they carried a hairline every few columns, boiled at 2.6x the
        # centre, and were cut against it rather than joined to it. Coverage
        # and staleness cannot see any of the three. These can.
        entry["hairlines"] = hairline_score(canvas, wing_w)
        entry["seam"] = seam_step(canvas, wing_w)
        run = burst(seg)
        entry["jitter"] = jitter_ratio(run, wing_w) if run else None

        claims = []
        for f in frames:
            try:
                claims += list(vision(PROMPT, image=wing_strip(f, wing_w)) or [])
            except Exception as e:                 # a model is not load-bearing
                entry["note"] = f"vision unavailable: {type(e).__name__}"
                break
        seen, uniq = set(), []
        for c in claims:
            k = str(c).strip().lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(str(c).strip())
        entry["claims"] = uniq[:4]
        entry["bad"] = bool([c for c in uniq if "acceptable" not in c.lower()])
        findings.append(entry)
        if verbose:
            flag = "BAD " if entry.get("bad") else "ok  "
            print(f"  shot {si:3d} {flag} streak {entry['streak']}  "
                  f"hairline {entry['hairlines']}  jitter {entry['jitter']}  "
                  f"seam {entry['seam']}  "
                  f"{'; '.join(entry['claims'])[:60]}", flush=True)

    report = dict(source=summary.get("source"), shots=len(findings),
                  findings=findings,
                  repairable=[f["shot"] for f in findings if f.get("bad")])
    (job / "polish_report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def notes_for(job_dir, shot):
    """Anything a person pinned to this shot."""
    try:
        import context as cx
        return [i.text for i in cx.DirectionStore(Path(job_dir)).for_shot(int(shot))
                if getattr(i, "text", "")]
    except Exception:
        return []


def brief_from(entry, notes=()):
    """
    Turn the complaints into an instruction the generator can act on.

    A pinned note leads. The person was looking at the frame and saying what
    belongs there; the model is describing what it sees wrong. Wanting takes
    precedence over complaining, and the note is also what makes the result
    DIRECTED rather than merely GENERATED.
    """
    notes = [n.strip() for n in (notes or []) if n and n.strip()]
    claims = [c for c in entry.get("claims", []) if "acceptable" not in c.lower()]
    if not notes and not claims:
        return None
    parts = []
    if notes:
        parts.append("; ".join(notes[:2]))
    else:
        parts.append("extend the scene sideways, same place and moment")
    if claims:
        parts.append("Avoid: " + "; ".join(claims[:3]))
    return ". ".join(parts)


def read_all(path):
    """Every frame of a segment, in order."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def best_segment(shots_dir, si):
    """
    The most finished version of a shot that exists.

    A shot can have been settled, then repainted, and the next pass has to pick
    up where the last one left off rather than reverting it. Repaint wins over
    settle because a repaint is applied ON TOP of a settled wall.
    """
    d = Path(shots_dir)
    for suffix in ("_master_polished.mp4", "_master_settled.mp4", "_master.mp4"):
        p = d / f"shot_{si:03d}{suffix}"
        if p.exists():
            return p
    return d / f"shot_{si:03d}_master.mp4"


def settle(job_dir, k=2, verbose=True):
    """
    Fix the wall with its own photography. Free, local, invents nothing.

    This is what the finishing pass should do first and almost always. The two
    complaints a finished film actually draws -- thin dark lines down the walls,
    and walls that shimmer while the picture is steady -- are not shortages of
    imagination that a generator is needed for. One was a bug in how a donor
    frame was sampled. The other is each frame assembling its wing from scratch,
    so the wall re-forms 24 times a second.

    Both are answerable from the footage already in hand: align each frame's
    neighbours to it and take the median of the wall. A hairline is one odd
    sample among five and the median steps over it. The frame-to-frame reforming
    is noise about a stable value and the median finds the value.

    Nothing here can invent a pixel. It refines the VALUE of pixels that were
    already recovered and never fills an empty one, so `filled`, the provenance
    map and `mean_real_wing` all come out exactly as they went in -- which is
    the whole reason to reach for this before reaching for a model. The report
    carries the before and after of all three measurements so the improvement is
    checkable rather than asserted.
    """
    import agent as ag
    import screenx_render as sx
    import wingcoverage as wc

    job = Path(job_dir)
    summary_path = job / "screenx_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    shots_dir = job / "deliverable" / "shots"
    if not shots_dir.is_dir():
        raise FileNotFoundError(f"no deliverable/shots in {job}")

    tracker = wc.Tracker()
    done = []
    for rec in summary.get("per_shot", []):
        si = int(rec.get("shot", 0))
        seg = best_segment(shots_dir, si)
        if not seg.exists():
            continue
        frames = read_all(seg)
        if len(frames) < 5:
            continue
        h, cw = frames[0].shape[:2]
        ww, fps = dims(summary, cw)
        if ww < 2 or cw - 2 * ww < 16:
            continue
        mid = len(frames) // 2

        before = dict(hairlines=hairline_score(frames[mid], ww),
                      jitter=jitter_ratio(frames[:40], ww),
                      seam=seam_step(frames[mid], ww))

        # The centre IS the source footage -- the fence guarantees it -- so the
        # geometry can be re-solved here without the original file, which may
        # be on another machine or gone.
        centres = [f[:, ww:cw - ww] for f in frames]
        Hs = wc.chain_homographies(tracker, centres)

        def warp_of(i, t, Hs=Hs):
            if Hs[i] is None or Hs[t] is None:
                return None
            return np.linalg.inv(Hs[i]) @ Hs[t]

        prov_stack = None
        prov_path = shots_dir / f"shot_{si:03d}_prov.npz"
        if prov_path.exists():
            try:
                prov_stack = np.load(prov_path)["prov"]
            except (OSError, ValueError, KeyError):
                prov_stack = None

        packed = []
        for i, f in enumerate(frames):
            if prov_stack is not None and i < len(prov_stack):
                real = np.isin(prov_stack[i], ag.PHOTOGRAPHIC)
            else:
                real = f.max(axis=2) > 6
            real = real.copy()
            real[:, ww:cw - ww] = True
            packed.append((f, real, None))

        settled = wc.settle_wings(packed, warp_of, ww, k=k)

        out = seg.with_name(f"shot_{si:03d}_master_settled.mp4")
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (cw, h))
        for (c, _f, _t), original in zip(settled, frames):
            c = c.copy()
            c[:, ww:cw - ww] = original[:, ww:cw - ww]      # the fence, again
            vw.write(c)
        vw.release()
        sx.to_h264(out, fps)

        got = read_all(out)
        after = dict(hairlines=hairline_score(got[mid], ww) if got else None,
                     jitter=jitter_ratio(got[:40], ww) if got else None,
                     seam=seam_step(got[mid], ww) if got else None)
        done.append(dict(shot=si, output=str(out), before=before, after=after))
        if verbose:
            print(f"  shot {si}: settled -> {out.name};  hairlines "
                  f"{before['hairlines']} -> {after['hairlines']}, jitter "
                  f"{before['jitter']} -> {after['jitter']}, seam "
                  f"{before['seam']} -> {after['seam']}", flush=True)

    if done:
        # `mean_real_wing` deliberately untouched: nothing was invented, so
        # there is nothing to charge. Recording the pass is not the same as
        # moving the number, and only one of the two would be a lie.
        summary["settled"] = [d["shot"] for d in done]
        summary_path.write_text(json.dumps(summary, indent=1, default=str),
                                encoding="utf-8")
        rebuild(job, verbose=verbose)

    rp = job / "polish_report.json"
    report = {}
    if rp.exists():
        try:
            report = json.loads(rp.read_text(encoding="utf-8"))
        except ValueError:
            report = {}
    report["settled"] = done
    rp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return done


def repair(job_dir, generator="wavespeed", shots=None, verbose=True,
           full=False):
    """
    Regenerate the flagged shots, and tell the truth about what changed.

    Any shot may be repaired, photographed or not. What the model repaints stops
    being photography: the changed pixels are relabelled GENERATED, or DIRECTED
    where a pinned note drove the change, the shot's provenance map is rewritten
    on disk, and the run summary is restated so `mean_real_wing` reflects the
    loss. The polish report carries the before and after.

    Two different jobs, and they aim differently:

        a fault   the model or the measurements found something broken. Paint
                  through the broken pixels only -- dead ones, hairlines,
                  whatever the provenance map already calls invented -- and
                  leave the photography alone. Costs a few percent.

        a note    a person said what belongs on that wall. Paint the whole wing,
                  land it on DIRECTED, and state what it cost. A shot needs no
                  fault to qualify: wanting something is reason enough.
    """
    job = Path(job_dir)
    rp = job / "polish_report.json"
    if not rp.exists():
        raise FileNotFoundError("run inspect() first")
    report = json.loads(rp.read_text(encoding="utf-8"))

    import agent as ag
    import provenance as P
    import screenx_render as sx
    make = sx.GENERATORS.get(generator)
    if make is None:
        raise ValueError(f"unknown generator {generator}")
    gen = make()

    summary_path = job / "screenx_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_shot = {int(r.get("shot", -1)): r for r in summary.get("per_shot", [])}
    shots_dir = job / "deliverable" / "shots"

    # Which shots to touch. A hosted generator bills per shot, and "repaint
    # everything the model complained about" is how a 34-shot trailer turns into
    # $7 -- so the caller may name the ones worth paying for. None means every
    # shot the inspection faulted, which is the old behaviour.
    only = None if shots is None else {int(s) for s in shots}

    done = []
    for entry in report["findings"]:
        if only is not None and int(entry["shot"]) not in only:
            continue

        # A pinned note is a request, not a second opinion.
        #
        # This used to skip any shot the model called acceptable, which meant a
        # person could stand in front of the frame, write down what belongs on
        # that wall, press the button and watch nothing happen -- because a
        # model had looked at the same wall and shrugged. The model's verdict is
        # about defects. A note is about content, and someone who knows the room
        # does not need a defect to be entitled to change it.
        notes = notes_for(job, entry["shot"])
        if not entry.get("bad") and not notes:
            continue
        brief = brief_from(entry, notes)
        if not brief:
            continue

        # the most finished version of this shot, so a repaint lands on top of
        # a settled wall rather than reverting it
        seg = best_segment(shots_dir, int(entry["shot"]))
        frames = read_all(seg)
        if not frames:
            continue

        # the centre IS the original frame -- the fence guarantees it byte for
        # byte -- so the source is recoverable from the deliverable itself
        h, cw = frames[0].shape[:2]
        wing_w, fps = dims(summary, cw)
        if wing_w < 2:
            continue
        centres = [f[:, wing_w:cw - wing_w] for f in frames]

        try:
            if hasattr(gen, "generate_shot"):
                widened = gen.generate_shot(centres, wing_w, brief)
            else:
                widened = []
                for c in centres:
                    try:
                        gen.prompt = brief
                    except AttributeError:
                        pass
                    widened.append(sx.synth_wings(c, wing_w, gen, P.GENERATED)[0])
        except Exception as e:
            if verbose:
                print(f"  shot {entry['shot']}: repair failed, original kept "
                      f"({type(e).__name__}: {e})"[:150], flush=True)
            continue

        # what a note drives is DIRECTED: a person who knows the room is not a
        # camera, but they are not the model's guess either
        label = P.DIRECTED if notes else P.GENERATED
        prov_path = shots_dir / f"shot_{int(entry['shot']):03d}_prov.npz"
        prov_stack = None
        if prov_path.exists():
            try:
                prov_stack = np.load(prov_path)["prov"]
            except (OSError, ValueError, KeyError):
                prov_stack = None

        out = shots_dir / f"shot_{int(entry['shot']):03d}_master_polished.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (cw, h))
        repainted, counted = 0, 0
        wing_changed, wing_total = 0, 0
        new_stack = []
        for i, (src, wide, original) in enumerate(zip(centres, widened, frames)):
            made = wide if wide.shape[:2] == (h, cw) else cv2.resize(wide, (cw, h))
            pm_i = (prov_stack[i] if prov_stack is not None and i < len(prov_stack)
                    else None)

            # Take the generator's work THROUGH the defects and nowhere else.
            #
            # Replacing the whole wing was the mistake that made this pass worth
            # avoiding: on a wall that was 99.07% photographed it redrew 62% to
            # fix 1.5% of columns, cost two thirds of the real-wing figure, and
            # invented a second copy of a logo that was already on screen. The
            # broken pixels are findable, so aim at them -- feathered, because a
            # hard-edged patch stands as a vertical seam for the whole shot.
            #
            # Where the generator returned nothing, keep what was there. A
            # hosted model expands to an aspect ratio rather than to our wing
            # width, so `fit_to_canvas` pads the shortfall with zeros; a
            # WaveSpeed pass on the cafe clip came back with 42px of dead black
            # down the outer edge of both wings. Black is not a repair.
            bad = defect_mask(original, wing_w, pm_i)
            bad &= made.max(axis=2) > 6

            # A note opens the whole wall; a fault opens only what is faulty.
            #
            # Aiming at defects is right when the complaint is a defect -- it is
            # what stops a hairline costing two thirds of a wall. It is wrong
            # when a person has said what should BE there: "a fire escape,
            # camera left" cannot be delivered through 2% of the wing, and
            # quietly painting a doorway into a hairline is not honouring the
            # note, it is ignoring it with extra steps. So a note widens the
            # target to the whole wing, everything it changes lands on DIRECTED
            # rather than GENERATED, and the cost is stated in full below like
            # any other repaint. Refusing would be the easier answer and the
            # worse one.
            if full or notes:
                wing_all = np.zeros_like(bad)
                wing_all[:, :wing_w] = True
                wing_all[:, cw - wing_w:] = True
                bad = wing_all & (made.max(axis=2) > 6)
            a = feather(bad)[..., None]
            canvas = (a * made.astype(np.float32)
                      + (1.0 - a) * original.astype(np.float32))
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
            canvas[:, wing_w:cw - wing_w] = src        # the fence, again

            # exactly which pixels moved, so exactly those get relabelled
            diff = np.abs(canvas.astype(np.int16)
                          - original.astype(np.int16)).max(axis=2)
            changed = diff > REPAINT_THRESHOLD
            changed[:, wing_w:cw - wing_w] = False     # the centre never changes

            wings = np.zeros_like(changed)
            wings[:, :wing_w] = True
            wings[:, cw - wing_w:] = True
            wing_changed += int((changed & wings).sum())
            wing_total += int(wings.sum())

            if pm_i is not None:
                pm = pm_i.copy()
                if pm.shape == changed.shape:
                    counted += int(np.isin(pm[wings], ag.PHOTOGRAPHIC).sum())
                    repainted += int((changed & wings
                                      & np.isin(pm, ag.PHOTOGRAPHIC)).sum())
                    pm[changed] = label
                new_stack.append(pm)
            vw.write(canvas)
        vw.release()
        sx.to_h264(out, fps)

        moved = (wing_changed / wing_total) if wing_total else 0.0
        lost, basis = _charge(counted, repainted, moved,
                              float(entry.get("photographed") or 0.0))

        rec = by_shot.get(int(entry["shot"]))
        if new_stack:
            # Keep the pre-repaint map, once.
            #
            # A repaint leaves the original segment on disk and calls itself
            # reversible, but it used to overwrite the provenance map in place
            # -- so shot_000_master.mp4 (99.07% photographed) ended up sitting
            # beside a map that said 37%, and the original accounting was gone
            # for good. You could rebuild the film from the original pixels and
            # then had nothing to describe them with. Snapshotted the same way
            # the summary is, and never overwritten by a second pass.
            keep = prov_path.with_name(prov_path.stem + ".prepolish.npz")
            if prov_stack is not None and not keep.exists():
                try:
                    np.savez_compressed(keep, prov=prov_stack)
                except OSError as e:
                    print(f"  pre-repaint provenance not kept: {e}", flush=True)
            np.savez_compressed(prov_path, prov=np.stack(new_stack))
            if rec is not None:
                rec["provenance"] = _report_over(new_stack, wing_w, cw - 2 * wing_w)
        elif rec is not None:
            # No map to rewrite, so the shot's provenance is moved by the same
            # share the pixels were: whatever `lost` charges comes off every
            # photographic rung and lands on the one this repaint earned.
            rec["provenance"] = _charge_report(rec.get("provenance") or {},
                                               lost, label)
        if rec is not None:
            rec["polished"] = dict(brief=brief, label=int(label), basis=basis,
                                   wing_pixels_moved=round(moved, 4),
                                   photographed_repainted=round(lost, 4))
        done.append(dict(shot=entry["shot"], brief=brief,
                         driven_by="note" if notes else "model",
                         label=("DIRECTED" if notes else "GENERATED"),
                         basis=basis, wing_pixels_moved=round(moved, 4),
                         photographed_repainted=round(lost, 4),
                         output=str(out)))
        if verbose:
            print(f"  shot {entry['shot']}: repolished by {'note' if notes else 'model'}"
                  f" -> {out.name}; {moved * 100:.1f}% of the wing was redrawn, "
                  f"{lost * 100:.1f}% of its photographed wing is now "
                  f"{'DIRECTED' if notes else 'GENERATED'} ({basis})", flush=True)

    if done:
        summary = _restate(summary, by_shot)
        summary_path.write_text(json.dumps(summary, indent=1, default=str),
                                encoding="utf-8")
        # and put it in the film. A repaint that only ever lands in a file
        # beside the original leaves the maker able to download a better shot
        # and unable to screen it.
        try:
            rebuilt = rebuild(job, verbose=verbose)
            report["rebuilt"] = rebuilt
        except Exception as e:
            report["rebuilt"] = dict(error=f"{type(e).__name__}: {e}")
            if verbose:
                print(f"  the deliverable was NOT rebuilt ({type(e).__name__}: {e}); "
                      f"the polished shots are on disk but the film still holds "
                      f"the old ones", flush=True)
    report["repaired"] = done
    rp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return done


def rebuild(job_dir, verbose=True):
    """
    Re-cut the deliverable so a polished shot reaches the film.

    Without this the pass is decoration: `repair` writes
    `shot_007_master_polished.mp4` beside the original and stops, so the maker
    can download a better shot and cannot screen it -- `master_widened.mp4` and
    the three projector feeds still hold the version that was faulted. A
    finishing pass whose output never reaches the print is not a finishing pass.

    Shots are re-read from their own segments rather than kept in memory,
    because the repaint may have happened in a different process, or yesterday.
    A polished segment is preferred wherever one exists; every other shot is
    passed through untouched.
    """
    import screenx_render as sx
    import walls as wl

    job = Path(job_dir)
    summary_path = job / "screenx_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    deliv = job / "deliverable"
    shots_dir = deliv / "shots"
    if not shots_dir.is_dir():
        raise FileNotFoundError(f"no deliverable/shots in {job}")

    theatre = wl.Theatre(**{k: v for k, v in (summary.get("theatre") or {}).items()
                            if k in wl.Theatre.__dataclass_fields__})

    order = [int(r.get("shot", i)) for i, r in enumerate(summary.get("per_shot", []))]
    used, canvases = [], []
    for si in order:
        src = best_segment(shots_dir, si)     # polished, else settled, else plain
        if not src.exists():
            continue
        cap = cv2.VideoCapture(str(src))
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            canvases.append((f, None))
            n += 1
        cap.release()
        if n:
            used.append((si, src.name, n))

    if not canvases:
        raise RuntimeError("no shot segments to rebuild from")

    # Both recorded by the renderer. `fps` in particular: guessing 24.0 here is
    # what silently re-timed a 30fps film by a quarter every time this ran.
    wing_w, fps = dims(summary, canvases[0][0].shape[1])
    sx.write_deliverable(deliv, canvases, wing_w, theatre, fps=fps)

    polished_shots = [si for si, name, _n in used if "_polished" in name]
    summary["deliverable_includes_polished"] = polished_shots
    summary["deliverable_includes_settled"] = [si for si, name, _n in used
                                               if "_settled" in name]
    summary_path.write_text(json.dumps(summary, indent=1, default=str),
                            encoding="utf-8")
    if verbose:
        print(f"  rebuilt the deliverable from {len(used)} shot(s), "
              f"{len(canvases)} frames; polished: "
              f"{polished_shots or 'none'}", flush=True)
    return dict(shots=len(used), frames=len(canvases), polished=polished_shots)


def _charge(counted, repainted, moved, photographed):
    """
    What the repaint cost, and on what evidence.

    With a provenance map beside the pixels this is exact: count the wing
    pixels that were photographed, count how many of those the model changed,
    divide. Without one the honest answer is not zero -- it is "unmeasured",
    and the first version of this returned 0.0 for it. That reported a repaint
    that redrew 92.6% of a wall as having cost nothing, which is precisely the
    lie the ladder exists to prevent, and it read as a measurement rather than
    as an absence of one.

    So the unmeasured case is CHARGED instead, at the share of the wing whose
    pixels actually moved. That can only overstate the loss, never understate
    it, and the direction matters: overstating makes the film look less
    photographed than it is, which costs coverage. Understating reports invented
    pixels as filmed, which is the one thing that must not happen.
    """
    if counted:
        return (repainted / counted), "provenance map"
    if not photographed:
        return 0.0, "wing held nothing photographed"
    return min(1.0, moved), "pixels moved (upper bound, no provenance map)"


def _charge_report(prov, lost, label):
    """
    Move `lost` of a shot's photographic share onto the rung this repaint earned.

    Used only when there is no per-pixel map to rewrite. Proportional across the
    photographic rungs, because without a map there is nothing to say which of
    them the model painted over -- so each gives up the same share.
    """
    import provenance as P
    photographic = ("primary", "recovered", "donated", "retrieved")
    earned = "directed" if label == P.DIRECTED else "generated"
    out = {k: float(v) for k, v in (prov or {}).items()}
    taken = 0.0
    for k in photographic:
        if out.get(k):
            give = out[k] * lost
            out[k] = round(out[k] - give, 6)
            taken += give
    out[earned] = round(out.get(earned, 0.0) + taken, 6)
    out["photographic"] = round(sum(out.get(k, 0.0) for k in photographic), 6)
    return {k: round(v, 4) for k, v in out.items()}


def _report_over(prov_stack, wing_w, w):
    """Average the per-rung report across a shot's frames."""
    import agent as ag
    acc = {}
    for pm in prov_stack:
        for k, v in ag.WingAgent.report(pm, wing_w, w).items():
            acc[k] = acc.get(k, 0.0) + v
    return {k: round(v / len(prov_stack), 4) for k, v in acc.items()}


def _restate(summary, by_shot):
    """
    Recompute the run's headline from the shots as they now stand.

    A polished run that kept its old mean_real_wing would be reporting pixels as
    photographed that a model drew thirty seconds ago.
    """
    recs = [by_shot[k] for k in sorted(by_shot)]
    provs = [r.get("provenance") for r in recs if r.get("provenance")]
    if provs:
        keys = set().union(*(p.keys() for p in provs))
        overall = {k: round(sum(p.get(k, 0.0) for p in provs) / len(provs), 4)
                   for k in keys}
        summary["provenance"] = overall
        summary["mean_real_wing"] = overall.get("photographic",
                                                summary.get("mean_real_wing"))
        summary["rungs_fired"] = sorted(
            k for k in ("primary", "recovered", "donated", "retrieved",
                        "referenced", "directed", "generated")
            if overall.get(k, 0.0) > 0.0005)
    summary["per_shot"] = recs
    summary["polished"] = True
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="settle, inspect and repair "
                                             "finished walls")
    ap.add_argument("job_dir")
    ap.add_argument("--no-settle", action="store_true",
                    help="skip the free local pass that fixes hairlines and "
                         "shimmer using the shot's own photography")
    ap.add_argument("--repair", nargs="?", const="wavespeed", default=None,
                    metavar="GENERATOR",
                    help="regenerate what is still faulted after settling "
                         "(costs money for hosted generators)")
    ap.add_argument("--full", action="store_true",
                    help="repaint the WHOLE wing rather than only its defects; "
                         "this is what costs a photographed wall its number")
    ap.add_argument("--shots", default="", metavar="N,N",
                    help="repaint only these shots; a hosted generator bills "
                         "per shot, so 'all the ones it complained about' is "
                         "rarely what you want to pay for")
    a = ap.parse_args()

    # Settle first, and by default. It is free, it invents nothing, and it
    # fixes the two things people actually complain about -- so inspecting
    # before it runs asks the model, and possibly your wallet, about defects
    # that were about to disappear anyway.
    if not a.no_settle:
        print("settling (free, no model, nothing invented):", flush=True)
        settle(a.job_dir)

    rep = inspect(a.job_dir)
    print("")
    print(f"{len(rep['repairable'])} shot(s) still faulted: "
          f"{rep['repairable']}")
    if a.repair:
        picked = [int(x) for x in a.shots.replace(" ", "").split(",") if x] or None
        repair(a.job_dir, a.repair, shots=picked, full=a.full)
