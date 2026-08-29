"""
cut_captures -- trim the raw screen recordings down to the part worth watching.

A raw take from tools/record_captures.py is mostly waiting: a page loading, a
consent dialog, then a model thinking. The signal is short and the dead air is
long, and the dead air is the same in every take, so it is found rather than
typed in -- hardcoded timestamps go stale the moment a take is re-recorded, and
a stale timestamp here silently cuts the answer in half.

    python tools/cut_captures.py

Writes media/captures/{agent,studio,ledger}.mp4.

HOW THE WINDOW IS FOUND
-----------------------
Frames are sampled and compared to their predecessor; a frame that differs by
more than a threshold is an EVENT -- text appearing, a panel filling, a stage
switching. Everything before the first event and after the last is throat
clearing. Between them, any gap longer than QUIET seconds is a wait, and waits
are sped up rather than cut, so the recording stays continuous and honest:
nothing is removed from the middle of the take, only compressed.

The video's captions name the speed-up wherever one happens.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "captures"

SAMPLE = 5          # compare every Nth frame
CHANGE = 0.010      # fraction of pixels that must differ to count as an event
QUIET = 1.5         # a gap longer than this is a wait
TARGET = 0.8        # what a wait is compressed to, in seconds
LEAD = 1.2          # seconds kept before the first event
TAIL = 2.6          # seconds kept after the last, so the answer can be read
MAX_SPEED = 8.0


def ffmpeg() -> str:
    """
    A real ffmpeg, not the first thing named ffmpeg on PATH.

    On this machine PATH turns up a shim in ~/Scripts that rejects filter
    options with "Error splitting the argument list", which surfaces here as a
    generic non-zero exit from a command that is perfectly valid. So candidates
    are probed with a trivial filter and the first one that survives wins.
    """
    cands = [
        "C:/Users/talig/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
        "ffmpeg-8.1.1-full_build/bin/ffmpeg.exe",
        shutil.which("ffmpeg"),
    ]
    for c in cands:
        if not c or not Path(c).is_file():
            continue
        try:
            probe = subprocess.run(
                [c, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=64x64:d=0.1",
                 "-filter:v", "setpts=PTS/2.0,fps=30", "-f", "null", "-"],
                capture_output=True, timeout=60)
            if probe.returncode == 0:
                return c
        except Exception:
            continue
    raise SystemExit("no working ffmpeg found (tried: %s)" % cands)


def events(path: Path):
    """Times, in seconds, where the picture changed materially."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out, prev, i = [], None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % SAMPLE == 0:
            g = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                d = float((cv2.absdiff(g, prev) > 18).mean())
                # the very first paint (blank -> page) is not an event worth keeping
                if CHANGE < d < 0.60:
                    out.append(i / fps)
            prev = g
        i += 1
    cap.release()
    return out, i / fps


def plan_settle(path: Path, want=12.0):
    """
    For a take whose payload is a STATIC end state rather than a sequence.

    The Studio's conversion report is a page of numbers that arrives and then
    sits there. Event-detection is the wrong tool -- it finds one transition and
    throws the report itself away as dead air, which is exactly backwards.
    Here the last big change is the report appearing, so the cut runs from just
    before it to the end of the take.
    """
    ev, dur = events(path)
    start = max(0.0, (ev[-1] if ev else dur) - 0.6)
    if dur - start < want * 0.5:                 # not enough tail, back off
        start = max(0.0, dur - want)
    return [(start, dur, 1.0)], dur - start


def plan(path: Path):
    """(segments, kept_seconds) where each segment is (start, end, speed)."""
    ev, dur = events(path)
    if not ev:
        return [(0.0, dur, 1.0)], dur
    a = max(0.0, ev[0] - LEAD)
    b = min(dur, ev[-1] + TAIL)

    segs, cursor = [], a
    for t in ev:
        if t <= cursor:
            continue
        gap = t - cursor
        if gap > QUIET:
            # keep a beat at normal speed, then compress the rest of the wait
            segs.append((cursor, cursor + 0.35, 1.0))
            speed = min(MAX_SPEED, max(1.0, (gap - 0.35) / TARGET))
            segs.append((cursor + 0.35, t, speed))
        else:
            segs.append((cursor, t, 1.0))
        cursor = t
    if b > cursor:
        segs.append((cursor, b, 1.0))

    segs = [(s, e, sp) for s, e, sp in segs if e - s > 0.05]
    kept = sum((e - s) / sp for s, e, sp in segs)
    return segs, kept


def content_box(path: Path):
    """
    The rectangle the page actually occupies, as (w, h, x, y).

    record_video_size does not scale the viewport up -- it composites the
    viewport into a canvas of that size and pads the remainder with flat grey
    (128,128,128). Recording a 1280x720 page into a 1920x1080 file therefore
    produces a small page in the corner of a grey field, which is worse than
    recording at 1920 directly. Cropping that padding away and letting the
    scaler take 1280 -> 1920 is what actually makes the UI bigger.

    The box is measured rather than assumed, so changing the viewport in
    record_captures.py does not silently mis-crop everything here.
    """
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.8))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    # Padding is uniform grey. Decide per ROW and per COLUMN by majority rather
    # than per pixel: video compression leaves ringing along the page's edge, so
    # "does this column contain any non-grey pixel" answers yes for the whole
    # frame and the crop silently becomes a no-op -- which is how the Studio
    # take came back 1920 wide when its page is 1280.
    grey = np.all(np.abs(fr.astype(int) - 128) <= 6, axis=2)
    col_pad = grey.mean(0) > 0.97
    row_pad = grey.mean(1) > 0.97
    cols, rows = np.where(~col_pad)[0], np.where(~row_pad)[0]
    if not len(cols) or not len(rows):
        return None
    cw, ch = int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)
    if cw >= w - 4 and ch >= h - 4:              # nothing was padded
        return None
    return cw - (cw % 2), ch - (ch % 2), int(cols.min()), int(rows.min())


def render(src: Path, dst: Path, segs):
    ff, tmp = ffmpeg(), dst.parent / "_seg"
    box = content_box(src)
    crop = f"crop={box[0]}:{box[1]}:{box[2]}:{box[3]}," if box else ""
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.mp4"):
        f.unlink()
    parts = []
    for n, (s, e, sp) in enumerate(segs):
        part = tmp / f"{dst.stem}_{n:03d}.mp4"
        subprocess.run(
            [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
             "-filter:v",
             # scale to width, preserving aspect: forcing 1920x1080 on a crop
             # that is not 16:9 stretches the UI vertically
             f"{crop}setpts=PTS/{sp:.4f},fps=30,scale=1920:-2:flags=lanczos",
             "-an", "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(part)],
            check=True)
        parts.append(part)
    listing = tmp / f"{dst.stem}.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(dst)], check=True)
    for p in parts:
        p.unlink()
    listing.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", default=str(OUT / "_raw"))
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_dir():
        raise SystemExit(f"no raw takes in {raw} -- run tools/record_captures.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("agent", "studio", "ledger"):
        takes = sorted((raw / name).glob("*.webm")) if (raw / name).is_dir() else []
        if not takes:
            print(f"  {name:8s} no take")
            continue
        src = takes[0]
        # the Studio's payload is a static report; the others are sequences
        segs, kept = (plan_settle(src) if name == "studio" else plan(src))
        sped = sum(1 for _, _, sp in segs if sp > 1.05)
        fastest = max((sp for _, _, sp in segs), default=1.0)
        dst = OUT / f"{name}.mp4"
        render(src, dst, segs)
        raw_dur = cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT) / 25.0
        b = content_box(src)
        print(f"  {name:8s} {raw_dur:5.1f}s raw -> {kept:5.1f}s   "
              f"{len(segs)} segments, {sped} sped, fastest {fastest:.1f}x"
              + (f", cropped to {b[0]}x{b[1]}" if b else ""))
    shutil.rmtree(OUT / "_seg", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
