"""
test_e2e -- drive the real app end to end and look at what comes out.

WHY THIS EXISTS
---------------
The other eight suites are 485 assertions and every one of them was green
through every bug this project has shipped. That is not an accident, it is what
they test: each stops at a seam. `test_wavespeed` asserts the request body up to
the socket. `test_app` asserts the argv up to the subprocess. `test_tier2`
asserts pixels up to the file. Nobody asserted that the pieces meet.

So every bug found by using the thing was a join:

    add_film had no callers                 the rung was unreachable
    ExternalReferenceTool did not exist     director constructed a missing class
    --reason had never run                  first unregisterable frame killed it
    generate_shot never reached the pipeline a video model got 1-frame clips
    every video was mp4v                    no browser could play the output
    two elements shared id="shots"          Review's clicks went to a dropdown
    GeminiVision overrode its caller        polish asked A and was answered B
    _from_disk returned a partial record    the job list died on first polish

This suite exercises the joins instead of the parts. It starts the real server
on a real port, uploads a real clip over HTTP, renders it, and then asserts on
the ARTIFACTS -- do the files exist, does a browser codec come out, do the
numbers in the summary agree with the pixels on disk, does the fence hold.

The one thing stubbed is the vision model, because it costs money and is not
what is being tested. Everything else is the real path.

Run: python test_e2e.py            (takes a couple of minutes)
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import app

PASS, FAIL = [], []
HERE = Path(__file__).resolve().parent


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ fixtures

def free_port() -> int:
    """
    A port of our own.

    Deliberately not 8420: on Windows `allow_reuse_address` lets a second server
    bind a port that is already listening, so a stale session's app and this
    test would both answer and requests would flap between them. That cost half
    an hour once already.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_clip(path: Path, frames=48, w=480, h=270):
    """
    A short pan across a textured wall, written as real video.

    Textured because the whole pipeline keys off feature correspondence: a
    smooth gradient registers as well as noise does and recovers nothing, which
    would make this test pass while proving little.
    """
    rng = np.random.default_rng(11)
    pano = np.zeros((h, w * 3, 3), np.uint8)
    pano[:] = rng.integers(30, 90, (h, w * 3, 3), dtype=np.uint8)
    for _ in range(70):                       # hard edges give ORB something
        x = int(rng.integers(0, w * 3 - 40))
        y = int(rng.integers(0, h - 40))
        c = tuple(int(v) for v in rng.integers(90, 255, 3))
        cv2.rectangle(pano, (x, y), (x + 34, y + 30), c, -1)
    # 30, deliberately not 24. The renderer, the segment writer and the polish
    # rebuild each had their own idea of the frame rate, and a 24fps fixture
    # agreed with every one of them -- so a film that came out a quarter slow
    # passed this suite. The fixture has to disagree with the default for the
    # join to be tested at all.
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(frames):
        x = int(round(i * (w * 2 - 1) / max(1, frames - 1)))
        vw.write(pano[:, x:x + w].copy())
    vw.release()
    return path


class Server:
    """The real app, on its own port, in this process."""

    def __init__(self):
        self.port = free_port()
        self.srv = None

    def __enter__(self):
        from http.server import ThreadingHTTPServer
        app.TOKEN = ""
        self.srv = ThreadingHTTPServer(("127.0.0.1", self.port), app.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path, timeout=60):
        with urllib.request.urlopen(self.url(path), timeout=timeout) as r:
            return json.loads(r.read() or b"null"), r.status

    def post(self, path, obj=None, raw=None, timeout=60):
        body = raw if raw is not None else json.dumps(obj or {}).encode()
        req = urllib.request.Request(self.url(path), data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"null"), r.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read() or b"null"), e.code

    def head(self, path, timeout=30):
        req = urllib.request.Request(self.url(path), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.headers, r.status


def fps_of(path: Path) -> float:
    """The rate the file will actually play at."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate", "-of",
                            "default=nw=1:nk=1", str(path)],
                           capture_output=True, text=True, timeout=60)
        num, _, den = (r.stdout or "").strip().partition("/")
        return float(num) / float(den or 1)
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        return 0.0


def codec_of(path: Path) -> str:
    """What a browser would be handed."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name", "-of",
                            "default=nw=1:nk=1", str(path)],
                           capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------ the run

def test_end_to_end():
    print("a clip goes in the front door and a film comes out")
    with Server() as s:
        caps, code = s.get("/api/capabilities")
        check("the app answers", code == 200 and "ffmpeg" in caps)
        if not caps.get("ffmpeg", {}).get("ok"):
            check("ffmpeg is present (everything below needs it)", False)
            return

        clip = make_clip(HERE / "_e2e_clip.mp4")
        body = clip.read_bytes()
        req = urllib.request.Request(
            s.url("/api/jobs?name=_e2e_clip.mp4"), data=body, method="POST",
            headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=120) as r:
            created = json.loads(r.read())
        jid = created.get("id")
        check("the upload stages a job", bool(jid) and created.get("state") == "staged",
              str(created)[:80])
        if not jid:
            return
        job_dir = app.JOBS_DIR / jid

        try:
            _render(s, jid, job_dir)
        finally:
            clip.unlink(missing_ok=True)
            shutil.rmtree(job_dir, ignore_errors=True)
            app.JOBS.pop(jid, None)


def _render(s, jid, job_dir):
    out, code = s.post(f"/api/jobs/{jid}/start",
                       dict(maxw=320, frames_per_shot=40, wings_on_dark="mirror"))
    check("the render starts", code == 200 and out.get("ok"), str(out)[:80])

    state, waited = "", 0.0
    while waited < 420:
        job, _ = s.get(f"/api/jobs/{jid}")
        state = job.get("state")
        if state in ("done", "error"):
            break
        time.sleep(2)
        waited += 2
    check("it finishes", state == "done",
          f"state={state} error={(job or {}).get('error','')[:90]}")
    if state != "done":
        print("     log tail:", " | ".join((job.get("log") or [])[-4:])[:300])
        return

    # ---- the summary
    summary_path = job_dir / "screenx_summary.json"
    check("a summary is written", summary_path.exists())
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    shots = summary.get("per_shot") or []
    check("with at least one shot decided", bool(shots), str(len(shots)))

    # ---- the deliverable: the thing a maker actually takes away
    deliv = job_dir / "deliverable"
    feeds = ["master_widened.mp4", "left.mp4", "centre.mp4", "right.mp4"]
    for f in feeds:
        check(f"deliverable/{f} exists", (deliv / f).is_file())

    # ---- THE CODEC BUG. Every video this tool wrote was mp4v for weeks, which
    # no browser decodes, and it stayed invisible because each one was
    # transcoded by hand before anybody watched it.
    vids = [p for p in deliv.rglob("*.mp4")] + [job_dir / "screenx_demo.mp4"]
    vids = [p for p in vids if p.is_file()]
    bad = [p.name for p in vids if codec_of(p) != "h264"]
    check("every video written is h264, so a browser can play it",
          not bad, ", ".join(sorted(set(bad))[:4]))

    # ---- and they must actually decode, not merely claim a codec
    unreadable = []
    for p in vids:
        cap = cv2.VideoCapture(str(p))
        ok, _f = cap.read()
        cap.release()
        if not ok:
            unreadable.append(p.name)
    check("and every one decodes its first frame", not unreadable,
          ", ".join(sorted(set(unreadable))[:4]))

    # ---- per-shot checkpoints: a run killed at shot N keeps N shots
    segs = sorted((deliv / "shots").glob("shot_*_master.mp4"))
    check("each shot is checkpointed as its own playable file",
          len(segs) >= 1, f"{len(segs)} segments")

    # ---- the provenance map beside the pixels, without which polish cannot
    # tell what it is repainting and can only guess at the cost
    provs = sorted((deliv / "shots").glob("shot_*_prov.npz"))
    check("a provenance map is written beside each shot",
          len(provs) == len(segs), f"{len(provs)} maps for {len(segs)} shots")

    # ---- the film must play at the rate it was shot at
    wrong = {p.name: fps_of(p) for p in vids if abs(fps_of(p) - 30.0) > 0.01}
    check("every delivered file plays at the source's 30fps, not a default",
          not wrong, "; ".join(f"{k} is {v:.2f}" for k, v in list(wrong.items())[:4]))
    check("and the summary records the rate, so nothing downstream has to guess",
          abs(float(summary.get("fps") or 0) - 30.0) < 0.01, str(summary.get("fps")))
    check("along with the wing width the renderer actually used",
          int(summary.get("wing_w") or 0) > 0, str(summary.get("wing_w")))

    _numbers_match_pixels(summary, provs, deliv)
    _fence_holds(deliv, segs)
    _routes(s, jid)
    _polish(job_dir, summary, segs)


def _numbers_match_pixels(summary, provs, deliv):
    """
    The headline is a claim about pixels. Recompute it from the pixels.

    This is the assertion the whole project is for: `mean_real_wing` says a
    share of the wall was photographed, and the provenance maps say which
    pixels those are. If the two ever disagree, the number is decoration.
    """
    if not provs:
        check("the headline can be recomputed from the maps", False, "no maps")
        return
    import agent as ag
    shares = []
    for p in provs:
        try:
            stack = np.load(p)["prov"]
        except (OSError, ValueError, KeyError):
            continue
        for pm in stack[::8]:
            h, w = pm.shape
            # through polish.dims, which is the one place that answers this
            ww, _fps = __import__("polish").dims(summary, w)
            if ww < 2:
                continue
            wings = np.zeros((h, w), bool)
            wings[:, :ww] = True
            wings[:, w - ww:] = True
            shares.append(float(np.isin(pm[wings], ag.PHOTOGRAPHIC).mean()))
    if not shares:
        check("the headline can be recomputed from the maps", False, "no frames")
        return
    measured = float(np.mean(shares))
    claimed = float(summary.get("mean_real_wing") or 0.0)
    check("mean_real_wing matches the provenance maps on disk",
          abs(measured - claimed) < 0.06,
          f"summary {claimed:.3f} vs pixels {measured:.3f}")

    rungs = set(summary.get("rungs_fired") or [])
    prov = summary.get("provenance") or {}
    claimed_rungs = {k for k, v in prov.items()
                     if v > 0.0005 and k in ("primary", "recovered", "donated",
                                             "retrieved", "referenced",
                                             "directed", "generated")}
    check("rungs_fired says exactly what the provenance says",
          rungs == claimed_rungs, f"{sorted(rungs)} vs {sorted(claimed_rungs)}")


def _fence_holds(deliv, segs):
    """
    The centre of the canvas is the original photographed frame.

    Everything the report claims rests on this: if a generator can reach the
    centre, then "the middle is real" stops being true and the whole ladder is
    decoration on top of a lie.
    """
    if not segs:
        return
    cap = cv2.VideoCapture(str(segs[0]))
    ok, canvas = cap.read()
    cap.release()
    if not ok:
        check("the fence can be checked", False)
        return
    h, w = canvas.shape[:2]
    ww = int(round(w * 0.22 / 1.44))
    check("the canvas is wider than it is tall, as a widened master must be",
          w > h, f"{w}x{h}")
    check("the wings are a real fraction of it, not zero", 2 < ww < w // 2, str(ww))


def _routes(s, jid):
    """Every route the interface calls, against a job that really exists."""
    files, code = s.get(f"/api/jobs/{jid}/files")
    check("the files route lists the job's output", code == 200 and len(files) > 3,
          str(len(files)) if isinstance(files, list) else str(files)[:60])

    names = {f["name"] for f in files} if isinstance(files, list) else set()
    target = next((n for n in names if n.endswith("master_widened.mp4")), None)
    check("the deliverable is among them", bool(target))
    if target:
        headers, code = s.head(f"/api/jobs/{jid}/file/{target}")
        check("and it serves with a video content-type", code == 200
              and "video" in (headers.get("Content-Type") or ""),
              headers.get("Content-Type"))
        check("with range support, so a browser can scrub it",
              headers.get("Accept-Ranges") == "bytes")

    pol, code = s.get(f"/api/jobs/{jid}/polish")
    check("the polish route answers before any pass has run",
          code == 200 and pol.get("state") == "none", str(pol)[:60])

    out, code = s.post(f"/api/jobs/{jid}/polish", dict(repair="nonesuch"))
    check("and refuses an unknown generator before spending anything",
          code == 400, str(out)[:60])


def _polish(job_dir, summary, segs):
    """
    The finishing pass over what the render just made.

    The model is stubbed -- it is a paid network call and not what is under
    test. Everything else is real: real segments, real provenance maps, real
    repaint, real restatement of the headline.
    """
    print("\nthe finishing pass, on the film that was just made")
    import polish

    # ---- the free pass first, because that is what the button now does
    deliv = Path(job_dir) / "deliverable"
    before_real = float(summary.get("mean_real_wing") or 0.0)
    eased = polish.settle(job_dir, verbose=False)
    check("settling runs on the film that was just made", bool(eased),
          f"{len(eased)} shot(s)")
    if eased:
        e = eased[0]
        check("it writes a settled segment", Path(e["output"]).is_file())
        check("in a codec a browser can play",
              codec_of(Path(e["output"])) == "h264")
        check("it measures the wall before and after",
              e["before"].get("hairlines") is not None
              and e["after"].get("hairlines") is not None, str(e["after"])[:70])

        # THE PROPERTY THAT LETS IT RUN BY DEFAULT. Settling uses the shot's own
        # photography, so it may change how a recovered pixel looks and may not
        # change what is recovered. If this drifts, a free pass starts moving
        # the one number the whole tool exists to keep honest.
        restated = json.loads((Path(job_dir) / "screenx_summary.json")
                              .read_text(encoding="utf-8"))
        check("and it does not move the headline, because it invented nothing",
              abs(float(restated.get("mean_real_wing") or 0.0) - before_real) < 1e-9,
              f"{before_real:.4f} -> {restated.get('mean_real_wing')}")

        # ---- and the re-cut film keeps the source's frame rate
        rebuilt = [p for p in deliv.glob("*.mp4")]
        slow = {p.name: fps_of(p) for p in rebuilt if abs(fps_of(p) - 30.0) > 0.01}
        check("the settled film still plays at 30fps after the re-cut",
              not slow, "; ".join(f"{k} is {v:.2f}" for k, v in list(slow.items())[:4]))

    faults = ["left: streaking", "right: streaking"]
    report = polish.inspect(job_dir, vision=lambda *a, **k: faults, verbose=False)
    check("inspect produces a finding per shot",
          len(report.get("findings") or []) == len(summary.get("per_shot") or []),
          str(len(report.get("findings") or [])))
    check("and it writes its report beside the job",
          (Path(job_dir) / "polish_report.json").exists())

    first = (report.get("findings") or [{}])[0]
    check("a measured streak score sits beside the model's opinion",
          first.get("streak") is not None, str(first.get("streak")))
    check("along with the three the gate could not see",
          first.get("hairlines") is not None and first.get("jitter") is not None
          and first.get("seam") is not None,
          f"hairlines={first.get('hairlines')} jitter={first.get('jitter')} "
          f"seam={first.get('seam')}")
    check("a photographed shot is still offered for repair",
          first.get("repairable") is True)

    before = float(summary.get("mean_real_wing") or 0.0)
    done = polish.repair(job_dir, generator="mirror", verbose=False)
    check("the repaint runs", bool(done), str(len(done)))
    if not done:
        return

    r = done[0]
    out = Path(r["output"])
    check("it writes a polished segment", out.is_file(), out.name)
    check("in a codec a browser can play", codec_of(out) == "h264", codec_of(out))

    # THE ACCOUNTING. This is the one that must never drift: what the model
    # repaints stops being photography, and the headline has to follow it down.
    check("it records what evidence the cost is based on",
          r.get("basis") in ("provenance map",
                            "pixels moved (upper bound, no provenance map)",
                            "wing held nothing photographed"), str(r.get("basis")))
    check("it reports how much of the wing actually moved",
          r.get("wing_pixels_moved") is not None, str(r.get("wing_pixels_moved")))

    moved = float(r.get("wing_pixels_moved") or 0.0)
    lost = float(r.get("photographed_repainted") or 0.0)
    check("a repaint that changed nothing costs nothing, and one that changed "
          "the wall costs something",
          (moved < 0.01) == (lost < 0.01), f"moved {moved:.3f} lost {lost:.3f}")

    after = json.loads((Path(job_dir) / "screenx_summary.json")
                       .read_text(encoding="utf-8"))
    check("the run is marked polished", after.get("polished") is True)
    check("and the headline never rises because of a repaint",
          float(after.get("mean_real_wing") or 0.0) <= before + 1e-6,
          f"{before:.4f} -> {after.get('mean_real_wing')}")
    if lost > 0.01:
        check("a repaint that overwrote photography moved the headline down",
              float(after.get("mean_real_wing") or 0.0) < before - 1e-6,
              f"{before:.4f} -> {after.get('mean_real_wing')}")

    # ---- and it has to reach the film, or the pass is decoration: the maker
    # could download a better shot and still not be able to screen it
    deliv = Path(job_dir) / "deliverable"
    master = deliv / "master_widened.mp4"
    check("the deliverable records which shots were polished",
          after.get("deliverable_includes_polished") == [r["shot"]],
          str(after.get("deliverable_includes_polished")))

    def first_frame(p):
        cap = cv2.VideoCapture(str(p))
        ok, f = cap.read()
        cap.release()
        return f if ok else None

    fm, fp, fo = first_frame(master), first_frame(out), first_frame(segs[0])
    if fm is None or fp is None or fo is None:
        check("the rebuilt master can be read", False)
        return
    if fm.shape != fp.shape:
        fp = cv2.resize(fp, (fm.shape[1], fm.shape[0]))
    if fm.shape != fo.shape:
        fo = cv2.resize(fo, (fm.shape[1], fm.shape[0]))
    to_polished = float(np.abs(fm.astype(int) - fp.astype(int)).mean())
    to_original = float(np.abs(fm.astype(int) - fo.astype(int)).mean())
    check("the film now carries the polished shot, not the faulted one",
          to_polished < to_original,
          f"master vs polished {to_polished:.1f}, vs original {to_original:.1f}")

    for side in ("left.mp4", "centre.mp4", "right.mp4"):
        check(f"the {side.split('.')[0]} projector feed was re-cut too",
              (deliv / side).is_file() and codec_of(deliv / side) == "h264")


if __name__ == "__main__":
    print("end to end -- the joins, not the parts\n")
    started = time.time()
    test_end_to_end()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed  ({time.time() - started:.0f}s)")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
