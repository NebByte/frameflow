"""
Frameflow Studio -- one interface over the whole toolkit.

    python app.py                 # http://localhost:8420
    python app.py --local         # bind 127.0.0.1 only
    python app.py --token SECRET  # require ?t=SECRET once, then a cookie

WHY THIS REPLACED THREE FRONT DOORS
-----------------------------------
There used to be `render.py` (18 flags), `demo.py` (11 of them) and
`serve.py` (5 of demo's). Each layer quietly dropped capability, so the browser
could not reach the 3D path, a second cut, the context layer or the reasoning
step at all. This exposes the render surface directly and loses nothing.

WHAT IT DOES NOT DO
-------------------
It does not import the pipeline. `serve.py` shelled out on purpose -- "so the
browser and the CLI cannot drift apart" -- and that is still right: a render
that dies takes a subprocess with it and not the server, and `render.py`
stays the single implementation of what a run means. The one thing gained over
scraping stdout is `--progress-json`, which hands back the per-shot record
itself rather than a line of text to be parsed back apart.

Stdlib only. `requirements.txt` is opencv + numpy and this does not add to it.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
JOBS_DIR = HERE / "jobs"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
GENERATORS = ("mirror", "inpaint", "diffusion", "hosted", "wavespeed", "gemini-edit")
MAX_BYTES = 4 << 30
CHUNK = 1 << 16

JOBS: dict[str, dict] = {}
# Keys the operator pasted for this session. Held in memory and passed to the
# render subprocess; never written to disk, never returned by any route, and
# gone when the server stops. A hosted generator needs a credential and the repo
# is not the place to keep one.
KEYS: dict[str, str] = {}
LOCK = threading.Lock()
SLOT = threading.Semaphore(1)      # the pipeline is CPU-bound: one clip at a time
TOKEN = ""


# ------------------------------------------------------------------ capability

def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def capabilities() -> dict:
    """
    What this machine can do right now, with the reason when it cannot.

    The UI greys out what is unavailable and shows the reason, instead of
    letting someone tick --prefer-3d on a GPU-less box and wait several minutes
    to be told the backend refuses.
    """
    caps = {}

    why = "no CUDA device visible"
    try:
        from frameflow import backends as bk
        cuda = bool(bk.GaussianBackend.available())
    except Exception as e:
        # Keep why it failed. This used to capture the exception and then
        # report "no CUDA device visible" anyway, so an ImportError in the
        # backend and a machine with no GPU produced the same message.
        cuda, why = False, f"{type(e).__name__}: {e}"[:120]
    caps["gpu"] = dict(ok=cuda, label="CUDA / gaussian backend",
                       reason="" if cuda else why,
                       enables=["prefer_3d"])

    try:
        from frameflow import sfm
        colmap = bool(sfm.colmap_available())
    except Exception:
        colmap = False
    caps["colmap"] = dict(ok=colmap, label="COLMAP",
                          reason="" if colmap else "colmap not on PATH",
                          enables=["sfm"])

    caps["ffmpeg"] = dict(ok=_has("ffmpeg"), label="ffmpeg",
                          reason="" if _has("ffmpeg") else "ffmpeg not on PATH",
                          enables=["web_encode"])

    ws = bool(KEYS.get("WAVESPEED_API_KEY") or os.environ.get("WAVESPEED_API_KEY")
              or os.environ.get("FRAMEFLOW_TOKEN")
              or os.environ.get("SCREENX_TOKEN"))   # pre-rename name still read
    caps["wavespeed"] = dict(ok=ws, label="WaveSpeed outpainter",
                             reason="" if ws else "paste a key below, or set "
                                                  "WAVESPEED_API_KEY",
                             enables=["wings_on_dark:wavespeed"],
                             wants_key="WAVESPEED_API_KEY")

    try:
        from frameflow import colabrun
        cb = colabrun.available()
    except Exception as e:
        cb = dict(ok=False, reason=f"{type(e).__name__}")
    caps["colab"] = dict(ok=bool(cb["ok"]), label="Colab runtime (remote GPU)",
                         reason=cb.get("reason", ""), enables=["remote"])

    gem, why = False, "no credential"
    try:
        from frameflow import gemini
        if os.environ.get("GEMINI_API_KEY"):
            gem, why = True, ""
        elif gemini.adc_project():
            gem, why = True, ""
            why = ""
    except Exception as e:
        why = type(e).__name__
    caps["gemini"] = dict(ok=gem, label="Gemini vision",
                          reason="" if gem else why, enables=["vision"])
    return caps


# ------------------------------------------------------------------ job model

def under(root: Path, rel: str) -> Path | None:
    """
    Resolve rel under root, or None if it escapes.

    An absolute path handed to `/` discards the left operand entirely, so this
    has to compare the resolved result rather than trust the join.
    """
    try:
        target = (root / rel).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return target


CONTEXT_EXT = {".srt", ".vtt", ".txt", ".fountain", ".json", ".md",
               ".png", ".jpg", ".jpeg"}


def context_name(raw: str) -> str:
    """
    A context file we are willing to write.

    Subtitles and screenplays bind to a shot and make its wings DIRECTED, which
    is a rung, not a decoration -- so the allow-list is explicit rather than
    "anything that is not a video".
    """
    raw = unquote(raw or "").replace("\\", "/").split("/")[-1]
    stem = Path(raw).stem[:60]
    ext = Path(raw).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "context"
    return stem + ext if ext in CONTEXT_EXT else ""


def read_span(cap, crop, a, b, maxw, n):
    """Evenly spaced frames from one shot, for the cheap analysis pass."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = crop
    out = []
    for i in np.linspace(a, max(a, b - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            break
        f = f[y0:y1, x0:x1]
        if f.shape[1] > maxw:
            sc = maxw / f.shape[1]
            f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def safe_name(raw: str) -> str:
    """A filename we are willing to write to disk. Ported from serve.py."""
    raw = unquote(raw or "").replace("\\", "/").split("/")[-1]
    stem = Path(raw).stem[:60]
    ext = Path(raw).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "clip"
    if ext not in VIDEO_EXT:
        return ""
    return stem + ext


def clamp(opts: dict) -> dict:
    """
    Server-side bounds. The browser is not trusted to have sent sane numbers,
    and a maxw of 40000 is a machine hang rather than a bad render.
    """
    def num(key, lo, hi, default):
        try:
            return max(lo, min(hi, int(opts.get(key, default))))
        except (TypeError, ValueError):
            return default

    def rng(key, lo, hi):
        """An optional float. Absent means the pipeline default stands."""
        raw = opts.get(key)
        if raw in (None, ""):
            return None
        try:
            return max(lo, min(hi, float(raw)))
        except (TypeError, ValueError):
            return None

    dark = opts.get("wings_on_dark") or None
    return dict(
        maxw=num("maxw", 160, 1920, 640),
        frames_per_shot=num("frames_per_shot", 20, 2000, 200),
        max_shots=num("max_shots", 0, 999, 0),
        rotate=int(opts.get("rotate", 0)) if int(opts.get("rotate", 0) or 0)
        in (0, 90, 180, 270) else 0,
        wings_on_dark=dark if dark in GENERATORS else None,
        sources=bool(opts.get("sources")),
        prefer_3d=bool(opts.get("prefer_3d")),
        sfm=bool(opts.get("sfm")),
        reason=bool(opts.get("reason")),
        vision=bool(opts.get("vision")),
        online=bool(opts.get("online")),
        library=(opts.get("library") or None),
        # the extended film is the product; the report is about it. Written
        # unless the caller explicitly asks not to.
        deliver=(opts.get("deliver", "1") not in ("", "0", False, None)),
        **{k: rng(k, lo, hi) for k, lo, hi in (
            ("wing", 0.05, 0.60),
            ("screen_width", 3.0, 40.0), ("screen_height", 2.0, 25.0),
            ("viewer_distance", 2.0, 60.0), ("wing_dim", 0.2, 1.0),
            ("gate_geometry", 0.0, 60.0),
            ("gate_full", 0.05, 1.0), ("gate_narrow", 0.01, 1.0),
            ("gate_detail", 0.0, 1.0), ("gate_stale", 0.05, 30.0))},
    )


def build_argv(clip: Path, outdir: Path, opts: dict) -> list:
    """
    The exact command line this job runs.

    Kept as one pure function so a test can assert the UI produces the same argv
    as the documented CLI line, rather than the two drifting the way demo.py and
    render.py did.
    """
    o = clamp(opts)
    argv = [sys.executable, "-u", "-m", "frameflow.render", str(clip),
            "-o", str(outdir), "--maxw", str(o["maxw"]),
            "--frames-per-shot", str(o["frames_per_shot"]),
            "--progress-json"]
    if o["max_shots"]:
        argv += ["--max-shots", str(o["max_shots"])]
    if o["rotate"]:
        argv += ["--rotate", str(o["rotate"])]
    if o["wings_on_dark"]:
        argv += ["--wings-on-dark", o["wings_on_dark"]]
    if o["sources"]:
        argv += ["--sources"]
    if o["sfm"]:
        argv += ["--sfm", str(outdir / "sfm")]
    if o["prefer_3d"]:
        argv += ["--prefer-3d"]
    if o["reason"]:
        argv += ["--reason"]
    if o["vision"]:
        argv += ["--vision"]
    if o["online"]:
        argv += ["--online"]
    if o["library"]:
        argv += ["--library", str(o["library"])]
    if o["deliver"]:
        argv += ["--deliver", "deliverable"]

    # geometry and gate: reachable in code since the beginning and never from a
    # command line, so in practice the wing width and the bar were constants
    for flag, key in (("--wing", "wing"), ("--screen-width", "screen_width"),
                      ("--screen-height", "screen_height"),
                      ("--viewer-distance", "viewer_distance"),
                      ("--wing-dim", "wing_dim"),
                      ("--gate-geometry", "gate_geometry"),
                      ("--gate-full", "gate_full"),
                      ("--gate-narrow", "gate_narrow"),
                      ("--gate-detail", "gate_detail"),
                      ("--gate-stale", "gate_stale")):
        if o.get(key) is not None:
            argv += [flag, str(o[key])]

    # attachments are files the operator uploaded beside the clip: another cut
    # for DONATED, another setup for RETRIEVED, context for DIRECTED
    for kind, flag in (("other_cut", "--other-cut"), ("also", "--also"),
                       ("context", "--context")):
        for extra in (opts.get(kind) or []):
            argv += [flag, str(extra)]
    return argv


def new_job(name: str) -> dict:
    with LOCK:
        jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{len(JOBS) + 1:02d}"
        job = dict(id=jid, name=name, state="staged", started=time.time(),
                   dir=str(JOBS_DIR / jid), log=[], shots=[], error="",
                   argv=[], summary=None, clip="", attachments={},
                   analysis=None)
        JOBS[jid] = job
    return job


def run_remote(job: dict, clip: Path, opts: dict):
    """
    Run this job on a Colab runtime instead of here.

    The reason to bother is narrow and real: the gaussian backend, the RETRIEVED
    rung and the diffusion generator all need CUDA, and no amount of patience on
    a laptop substitutes. Everything else is faster locally once the upload is
    counted.

    An accelerator is not assumed. Colab hands out CPU runtimes while refusing
    GPUs to an account over quota, so if a GPU was asked for and a CPU arrived,
    that is reported and the 3D flags are dropped rather than being sent to a
    machine that cannot honour them -- a 3D run silently served by the CPU path
    would look like a result and prove nothing.
    """
    from frameflow import colabrun as cr
    outdir = Path(job["dir"])
    scratch = outdir / "remote"
    log = job["log"].append

    log("allocating a Colab runtime...")
    got = cr.allocate("T4")
    if not got["ok"]:
        job["state"], job["error"] = "error", f"no runtime: {got['note']}"
        return
    job["accelerator"] = got["accelerator"]
    if got["note"]:
        log(got["note"])
    log(f"runtime: {got['accelerator']}")

    wants_gpu = bool(opts.get("prefer_3d") or opts.get("sfm"))
    if wants_gpu and got["accelerator"] == "CPU":
        opts = dict(opts, prefer_3d=False, sfm=False)
        log("3D flags dropped: this runtime has no GPU, and running the CPU "
            "path under a 3D label would prove nothing")

    log("uploading toolkit and clip...")
    cr.upload(cr.toolkit_zip(scratch / "tk.zip"), "/content/tk.zip")
    cr.upload(clip, f"/content/{clip.name}")

    code, out = cr.exec_file(cr.script(cr.SETUP % dict(need_gpu=wants_gpu),
                                       scratch, "setup.py"), timeout=1800)
    for line in (out or "").splitlines()[-8:]:
        log(line)
    if "READY" not in (out or ""):
        job["state"], job["error"] = "error", "remote setup failed"
        return

    # the same argv the local runner builds, minus the paths it owns
    argv = build_argv(clip, outdir, opts)
    flags = []
    it = iter(argv[4:])
    for a in it:
        if a == "-o":
            next(it, None)
            continue
        flags.append(a)
    job["argv"] = argv
    cr.exec_file(cr.script(cr.LAUNCH % dict(clip=clip.name, flags=" ".join(flags)),
                           scratch, "launch.py"), timeout=300)

    poll = cr.script(cr.POLL, scratch, "poll.py")
    seen = 0
    for _ in range(180):
        time.sleep(10)
        code, out = cr.exec_file(poll, timeout=200)
        for rec in cr.parse_shots(out or "")[seen:]:
            job["shots"].append(rec)
            seen += 1
        if "@@IDLE" in (out or ""):
            break

    log("downloading results...")
    ok = False
    for remote, local in (("/content/out/screenx_summary.json",
                           outdir / "screenx_summary.json"),
                          ("/content/out/screenx_demo.mp4", outdir / "screenx_demo.mp4"),
                          ("/content/out/deliverable/master_widened.mp4",
                           outdir / "deliverable" / "master_widened.mp4"),
                          ("/content/out/deliverable/left.mp4",
                           outdir / "deliverable" / "left.mp4"),
                          ("/content/out/deliverable/centre.mp4",
                           outdir / "deliverable" / "centre.mp4"),
                          ("/content/out/deliverable/right.mp4",
                           outdir / "deliverable" / "right.mp4")):
        c, _o = cr.download(remote, Path(local))
        ok = ok or (c == 0 and Path(local).exists())

    summary = outdir / "screenx_summary.json"
    if summary.exists():
        try:
            job["summary"] = json.loads(summary.read_text(encoding="utf-8"))
            job["state"] = "done"
        except ValueError as e:
            job["state"], job["error"] = "error", f"summary unreadable: {e}"
    else:
        job["state"], job["error"] = "error", "remote run produced no summary"
    cr.stop()
    log("runtime released")


def run_job(job: dict, clip: Path, opts: dict):
    """Drive one render to completion. Runs on its own thread."""
    if opts.get("remote"):
        with SLOT:
            job["state"] = "running"
            try:
                return run_remote(job, clip, opts)
            except Exception as e:
                job["state"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"
                return
    outdir = Path(job["dir"])
    argv = build_argv(clip, outdir, opts)
    job["argv"] = argv
    with SLOT:
        job["state"] = "running"
        job["started"] = time.time()
        try:
            env = {**os.environ, **KEYS}      # session keys, never persisted
            proc = subprocess.Popen(argv, cwd=str(HERE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, env=env)
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("@@SHOT "):
                    try:
                        job["shots"].append(json.loads(line[7:]))
                        continue
                    except ValueError:
                        pass
                job["log"].append(line)
                del job["log"][:-400]
            code = proc.wait()
        except Exception as e:
            job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
            return

        summary = outdir / "screenx_summary.json"
        if code == 0 and summary.exists():
            try:
                job["summary"] = json.loads(summary.read_text(encoding="utf-8"))
            except ValueError as e:
                job["error"] = f"summary unreadable: {e}"
            job["state"] = "done"
        else:
            job["state"] = "error"
            job["error"] = job["error"] or (
                "no shots rendered" if code else "render produced no summary")


def run_polish(job: dict, repair: str | None, shots: str = "",
               settle: bool = True, full: bool = False):
    """
    The finishing pass over a film that is already rendered. Own thread.

    A subprocess for the same reason a render is one: `polish.py` stays the
    single implementation of what inspection and repair mean, so the browser
    cannot drift from the terminal, and a vision call that hangs or a generator
    that dies takes a subprocess with it rather than the server.

    Repair rewrites the run's summary -- that is the whole point of it, since
    repainted pixels stop counting as photographed -- so the fresh summary is
    read back here. A report screen still showing the pre-polish real-wing
    figure would be reporting pixels as filmed that a model drew a minute ago.
    """
    d = Path(job["dir"])
    argv = [sys.executable, "-m", "frameflow.polish", str(d)]
    # The settle pass runs unless it is explicitly waived. It is free, calls
    # nothing hosted, invents no pixel and cannot move the real-wing figure --
    # so there is no version of "finish this film" where skipping it by default
    # is the right guess.
    if settle is False:
        argv += ["--no-settle"]
    if repair:
        argv += ["--repair", repair]
        if full:
            argv += ["--full"]
        if shots:
            argv += ["--shots", shots]
    p = job["polish"] = dict(state="running", repairing=bool(repair),
                             settling=settle is not False, log=[],
                             report=None, repaired=[], settled=[], error="",
                             argv=argv)
    code = 0
    with SLOT:
        try:
            env = {**os.environ, **KEYS}      # session keys, never persisted
            proc = subprocess.Popen(argv, cwd=str(HERE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, env=env)
            for line in proc.stdout:
                p["log"].append(line.rstrip())
                del p["log"][:-200]
            code = proc.wait()
        except Exception as e:
            p["state"], p["error"] = "error", f"{type(e).__name__}: {e}"
            return

    report = d / "polish_report.json"
    if report.exists():
        try:
            p["report"] = json.loads(report.read_text(encoding="utf-8"))
            p["repaired"] = p["report"].get("repaired") or []
            p["settled"] = p["report"].get("settled") or []
        except ValueError as e:
            p["error"] = f"report unreadable: {e}"

    # A settle re-cuts the deliverable too, so the players must be pointed at
    # the new files even though no number changed.
    if p["repaired"] or p["settled"]:
        summary = d / "screenx_summary.json"
        try:
            job["summary"] = json.loads(summary.read_text(encoding="utf-8"))
            job["shots"] = job["summary"].get("per_shot", job.get("shots") or [])
        except (OSError, ValueError) as e:
            p["error"] = p["error"] or f"restated summary unreadable: {e}"

    if code and not p["report"]:
        p["state"] = "error"
        p["error"] = p["error"] or f"polish exited {code}"
    else:
        p["state"] = "done"


def known_jobs() -> list:
    """Live jobs plus anything already on disk, so a restart loses nothing."""
    out = {j["id"]: dict(id=j["id"], name=j["name"], state=j["state"],
                         started=j["started"],
                         real=(j.get("summary") or {}).get("mean_real_wing"))
           for j in JOBS.values()}
    if JOBS_DIR.exists():
        for d in sorted(JOBS_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d / "screenx_summary.json").exists() and d.name not in out:
                try:
                    s = json.loads((d / "screenx_summary.json").read_text(encoding="utf-8"))
                except ValueError:
                    continue
                out[d.name] = dict(id=d.name, name=s.get("source", d.name),
                                   state="done", started=d.stat().st_mtime,
                                   real=s.get("mean_real_wing"))
    return sorted(out.values(), key=lambda j: j["started"], reverse=True)[:40]


# ------------------------------------------------------------------ http

class Handler(BaseHTTPRequestHandler):
    server_version = "ScreenXStudio"

    def log_message(self, fmt, *args):
        pass                                    # the render log is the useful one

    # -- plumbing

    def _write(self, buf: bytes):
        # A client that closes mid-download is ordinary -- a browser seeking in
        # a video does it on every scrub. Windows reports that as
        # ConnectionAbortedError (WinError 10053) rather than BrokenPipe, which
        # was not in this tuple, so the server printed a traceback for routine
        # behaviour. Noise in a log is not free: it is what a real error hides in.
        try:
            self.wfile.write(buf)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise

    def _send(self, code, body: bytes, ctype="application/json", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self._write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def _file(self, path: Path):
        """Static send with Range support, so video scrubbing works."""
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200

        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if m and size:
            lo, hi = m.group(1), m.group(2)
            if lo:
                start = min(int(lo), size - 1)
                end = min(int(hi), size - 1) if hi else size - 1
            elif hi:
                start = max(0, size - int(hi))
            if start <= end:
                code = 206

        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                buf = fh.read(min(CHUNK, left))
                if not buf:
                    break
                self._write(buf)
                left -= len(buf)

    def _allowed(self) -> bool:
        if not TOKEN:
            return True
        q = parse_qs(urlparse(self.path).query)
        return (q.get("t", [""])[0] == TOKEN
                or f"sxtoken={TOKEN}" in (self.headers.get("Cookie") or ""))

    # -- routes

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = urlparse(self.path).path
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")

        if route in ("/", "/index.html"):
            extra = ()
            if TOKEN and parse_qs(urlparse(self.path).query).get("t"):
                extra = (("Set-Cookie", f"sxtoken={TOKEN}; Path=/; SameSite=Lax"),)
            return self._file_or_404(STATIC / "index.html", extra)

        if route == "/api/capabilities":
            return self._json(capabilities())

        if route == "/api/jobs":
            return self._json(known_jobs())

        m = re.match(r"^/api/jobs/([\w.-]+)$", route)
        if m:
            job = JOBS.get(m.group(1)) or self._from_disk(m.group(1))
            return self._json(job or {"error": "no such job"}, 200 if job else 404)

        m = re.match(r"^/api/jobs/([\w.-]+)/events$", route)
        if m:
            return self._events(m.group(1))

        m = re.match(r"^/api/jobs/([\w.-]+)/files$", route)
        if m:
            return self._files(m.group(1))

        m = re.match(r"^/api/jobs/([\w.-]+)/polish$", route)
        if m:
            return self._polish_state(m.group(1))

        m = re.match(r"^/api/jobs/([\w.-]+)/file/(.+)$", route)
        if m:
            root = JOBS_DIR / m.group(1)
            target = under(root, m.group(2))
            if target is None or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            return self._file(target)

        if route.startswith("/static/"):
            target = under(STATIC, route[len("/static/"):])
            if target is None or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            return self._file(target)

        return self._send(404, b"not found", "text/plain")

    def _file_or_404(self, path: Path, extra=()):
        if not path.is_file():
            return self._send(404, b"static/index.html missing", "text/plain")
        body = path.read_bytes()
        return self._send(200, body, "text/html; charset=utf-8", extra)

    @staticmethod
    def _from_disk(jid: str):
        """
        Rehydrate a finished job from what it left on disk.

        The shape has to match `new_job`'s exactly, because a caller is entitled
        to put this back into JOBS -- polish does, so that a pass started on a
        job the server has never seen can be polled. A record missing a key that
        `known_jobs` reads takes the whole job list down for the life of the
        process, and it does it the moment someone polishes rather than at the
        point the record was built.
        """
        d = JOBS_DIR / jid
        f = d / "screenx_summary.json"
        if not f.exists():
            return None
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            return None
        clip = next((str(p) for p in sorted(d.glob("*"))
                     if p.suffix.lower() in VIDEO_EXT), "")
        return dict(id=jid, name=s.get("source", jid), state="done",
                    started=d.stat().st_mtime, dir=str(d), log=[],
                    shots=s.get("per_shot", []), summary=s, error="", argv=[],
                    clip=clip, attachments={}, analysis=None)

    def _events(self, jid: str):
        """
        Server-sent events: one message per shot as it lands, then the summary.

        Polls the job's own lists rather than holding the render thread, so a
        client that disconnects mid-render costs nothing.
        """
        job = JOBS.get(jid)
        if job is None:
            return self._send(404, b"no such job", "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sent_shots, sent_log = 0, 0
        try:
            while True:
                while sent_shots < len(job["shots"]):
                    rec = job["shots"][sent_shots]
                    sent_shots += 1
                    self._write(b"event: shot\ndata: "
                                + json.dumps(rec, default=str).encode() + b"\n\n")
                while sent_log < len(job["log"]):
                    line = job["log"][sent_log]
                    sent_log += 1
                    self._write(b"event: log\ndata: "
                                + json.dumps(line).encode() + b"\n\n")
                if job["state"] in ("done", "error"):
                    payload = dict(state=job["state"], error=job["error"],
                                   summary=job.get("summary"))
                    self._write(b"event: end\ndata: "
                                + json.dumps(payload, default=str).encode() + b"\n\n")
                    return
                self._write(b": keepalive\n\n")
                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        route = urlparse(self.path).path
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")

        if route == "/api/jobs":
            return self._create_job()

        if route == "/api/keys":
            return self._set_keys()

        for pattern, method in (
                (r"^/api/jobs/([\w.-]+)/notes$", self._add_note),
                (r"^/api/jobs/([\w.-]+)/attach$", self._attach),
                (r"^/api/jobs/([\w.-]+)/start$", self._start),
                (r"^/api/jobs/([\w.-]+)/analyse$", self._analyse),
                (r"^/api/jobs/([\w.-]+)/polish$", self._polish)):
            m = re.match(pattern, route)
            if m:
                return method(m.group(1))

        return self._send(404, b"not found", "text/plain")

    def do_DELETE(self):
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")
        m = re.match(r"^/api/jobs/([\w.-]+)$", urlparse(self.path).path)
        if m:
            return self._delete(m.group(1))
        return self._send(404, b"not found", "text/plain")

    def _create_job(self):
        q = parse_qs(urlparse(self.path).query)
        name = safe_name(q.get("name", [""])[0])
        if not name:
            return self._json({"error": "unsupported file type"}, 400)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BYTES:
            return self._json({"error": "bad content length"}, 400)

        job = new_job(name)
        outdir = Path(job["dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        clip = outdir / name
        self._recv_to(clip, length)
        job["clip"] = str(clip)
        # staged, not started: the operator may still attach another cut, a
        # second setup, or context files, and each of those changes the run
        return self._json({"id": job["id"], "state": job["state"]})

    def _recv_to(self, path: Path, length: int):
        left = length
        with open(path, "wb") as fh:
            while left > 0:
                buf = self.rfile.read(min(CHUNK, left))
                if not buf:
                    break
                fh.write(buf)
                left -= len(buf)

    def _attach(self, jid: str):
        """
        A file that rides along with the clip.

        other_cut -> DONATED, also -> RETRIEVED, context -> DIRECTED. Three
        different claims about where pixels may come from, so they stay three
        kinds rather than one "extra files" bucket.
        """
        job = JOBS.get(jid)
        if job is None or job["state"] != "staged":
            return self._json({"error": "no staged job"}, 404)
        q = parse_qs(urlparse(self.path).query)
        kind = q.get("kind", [""])[0]
        if kind not in ("other_cut", "also", "context"):
            return self._json({"error": "unknown attachment kind"}, 400)

        raw = q.get("name", [""])[0]
        name = context_name(raw) if kind == "context" else safe_name(raw)
        if not name:
            return self._json({"error": "unsupported file type"}, 400)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BYTES:
            return self._json({"error": "bad content length"}, 400)

        dest = Path(job["dir"]) / "attached" / kind
        dest.mkdir(parents=True, exist_ok=True)
        self._recv_to(dest / name, length)
        job["attachments"].setdefault(kind, []).append(str(dest / name))
        return self._json({"ok": True, "kind": kind, "name": name,
                           "attachments": job["attachments"]})

    def _start(self, jid: str):
        job = JOBS.get(jid)
        if job is None:
            return self._json({"error": "no such job"}, 404)
        if job["state"] not in ("staged", "error", "done"):
            return self._json({"error": "job is " + job["state"]}, 409)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            opts = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)
        opts = dict(opts)
        opts.update(job["attachments"])          # the files, not the form
        job["state"] = "queued"
        job["shots"], job["log"], job["error"] = [], [], ""
        # a finding describes the walls of the render that produced it. Re-render
        # and those walls are gone, so keeping the report would leave the report
        # screen faulting pixels that no longer exist.
        job.pop("polish", None)
        (JOBS_DIR / jid / "polish_report.json").unlink(missing_ok=True)
        threading.Thread(target=run_job, args=(job, Path(job["clip"]), opts),
                         daemon=True).start()
        return self._json({"ok": True, "id": jid})

    def _analyse(self, jid: str):
        """
        Shot detection before committing to a render.

        The cheap pass: where the cuts are, how each shot moves, how far the
        camera travels. Displacement is the number worth reading -- across every
        clip measured it tracked effective coverage -- so a film of locked-off
        shots can be recognised as a poor candidate in seconds instead of after
        a full render.
        """
        job = JOBS.get(jid)
        if job is None or not job.get("clip"):
            return self._json({"error": "no such job"}, 404)
        try:
            import cv2
            from frameflow import shotdetect as sd
            from frameflow import wingcoverage as wc
            path = job["clip"]
            seg = sd.segment(path)
            shots = [t for t in seg["shots"] if t[1] - t[0] >= 12]
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            tracker = wc.Tracker()
            out = []
            for si, (a, b) in enumerate(shots[:40]):
                frames = read_span(cap, seg["crop"], a, b, 320, 8)
                if len(frames) >= 2:
                    kind, stats = wc.classify_motion(tracker, frames)
                else:
                    kind, stats = "LOCKED", {}
                out.append(dict(shot=si, start=a, frames=b - a,
                                seconds=round((b - a) / fps, 2), motion=kind,
                                displacement=stats.get("displacement", 0.0)))
            cap.release()
            job["analysis"] = dict(crop=list(seg["crop"]), fps=round(fps, 2),
                                   shots=out, total=len(shots))
        except Exception as e:
            return self._json({"error": type(e).__name__ + ": " + str(e)}, 500)
        return self._json(job["analysis"])

    def _polish(self, jid: str):
        """
        Look at the finished walls, and optionally repaint what a model faults.

        A render decides what can be earned; nothing in it asks whether the
        result looks good, because coverage and staleness are statistics over
        pixels and neither notices a wall that is one column smeared forty
        times. This is the pass that asks.

        Repair is NOT refused on photographed wings. A streaked recovered wall
        is a real defect and a maker should be able to fix it -- so what it
        costs is stated instead of forbidden: every pixel the model changes is
        relabelled GENERATED, or DIRECTED where a pinned note drove it, and the
        run's real-wing figure falls by exactly the repainted share. The film
        gets better, the number stays true, and the report carries both.
        """
        job = JOBS.get(jid) or self._from_disk(jid)
        if job is None:
            return self._json({"error": "no such job"}, 404)
        job.setdefault("dir", str(JOBS_DIR / jid))
        JOBS.setdefault(jid, job)
        if job.get("state") == "running":
            return self._json({"error": "job is still rendering"}, 409)
        if (job.get("polish") or {}).get("state") == "running":
            return self._json({"error": "already polishing"}, 409)
        if not (JOBS_DIR / jid / "screenx_summary.json").exists():
            return self._json({"error": "nothing to polish yet — convert first"}, 409)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)

        repair = body.get("repair") or None
        if repair is True:
            repair = "wavespeed"
        if repair and repair not in GENERATORS:
            return self._json({"error": f"unknown generator {repair}"}, 400)

        # Which shots to pay for. A hosted generator bills per shot, and a
        # fast-cut trailer refuses most of them -- 34 shots at $0.20 is $7 spent
        # on one-second fragments. Empty means every faulted shot, as before.
        raw = body.get("shots")
        if isinstance(raw, list):
            raw = ",".join(str(s) for s in raw)
        shots = re.sub(r"[^\d,]", "", str(raw or ""))
        # settle defaults ON; only an explicit false turns it off
        settle = body.get("settle") is not False
        full = bool(body.get("full"))
        threading.Thread(target=run_polish, args=(job, repair, shots, settle, full),
                         daemon=True).start()
        return self._json({"ok": True, "repairing": bool(repair),
                           "settling": settle,
                           "shots": shots or "all faulted"})

    def _polish_state(self, jid: str):
        """
        Whatever the last pass found. Falls back to the report on disk so a
        server restart does not lose a finished inspection.
        """
        job = JOBS.get(jid)
        p = (job or {}).get("polish")
        if p:
            return self._json(p)
        f = JOBS_DIR / jid / "polish_report.json"
        if not f.exists():
            return self._json(dict(state="none", report=None, log=[],
                                   repaired=[], error="", repairing=False))
        try:
            report = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as e:
            return self._json({"error": f"report unreadable: {e}"}, 500)
        return self._json(dict(state="done", report=report, log=[],
                               repaired=report.get("repaired") or [],
                               error="", repairing=False))

    def _files(self, jid: str):
        d = JOBS_DIR / jid
        if not d.is_dir():
            return self._json({"error": "no such job"}, 404)
        out = [dict(name=f.relative_to(d).as_posix(), bytes=f.stat().st_size)
               for f in sorted(d.rglob("*")) if f.is_file()]
        return self._json(out)

    def _delete(self, jid: str):
        d = JOBS_DIR / jid
        if under(JOBS_DIR, jid) is None or not d.is_dir():
            return self._json({"error": "no such job"}, 404)
        shutil.rmtree(d, ignore_errors=True)
        JOBS.pop(jid, None)
        return self._json({"ok": True})

    def _set_keys(self):
        """
        Accept credentials for this session only.

        Held in memory, handed to the render subprocess, never written to disk
        and never read back out: the response says which names are set, not what
        they are. A key that reaches a file ends up in a backup, a zip or a
        repository, and this one bills a card.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)
        allowed = {"WAVESPEED_API_KEY", "GEMINI_API_KEY", "SCREENX_ENDPOINT",
                   "SCREENX_TOKEN"}
        for name, value in (body or {}).items():
            if name not in allowed:
                return self._json({"error": f"unknown credential {name}"}, 400)
            value = (value or "").strip()
            if value:
                KEYS[name] = value
            else:
                KEYS.pop(name, None)
        return self._json({"set": sorted(KEYS)})

    def _add_note(self, jid: str):
        """
        A human pinning what belongs off-frame in one shot.

        context.DirectionStore has always persisted these and render
        reads them on every run -- there has simply never been a way to write
        one. Pixels driven by a note are labelled DIRECTED, which is outside
        PHOTOGRAPHIC: someone who knows the place is still not a camera.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json({"error": "bad json"}, 400)
        shot, text = body.get("shot"), (body.get("text") or "").strip()
        if shot is None or not text:
            return self._json({"error": "shot and text required"}, 400)
        try:
            from frameflow import context as cx
            store = cx.DirectionStore(Path(JOBS_DIR / jid))
            store.add(int(shot), text, body.get("author") or "operator")
            store.save()
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"ok": True})


# ------------------------------------------------------------------ main

def lan_addresses():
    out = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        out.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return out


def main():
    global TOKEN
    ap = argparse.ArgumentParser(description="Frameflow Studio")
    ap.add_argument("-p", "--port", type=int, default=8420)
    ap.add_argument("--host", default=None)
    ap.add_argument("--local", action="store_true", help="bind 127.0.0.1 only")
    ap.add_argument("--token", default="", help="require ?t=TOKEN once")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    TOKEN = a.token
    JOBS_DIR.mkdir(exist_ok=True)
    host = a.host or ("127.0.0.1" if a.local else "0.0.0.0")
    srv = ThreadingHTTPServer((host, a.port), Handler)

    url = f"http://localhost:{a.port}/" + (f"?t={TOKEN}" if TOKEN else "")
    print(f"Frameflow Studio on {url}")
    if host == "0.0.0.0":
        for ip in lan_addresses():
            print(f"  on this network: http://{ip}:{a.port}/"
                  + (f"?t={TOKEN}" if TOKEN else ""))
    caps = capabilities()
    print("  " + "  ".join(
        f"{k}:{'yes' if v['ok'] else 'no'}" for k, v in caps.items()))
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
