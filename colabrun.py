"""
colabrun -- offload a render to a Colab runtime.

WHY THIS EXISTS
---------------
Two rungs and one backend need hardware this laptop does not have: RETRIEVED
needs COLMAP and CUDA, the gaussian backend needs CUDA, and the diffusion
generator needs both. `remote.py` has always described that path -- pack a job,
ship it to a GPU host, read the result back -- and every module that hits the
wall points at it, but nothing ever drove it and it needs an SSH host somebody
has to own.

Google's Colab CLI removes the host problem: `colab new --gpu T4` allocates one,
`colab exec` runs code on it, `colab download` brings the result back. This
wraps that into the same shape the local runner has, so the interface can offer
"run this on a GPU" as a checkbox rather than a weekend.

WHAT IT DOES NOT DO
-------------------
It does not authenticate. The CLI's first run needs a browser and a pasted code,
which is a person's job, not a server's -- `available()` reports whether that
has already happened rather than trying to do it.

It also does not pretend a GPU is there when it is not. Colab refuses an
accelerator when the account is over quota while still handing out CPU runtimes,
so `allocate()` reports which it got and the caller decides whether that is
worth proceeding with. Silently running the CPU path on a "GPU" job would
produce a result that looks right and proves nothing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = "colab"
AUTH = ("--auth", "adc")           # ADC: no key, and the token renews itself
TIMEOUT = 900


def _run(args, timeout=TIMEOUT):
    """One CLI call. Returns (code, combined output)."""
    try:
        p = subprocess.run([CLI, *AUTH, *args], capture_output=True, text=True,
                           timeout=timeout, shell=(shutil.which(CLI) is None))
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"{type(e).__name__}: {e}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def available() -> dict:
    """
    Can this machine drive a Colab runtime right now, and with what?

    Distinguishes three states a caller cares about differently: the CLI is
    missing, the CLI is there but nobody has logged in, or it is ready.
    """
    if shutil.which(CLI) is None:
        return dict(ok=False, reason="colab CLI not installed "
                                     "(uv tool install google-colab-cli)")
    code, out = _run(["sessions"], timeout=120)
    if "authoriz" in out.lower() or "authorization code" in out.lower():
        return dict(ok=False, reason="colab CLI is not logged in; run "
                                     "`colab new --gpu T4` once in a terminal")
    if code != 0 and "No active sessions" not in out:
        return dict(ok=False, reason=out.strip().splitlines()[-1][:120]
                    if out.strip() else "colab CLI failed")
    return dict(ok=True, reason="")


def allocate(gpu="T4"):
    """
    Get a runtime. Falls back to CPU and says so rather than failing outright.

    Colab hands out CPU runtimes while refusing accelerators when an account is
    over its GPU quota, and the two failures need different responses: a CPU
    runtime is useless for the 3D path and perfectly good for everything else.
    """
    code, out = _run(["new", "--gpu", gpu], timeout=600)
    if code == 0 and "READY" in out:
        return dict(ok=True, accelerator=gpu, note="")
    refused = "Unavailable" in out or "TooManyAssignments" in out or code != 0
    code, out2 = _run(["new"], timeout=600)
    if code == 0 and "READY" in out2:
        return dict(ok=True, accelerator="CPU",
                    note=f"{gpu} refused ({'quota or none free' if refused else 'unknown'});"
                         f" got a CPU runtime instead")
    return dict(ok=False, accelerator=None,
                note=(out2 or out).strip().splitlines()[-1][:160] if (out2 or out)
                else "no runtime")


def toolkit_zip(dest: Path) -> Path:
    """The pipeline, without the media or the job folders."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(HERE.glob("*.py")):
            z.write(f, f.name)
    return dest


def upload(local: Path, remote: str):
    return _run(["upload", str(local), remote], timeout=1800)


def exec_file(path: Path, timeout=TIMEOUT):
    return _run(["exec", "--file", str(path), "--timeout", str(int(timeout))],
                timeout=timeout + 120)


def download(remote: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    return _run(["download", remote, str(local)], timeout=1800)


def stop():
    return _run(["stop"], timeout=300)


SETUP = '''
import subprocess, os, zipfile, pathlib

def sh(cmd, timeout=2400):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                       timeout=timeout)
    print("$", cmd, "->", p.returncode)
    tail = ((p.stdout or "") + (p.stderr or "")).strip()[-400:]
    if tail:
        print("   ", tail.replace(chr(10), " | ")[:400])

pathlib.Path("/content/toolkit").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/tk.zip") as z:
    z.extractall("/content/toolkit")
print("py files:", len(list(pathlib.Path("/content/toolkit").glob("*.py"))))

NEED_GPU = %(need_gpu)s
if NEED_GPU:
    sh("apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y colmap >/dev/null 2>&1; which colmap")
    sh("pip install -q gsplat 2>&1 | tail -2")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch:", e)
print("READY")
'''

LAUNCH = '''
import subprocess, os, time
os.makedirs("/content/logs", exist_ok=True)
cmd = ("cd /content/toolkit && setsid nohup python screenx_render.py "
       "/content/%(clip)s -o /content/out %(flags)s "
       "< /dev/null > /content/logs/run.log 2>&1 & echo $! > /content/logs/run.pid")
subprocess.run(["bash", "-lc", cmd], stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL, timeout=60, start_new_session=True)
time.sleep(3)
print("pid", open("/content/logs/run.pid").read().strip())
'''

POLL = '''
import subprocess, os, json
out = subprocess.run(["bash", "-lc",
    # -a and the bash filter: a bare `pgrep -f` matches the very command
    # running it, so the job looks alive forever
    "pgrep -af screenx_render.py | grep -v 'bash -lc' | head -1; echo '@@log'; "
    "tail -c 1400 /content/logs/run.log 2>/dev/null"],
    capture_output=True, text=True, timeout=120).stdout
alive, _, log = out.partition("@@log")
print("@@ALIVE" if alive.strip() else "@@IDLE")
print(log)
print("@@SUMMARY", os.path.exists("/content/out/screenx_summary.json"))
'''


def script(body: str, scratch: Path, name: str) -> Path:
    """Write one of the templates above to a file the CLI can execute."""
    scratch.mkdir(parents=True, exist_ok=True)
    p = scratch / name
    p.write_text(body, encoding="utf-8")
    return p


def parse_shots(log: str):
    """The per-shot lines the render prints, for a caller showing progress."""
    out = []
    for m in re.finditer(r"^\s{2}shot\s+(\d+)\s+(\w+)\s+(\w+)\s+geom\s+([\d.]+)dB"
                         r"\s+eff\s+([\d.]+)%", log, re.M):
        out.append(dict(shot=int(m.group(1)), motion=m.group(2),
                        state=m.group(3), geometry=float(m.group(4)),
                        effective=float(m.group(5)) / 100.0))
    return out


if __name__ == "__main__":
    import sys
    a = available()
    print("colab CLI:", "ready" if a["ok"] else a["reason"])
    if a["ok"] and "--allocate" in sys.argv:
        got = allocate()
        print("runtime:", got)
        if got["ok"]:
            print(json.dumps(got, indent=1))
