"""
remote -- run the GPU tiers somewhere else.

Tier 2 needs CUDA and this machine has none. Rather than making the pipeline
depend on where it runs, a shot is packaged into a job file, executed by a
worker on a GPU host, and the result is read back. The job file is the only
interface, which means the SAME artefact works over SSH, over a shared drive,
or pasted into a notebook cell -- see the transport notes below.

    local                          GPU host
    -----                          --------
    pack_job(frames, ...)  --->    python remote.py worker job/
                                     -> fits splats, renders widened
    read_result(job/)      <---    result.npz

WHY A JOB FILE AND NOT AN RPC
-----------------------------
A shot is an independent unit of work that takes minutes, not milliseconds. A
job file survives a dropped connection, a pre-empted spot instance, and a Colab
runtime that recycled itself halfway through -- all of which are normal, not
exceptional, on rented GPUs. It also means the expensive half can be re-run
against a fixed input when a number looks wrong, which an RPC cannot.

TRANSPORT: WHAT ACTUALLY WORKS
------------------------------
**Google Colab, free tier: SSH is not allowed.** Colab's FAQ lists "remote
control such as SSH shells, remote desktops" among the disallowed activities on
free managed runtimes, and says such sessions "may be terminated at any time
without warning". Tools like colab-ssh work by tunnelling around this; using
them risks the account, and the colab-ssh project is itself marked inactive.
Do not build on it.

**Google Colab, paid: allowed.** The same FAQ says "You can remove these types
of restrictions by purchasing one of our paid plans" -- Pro, Pro+, or Pay As You
Go with a positive compute-unit balance. So SSH into Colab is a licensing
question, not a technical one.

**Colab without SSH at all works fine here, and is the recommended Colab path.**
Upload `job.npz` plus this repo to the runtime, run the worker in a cell, and
download `result.npz`. Nothing in the job protocol needs a shell:

    !pip -q install gsplat torch opencv-python-headless
    !python remote.py worker /content/job

**For anything long-running, a real GPU host beats Colab regardless of tier.**
Colab recycles runtimes and its filesystem is ephemeral; a 78-shot film is hours
of fitting. RunPod, Vast.ai, and Lambda give you SSH as a product feature with
persistent volumes, which is what `RemoteGPU` below assumes.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

JOB = "job.npz"
RESULT = "result.npz"
META = "meta.json"


# ---------------------------------------------------------------- job format

def pack_job(job_dir, frames, wing_w, *, iters=3000, colmap_dir=None,
             mask_dynamics=True, alpha_thresh=0.5, allow_untrusted_poses=False,
             scene_id=None):
    """
    Write a self-contained unit of GPU work.

    Frames are stored as a single uint8 array rather than PNGs: a shot is 30-90
    frames and the array compresses adequately, while a directory of images
    invites the frame ORDER to drift, which silently corrupts pose association.
    """
    d = Path(job_dir)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(f) for f in frames])
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    np.savez_compressed(d / JOB, frames=arr)

    meta = dict(n_frames=int(arr.shape[0]), height=int(arr.shape[1]),
                width=int(arr.shape[2]), wing_w=int(wing_w), iters=int(iters),
                mask_dynamics=bool(mask_dynamics),
                alpha_thresh=float(alpha_thresh),
                allow_untrusted_poses=bool(allow_untrusted_poses),
                colmap_dir=str(colmap_dir) if colmap_dir else None,
                scene_id=scene_id)
    (d / META).write_text(json.dumps(meta, indent=2))
    return d


def read_result(job_dir):
    """-> list of (canvas, filled, tmap), the Backend.propagate contract."""
    d = Path(job_dir)
    f = d / RESULT
    if not f.exists():
        err = d / "error.txt"
        if err.exists():
            raise RuntimeError(f"worker failed:\n{err.read_text()[:2000]}")
        raise FileNotFoundError(f"no {RESULT} in {d}; has the worker run?")
    z = np.load(f)
    canvases, filled, tmaps = z["canvas"], z["filled"], z["tmap"]
    return [(canvases[i], filled[i].astype(bool), tmaps[i].astype(np.int32))
            for i in range(len(canvases))]


def write_result(job_dir, rendered):
    d = Path(job_dir)
    np.savez_compressed(
        d / RESULT,
        canvas=np.stack([r[0] for r in rendered]),
        filled=np.stack([r[1] for r in rendered]),
        tmap=np.stack([r[2] for r in rendered]))
    return d / RESULT


# ---------------------------------------------------------------- the worker

def worker(job_dir, verbose=True):
    """
    Runs ON the GPU host. Reads job.npz, writes result.npz.

    Failures are written to error.txt rather than only raised, because the most
    common way to run this is detached in a session you will not be watching.
    """
    from . import backends as bk
    d = Path(job_dir)
    meta = json.loads((d / META).read_text())
    frames = list(np.load(d / JOB)["frames"])

    if verbose:
        print(f"job {d}: {len(frames)} frames "
              f"{meta['width']}x{meta['height']}, wing {meta['wing_w']}px, "
              f"{meta['iters']} iters")
    try:
        backend = bk.GaussianBackend(
            iters=meta["iters"], mask_dynamics=meta["mask_dynamics"],
            colmap_dir=meta["colmap_dir"], alpha_thresh=meta["alpha_thresh"],
            allow_untrusted_poses=meta["allow_untrusted_poses"])
        rendered = backend.propagate(frames, meta["wing_w"])
        out = write_result(d, rendered)
    except Exception as e:                    # noqa: BLE001 -- recorded, re-raised
        (d / "error.txt").write_text(f"{type(e).__name__}: {e}")
        if verbose:
            print(f"FAILED: {type(e).__name__}: {e}")
        raise
    if verbose:
        cov = float(np.mean([r[1].mean() for r in rendered]))
        print(f"wrote {out}  mean coverage {cov * 100:.1f}%")
    return out


# ---------------------------------------------------------------- transport

class RemoteGPU:
    """
    Push a job over SSH, run it, pull the result.

    Assumes key-based auth and that the repo already exists at `remote_root` on
    the host. It deliberately does NOT install anything or sync code: silently
    running a different version of the toolkit than the one you are measuring is
    a good way to spend a day. `bootstrap()` prints what to run once, by hand.
    """

    def __init__(self, host, remote_root="~/frameflow", python="python3",
                 ssh_opts=(), scp="scp", ssh="ssh"):
        self.host = host
        self.remote_root = remote_root
        self.python = python
        self.ssh_opts = list(ssh_opts)
        self.scp_bin = scp
        self.ssh_bin = ssh

    def _run(self, argv):
        p = subprocess.run(argv, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                f"{' '.join(shlex.quote(a) for a in argv)}\n"
                f"exit {p.returncode}\n{p.stderr[-2000:]}")
        return p.stdout

    def bootstrap(self) -> str:
        """One-time host setup, as a string to run yourself rather than for me
        to run for you -- it installs packages and clones code."""
        return (
            f"ssh {self.host}\n"
            f"  git clone <your-remote> {self.remote_root}\n"
            f"  pip install torch gsplat opencv-python-headless numpy\n"
            f"  # for real poses, which the backend requires by default:\n"
            f"  sudo apt-get install -y colmap\n")

    def submit(self, job_dir, remote_name=None, timeout=None):
        """Copy the job up, run the worker, copy the result back."""
        local = Path(job_dir)
        name = remote_name or local.name
        rdir = f"{self.remote_root}/jobs/{name}"

        self._run([self.ssh_bin, *self.ssh_opts, self.host, f"mkdir -p {rdir}"])
        for f in (JOB, META):
            self._run([self.scp_bin, *self.ssh_opts, str(local / f),
                       f"{self.host}:{rdir}/{f}"])

        cmd = f"cd {self.remote_root} && {self.python} remote.py worker {rdir}"
        out = self._run([self.ssh_bin, *self.ssh_opts, self.host, cmd])

        self._run([self.scp_bin, *self.ssh_opts,
                   f"{self.host}:{rdir}/{RESULT}", str(local / RESULT)])
        return out


# ---------------------------------------------------------------- cli

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage:\n"
              "  python remote.py worker <job_dir>        run on the GPU host\n"
              "  python remote.py submit <job_dir> <host> push, run, pull\n"
              "  python remote.py bootstrap <host>        print host setup steps\n")
        return 0
    cmd = argv[0]
    if cmd == "worker":
        worker(argv[1])
        return 0
    if cmd == "submit":
        print(RemoteGPU(argv[2]).submit(argv[1]))
        return 0
    if cmd == "bootstrap":
        print(RemoteGPU(argv[1]).bootstrap())
        return 0
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
