"""
sfm -- build the COLMAP reconstructions Tier 2 requires.

`GaussianBackend` refuses to render from anything but COLMAP poses, and
`SameLocationTool` needs a reconstruction spanning EVERY setup of a location at
once. Nothing built those. This does.

    scene_dir/
      images/s0007_f0000.png ...   every frame of every setup, one flat folder
      manifest.json                (shot, frame) for each image, in name order
      database.db                  COLMAP's
      sparse/0/{cameras,images,points3D}.txt

ONE RECONSTRUCTION PER SCENE, NOT PER SHOT
------------------------------------------
The obvious build is "for each shot, reconstruct it plus its donors", which
re-solves the same location once per shot and produces reconstructions in
different arbitrary coordinate frames that cannot be compared. Instead every
setup of a location goes into one solve, and `manifest.json` records which
(shot, frame) each view is. A shot then finds its own poses by lookup.

WHY THE FILENAMES MATTER
------------------------
`splat.poses_from_colmap` sorts images by filename, because COLMAP's images.txt
is in registration order, not capture order. `s{shot:04d}_f{frame:04d}.png`
makes filename order identical to (shot, frame) order, so the sorted poses line
up with the manifest by construction rather than by hope. Zero-padding to four
digits is what keeps s10 after s9.

WHAT COLMAP WILL REFUSE, AND WHY THAT IS INFORMATION
----------------------------------------------------
A scene whose setups share no visual overlap will not register, and the mapper
will return fewer images than it was given or several disconnected models. That
is not a failure to work around -- it is the measurement that those two setups
cannot be bridged, which is the same thing the 1-in-400 homography result said.
`build_scene` reports the registration fraction and `SceneModel.usable` refuses
partial reconstructions rather than silently rendering from whatever registered.
"""
from __future__ import annotations

import json
import re
import shutil
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COLMAP = "colmap"
MIN_REGISTERED = 0.80          # fraction of submitted images that must register


class ColmapMissing(RuntimeError):
    pass


def colmap_available(binary=COLMAP) -> bool:
    return shutil.which(binary) is not None


def require_colmap(binary=COLMAP):
    if not colmap_available(binary):
        raise ColmapMissing(
            f"{binary!r} is not on PATH. Tier 2 needs real poses: an "
            f"essential-matrix chain has no recovered scale and the backend "
            f"refuses to render from it.\n"
            f"  linux : sudo apt-get install -y colmap\n"
            f"  macos : brew install colmap\n"
            f"  win   : https://github.com/colmap/colmap/releases\n"
            f"On a rented GPU host this is part of the one-time bootstrap; see "
            f"remote.RemoteGPU.bootstrap().")


# ---------------------------------------------------------------- manifest

@dataclass
class SceneModel:
    """A built reconstruction plus the map from (shot, frame) back to view."""
    scene_id: str
    sparse_dir: Path
    manifest: list                 # [[shot, frame], ...] SUBMITTED, in write order
    n_submitted: int
    n_registered: int
    registered: list | None = None  # [[shot, frame], ...] REGISTERED, in POSE order

    @property
    def views(self) -> list:
        """
        The (shot, frame) behind each pose, in the order `poses()` returns them.

        This is NOT `manifest`. `manifest` is everything handed to COLMAP;
        `poses_from_colmap` returns only the images COLMAP actually registered,
        sorted by filename. The two coincide only at 100% registration -- and
        the gate accepts anything at or above MIN_REGISTERED, so at 80-99% the
        manifest index of a view is not its pose index, and using one for the
        other pairs each frame with some other frame's camera. Falls back to
        the manifest for models built before this field existed, which is
        exactly the fully-registered case where they are the same list.
        """
        return self.manifest if self.registered is None else self.registered

    @property
    def registered_fraction(self) -> float:
        return self.n_registered / max(self.n_submitted, 1)

    @property
    def usable(self) -> bool:
        """
        Partial reconstructions are refused.

        If COLMAP registered 60% of the views, the poses that exist may be
        perfect -- but the frames that failed are exactly the ones whose
        geometry was ambiguous, and rendering the rest anyway means the wing
        recovery silently covers only the easy shots while the report reads as
        though it covered the scene.
        """
        return self.registered_fraction >= MIN_REGISTERED

    def index_of(self, shot, frame):
        """View index for one (shot, frame), or None if it did not register."""
        try:
            return self.views.index([int(shot), int(frame)])
        except ValueError:
            return None

    def poses(self):
        import splat as sp
        return sp.poses_from_colmap(self.sparse_dir)

    def to_json(self):
        return dict(scene_id=self.scene_id, sparse_dir=str(self.sparse_dir),
                    manifest=self.manifest, n_submitted=self.n_submitted,
                    n_registered=self.n_registered, registered=self.registered,
                    registered_fraction=round(self.registered_fraction, 4),
                    usable=self.usable)


def load_scene(scene_dir) -> SceneModel:
    d = Path(scene_dir)
    meta = json.loads((d / "scene.json").read_text())
    reg = meta.get("registered")
    return SceneModel(meta["scene_id"], Path(meta["sparse_dir"]),
                      [list(x) for x in meta["manifest"]],
                      meta["n_submitted"], meta["n_registered"],
                      None if reg is None else [list(x) for x in reg])


# ---------------------------------------------------------------- build

def write_images(scene_dir, setups, max_frames_per_setup=None):
    """
    Lay every setup's frames into one flat folder with sortable names.

    `setups` is [{"shot": int, "frames": [...]}, ...] -- the shape filmindex
    already serves. Returns the manifest.
    """
    d = Path(scene_dir)
    img = d / "images"
    img.mkdir(parents=True, exist_ok=True)
    manifest = []
    for st in sorted(setups, key=lambda s: int(s["shot"])):
        shot = int(st["shot"])
        frames = list(st["frames"])
        if max_frames_per_setup:
            # even sampling, not the first N: the first N of a moving shot is a
            # fraction of its baseline, and baseline is what SfM needs
            k = min(max_frames_per_setup, len(frames))
            pick = np.linspace(0, len(frames) - 1, k).astype(int)
        else:
            pick = range(len(frames))
        for f_i in pick:
            name = f"s{shot:04d}_f{int(f_i):04d}.png"
            cv2.imwrite(str(img / name), frames[int(f_i)])
            manifest.append([shot, int(f_i)])
    return manifest


_GPU_OPTS = {}


def gpu_option_names(binary=None):
    """
    What this COLMAP calls the CPU/GPU switch.

    4.x renamed SiftExtraction/SiftMatching to FeatureExtraction/
    FeatureMatching, and an unknown option is fatal rather than ignored:
    "Failed to parse options - unrecognised option". That surfaced as
    `colmap features failed`, which is also what a missing display produces, so
    a version mismatch and the headless problem were indistinguishable from the
    log. Asked once per binary and cached.
    """
    binary = binary or COLMAP
    if binary in _GPU_OPTS:
        return _GPU_OPTS[binary]
    pair = ("--SiftExtraction.use_gpu", "--SiftMatching.use_gpu")
    try:
        out = subprocess.run([binary, "feature_extractor", "-h"],
                             capture_output=True, text=True, timeout=60)
        text = (out.stdout or "") + (out.stderr or "")
        if "--FeatureExtraction.use_gpu" in text:
            pair = ("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu")
    except (OSError, subprocess.SubprocessError):
        pass                                   # the run itself will report it
    _GPU_OPTS[binary] = pair
    return pair


def build_scene(scene_dir, setups, scene_id="scene", max_frames_per_setup=24,
                binary=COLMAP, single_camera_per_shot=True, verbose=True,
                timeout=None, gpu_sift=None):
    """
    Run COLMAP over every setup of one location. -> SceneModel.

    Cached: if `sparse/0` already holds a reconstruction and the manifest
    matches, the solve is skipped. A scene solve is minutes to hours.
    """
    require_colmap(binary)
    d = Path(scene_dir)
    d.mkdir(parents=True, exist_ok=True)
    sparse = d / "sparse" / "0"

    manifest = write_images(d, setups, max_frames_per_setup)
    cached = d / "scene.json"
    if sparse.exists() and (sparse / "images.txt").exists() and cached.exists():
        prev = load_scene(d)
        if prev.manifest == manifest:
            if verbose:
                print(f"  {scene_id}: cached ({prev.n_registered}/"
                      f"{prev.n_submitted} registered)")
            return prev

    db = d / "database.db"
    if db.exists():
        db.unlink()
    (d / "sparse").mkdir(exist_ok=True)

    def run(args, what):
        if verbose:
            print(f"  {scene_id}: {what}")
        p = subprocess.run([binary, *args], capture_output=True, text=True,
                           timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"colmap {what} failed:\n{p.stdout[-1500:]}\n"
                               f"{p.stderr[-1500:]}")
        return p.stdout

    # single_camera_per_shot: one intrinsic per setup, which is what a setup IS
    # -- one camera, one lens, locked for the duration. Solving a separate
    # camera per FRAME would be badly conditioned; solving ONE camera for the
    # whole scene would force two different lenses into one focal length.
    # COLMAP's SIFT runs on the GPU by default, and that path wants a GL
    # context. A headless box -- Colab, any rented GPU host, the exact machines
    # this tier is built for -- has none, so feature extraction dies with the
    # CUDA device sitting idle beside it. Measured: `colmap features failed` on
    # a T4 with 48 perfectly good images. CPU SIFT is slower and works
    # everywhere, so it is the default unless a display exists or the caller
    # insists otherwise.
    use_gpu = (bool(os.environ.get("DISPLAY")) if gpu_sift is None else gpu_sift)
    gpu = "1" if use_gpu else "0"

    extract_opt, match_opt = gpu_option_names(binary)

    run(["feature_extractor", "--database_path", str(db),
         "--image_path", str(d / "images"),
         "--ImageReader.single_camera", "0" if single_camera_per_shot else "1",
         "--ImageReader.camera_model", "SIMPLE_RADIAL",
         extract_opt, gpu], "features")
    run(["exhaustive_matcher", "--database_path", str(db),
         match_opt, gpu], "matching")
    run(["mapper", "--database_path", str(db), "--image_path", str(d / "images"),
         "--output_path", str(d / "sparse")], "mapping")

    if not (d / "sparse" / "0").exists():
        raise RuntimeError(
            f"{scene_id}: COLMAP produced no model. The setups in this scene "
            f"share too little visual overlap to register -- which is the same "
            f"finding as the 1-in-400 cross-setup homography result, not a bug.")

    run(["model_converter", "--input_path", str(sparse),
         "--output_path", str(sparse), "--output_type", "TXT"], "converting")

    names = _registered_names(sparse / "images.txt")
    n_reg = len(names)
    model = SceneModel(scene_id, sparse, manifest, len(manifest), n_reg,
                       views_from_names(names))
    (d / "scene.json").write_text(json.dumps(model.to_json(), indent=2))
    if verbose:
        print(f"  {scene_id}: {n_reg}/{len(manifest)} registered "
              f"({model.registered_fraction * 100:.0f}%), "
              f"{'usable' if model.usable else 'REFUSED, partial'}")
    return model


VIEW_NAME = re.compile(r"^s(\d+)_f(\d+)\.")


def _registered_names(images_txt) -> list:
    """
    The filenames COLMAP registered, in the order `poses_from_colmap` yields.

    Same structural line test used there -- an image line is exactly 10 tokens
    and a POINTS2D line never is -- and the same filename sort, because that
    sort IS the view order the poses come back in.
    """
    names = []
    for line in Path(images_txt).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) == 10 and p[0].isdigit() and p[8].isdigit():
            names.append(p[-1])
    names.sort()
    return names


def _count_registered(images_txt) -> int:
    return len(_registered_names(images_txt))


def views_from_names(names) -> list | None:
    """
    [[shot, frame], ...] parsed from `write_images` filenames, or None.

    None when any name is not ours -- a reconstruction someone else built, or a
    test fixture. The caller then falls back to the manifest, which is the old
    behaviour, rather than crashing or inventing a mapping.
    """
    out = []
    for n in names:
        m = VIEW_NAME.match(n)
        if not m:
            return None
        out.append([int(m.group(1)), int(m.group(2))])
    return out


def build_film(index, out_dir, min_setups=2, verbose=True, **kw):
    """
    Build every multi-setup scene in a FilmIndex. -> {scene_id: SceneModel}.

    Single-setup scenes are skipped: there is no other setup to bridge to, so a
    reconstruction of one shot adds nothing `GaussianBackend` cannot get itself.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models, skipped = {}, 0
    scenes = index.scenes()
    if verbose:
        print(f"building {len(scenes)} scene(s) into {out}")
    for scene_id, members in scenes.items():
        if len(members) < min_setups:
            skipped += 1
            continue
        # frames_for_sfm, not shots[i]["frames"] -- the latter is three 320px
        # thumbnails kept for appearance matching, and reconstructing from those
        # is what made every scene come back 0 usable
        setups = [dict(shot=index.shots[i]["shot"], frames=index.frames_for_sfm(i))
                  for i in members]
        try:
            models[scene_id] = build_scene(out / scene_id, setups, scene_id,
                                           verbose=verbose, **kw)
        except (RuntimeError, ColmapMissing) as e:
            if verbose:
                print(f"  {scene_id}: {str(e).splitlines()[0]}")
    if verbose:
        usable = sum(1 for m in models.values() if m.usable)
        print(f"{usable} usable, {len(models) - usable} partial, "
              f"{skipped} single-setup scenes skipped")
    return models


def colmap_dirs(models):
    """{scene_id: sparse_dir} for the usable models, for SameLocationTool."""
    return {sid: m.sparse_dir for sid, m in models.items() if m.usable}
