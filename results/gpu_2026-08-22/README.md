# First verified GPU run on real footage -- 2026-08-22

Colab T4, torch 2.11.0+cu128, gsplat 1.5.3, COLMAP from apt.
Footage: "Drone flight over park" (Wikimedia Commons, CC BY-SA 4.0),
1280x720, 664 frames, 33.2s, transcoded to h264. Run as:

    python screenx_render.py drone.mp4 -o out --maxw 480 \
        --sfm sfm --prefer-3d --sources --max-shots 3

## What is new here

`build_film` produced **usable reconstructions for the first time**: 24/24
registered (100%) on all three scenes, versus 48% and two disconnected models
on the two-setup phone shoot. Single-setup scenes reach COLMAP at all only
because `--prefer-3d` passes `min_setups=1`.

Two of three shots then **cleared the 20 dB geometry gate** -- every earlier
run on real footage scored 14.5 dB.

| shot | geometry | verdict | reason | effective | coverage | displacement |
|------|----------|---------|--------|-----------|----------|--------------|
| 0 | 17.3 dB | OFF | geometry unverified (< 20) | 0.96% | 1.00 | 5.07 |
| 1 | **23.6 dB** | OFF | wing carries no detail (3%) | 2.92% | 1.00 | 8.54 |
| 2 | **21.7 dB** | OFF | effective 16% below 25% | 15.5% | 0.87 | 17.6 |

`wings_on: 0`, `wings_generated: 0`, `mean_real_wing: 0.0`, mean_effective
0.0647. All three still gate OFF -- but only shot 0 fails on geometry now. The
other two fail on *content*: the rasteriser filled the wing and there was
little out there worth keeping. Effective coverage tracks camera displacement
(5.07 -> 8.54 -> 17.6 gives 0.96% -> 2.9% -> 15.5%), which says the next lever
is lateral camera motion, not a better backend.

RETRIEVED cannot fire on this footage and that is structural: it means
borrowing periphery from ANOTHER SETUP of a location via SameLocationTool, and
these are single-setup shots. The reachable prize here was wings ON with
recovered pixels.

## verify_gpu.py, same session

| check | result |
|-------|--------|
| left / right wing coverage | 100.0% |
| primary region byte-for-byte identical | pass |
| primary marked filled, zero staleness | pass |
| **placement vs known truth** | **22.7 dB** |
| staleness bounded by shot length | pass, max 1 of 16 |
| wings staler than centre | pass, mean 0.3 frames |
| leave_one_out_3d above gate | FAIL -- 0.0 dB from 0 probes |
| a good 3D shot no longer gated OFF | FAIL -- OFF |

Both failures are harness gaps, not geometry: `verify_gpu.py`'s last stage
never passes a `colmap_dir`, so the backend meets scale-free essential poses
(inlier_ratio 0.86) and refuses, and the leave-one-out then has no
reconstruction to hold a frame out of -- 0 probes, so 0.0 dB means "not
measured", not "bad".

## Bug found and fixed by this run

`poses_from_colmap` returns one pose per REGISTERED view of the whole scene,
which is not the shot's frame list: `build_scene` submits at most
max_frames_per_setup (24) while the render loads SFM_FRAMES (40), and a
multi-setup scene carries the other setup's views too. `seed_points` pairs
frames[i] with pose i. Fixed in `screenx_render.align_to_poses` using
`model.views` as the authority. Pinned by `test_render_frames_match_pose_order`.

The IndexError was the lucky failure: it fires only when the lists differ in
length. Differing in *content* at equal length fits and renders a scene turned
inside out, silently.
