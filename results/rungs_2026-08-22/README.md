# Six of seven rungs fired on real footage -- 2026-08-22

One run of a 53.8 s handheld apartment walk, everything switched on:

    python screenx_render.py IMG_0803.MOV -o out --maxw 480 \
        --frames-per-shot 120 --context apartment.srt \
        --wings-on-dark mirror --sources --online

```
rungs_fired: ['directed', 'donated', 'generated', 'recovered', 'referenced']

recovered   65.36%     this shot's own periphery
directed    23.12%     a real .srt, bound to shots by timecode
donated      8.44%     first time in the project
referenced   2.34%     a CC0 plate, licence-checked
generated    0.73%
```

Per shot, `same_take[real]` ran on all five and `external_reference` on shot 0.

## Two bugs this run exposed

**1. `ExternalReferenceTool` did not exist.** `director.py:160` has always
constructed `ag.ExternalReferenceTool(fetcher, policy)` and the class was absent
from `agent.py`, so `--online` and `--library` did not fail quietly -- they
raised `AttributeError` on contact. Every rung is reached through the same
scout, which is why nothing noticed until the flag was actually used.

Written now, on rung 4, deliberately outside `PHOTOGRAPHIC`: a licensed
photograph of *a* food court is a camera pointed at *a* place, not at *this*
one. Unlicensed material is refused rather than used and flagged.

**2. The external query was narrower than the corpus.** `describe` builds
"dim warm exterior background plate". Measured against the live API:

```
dim warm exterior background plate   0 results
warm exterior background plate       0
exterior background plate           73
background plate                   240
```

`OpenverseFetcher.widen` now drops adjectives first and ends on the head noun,
and the asset records which phrasing found it, so the report can say how loosely
the material was matched.

## Also added: the summary says which rungs fired

A run reported `mean_real_wing` and nothing else about provenance, so
"DONATED fired" was not a checkable claim -- the label lived in the pixels and
never in the summary. `screenx_summary.json` now carries `provenance` and
`rungs_fired`, per shot and overall, computed from `agent.WingAgent.report`
rather than recounted by hand. The interface shows both.

## The seventh rung, and what it actually needs

`RETRIEVED` needs **two** things, and only one of them is COLMAP:

| stage | needs | status here |
|-------|-------|-------------|
| registration | COLMAP, CPU | not installed |
| rendering | gsplat, CUDA | no GPU on this machine |

`SameLocationTool.run` returns immediately when `self.backend is None`
(agent.py:120) and renders with `device="cuda"` (agent.py:145), so COLMAP alone
would leave the rung unable to draw anything.

Attempted tonight and blocked:

- **Colab T4**: `Service Unavailable` on three consecutive allocation attempts,
  after several T4 sessions the same day -- most likely the free tier's daily
  GPU quota.
- **COLMAP 4.1.1 windows-nocuda** (114.6 MB, official repo): download failed,
  then all DNS resolution on the machine stopped -- github, pypi, openverse and
  wavespeed alike, including hosts that had answered minutes earlier.

## To finish it

When the network and a GPU are back, in this order:

1. `winget install colmap` (winget 1.29 is present) or the official
   windows-nocuda zip. This alone answers the open question: **does cross-setup
   registration now clear 80%?** It failed at 48% and 50% before the
   spread-sampling fix, and registration is CPU work.
2. If it clears, one Colab session renders it: `--sfm --prefer-3d --also <wide>`.
3. If it still fails, the answer is footage, not tooling: one continuous move
   from a wide into a close-up gives COLMAP a connected chain instead of two
   islands.
