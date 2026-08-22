# Quickstart

    pip install -r requirements.txt
    python app.py

ScreenX Studio opens in your browser. Drop a clip, watch each shot's verdict land as it is
decided, review what the planner actually managed on each one, and read the conversion
report. Everything the pipeline can do is reachable from it -- the 3D path, a second cut, a
second setup, the context layer, the reasoning step -- which was not true of the three
partial front ends this replaced.

It does not import the pipeline. Every job shells out to `screenx_render.py`, so the browser
and the command line cannot drift apart, and a render that dies takes a subprocess with it
rather than the server. Jobs land in `jobs/<id>/`.

    python app.py --local            this machine only
    python app.py --token wings26    require a passcode; the printed link carries it
    python app.py -p 9000            different port

The same run from the command line, which is what the browser builds underneath:

    python screenx_render.py clip.mp4 -o jobs/mine --maxw 640 --frames-per-shot 200

On Windows the firewall blocks inbound until you allow the port once, in an **admin**
PowerShell:

    New-NetFirewallRule -DisplayName "ScreenX Studio 8420" -Direction Inbound `
      -Action Allow -Protocol TCP -LocalPort 8420 -Profile Private

Know what that opens: no accounts, every upload listed to everyone, and any visitor can
spend this machine's CPU and disk. Fine for a team on a private network, wrong for a café.
Uploads stop at 4 GB each and 24 GB total.

Budget the wait: measured here, a frame costs roughly **0.2–0.9 s** at 480 px depending on
what else the CPU is doing, and `--frames-per-shot` caps the work on any one take. A clip
with many cuts pays that cost per cut.

The pipeline on its own:

    python3 screenx_render.py yourclip.mp4 --maxw 640 --frames-per-shot 200

Writes `screenx_demo.mp4` — three panels that play as the experience, then reveal what was
invented — plus `screenx_summary.json`.

**Use `--frames-per-shot 200` or higher on long takes.** The default truncates, discarding
the length that makes a long shot valuable. Measured cost on a 24s clip: 10 points of
effective coverage, 17 points of real-pixel fraction.

Other flags: `--rotate 0|90|180|270` for clips carrying no rotation metadata,
`--max-shots N`.

## Layout

    *.py          the toolkit (also duplicated in code/)
    README.md     every measurement, and the two calls I got wrong
    ROADMAP.md    what is built, what is stubbed, what each next step costs
    results/      contact sheets, reports, per-shot measurements
    demos/        rendered three-panel videos

## Modules

| file | does |
|---|---|
| `shotdetect.py` | cuts + letterbox; survives modern colour grading |
| `wingcoverage.py` | motion class, propagation, the four metrics |
| `backends.py` | mosaic / layered / gaussian behind one interface |
| `gating.py` | leave-one-out self-check, FULL/NARROW/OFF verdict |
| `fill.py` | the fence: generated pixels cannot overwrite recovered ones |
| `agent.py` | six-rung provenance ladder, tools, licence gate |
| `splat.py` | poses, seeding, the widened-frustum render (GPU for the fit only) |
| `test_splat.py` | 52 assertions over everything in `splat.py` that runs on CPU |
| `remote.py` | job files, so a shot can be fitted on a rented GPU |
| `sfm.py` | COLMAP reconstructions, one per location, with a (shot, frame) manifest |
| `verify_gpu.py` | the one command that proves the CUDA path; run it on the GPU host |
| `colab_verify.ipynb` | that command as a free-Colab notebook — T4, upload, run |
| `test_tier2.py` | 46 assertions over the scene layer, the hold-out dispatch, the refusals |
| `director.py` | goal-driven planner over those tools |
| `walls.py` | theatre geometry, three projector panels |
| `crosscut.py` / `crossres.py` | same-take matching across cuts of one film |
| `screenx_render.py` | end to end, video out |
| `demo_ui.py` | the QC report |
| `app.py` | the interface: load, render, review, report |
| `screenx_render.py` | the pipeline itself, and the single implementation of a run |

Ground-truth check: `python3 make_test_clip.py && python3 validate.py`

## Headline measurements

| footage | effective | real wing |
|---|---|---|
| cafe, 6s landscape | 63.7% | 91.5% |
| apartment, 24s landscape | 48.7% | **94.4%** |
| room, 21s portrait | 32.3% | 30.6% |
| Thunderbolts* trailer | **6.15%** | — |

GPU is optional — it unlocks `backends.GaussianBackend` (parallax shots) and
`fill.DiffusionGenerator`. Everything else runs on CPU.

The demo notebook `ScreenX_Demo.ipynb` is standalone, ships separately, and does not
need this archive.
