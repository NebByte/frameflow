# Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Frameflow Studio opens in your browser. Drop a clip, watch each shot's verdict land as it
is decided, review what the planner managed on each one, and read the conversion report.
Everything the pipeline can do is reachable from it — the 3D path, a second cut, a second
setup, the context layer, the reasoning step.

It does not import the pipeline. Every job shells out to `python -m frameflow.render`, so
the browser and the command line cannot drift apart, and a render that dies takes a
subprocess with it rather than the server. Jobs land in `jobs/<id>/`.

```bash
python app.py --local            # this machine only
python app.py --token wings26    # require a passcode; the printed link carries it
python app.py -p 9000            # different port
```

## The same run from the command line

Which is what the browser builds underneath:

```bash
python -m frameflow.render clip.mp4 -o jobs/mine --maxw 640 --frames-per-shot 200
```

Writes `frameflow_demo.mp4` — three panels that play as the experience, then reveal what
was invented — plus `frameflow_summary.json`.

**Ask before you render.** Triage answers "is this worth converting at all" in seconds per
shot instead of hours per film, and it understates, so a shot it clears is one a render
will clear:

```bash
python -m frameflow.triage clip.mp4
```

**Use `--frames-per-shot 200` or higher on long takes.** It is a cap, not a target: a shot
longer than it is cut off at it. Measured cost of truncating a 24s clip: 10 points of
effective coverage, 17 points of real-pixel fraction.

Other flags: `--rotate 0|90|180|270` for clips carrying no rotation metadata, `--max-shots N`.

Budget the wait: a frame costs roughly **0.2–0.9 s** at 480 px depending on what else the
CPU is doing. A clip with many cuts pays that per cut.

## Serving it to other people

On Windows the firewall blocks inbound until you allow the port once, in an **admin**
PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Frameflow Studio 8420" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8420 -Profile Private
```

Know what that opens: no accounts, every upload listed to everyone, and any visitor can
spend this machine's CPU and disk. Fine for a team on a private network, wrong for a café.
Uploads stop at 4 GB each and 24 GB total.

## The agent

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=us-central1
python server.py     # agent on :8080/dev-ui/ , studio on :8080/studio/
```

## Layout

```
frameflow/     the pipeline
agent_service/ the ADK agent and its tools
tests/         106 checks
tools/         synthetic footage, ground-truth validation, GPU checks
results/       contact sheets, reports, per-shot measurements
demos/         rendered three-panel videos
app.py         Frameflow Studio
server.py      agent + studio on one port
```

| file | does |
|---|---|
| `frameflow/shotdetect.py` | cuts + letterbox; survives modern colour grading |
| `frameflow/wingcoverage.py` | motion class, propagation, settling, the four metrics |
| `frameflow/backends.py` | mosaic / layered / gaussian behind one interface |
| `frameflow/gating.py` | leave-one-out self-check, FULL/NARROW/OFF verdict |
| `frameflow/triage.py` | every shot's verdict without rendering it |
| `frameflow/fill.py` | the fence: generated pixels cannot overwrite recovered ones |
| `frameflow/agent.py` | six-rung provenance ladder, tools, licence gate |
| `frameflow/director.py` | goal-driven planner over those tools |
| `frameflow/polish.py` | inspect, settle, and aimed repaint |
| `frameflow/splat.py` | poses, seeding, the widened-frustum render (GPU for the fit only) |
| `frameflow/sfm.py` | COLMAP reconstructions, one per location, with a manifest |
| `frameflow/walls.py` | theatre geometry, three projector panels |
| `frameflow/crosscut.py` · `crossres.py` | same-take matching across cuts of one film |
| `frameflow/ledger.py` | the ClickHouse schema and writer |
| `frameflow/render.py` | end to end, video out |
| `frameflow/remote.py` · `colabrun.py` | fitting a shot on a rented or free GPU |
| `tools/verify_gpu.py` | the one command that proves the CUDA path |
| `colab_verify.ipynb` | that command as a free-Colab notebook — T4, upload, run |

Ground-truth check:

```bash
python tools/make_test_clip.py && python tools/validate.py
```

## Headline measurements

| footage | effective | real wing |
|---|---|---|
| café, 6s landscape | 63.7% | 91.5% |
| apartment, 24s landscape | 48.7% | **94.4%** |
| gym pan, 27s at 1024px | 52.5% | **90.4%** |
| room, 21s portrait | 32.3% | 30.6% |
| Thunderbolts\* trailer | **6.15%** | — |

GPU is optional — it unlocks `backends.GaussianBackend` (parallax shots) and
`fill.DiffusionGenerator`. Everything else runs on CPU.

Every measurement above, and the two calls that turned out wrong, are in
[`docs/RESEARCH-LOG.md`](docs/RESEARCH-LOG.md).
