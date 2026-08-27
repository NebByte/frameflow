# Which shots can we earn?

**An agent that triages a film for 270° immersive conversion, recovers the side
walls from the footage itself, and never lies about what it invented.**

Built for [Agentic Cinema: The Blockbuster Hackathon](https://agentic-cinema.devpost.com/)
— ClickHouse track.

| | |
|---|---|
| **Try it** | https://screenx-agent-460687416455.us-central1.run.app |
| **The agent** | `/dev-ui/` — ask it about a film |
| **The converter** | `/studio/` — upload, convert, watch the three projector feeds |
| **Stack** | Gemini · Google Cloud Agent Builder (ADK) · Cloud Run · ClickHouse MCP |

---

## The problem

ScreenX puts a film on three walls instead of one — a main screen plus the left
and right of the auditorium. Converting a film to it takes **about two months
per title**: two to three weeks moving assets, four or more weeks of CG, two
weeks of QC. Even then only part of a film gets converted — *Bohemian Rhapsody*
got 43 minutes of 134.

The labour is artists pulling frames from alternate takes and B-roll and
rotoscoping them into the side panels. Wikipedia is blunt about the
consequence: the format "has been rarely produced for Hollywood studio films
due to the complexity of the additional CGI work."

But a large share of that work is spent answering a question that is not an art
problem at all: **which shots are even possible.** A panning camera already
photographed the side walls — the crop threw them away. A locked-off close-up
never filmed them, and no amount of artist time recovers what was never
photographed.

Deciding which is which is geometry. It is cheap. And doing it first is the
difference between converting 43 minutes and converting the film.

## What this does

```
triage    every shot's verdict, without rendering    ~7s per shot
render    recover the walls that can be recovered    minutes to hours
settle    fix the walls using their own photography  free, invents nothing
ledger    every verdict, queryable across a catalogue
```

Ask the agent *"is this worth converting?"* and it runs the same motion
classifier, geometry hold-out probe and gate a real render uses — on a window
of each shot — and tells you. On a 27-second handheld pan, triage took **91
seconds** and predicted 50.7% effective coverage; the full render took **two
hours** and delivered 52.54%.

Triage understates by construction. It judges each shot on a consecutive
window, and a window holds fewer donor frames than the whole shot, so a render
recovers more — never less. Understating costs you a shot you could have had.
Overstating costs you the artist-month you committed on the strength of it.

## The line this refuses to blur

Every pixel in a side wall is either **photographed** or **invented**, and the
whole toolkit is built to keep those apart:

| rung | what it means |
|---|---|
| `primary` | the frame itself |
| `recovered` | this camera filmed it, earlier in the same shot |
| `donated` / `retrieved` | filmed at this location, in another take or cut |
| `directed` | a model drew it, steered by a person who knows the room |
| `generated` | a model drew it |

`mean_real_wing` is the fraction that is genuinely photography. When a model
repaints something, that number falls by exactly the repainted share, and the
provenance map on disk is rewritten per pixel to match. **A refusal is a real
answer** — `OFF` and `LOCKED` shots mean the camera never filmed anything out
there, and the tool says so rather than inventing something and calling it a
conversion.

This is also why ClickHouse is load-bearing rather than decorative. The studio
question is not "how did this shot do" — it is *"across everything we own, what
converts without inventing anything?"* That is a query.

```sql
SELECT source, count() AS shots,
       countIf(state IN ('FULL','NARROW','BORROWED')) AS earned,
       round(avg(photographic), 4) AS mean_real_wing
FROM screenx_shots GROUP BY source ORDER BY earned DESC
```

Refused shots are rows too. A ledger of successes answers "what did we convert"
and never "what could we have".

## Measured

Three real rooms, three different days, CPU only, no model involved:

| clip | real wall |
|---|---|
| café | 98.7% |
| apartment walk | up to 99.3% |
| gym pan (1024px, whole take) | **90.4%** |

And the artefacts that made earlier cuts unwatchable, on the same footage:

| | before | after |
|---|---|---|
| thin dark lines | 2.67% of wall columns | **0.00%** |
| wall shimmer vs the picture | 1.22× | **1.00×** |
| delivered frame rate | 24 (source was 30) | **30** |

The dark lines were a bug, not a limitation: the donor frame was sampled with
bilinear interpolation against a black border while its validity mask was
sampled nearest-neighbour, so every donor's edge column composited at half
brightness and was labelled as real photography.

## Run it

```bash
pip install -r requirements.txt
python make_test_clip.py            # synthetic clips with known ground truth

python triage.py media/pan_flat.mp4       # earned
python triage.py media/locked_off.mp4     # refused, and cheap to learn

python screenx_render.py media/pan_flat.mp4 -o jobs/demo --deliver
python polish.py jobs/demo                # settle the walls; free
python app.py                             # the studio, on :8420
```

The agent needs Google Cloud:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=us-central1
python server.py                          # agent on :8080/dev-ui/
```

The ledger needs a (free) ClickHouse Cloud service. Without it everything else
still runs, and the agent says the ledger is not connected rather than
answering catalogue questions from one job:

```bash
export CLICKHOUSE_HOST=... CLICKHOUSE_USER=... CLICKHOUSE_PASSWORD=...
python ledger.py --write jobs/demo
python ledger.py --examples
```

## How it fits together

```
      you ─────► ADK agent (Gemini, Cloud Run)
                     │
                     ├── triage_film ──► shot detect, motion class,
                     │                    hold-out geometry, gate
                     ├── render_film ──► propagate walls, settle, deliver
                     ├── settle_walls ─► median each frame against its
                     │                    own aligned neighbours
                     ├── record_run ───► ClickHouse (writes)
                     └── ledger.* ─────► ClickHouse MCP server (reads)
```

Every tool is a thin wrapper over code that is already tested. An agent that
reimplements the pipeline inside its tool layer is an agent whose numbers
nobody has checked.

## What is still broken

**Depth.** A single homography cannot place off-plane content, so a recovered
wall can repeat signage that is still on screen in the centre. Measured: 3.0%
of wall patches echo the centre on a recovered wall. `GaussianBackend`
(`backends.py`) reconstructs the scene properly and is written — it has never
run, because it needs CUDA. It refuses to fall back rather than putting
homography pixels behind a 3D label.

**Nobody has seen any of this in a theatre.** Every number here was measured at
1:1 on a monitor, which is the harshest possible viewing condition for content
that is watched at 40° off-axis in a dark room. That test has not been done and
is not something a measurement can substitute for.

## Tests

```bash
python test_agent_service.py     # 44 — agent, ledger, triage
python test_polish.py            # 62 — the finishing pass
python -m pytest test_tier2.py test_app.py test_splat.py -q   # 45
python test_e2e.py               # 52 — the joins, which is where bugs live
```

## Layout

| | |
|---|---|
| `triage.py` | verdicts without rendering |
| `ledger.py` | the ClickHouse schema and writer |
| `agent_service/` | the ADK agent and its tools |
| `server.py` | agent + studio on one port |
| `screenx_render.py` | the conversion pipeline |
| `wingcoverage.py` | propagation, settling, coverage metrics |
| `gating.py` | the hold-out geometry probe and the gate |
| `polish.py` | inspect, settle, and aimed repaint |
| `walls.py` | auditorium geometry |
| [`docs/RESEARCH-LOG.md`](docs/RESEARCH-LOG.md) | how every one of these was arrived at, including the negative results |

## Licence

MIT. See [LICENSE](LICENSE).
