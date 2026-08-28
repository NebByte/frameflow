"""
test_wavespeed -- the outpainting adapter and the shot-level contract.

Run: python test_wavespeed.py

No network, so the HTTP is injected. That still covers what breaks in an async
job adapter: submission shape, polling through pending states, failure and
timeout, the mp4 round trip, and -- the one that matters -- that a model
returning a re-imagined centre cannot corrupt the metric, because compositing
happens on our side of the wire.
"""
from __future__ import annotations

import base64

import numpy as np

import agent as ag
import screenx_render as sx
import wavespeed as ws

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def clip(n=6, w=64, h=36, val=120):
    return [np.full((h, w, 3), val + i, np.uint8) for i in range(n)]


def scripted(steps, sink=None):
    """Return each canned reply in turn; repeat the last forever."""
    calls = {"n": 0}

    def transport(url, body=None, headers=None, timeout=ws.TIMEOUT):
        i = min(calls["n"], len(steps) - 1)
        calls["n"] += 1
        if sink is not None:
            sink.setdefault("urls", []).append(url)
            if body is not None:
                sink["body"] = body
            sink["headers"] = headers
        step = steps[i]
        if isinstance(step, Exception):
            raise step
        return step
    return transport


def done_with(frames, w, h):
    """A completed job whose output is an inline mp4 of `frames`."""
    blob = ws.WaveSpeedOutpainter.encode_clip(frames)
    return {"data": {"status": "completed",
                     "outputs": [f"data:video/mp4;base64,{blob}"]}}


def test_submission():
    sink = {}
    frames = clip()
    wide = [np.full((36, 64 + 2 * 14, 3), 200, np.uint8) for _ in frames]
    g = ws.WaveSpeedOutpainter(token="k", sleep=lambda s: None,
                               transport=scripted([{"data": {"id": "job-1"}},
                                                   done_with(wide, 92, 36)], sink))
    g.generate_shot(frames, 14, "a lit street to the left")

    check("submits to the model path", ws.DEFAULT_MODEL in sink["urls"][0],
          sink["urls"][0].split("/")[-1])
    check("bearer token is sent",
          sink["headers"].get("Authorization", "").startswith("Bearer "))
    b = sink["body"]
    check("the prompt is sent", b["prompt"] == "a lit street to the left")
    check("the clip goes as one video, not N frames",
          b["video"].startswith("data:video/mp4;base64,"))
    # The live schema is additionalProperties:false and takes no per-side
    # expansion: video, prompt, aspect_ratio, seed. Sending expand_left/
    # target_width -- which an earlier version of this adapter did -- is a
    # rejected request, not an ignored field.
    check("only the fields the API declares are sent",
          set(b) == {"video", "prompt", "aspect_ratio", "seed"}, str(sorted(b)))
    check("the aspect asked for is one the API accepts",
          b["aspect_ratio"] in ws.WaveSpeedOutpainter.ASPECTS, b["aspect_ratio"])
    check("the aspect is wider than the source, never narrower",
          ws.WaveSpeedOutpainter.ASPECTS[b["aspect_ratio"]] >= 64 / 36)
    check("a seed is sent so a run can be repeated", b["seed"] == -1)


def test_canvas_fit():
    """
    What comes back is an aspect ratio, not our wing width.

    Squashing it to fit would rescale the picture horizontally and the centre
    would stop lining up with the real frame composited over it -- the wings
    would no longer continue the shot.
    """
    print("fitting the returned clip to the wing canvas")
    fit = ws.WaveSpeedOutpainter.fit_to_canvas
    w, h, ww = 64, 36, 14                       # canvas is 92 wide

    wide = np.full((72, 400, 3), 7, np.uint8)   # double height, far too wide
    out = fit(wide, w, h, ww)
    check("height is matched and the surplus trimmed",
          out.shape == (36, 92, 3), str(out.shape))

    narrow = np.full((36, 50, 3), 9, np.uint8)  # narrower than the canvas
    out2 = fit(narrow, w, h, ww)
    check("a short return is padded, not stretched",
          out2.shape == (36, 92, 3) and int(out2[0, 46, 0]) == 9, str(out2.shape))

    exact = np.full((36, 92, 3), 5, np.uint8)
    check("an exact match is passed through untouched",
          fit(exact, w, h, ww) is exact)


def test_polling():
    frames = clip(4)
    wide = [np.full((36, 92, 3), 200, np.uint8) for _ in frames]
    steps = [{"data": {"id": "j"}},
             {"data": {"status": "processing"}},
             {"data": {"status": "processing"}},
             done_with(wide, 92, 36)]
    slept = []
    g = ws.WaveSpeedOutpainter(token="k", transport=scripted(steps),
                               poll_every=0.5, sleep=slept.append)
    out = g.generate_shot(frames, 14)
    check("polls through pending states until done", len(out) == len(frames),
          f"{len(slept)} waits, {len(out)} frames")
    check("waits between polls", slept and all(x == 0.5 for x in slept))


def test_failures():
    frames = clip(3)
    g = ws.WaveSpeedOutpainter(token="k", sleep=lambda s: None,
                               transport=scripted([{"data": {"id": "j"}},
                                                   {"data": {"status": "failed",
                                                             "error": "bad input"}}]))
    try:
        g.generate_shot(frames, 10)
        check("a failed job raises", False)
    except RuntimeError as e:
        check("a failed job raises", "failed" in str(e), str(e)[:40])

    g2 = ws.WaveSpeedOutpainter(token="k", sleep=lambda s: None, poll_every=1.0,
                                poll_limit=3.0,
                                transport=scripted([{"data": {"id": "j"}},
                                                    {"data": {"status": "processing"}}]))
    try:
        g2.generate_shot(frames, 10)
        check("a stuck job times out", False)
    except TimeoutError:
        check("a stuck job times out", True)

    g3 = ws.WaveSpeedOutpainter(token="k", sleep=lambda s: None,
                                transport=scripted([{"no": "id"}]))
    try:
        g3.generate_shot(frames, 10)
        check("a response with no job id raises", False)
    except RuntimeError as e:
        check("a response with no job id raises", "job id" in str(e))

    check("no credential refuses before spending anything",
          _refuses(ws.WaveSpeedOutpainter(token="")))


def _refuses(g):
    try:
        g.generate_shot(clip(2), 8)
        return False
    except RuntimeError as e:
        return "credential" in str(e).lower()


def test_clip_roundtrip():
    frames = [np.full((36, 92, 3), 40 + i * 20, np.uint8) for i in range(5)]
    raw = base64.b64decode(ws.WaveSpeedOutpainter.encode_clip(frames))
    back = ws.WaveSpeedOutpainter.decode_clip(raw, expect=5)
    check("clip survives the mp4 round trip", len(back) == 5, f"{len(back)} frames")
    check("and keeps its shape", back[0].shape == frames[0].shape, str(back[0].shape))

    short = ws.WaveSpeedOutpainter.decode_clip(raw, expect=8)
    check("a short result is padded, not truncated to nothing", len(short) == 8)


def test_shot_generator_fence():
    """A model that re-imagines the centre must not be able to corrupt it."""
    frames = clip(4, w=64, h=36, val=100)
    ww = 14
    vandal = [np.full((36, 64 + 2 * ww, 3), 250, np.uint8) for _ in frames]

    class Vandal:
        prompt = "x"

        def generate_shot(self, frames, wing_w, prompt=None):
            return vandal

    out = sx._shot_generate(Vandal(), frames, ww, "x")
    check("every frame comes back", len(out) == len(frames))
    canvas, prov = out[0]
    check("the filmed centre is untouched",
          np.array_equal(canvas[:, ww:ww + 64], frames[0]))
    check("the wings took the model's pixels",
          int(canvas[0, 0, 0]) == 250, str(canvas[0, 0, 0]))
    check("centre is PRIMARY", int(prov[0, ww]) == ag.PRIMARY)
    check("wings are not", int(prov[0, 0]) != ag.PRIMARY)


def test_pipeline_prefers_shot_path():
    """With a temporal generator the pipeline must submit once, not per frame."""
    calls = {"shot": 0, "frame": 0}
    ww_seen = {}

    class Counting:
        prompt = "x"

        def generate_shot(self, frames, wing_w, prompt=None):
            calls["shot"] += 1
            ww_seen["w"] = wing_w
            h, w = frames[0].shape[:2]
            return [np.full((h, w + 2 * wing_w, 3), 210, np.uint8) for _ in frames]

        def __call__(self, canvas, hole, conf):
            calls["frame"] += 1
            return canvas.copy()

    import wingcoverage as wc
    frames = [np.random.randint(0, 255, (36, 64, 3), dtype=np.uint8) for _ in range(12)]
    pairs, rec = sx.process_shot(frames, wc.Tracker(), dark_generator=Counting())
    check("one submission for the whole shot", calls["shot"] == 1, str(calls))
    check("and not one per frame", calls["frame"] == 0)
    check("all frames rendered", len(pairs) == len(frames))
    check("wing width passed through", ww_seen.get("w") == int(64 * sx.WING))


def test_registry_agrees():
    cap = next(c for c in ag.REGISTRY if c.name == "wavespeed-outpainter")
    check("the registry entry is anchored", cap.conditions_on_known)
    check("and temporal", cap.temporal)
    check("select_provider picks a hosted anchored one when asked",
          ag.select_provider(690, need_anchor=True, allow_hosted=True) is not None)


if __name__ == "__main__":
    print("submission")
    test_submission()
    print("polling")
    test_canvas_fit()
    test_polling()
    print("failures")
    test_failures()
    print("codec")
    test_clip_roundtrip()
    print("the fence at shot level")
    test_shot_generator_fence()
    print("pipeline")
    test_pipeline_prefers_shot_path()
    print("registry")
    test_registry_agrees()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
