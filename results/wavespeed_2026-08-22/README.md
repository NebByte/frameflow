# WaveSpeed outpainter: live wire verified -- 2026-08-22

First real call to a hosted generation model in this project. 16 frames of the
café clip at 480x270, wing 105px, one job, 79 seconds, $0.20.

```
aspect asked : 21:9
returned     : 16 frames at 690x270   (= 480 + 2*105, exact)
centre vs source frame: 4.1 mean abs diff  (recompression only)
```

`ws_compare.png` is the source frame on an empty canvas above the model's
output. It continued the counter and wall to the left and added shelving and a
figure to the right -- scene extension, not a smear.

## Two defects the first call would have hit, both found for $0

The adapter had never run. Probing the catalogue and the schema before spending
anything turned up both:

1. **Wrong model id.** `wavespeed-ai/video-outpainting` does not exist. The real
   one is `wavespeed-ai/video-outpainter` ($0.20, video-to-video).
2. **Six rejected fields.** The adapter sent `expand_left`, `expand_right`,
   `expand_top`, `expand_bottom`, `target_width`, `target_height`. The live
   schema is `additionalProperties: false` and accepts only
   `video`, `prompt`, `aspect_ratio`, `seed` -- so those are a rejected request,
   not ignored fields. The API takes no per-side expansion at all.

Auth was confirmed separately at zero cost: a GET on a nonexistent prediction
returns 404 with the key and 401 without it.

## What changed

`aspect_for()` picks the enum ratio nearest the wing canvas and never narrower
than the source -- a narrower ratio would crop the picture instead of extending
it. `fit_to_canvas()` height-matches the return and trims or pads symmetrically,
because the model expands to a ratio rather than to our wing width: rescaling to
fit would shift the centre out of alignment with the real frame composited over
it, and the wings would stop continuing the shot.

Pinned by `test_canvas_fit` and a request-shape test that asserts the body is
exactly the four declared fields. 32 assertions in `test_wavespeed.py` pass.

## Cost note

$0.20 per shot, per job. A 6-shot clip is $1.20. Ask before a full run.
