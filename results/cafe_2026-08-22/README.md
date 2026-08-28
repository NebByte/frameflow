# The café clip, re-run on the current build -- 2026-08-22

Source: `IMG_0683.mov`, 1920x1080 landscape, 178 frames, 5.9s, one continuous
handheld pan across a food court. The same footage behind `frameflow_cafe_demo.mp4`
from the earlier build.

    python -m frameflow.render IMG_0683.mov -o out --maxw 640 --frames-per-shot 200

No GPU, no COLMAP, no --prefer-3d, no API keys. CPU propagation path.

```
shot 0   ROTATION   NARROW   geom 23.5 dB   effective 54.3%

wings_on:        1        mean_real_wing:  0.9869
wings_generated: 0        mean_effective:  0.5434
walls 2.86 m (23.8% of room, vertical-limited)
```

## Why this run matters

**98.69% of the wing pixels are genuinely filmed.** Not mirrored, not generated:
`wings_generated` is 0. This is the highest real-wing fraction measured in the
project, above the old build's 91.5% on this same clip and above the apartment
clip's 94.4%, which had been the record.

It is also the first time the CURRENT build -- with the provenance ladder, the
fence, Tier 2 and the context layer on top -- has been shown to still do the
thing the original one did. That was untested until now.

## Against the old build, same footage

| | old build | this build |
|---|---|---|
| effective coverage | 63.7% | 54.3% |
| real wing | 91.5% | **98.7%** |
| verdict | FULL | NARROW |
| geometry | 26.9 dB | 23.5 dB |
| wall depth | 2.86 m (24%) | 2.86 m (23.8%) |

Wall geometry reproduces exactly. Coverage is ~9 points lower and the verdict
drops FULL -> NARROW because 54.3% sits just under the 55% eff_full threshold --
worth chasing, since the old run cleared it on identical footage. The likely
difference is the render width (--maxw 640 here) or shot-boundary detection, not
the gate: the thresholds are byte-identical between builds.

The real-wing fraction going UP while coverage went DOWN is consistent: this
build fills less wall, and what it does fill is almost entirely photographed.

## What it settles

The 3D/COLMAP path verified on GPU the same night (see gpu_2026-08-22) scored
23.6 and 21.7 dB of trustworthy geometry but filled no wings on drone footage.
This run fills them on CPU alone. Lateral camera motion through a textured space
at conversational distance is what produces real wing pixels -- not
reconstruction quality.


## Correction, same day

`mean_real_wing` was measured over the **left wing only** (`[:, :ww]`) while
`agent.WingAgent.report` had always used both. On a pan the two wings are not
interchangeable -- the trailing one recovers and the leading one cannot -- so the
headline number depended on which side happened to be left. Now both:
**0.9869**, from 0.984. The verdict, geometry and effective coverage are
unchanged, so nothing about the run itself moved.
