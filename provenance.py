"""
provenance — the labels, defined once.

These lived in two places. `fill.py` had a three-level set for the fence and
`agent.py` a five-level one for the ladder, and they disagreed on the value 2:
GENERATED in one file, DONATED in the other. DONATED counts toward `real`, so
routing generated pixels through the fence's labels would have reported
invented pixels as real photography -- silently, and in the one number the
whole project rests on.

Nothing imported the fence's constants yet, so no measurement published in the
README was ever affected. Both modules now import from here.

THE LADDER, ordered by how much you can trust a pixel:

    PRIMARY    the frame itself
    RECOVERED  propagated from elsewhere in THIS shot         (same camera)
    DONATED    same take, another cut, geometrically verified (same camera)
    RETRIEVED  another setup of the same location, 3D-verified(same set)
    REFERENCED licensed external material, UNVERIFIED         (somewhere else)
    DIRECTED   invented to satisfy an assertion about this place
    GENERATED  invented, unconstrained

WHY REFERENCED EXISTS
---------------------
`ExternalReferenceTool` composites licensed stills and plates. Those are real
photons, so calling them GENERATED is a lie in one direction -- but the tool
performs no check that the asset depicts THIS location, so calling them
RETRIEVED is a lie in the other, and a worse one, because RETRIEVED is inside
PHOTOGRAPHIC and PHOTOGRAPHIC is the headline number.

Measured: with a dummy library wired in, a flat colour plate moved
`mean_real_wing` from 3.36% to 5.34%. Nothing about the wing got more true.

So it gets its own rung, below RETRIEVED and outside PHOTOGRAPHIC. This is not
a rejection of verification -- it is the label that is honest WHILE
verification does not exist. When the 3D backend lands (ROADMAP Tier 2.1), an
asset that registers to the shot is promoted REFERENCED -> RETRIEVED and earns
its way into the number. Until then it does not.

WHY DIRECTED EXISTS
-------------------
A script page reading "fire escape, camera left", and a person pausing the shot
to type "there is a fire escape there", are the same kind of thing: an assertion
about what was in the room, from someone who may well know, with no photons
behind it. Generation driven by either is better informed than free invention,
and is still invention.

So it sits above GENERATED -- something constrained the pixels -- and below
REFERENCED, which at least involved a camera being pointed at something. It
stays outside PHOTOGRAPHIC, because a claim is not evidence. The director being
right does not make the wall filmed.
"""
from __future__ import annotations

PRIMARY, RECOVERED, DONATED, RETRIEVED, REFERENCED, DIRECTED, GENERATED = (
    0, 1, 2, 3, 4, 5, 6)

PROV_NAMES = {PRIMARY: "primary", RECOVERED: "recovered", DONATED: "donated",
              RETRIEVED: "retrieved", REFERENCED: "referenced",
              DIRECTED: "directed", GENERATED: "generated"}

# what each rung counts as when reporting "real"
REAL_LEVELS = (PRIMARY, RECOVERED, DONATED)      # same camera, same moment
PHOTOGRAPHIC = REAL_LEVELS + (RETRIEVED,)        # real photons, THIS place

# Everything PHOTOGRAPHIC excludes -- derived, not listed, because the last two
# times a rung was inserted the hand-written membership was the thing that got
# missed. A budget on invention must cover all of these: otherwise "at most 15%
# generated" is satisfiable with a wing that is 85% unverified stock, or 85%
# whatever somebody typed into a note.
NOT_THIS_PLACE = tuple(k for k in PROV_NAMES if k not in PHOTOGRAPHIC)

