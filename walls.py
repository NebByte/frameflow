"""
walls — project the recovered canvas onto real theatre geometry.

THE PROBLEM WITH A WIDE FLAT IMAGE
----------------------------------
The canvas produced by propagation is a single pinhole image. Widening it does
not give you ScreenX: a pinhole cannot span 270 degrees, because horizontal
extent goes as tan(theta) and blows up at 90. Displaying a wide flat strip on
side walls looks smeared, and that smear is why naive outpainting demos read as
"stretched panorama" rather than "theatre".

The side walls sit at ~90 degrees to the main screen. Each needs its OWN
projection, computed for a viewer sitting in the auditorium.

WHY A HOMOGRAPHY IS EXACT HERE
------------------------------
The source canvas is a plane (the pinhole image plane). Each wall is a plane.
The viewer is a single centre of projection. A plane-to-plane map through a
common centre of projection IS a homography -- so each wall is one exact 3x3,
no resampling approximation, no spherical intermediate needed.

HOW FAR BACK THE WINGS CAN REACH
--------------------------------
Recovered periphery is finite, so the wings cannot run the full length of the
auditorium. With screen at distance D, main image width w and wing width
wing_w, the walls can be covered from the screen back to

    L = D * wing_w / (w/2 + wing_w)          (derived in wall_extent)

so a wing ratio r = wing_w/w gives L = D * r / (0.5 + r):

    r = 0.15  ->  L = 0.23 * D
    r = 0.25  ->  L = 0.33 * D
    r = 0.75  ->  L = 0.60 * D

That formula ties your coverage metric directly to a physical distance in a
room, which is the number a theatre operator actually cares about.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from provenance import NOT_THIS_PLACE


@dataclass
class Theatre:
    """Auditorium geometry, in arbitrary consistent units (metres by default)."""
    screen_width: float = 14.0
    screen_height: float = 6.0
    viewer_distance: float = 12.0     # viewer to screen plane
    panel_width: int = 640            # output px per side panel
    panel_height: int = 360
    wing_dim: float = 0.82            # side walls run slightly darker, as ScreenX does
    feather_px: int = 24              # seam blend width


def wall_extent(w, wing_w, theatre: Theatre, h=None):
    """
    How far back from the screen the side walls can be covered.

    TWO constraints, and the second is the one that actually binds.

    HORIZONTAL: a wall point at (+/-Wm/2, y, z) projects to canvas offset
    f*(Wm/2)/(-z). The canvas holds w/2 + wing_w of horizontal offset, giving
        z_h = D * (w/2) / (w/2 + wing_w)

    VERTICAL: the same wall point at height +/-Hm/2 projects to f*(Hm/2)/(-z),
    and the canvas holds only h/2 -- the ORIGINAL frame height, because
    propagation widens the canvas but never makes it taller. That gives
        z_v = D * (w/h) * (Hm/Wm)
    which is independent of wing_w entirely.

    The wall can only be covered back to max(z_h, z_v). Past a wing ratio of
    roughly 0.25 the vertical term dominates and widening the wings stops
    buying any additional reach -- visible as black wedges above and below the
    wall image. To go deeper you need vertical periphery, not horizontal.

    Which means format variants (open-matte, IMAX 1.90) are useful after all:
    they differ from scope in HEIGHT, exactly the axis that binds here.
    """
    D = theatre.viewer_distance
    h = h or int(w * 9 / 16)
    half = w / 2.0 + wing_w
    z_h = D * (w / 2.0) / half
    z_v = D * (w / float(h)) * (theatre.screen_height / theatre.screen_width)
    z_min = max(z_h, z_v)
    binding = "vertical" if z_v >= z_h else "horizontal"
    return dict(
        depth=max(0.0, D - z_min),
        z_near=z_min,
        fraction_of_room=max(0.0, D - z_min) / D,
        focal=w * D / theatre.screen_width,
        z_horizontal=z_h,
        z_vertical=z_v,
        binding=binding,
        depth_if_horizontal_only=D - z_h,
    )


def _canvas_of(P, focal, cx, cy):
    """Project a 3D point (viewer at origin, screen toward -z) into canvas px."""
    X, Y, Z = P
    u = cx + focal * (X / -Z)
    v = cy + focal * (Y / -Z)
    return [u, v]


def wall_homographies(canvas_shape, wing_w, theatre: Theatre):
    """
    One homography per panel, mapping panel pixel -> canvas pixel.
    Returns dict with 'left', 'centre', 'right' and the extent info.
    """
    h, cw = canvas_shape[:2]
    w = cw - 2 * wing_w
    ext = wall_extent(w, wing_w, theatre, h)
    f = ext["focal"]
    cx, cy = cw / 2.0, h / 2.0

    D = theatre.viewer_distance
    Wm, Hm = theatre.screen_width, theatre.screen_height
    z_far, z_near = -D, -ext["z_near"]
    y_top, y_bot = Hm / 2.0, -Hm / 2.0

    pw, ph = theatre.panel_width, theatre.panel_height
    dst = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])

    out = {}

    # centre: the main screen itself, a straight crop of the canvas
    centre_src = np.float32([
        _canvas_of((-Wm / 2, y_top, z_far), f, cx, cy),
        _canvas_of((Wm / 2, y_top, z_far), f, cx, cy),
        _canvas_of((Wm / 2, y_bot, z_far), f, cx, cy),
        _canvas_of((-Wm / 2, y_bot, z_far), f, cx, cy),
    ])
    out["centre"] = cv2.getPerspectiveTransform(dst, centre_src)

    # left wall: x = -Wm/2, running from the screen (z_far) toward viewer (z_near).
    # Panel x=0 is at the screen edge so the seam lines up with the main image.
    left_src = np.float32([
        _canvas_of((-Wm / 2, y_top, z_far), f, cx, cy),
        _canvas_of((-Wm / 2, y_top, z_near), f, cx, cy),
        _canvas_of((-Wm / 2, y_bot, z_near), f, cx, cy),
        _canvas_of((-Wm / 2, y_bot, z_far), f, cx, cy),
    ])
    # panel is mirrored: its right edge meets the screen
    left_dst = np.float32([[pw, 0], [0, 0], [0, ph], [pw, ph]])
    out["left"] = cv2.getPerspectiveTransform(left_dst, left_src)

    right_src = np.float32([
        _canvas_of((Wm / 2, y_top, z_far), f, cx, cy),
        _canvas_of((Wm / 2, y_top, z_near), f, cx, cy),
        _canvas_of((Wm / 2, y_bot, z_near), f, cx, cy),
        _canvas_of((Wm / 2, y_bot, z_far), f, cx, cy),
    ])
    right_dst = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    out["right"] = cv2.getPerspectiveTransform(right_dst, right_src)

    out["extent"] = ext
    return out


def _feather(panel_w, panel_h, side, px):
    """Alpha ramp so the wall image fades in away from the screen seam."""
    a = np.ones((panel_h, panel_w), np.float32)
    if px <= 0:
        return a
    ramp = np.linspace(0.0, 1.0, px, dtype=np.float32)
    if side == "left":          # seam on the RIGHT edge of the left panel
        a[:, :px] *= ramp[None, :]
    elif side == "right":       # seam on the LEFT edge
        a[:, -px:] *= ramp[::-1][None, :]
    return a


def auto_panels(canvas_shape, wing_w, theatre: Theatre, height_px=300):
    """
    Panel pixel dimensions from PHYSICAL wall shape, not a hardcoded landscape.

    A side wall is (wall depth) wide by (screen height) tall. With depth 3.7m and
    a 6m screen that is TALLER than it is wide -- rendering it into a 420x236
    landscape panel squashes the image so hard it reads as rotated. That was the
    bug in the first room render, not the source orientation.
    """
    h, cw = canvas_shape[:2]
    w = cw - 2 * wing_w
    e = wall_extent(w, wing_w, theatre, h)
    centre_w = int(height_px * theatre.screen_width / theatre.screen_height)
    wall_w = max(24, int(height_px * max(e["depth"], 0.01) / theatre.screen_height))
    return centre_w, wall_w, height_px, e


def render(canvas, wing_w, theatre: Theatre = None, filled=None,
           provenance=None, mark_generated=False, auto_aspect=True):
    """
    Render the three projector feeds.

    filled / provenance are optional. With mark_generated=True, pixels that were
    not photographed at this location are tinted so a reviewer can see at a
    glance what the audience is being shown -- useful for the worklist view, off
    for a real screening.

    That covers GENERATED and REFERENCED both. Licensed external material is a
    real photograph, but not of this place, so a reviewer checking "what am I
    inventing" needs to see it lit. The tint follows NOT_THIS_PLACE rather than
    a literal, which is what this used to be: `provenance >= 4` meant GENERATED
    under the old numbering and silently changed meaning when the REFERENCED
    rung was inserted.
    """
    theatre = theatre or Theatre()
    if auto_aspect:
        cwid, wwid, hpx, _ = auto_panels(canvas.shape, wing_w, theatre)
        theatre = Theatre(**{**theatre.__dict__,
                             "panel_width": wwid, "panel_height": hpx})
    Hs = wall_homographies(canvas.shape, wing_w, theatre)
    pw, ph = theatre.panel_width, theatre.panel_height

    src = canvas.copy()
    if mark_generated and provenance is not None:
        gen = np.isin(provenance, NOT_THIS_PLACE)
        tint = src.astype(np.float32)
        tint[gen] = tint[gen] * 0.55 + np.array([120, 0, 120], np.float32) * 0.45
        src = np.clip(tint, 0, 255).astype(np.uint8)

    centre_w = int(ph * theatre.screen_width / theatre.screen_height)
    panels = {}
    for side in ("left", "centre", "right"):
        M = Hs[side]
        tw = centre_w if side == "centre" else pw
        if side == "centre":
            M = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [tw, 0], [tw, ph], [0, ph]]),
                cv2.perspectiveTransform(
                    np.float32([[[0, 0], [pw, 0], [pw, ph], [0, ph]]]), M)[0])
        p = cv2.warpPerspective(src, np.linalg.inv(M), (tw, ph),
                                flags=cv2.INTER_CUBIC)
        if side != "centre":
            a = _feather(tw, ph, side, theatre.feather_px)[..., None]
            p = (p.astype(np.float32) * a * theatre.wing_dim).astype(np.uint8)
        panels[side] = p

    panels["extent"] = Hs["extent"]
    return panels


def contact_sheet(panels, gap=6, label=True):
    """Single wide image: left | centre | right, as a ScreenX master is laid out."""
    l, c, r = panels["left"], panels["centre"], panels["right"]
    h = max(l.shape[0], c.shape[0], r.shape[0])
    sep = np.zeros((h, gap, 3), np.uint8)
    sheet = np.hstack([l, sep, c, sep, r])
    if label:
        bar = np.zeros((26, sheet.shape[1], 3), np.uint8)
        e = panels["extent"]
        txt = (f"LEFT WALL  |  MAIN SCREEN  |  RIGHT WALL     "
               f"walls cover {e['depth']:.1f}m from screen "
               f"({e['fraction_of_room']*100:.0f}% of room; {e['binding']}-limited)")
        cv2.putText(bar, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        sheet = np.vstack([bar, sheet])
    return sheet
