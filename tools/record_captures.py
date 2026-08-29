"""
record_captures -- screen recordings of the deployed service, for the demo video.

The contest asks for a video "showing your project functioning as built -- not a
cinematic trailer". The demo video used to typeset its agent transcripts onto
slides: the words were real, but a script drew them, and a judge never saw the
software run. These are the real thing -- Chromium driving the live Cloud Run
service, recorded.

    python tools/record_captures.py                    # all three
    python tools/record_captures.py --only agent
    python tools/record_captures.py --base http://127.0.0.1:8080

Writes media/captures/{agent,studio,ledger}.mp4, which make_demo_video.py plays.
That directory is gitignored: it is a recording of a running service, not source.

WHY A SMALL VIEWPORT AND A 2x SCALE FACTOR
------------------------------------------
Recorded at 1920 directly, the ADK dev UI's body text lands around nine pixels
tall in the finished 1080p video -- legible on a monitor, mush after YouTube's
encoder. So the page is laid out at 1280 wide, which makes every element ~1.5x
larger relative to the frame, and rendered at device_scale_factor=2 so the
downscale to 1920 is a genuine supersample rather than an upscale of soft
pixels. Bigger AND sharper, from the same page.

WHAT IS EDITED
--------------
Trimming, and speeding up the waits. Nothing is redrawn or re-typed, and every
scene's caption in the video says when a wait was sped. The questions are typed
a character at a time because a form that fills instantly reads as a fake.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "captures"
BASE = "https://frameflow-460687416455.us-central1.run.app"

REFUSAL = "Is media/locked_off.mp4 worth converting?"
CATALOGUE = "Across everything analysed so far, what converts without inventing anything?"


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    guess = Path("C:/Users/talig/AppData/Local/Microsoft/WinGet/Packages/"
                 "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
                 "ffmpeg-8.1.1-full_build/bin/ffmpeg.exe")
    if guess.is_file():
        return str(guess)
    raise SystemExit("ffmpeg not found on PATH")


def _dismiss_telemetry(pg):
    """ADK asks about usage data on first load; it would sit over the demo."""
    try:
        pg.get_by_role("button", name="No Thanks").click(timeout=8000)
    except Exception:
        pass


def _type(pg, q, delay=0.033):
    ta = pg.locator("textarea").first
    ta.click()
    for ch in q:
        ta.type(ch, delay=0)
        time.sleep(delay)
    pg.wait_for_timeout(650)
    ta.press("Enter")


def _wait_idle(pg, limit=300):
    """Generating shows a stop icon; idle shows send. Wait for send to return."""
    t0 = time.time()
    pg.wait_for_timeout(2500)
    while time.time() - t0 < limit:
        if "stop" not in [i.strip() for i in pg.locator("mat-icon").all_inner_texts()]:
            return time.time() - t0
        pg.wait_for_timeout(1200)
    return -1.0


def _context(br, tmp):
    return br.new_context(viewport={"width": 1280, "height": 720},
                          device_scale_factor=2,
                          record_video_dir=str(tmp),
                          record_video_size={"width": 1920, "height": 1080})


def record_agent(br, tmp, base):
    ctx = _context(br, tmp)
    pg = ctx.new_page()
    pg.goto(f"{base}/dev-ui/", wait_until="networkidle", timeout=90000)
    pg.wait_for_timeout(3500)
    _dismiss_telemetry(pg)
    pg.wait_for_timeout(1200)
    mark = time.time()
    _type(pg, REFUSAL)
    took = _wait_idle(pg)
    pg.wait_for_timeout(3000)
    ctx.close()
    print(f"    answered in {took:.0f}s")
    return time.time() - mark


def record_ledger(br, tmp, base):
    # Warm the ledger first: ClickHouse scales to zero, and a cold start is
    # 30+ seconds of spinner that says nothing about the product.
    warm = br.new_context(viewport={"width": 1024, "height": 700}).new_page()
    warm.goto(f"{base}/dev-ui/", wait_until="networkidle", timeout=90000)
    warm.wait_for_timeout(3000)
    _dismiss_telemetry(warm)
    warm.wait_for_timeout(1000)
    _type(warm, "How many shots are in the ledger?", delay=0.0)
    print(f"    warm-up {_wait_idle(warm):.0f}s")
    warm.context.close()

    ctx = _context(br, tmp)
    pg = ctx.new_page()
    pg.goto(f"{base}/dev-ui/", wait_until="networkidle", timeout=90000)
    pg.wait_for_timeout(3500)
    _dismiss_telemetry(pg)
    pg.wait_for_timeout(1200)
    _type(pg, CATALOGUE)
    took = _wait_idle(pg)
    pg.wait_for_timeout(3500)
    ctx.close()
    print(f"    answered in {took:.0f}s")


def record_studio(br, tmp, base):
    ctx = _context(br, tmp)
    pg = ctx.new_page()
    pg.goto(f"{base}/studio/", wait_until="networkidle", timeout=90000)
    pg.wait_for_timeout(6000)                       # the capability panel filling
    pg.click("text=pan_flat", timeout=25000)
    pg.wait_for_timeout(5000)
    for stage in ("REVIEW", "REPORT"):
        try:
            pg.click(f"button.stage:has-text('{stage}')", timeout=12000)
            pg.wait_for_timeout(6000)
        except Exception as e:
            print(f"    {stage}: {type(e).__name__}")
    pg.wait_for_timeout(2500)
    ctx.close()


JOBS = {"agent": record_agent, "ledger": record_ledger, "studio": record_studio}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--only", choices=sorted(JOBS))
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("needs playwright:  pip install playwright"
                         "  &&  python -m playwright install chromium")

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "_raw"
    raw.mkdir(exist_ok=True)
    names = [args.only] if args.only else list(JOBS)

    with sync_playwright() as p:
        br = p.chromium.launch()
        for name in names:
            print(f"  {name}")
            tmp = raw / name
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            JOBS[name](br, tmp, args.base.rstrip("/"))
            webm = sorted(tmp.glob("*.webm"))
            if not webm:
                print(f"    ! nothing recorded")
                continue
            print(f"    raw {webm[0].stat().st_size / 1048576:.1f} MB -> {tmp}")
        br.close()

    print(f"\n  raw takes are in {raw}")
    print("  trim them with tools/cut_captures.py, which writes the mp4s the")
    print("  video builder plays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
