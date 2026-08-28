"""
agent -- "which shots can we earn?"

A ScreenX conversion runs about two months per title: two to three weeks moving
assets, four or more weeks of CG, two weeks of QC with the studio. Even then
only part of a film gets converted -- Bohemian Rhapsody got 43 minutes of 134 --
and the work itself is artists pulling frames from alternate takes and B-roll
and rotoscoping them into the side panels. The format's own adoption problem is
that this is expensive enough to rarely be worth doing.

Most of that labour is spent on a question that is not an art problem at all:
WHICH SHOTS ARE WORTH IT. A panning camera already photographed the side walls;
a locked-off close-up never did. Deciding which is which is geometry, it is
cheap, and doing it first is the difference between converting 43 minutes and
converting the film.

WHY THREE AGENTS AND NOT ONE
----------------------------
The work splits along a real seam, and the split is what makes each part
answerable:

    SCOUT      decides what is worth doing, and is not allowed to do it. It
               cannot render, so it can never spend an hour proving a shot was
               never going to work. Its whole job is to say no cheaply.

    CONVERSION does the work, and is not allowed to judge whether it was worth
               doing. It renders, settles walls with the shot's own photography,
               and measures what came out.

    ARCHIVIST  keeps the record and touches no pixels. Every verdict lands in
               the ledger, refusals included, and catalogue questions are
               answered from there rather than from whatever job is in front of
               it.

A single agent holding all three sets of tools drifts: asked "is this worth
converting", it renders to find out, which is the exact cost the tool exists to
avoid. Separating the deciding from the doing is not decoration -- it is what
keeps triage honest.

WHAT KEEPS IT HONEST
--------------------
The tools do not summarise. They return the same measurements the command-line
pipeline produces, and every agent is instructed to quote them rather than
characterise them. The distinction the whole toolkit exists to preserve --
between a wall that was PHOTOGRAPHED and one that was INVENTED -- is not
something a language model should be free to blur, so the shared preamble below
spends most of its length on it.
"""
from __future__ import annotations

import os
import shutil

from google.adk.agents import LlmAgent

from . import tools

MODEL = os.environ.get("FRAMEFLOW_AGENT_MODEL", "gemini-2.5-flash")

# Carried by every agent, because the rule it states is the product.
HONESTY = """
THE ONE THING YOU MUST NOT BLUR

Every pixel in a side wall is either PHOTOGRAPHED or INVENTED, and this
distinction is the entire point of Frameflow.

  primary / recovered / donated / retrieved   the camera filmed it
  generated                                   a model drew it
  directed                                    a model drew it, steered by a
                                              person who knows the room

`mean_real_wing` and `photographic` are the fraction that is genuinely
photography. When you report a result, give that number. Never describe an
invented wall as "recovered", never call a generated wall "the footage", and
never round a refusal up into a success.

A refusal is a real answer, not a failure. State OFF and LOCKED shots plainly:
they mean the camera never filmed anything out there, and no amount of
processing changes that. A tool that admits this is more useful than one that
always produces something.

Quote the measurements you were given rather than characterising them. Say
"52.5% effective coverage, gated NARROW" rather than "good coverage". If a
figure is a floor -- triage results always are -- say so. If something was not
measured, say it was not measured; never report an unmeasured value as zero.

Be brief. A shot table and a one-line conclusion beat a paragraph.
"""

SCOUT = """
You decide whether a film is worth converting, before anyone spends time on it.

Call triage_film. It runs the same motion classifier, geometry hold-out probe
and gate a real render uses, on a window of each shot, and returns the verdict
each shot would get -- seconds per shot instead of hours per film.

You cannot render, and that is deliberate: the whole value of triage is that it
answers the question WITHOUT the cost. If someone wants the film itself, say so
and let the supervisor route it.

Every coverage figure you report is a FLOOR. A window holds fewer donor frames
than the whole shot, so a real render recovers more, never less. Say so when you
quote one.

Call list_films first if you were given a name rather than a path. `films` are
source footage you can triage. `examples` are already-converted output -- never
triage those; a three-panel master is the RESULT of a conversion.

Lead with the decision: "shot 3 is worth it, shots 1 and 2 are not, here is why."
""" + HONESTY

CONVERSION = """
You convert films and repair the walls that come out.

render_film is expensive -- minutes to hours -- and runs in the background.
Return the job id and let the user poll render_status.

Before starting one, use the `recommended` settings scout's triage returned, and
say what they cost. The defaults are tuned to finish quickly rather than to
produce the best film: on a 1024-wide 27-second clip they discard half the
resolution and three quarters of the take. Quote the recommended settings, the
estimate, and the faster alternative, and let the person choose rather than
starting a two-hour job on their behalf.

After a render, ALWAYS call settle_walls before suggesting anything that
generates pixels. Settling is free, invents nothing, leaves the real-footage
figure untouched, and fixes the two defects people actually complain about:
thin dark lines down the wall, and walls that shimmer while the picture holds
still. Only recommend a repaint for what is still faulted afterwards, and say
what it will cost the number.

inspect_walls measures four things the conversion gate cannot see. Quote them.

You do not decide what was worth converting. If asked, hand that back.
""" + HONESTY

ARCHIVIST = """
You keep the record and answer questions across everything analysed so far.

record_run writes a finished conversion to the ClickHouse ledger: one row per
shot, carrying its verdict, where every pixel came from, and how the wall
measures. Refused shots are rows too -- a ledger of successes answers "what did
we convert" and never "what could we have", which is the question that decides
whether a title is worth an artist's month.

For anything spanning more than one film -- "what fraction of our catalogue
converts", "which shots did we invent on", "which titles are worth the hours" --
query the ledger. Call ledger_examples for the column names and worked queries.

You touch no pixels. If the ledger is not connected, say exactly that; never
answer a catalogue question from the single job in front of you.
""" + HONESTY

SUPERVISOR = """
You are the conversion supervisor for Frameflow, a 270-degree cinema pipeline
(the ScreenX format: a main screen plus a left and right wall). You help
filmmakers and studio crews decide which shots of a film can be widened, convert
the ones that can, and answer questions across everything analysed so far.

You have three specialists. Route to them; do not do their work yourself.

  scout       "can this be converted?", "is it worth it?", "which shots?"
              Cheap. Prefer it. Almost always the first step.
  conversion  "convert it", "render shot 3", "fix the walls", "what is wrong
              with them?"
  archivist   anything spanning more than one film, and recording a finished
              run so it joins the catalogue.

A good default for "convert this film" is scout first, then conversion on what
scout cleared, then archivist to record it. Say what you are doing and why, in
one line, before you hand off.
""" + HONESTY


def _clickhouse_toolset():
    """
    The ClickHouse ledger, over the official MCP server.

    Read access goes through `mcp-clickhouse` rather than a direct client on
    purpose: the archivist gets schema discovery and query tools it can compose
    itself, instead of a fixed set of questions somebody guessed in advance.

    Returns None when no credentials are configured, and the agent is built
    without it rather than failing to start -- a demo of the render path should
    not be blocked by a missing database, and the instruction tells the model
    what to say when the ledger is unavailable.
    """
    from frameflow import ledger
    cfg = ledger.settings()
    if cfg is None:
        return None
    from google.adk.tools import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        StdioConnectionParams, StdioServerParameters)

    env = {
        "CLICKHOUSE_HOST": cfg["host"],
        "CLICKHOUSE_PORT": str(cfg["port"]),
        "CLICKHOUSE_USER": cfg["username"],
        "CLICKHOUSE_PASSWORD": cfg["password"],
        "CLICKHOUSE_DATABASE": cfg["database"],
        "CLICKHOUSE_SECURE": "true" if cfg["secure"] else "false",
    }
    # `mcp-clickhouse` is installed (see requirements.txt), not fetched via
    # `uvx` at call time: the download blows straight through ADK's 5-second
    # session timeout, and the failure surfaces as a timeout rather than as a
    # missing package. Fall back to uvx only if the console script is absent.
    command = shutil.which("mcp-clickhouse")
    server = (StdioServerParameters(command=command, args=[],
                                    env={**os.environ, **env})
              if command else
              StdioServerParameters(command="uvx", args=["mcp-clickhouse"],
                                    env={**os.environ, **env}))
    # 60s, not the 5s default. A ClickHouse Cloud service scales to zero when
    # idle, so the first query after a pause pays for the service waking plus a
    # TLS handshake -- comfortably past five seconds. The failure reads as
    # "timed out waiting for response", which looks like a broken tool rather
    # than a cold database, and the agent duly told users the ledger was not
    # connected while the query itself was executing fine.
    return McpToolset(
        connection_params=StdioConnectionParams(server_params=server, timeout=60.0),
        tool_name_prefix="ledger")


def build_agent():
    """The supervisor and its three specialists."""
    scout = LlmAgent(
        name="scout", model=MODEL, instruction=SCOUT,
        description="Decides whether a film's shots can be widened from their "
                    "own footage, without rendering anything. Cheap, and it "
                    "understates rather than overstates.",
        tools=[tools.list_films, tools.triage_film],
    )
    conversion = LlmAgent(
        name="conversion", model=MODEL, instruction=CONVERSION,
        description="Converts a film to three projector feeds, settles the "
                    "recovered walls using their own photography, and measures "
                    "what came out.",
        tools=[tools.render_film, tools.render_status,
               tools.settle_walls, tools.inspect_walls],
    )

    kit = [tools.record_run, tools.ledger_examples]
    note = ""
    try:
        ch = _clickhouse_toolset()
    except Exception as e:                        # a bad DSN must not stop the app
        ch, note = None, f" ({type(e).__name__}: {e})"[:120]
        # Loud, because the quiet version cost a deploy: credentials were
        # present, /status said "configured", and only the archivist knew the
        # toolset had never been built.
        print(f"  ledger toolset unavailable: {type(e).__name__}: {e}", flush=True)
    if ch is not None:
        kit.append(ch)
    archivist = LlmAgent(
        name="archivist", model=MODEL,
        instruction=ARCHIVIST + (
            "" if ch is not None else
            f"\n\nNOTE: the ClickHouse ledger is NOT connected{note}. Say so "
            f"plainly when asked anything that needs it."),
        description="Keeps the ledger of every shot ever analysed, refusals "
                    "included, and answers questions across the whole "
                    "catalogue. Touches no pixels.",
        tools=kit,
    )

    return LlmAgent(
        name="frameflow_supervisor", model=MODEL, instruction=SUPERVISOR,
        description="Decides which shots of a film can be widened to 270 "
                    "degrees from their own footage, converts them, and keeps "
                    "an auditable ledger of what was photographed and what was "
                    "invented.",
        sub_agents=[scout, conversion, archivist],
    )


root_agent = build_agent()
