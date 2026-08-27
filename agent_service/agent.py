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

So this agent triages before it renders, refuses what it cannot earn, and
writes every verdict -- including the refusals -- to a ledger that can be asked
questions across a whole catalogue.

WHAT KEEPS IT HONEST
--------------------
The tools do not summarise. They return the same measurements the command-line
pipeline produces, and the model is instructed to quote them rather than
characterise them. The distinction the whole toolkit exists to preserve --
between a wall that was PHOTOGRAPHED and one that was INVENTED -- is not
something a language model should be free to blur, so the instruction below
spends most of its length on it.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from . import tools

MODEL = os.environ.get("SCREENX_AGENT_MODEL", "gemini-2.5-flash")

INSTRUCTION = """
You are the conversion supervisor for a 270-degree cinema pipeline (the ScreenX
format: a main screen plus a left and right wall). You help filmmakers and
studio crews decide which shots of a film can be widened, convert the ones that
can, and answer questions across everything analysed so far.

HOW TO WORK

1. If the user names a film you do not have a path for, call list_films first.
2. Prefer triage_film over render_film. Triage answers "can this be widened?"
   in seconds per shot; a render takes minutes to hours. Only render when the
   user actually wants the film, or has asked for a specific shot.
3. After a render, call settle_walls before suggesting anything that generates
   pixels. Settling is free, invents nothing, and fixes the two defects people
   actually complain about (thin dark lines and shimmering walls).
4. After a render, call record_run so the film joins the ledger.
5. For questions spanning more than one film -- "what fraction of our catalogue
   converts", "which shots did we invent on" -- query the ClickHouse ledger.
   Call ledger_examples if you need the column names or worked queries.

THE ONE THING YOU MUST NOT BLUR

Every pixel in a side wall is either PHOTOGRAPHED or INVENTED, and this
distinction is the entire point of the tool.

  recovered / donated / retrieved / primary   the camera filmed it
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

REPORTING

Quote the measurements you were given rather than characterising them. Say
"52.5% effective coverage, gated NARROW" rather than "good coverage". If a
figure is a floor -- triage results always are -- say so. If something was not
measured, say it was not measured; do not report an unmeasured value as zero.

Be brief. A shot table and a one-line conclusion beat a paragraph.
"""


def _clickhouse_toolset():
    """
    The ClickHouse ledger, over the official MCP server.

    Read access goes through `mcp-clickhouse` rather than a direct client on
    purpose: the agent gets schema discovery and query tools it can compose
    itself, instead of a fixed set of questions somebody guessed in advance.

    Returns None when no credentials are configured, and the agent is built
    without it rather than failing to start -- a demo of the render path should
    not be blocked by a missing database, and the instruction tells the model
    what to say if the ledger is unavailable.
    """
    import ledger
    cfg = ledger.settings()
    if cfg is None:
        return None
    from google.adk.tools import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters

    env = {
        "CLICKHOUSE_HOST": cfg["host"],
        "CLICKHOUSE_PORT": str(cfg["port"]),
        "CLICKHOUSE_USER": cfg["username"],
        "CLICKHOUSE_PASSWORD": cfg["password"],
        "CLICKHOUSE_DATABASE": cfg["database"],
        "CLICKHOUSE_SECURE": "true" if cfg["secure"] else "false",
    }
    return McpToolset(
        connection_params=StdioServerParameters(
            command="uvx", args=["mcp-clickhouse"],
            env={**os.environ, **env},
        ),
        tool_name_prefix="ledger",
    )


def build_agent():
    """The agent, with the ledger attached when there is one to attach."""
    kit = [
        tools.list_films,
        tools.triage_film,
        tools.render_film,
        tools.render_status,
        tools.inspect_walls,
        tools.settle_walls,
        tools.record_run,
        tools.ledger_examples,
    ]
    note = ""
    try:
        ch = _clickhouse_toolset()
    except Exception as e:                        # a bad DSN must not stop the app
        ch, note = None, f" (ledger unavailable: {type(e).__name__})"
    if ch is not None:
        kit.append(ch)
    else:
        note = note or " (ledger unavailable: no ClickHouse credentials set)"

    return LlmAgent(
        name="screenx_supervisor",
        model=MODEL,
        description="Decides which shots of a film can be widened to 270 "
                    "degrees from their own footage, converts them, and keeps "
                    "an auditable ledger of what was photographed and what was "
                    "invented.",
        instruction=INSTRUCTION + (
            f"\n\nNOTE: the ClickHouse ledger is not connected{note}. If asked a "
            f"question that needs it, say it is not connected rather than "
            f"answering from the single job in front of you."
            if ch is None else ""),
        tools=kit,
    )


root_agent = build_agent()
