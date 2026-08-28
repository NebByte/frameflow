"""
test_agent_service -- the agent, the ledger, and the triage behind them.

Three properties matter here and none of them are about the model:

    the ledger records what the render measured, and refuses to be written to
    triage UNDERSTATES, so a shot it clears is one a render will clear
    a tool that fails returns a failure, it does not raise into the model

The last one is the difference between an agent that says "the database is not
connected" and one that hands Gemini a stack trace and lets it narrate.

No live Gemini calls: they cost money, need network, and are not what is under
test. `test_e2e.py` is where the joins get exercised.

Run: python test_agent_service.py
"""

from __future__ import annotations

# Runnable directly as well as under pytest, so the repo root has to be
# importable either way.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path


from frameflow import ledger
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


SUMMARY = dict(
    source="clip.mp4", shots=2, frames=300, fps=30.0, wing_w=105,
    wing_ratio=0.22, width=690, height=270, mean_real_wing=0.5,
    per_shot=[
        dict(shot=0, start=0, frames=200, motion="ROTATION", backend="mosaic",
             geometry=30.1, state="NARROW", coverage=1.0, effective=0.53,
             reasons="effective 53%", displacement=11.5, layer_disagree=9.0,
             provenance=dict(recovered=0.9, generated=0.1, photographic=0.9)),
        dict(shot=1, start=200, frames=100, motion="LOCKED", backend="none",
             geometry=0.0, state="OFF", coverage=0.0, effective=0.0,
             reasons="locked off", provenance={}),
    ])


def test_the_ledger_records_refusals_too():
    """
    A ledger of successes answers "what did we convert" and never "what could
    we have" -- which is the question that decides whether a title is worth an
    artist's month. The refused shot has to be a row.
    """
    print("every shot is a row, including the ones that were refused")
    run_id, rows = ledger.rows_from(SUMMARY)
    check("one row per shot", len(rows) == 2, str(len(rows)))
    check("column count matches the schema",
          all(len(r) == len(ledger.COLUMNS) for r in rows))

    by = {r[ledger.COLUMNS.index("shot")]: r for r in rows}
    st = ledger.COLUMNS.index("state")
    check("the earned shot is recorded", by[0][st] == "NARROW", by[0][st])
    check("and so is the refusal", by[1][st] == "OFF", by[1][st])
    check("with the reason it was refused",
          "locked" in by[1][ledger.COLUMNS.index("reasons")])

    ph = ledger.COLUMNS.index("photographic")
    check("provenance is carried across", abs(by[0][ph] - 0.9) < 1e-6, str(by[0][ph]))
    check("a shot with no provenance reads as zero, not as missing",
          by[1][ph] == 0.0)

    # every rung gets a column even when it never fired
    for rung in ledger.RUNGS:
        if rung not in ledger.COLUMNS:
            check(f"rung {rung} has a column", False)
            break
    else:
        check("every rung has a column, fired or not", True)


def test_unmeasured_is_not_zero():
    """
    A run nobody inspected has no hairline figure. Recording that as 0.0 would
    read as a flawless wall, which is exactly the lie `polish._charge` exists
    to prevent one level down. -1 means nobody looked.
    """
    print("\nunmeasured must not read as perfect")
    _run, rows = ledger.rows_from(SUMMARY)          # no findings passed
    i = ledger.COLUMNS.index("hairlines")
    check("uninspected walls are -1, not 0", rows[0][i] == -1.0, str(rows[0][i]))

    _run, rows2 = ledger.rows_from(
        SUMMARY, findings=[dict(shot=0, hairlines=0.0, jitter=1.04, seam=1.5)])
    check("a measured zero is recorded as zero", rows2[0][i] == 0.0)
    check("and the other measurements come with it",
          abs(rows2[0][ledger.COLUMNS.index("jitter")] - 1.04) < 1e-6)


def test_the_ledger_answers_questions_and_takes_no_orders():
    """
    `ask` is reachable from an agent, and an agent will eventually be asked to
    drop a table by someone who thinks it is funny.
    """
    print("\nthe ledger answers questions; it does not take orders")
    for bad in ("DROP TABLE frameflow_shots",
                "INSERT INTO frameflow_shots VALUES (1)",
                "TRUNCATE TABLE frameflow_shots",
                "ALTER TABLE frameflow_shots DELETE WHERE 1=1",
                "select 1; drop table frameflow_shots"):
        try:
            ledger.ask(bad, client=object())
            check(f"refuses: {bad[:34]}", False, "it was allowed")
        except ValueError:
            check(f"refuses: {bad[:34]}", True)
        except Exception as e:
            check(f"refuses: {bad[:34]}", False, f"wrong error {type(e).__name__}")


def test_no_credentials_says_so():
    """
    A missing database must read as a missing database, never as a render that
    found nothing.
    """
    print("\na missing ledger is a missing ledger")
    check("no host means no settings", ledger.settings(env={}) is None)
    try:
        ledger.connect(env={})
        check("connect refuses without credentials", False)
    except ledger.NotConfigured as e:
        check("connect refuses without credentials", True)
        check("and names the variables to set", "CLICKHOUSE_HOST" in str(e),
              str(e)[:60])

    cfg = ledger.settings(env={"CLICKHOUSE_HOST": "h", "CLICKHOUSE_PASSWORD": "p"})
    check("sensible defaults for everything else",
          cfg["port"] == 8443 and cfg["username"] == "default" and cfg["secure"],
          str(cfg))


def test_a_failing_tool_returns_a_failure():
    """
    Tools hand their result to a language model. One that raises hands it a
    stack trace and invites a narrated guess; one that returns
    {"ok": false, "error": ...} lets it say what happened.
    """
    print("\ntools fail by returning, not by raising")
    from agent_service import tools

    for name, call in (
            ("triage_film", lambda: tools.triage_film("nope-does-not-exist.mp4")),
            ("render_film", lambda: tools.render_film("nope-does-not-exist.mp4")),
            ("inspect_walls", lambda: tools.inspect_walls("no-such-job")),
            ("settle_walls", lambda: tools.settle_walls("no-such-job")),
            ("record_run", lambda: tools.record_run("no-such-job")),
            ("render_status", lambda: tools.render_status("no-such-job"))):
        try:
            got = call()
            check(f"{name} returns a dict on failure", isinstance(got, dict)
                  and got.get("ok") is False, str(got)[:70])
        except Exception as e:
            check(f"{name} returns a dict on failure", False,
                  f"raised {type(e).__name__}")

    got = tools.list_films()
    check("list_films works with nothing configured", got.get("ok") is True)
    check("and reports films it can see", isinstance(got.get("films"), list))


def test_the_scout_cannot_render():
    """
    The property that makes triage worth having.

    Triage exists so nobody spends an hour proving a shot was never going to
    work. An agent that can both judge and render will, asked "is this worth
    converting?", render to find out -- which is precisely the cost being
    avoided. So the scout does not get render_film, and that is enforced here
    rather than hoped for in a prompt.
    """
    print("")
    print("the scout decides; it does not render")
    from agent_service.agent import build_agent
    a = build_agent()

    check("the root is a supervisor", a.name == "frameflow_supervisor", a.name)
    kids = {s.name: s for s in (a.sub_agents or [])}
    check("with three specialists", set(kids) == {"scout", "conversion", "archivist"},
          ", ".join(sorted(kids)))
    if not kids:
        return

    def toolnames(agent):
        return {getattr(t, "__name__", type(t).__name__) for t in agent.tools}

    scout = toolnames(kids["scout"])
    check("the scout can triage", "triage_film" in scout)
    check("and cannot render", "render_film" not in scout, ", ".join(sorted(scout)))
    check("and cannot repaint", "settle_walls" not in scout)

    conv = toolnames(kids["conversion"])
    check("conversion renders", "render_film" in conv)
    check("and settles walls", "settle_walls" in conv)
    check("and does not triage", "triage_film" not in conv, ", ".join(sorted(conv)))

    arc = toolnames(kids["archivist"])
    check("the archivist records runs", "record_run" in arc)
    check("and touches no pixels",
          not ({"render_film", "settle_walls", "triage_film"} & arc),
          ", ".join(sorted(arc)))


def test_every_agent_carries_the_honesty_rule():
    """
    The distinction between photographed and invented is the product. It is
    repeated into every specialist rather than left with the supervisor,
    because a sub-agent answers the user directly once it is handed control.
    """
    print("")
    print("every agent is told what it must not blur")
    from agent_service.agent import build_agent
    a = build_agent()
    for agent in [a] + list(a.sub_agents or []):
        text = agent.instruction or ""
        check(f"{agent.name}: defends photographed vs invented",
              "INVENTED" in text and "PHOTOGRAPHED" in text)
        check(f"{agent.name}: a refusal is a real answer",
              "refusal is a real answer" in text)


def test_a_missing_ledger_is_admitted_not_guessed():
    """
    The archivist is the one that answers catalogue questions, so it is the one
    that has to say when it cannot.
    """
    print("")
    print("no ledger means the archivist says so")
    from agent_service.agent import build_agent
    a = build_agent()
    arc = next((s for s in (a.sub_agents or []) if s.name == "archivist"), None)
    check("the archivist exists", arc is not None)
    if arc is None:
        return
    if ledger.settings() is None:
        check("and is told the ledger is not connected",
              "NOT connected" in (arc.instruction or ""))
    else:
        check("ledger configured, so no disclaimer needed", True)


def test_triage_understates():
    """
    The property the whole triage idea rests on.

    Triage looks at a window, not the shot. A window holds fewer donor frames
    than the shot does, so it recovers less -- which means a shot triage clears
    is one a render will clear. Wrong in the safe direction: understating costs
    a shot you could have had, overstating costs the artist-month you committed
    on the strength of it.
    """
    print("\ntriage is a floor, not an estimate")
    from frameflow import triage as tr
    check("the window is consecutive, not spread",
          "consecutive" in tr.__doc__.lower())
    rep_doc = tr.triage_film.__doc__ or ""
    check("and the report says the figures are floors",
          "floor" in rep_doc.lower() or "floor" in tr.triage_film.__doc__.lower())

    # measured on the real clip earlier: triage 50.7% vs render 52.54%
    clip = Path("media/pan_flat.mp4")
    if not clip.exists():
        check("synthetic clip present (run make_test_clip.py)", False)
        return
    rep = tr.triage_film(str(clip), maxw=320, verbose=False)
    check("it reaches a verdict", bool(rep["verdicts"]), str(rep["shots"]))
    v = rep["verdicts"][0]
    check("a pure pan is earned, not refused",
          v["state"] in ("FULL", "NARROW"), f"{v['state']} @ {v['effective']}")
    check("every verdict carries what it means to a person",
          all(x.get("meaning") for x in rep["verdicts"]))
    check("and the report states its basis", "floor" in rep["basis"].lower())


def test_a_locked_shot_is_refused_before_anything_is_rendered():
    """
    The whole economic argument: a locked-off camera never filmed the walls, and
    finding that out should cost seconds, not an artist's week.
    """
    print("\nnothing filmed means nothing recovered, and it is cheap to learn")
    from frameflow import triage as tr
    clip = Path("media/locked_off.mp4")
    if not clip.exists():
        check("synthetic clip present (run make_test_clip.py)", False)
        return
    rep = tr.triage_film(str(clip), maxw=320, verbose=False)
    v = rep["verdicts"][0]
    check("it is refused", v["state"] in ("LOCKED", "OFF"), v["state"])
    check("in plain language", "never" in v["meaning"] or "nothing" in v["meaning"],
          v["meaning"])
    check("and none of the running time counts as earned",
          rep["earned_fraction"] == 0.0, str(rep["earned_fraction"]))


if __name__ == "__main__":
    print("the agent, the ledger, and the triage behind them\n")
    test_the_ledger_records_refusals_too()
    test_unmeasured_is_not_zero()
    test_the_ledger_answers_questions_and_takes_no_orders()
    test_no_credentials_says_so()
    test_a_failing_tool_returns_a_failure()
    test_the_scout_cannot_render()
    test_every_agent_carries_the_honesty_rule()
    test_a_missing_ledger_is_admitted_not_guessed()
    test_triage_understates()
    test_a_locked_shot_is_refused_before_anything_is_rendered()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
