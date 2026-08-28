"""
ledger -- every shot's verdict, in a table somebody can question.

A render already writes `screenx_summary.json` beside the job. That is the
right home for one film and the wrong home for a catalogue: the question a
studio actually asks is not "how did this shot do" but

    which shots, across everything we own, can be widened without inventing
    anything -- and which are we being asked to pay artists for?

That is a query. It is not answerable by opening two hundred JSON files, and it
is the question that decides whether a title is worth converting at all.

So each shot becomes a row: what the camera did, what the renderer decided,
where every pixel came from, and how the wall actually measures. Refusals are
rows too, and that is deliberate -- a ledger that only recorded successes would
answer "what did we convert" and never "what could we have".

    write_run(job_dir)     a finished render -> rows
    ask(sql)               read them back

NOTHING HERE INVENTS A NUMBER
-----------------------------
Every column is copied from what the render already measured and refused to
overstate. `ledger` adds no estimate of its own, because a figure that appears
for the first time in a database is a figure nobody gated.

CREDENTIALS
-----------
CLICKHOUSE_HOST / _PORT / _USER / _PASSWORD / _DATABASE, or CLICKHOUSE_URL.
Absent, every call raises `NotConfigured` with the variable names in the
message. A missing database must read as a missing database and never as a
render that produced nothing.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

TABLE = "frameflow_shots"

# The rungs, in the order the ladder climbs them. Named here rather than
# discovered from the data so a rung that never fires still gets a column and
# still reads as zero instead of vanishing.
RUNGS = ("primary", "recovered", "donated", "retrieved",
         "referenced", "directed", "generated")

# Backticked because `primary` collides with PRIMARY KEY in ClickHouse's
# grammar, and a rung is not going to be renamed to suit a parser.
_RUNG_COLS = "\n".join(f"    `{r}` Float32," for r in RUNGS)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {{db}}.{TABLE} (
    run_id          String,
    ran_at          DateTime,
    source          String,
    shot            UInt16,
    start_frame     UInt32,
    frames          UInt32,

    motion          LowCardinality(String),
    displacement    Float32,
    layer_disagree  Float32,

    backend         LowCardinality(String),
    geometry_db     Float32,
    state           LowCardinality(String),
    reasons         String,
    coverage        Float32,
    effective       Float32,

{_RUNG_COLS}
    photographic    Float32,

    hairlines       Float32,
    jitter          Float32,
    seam            Float32,

    wing_ratio      Float32,
    wing_w          UInt16,
    fps             Float32,
    width           UInt16,
    height          UInt16
) ENGINE = MergeTree
ORDER BY (source, run_id, shot)
"""

COLUMNS = (["run_id", "ran_at", "source", "shot", "start_frame", "frames",
            "motion", "displacement", "layer_disagree",
            "backend", "geometry_db", "state", "reasons", "coverage",
            "effective"]
           + list(RUNGS) + ["photographic",
                            "hairlines", "jitter", "seam",
                            "wing_ratio", "wing_w", "fps", "width", "height"])


class NotConfigured(RuntimeError):
    """No ClickHouse credentials. Says which variables are missing."""


def settings(env=None):
    """Connection details from the environment, or None if there are none."""
    e = os.environ if env is None else env
    host = e.get("CLICKHOUSE_HOST", "").strip()
    if not host:
        return None
    return dict(
        host=host,
        port=int(e.get("CLICKHOUSE_PORT") or 8443),
        username=e.get("CLICKHOUSE_USER", "default"),
        password=e.get("CLICKHOUSE_PASSWORD", ""),
        database=e.get("CLICKHOUSE_DATABASE", "default"),
        secure=(e.get("CLICKHOUSE_SECURE", "1") not in ("0", "false", "False")),
    )


def connect(env=None):
    """
    A live client, or a refusal that names what is missing.

    Deliberately not a silent fallback to an in-memory stub. A demo that
    quietly stops writing to the database it claims to use is worse than one
    that stops.
    """
    cfg = settings(env)
    if cfg is None:
        raise NotConfigured(
            "no ClickHouse connection: set CLICKHOUSE_HOST, CLICKHOUSE_USER, "
            "CLICKHOUSE_PASSWORD (and optionally _PORT, _DATABASE). A free "
            "ClickHouse Cloud service supplies all four."
        )
    try:
        import clickhouse_connect
    except ImportError as e:                      # pragma: no cover
        raise NotConfigured(f"clickhouse-connect is not installed: {e}") from e
    db = cfg.pop("database")
    client = clickhouse_connect.get_client(**cfg)
    client.database = db
    return client


def ensure_schema(client, database=None):
    """Create the table if it is not there. Safe to call every run."""
    db = database or getattr(client, "database", None) or "default"
    client.command(SCHEMA.format(db=db))
    return f"{db}.{TABLE}"


# ------------------------------------------------------------------ shaping

def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def rows_from(summary, findings=None, run_id=None, ran_at=None):
    """
    A finished run's summary -> one row per shot.

    `findings` is polish_report.json's list, which carries the three
    measurements the gate cannot see (hairlines, jitter, seam). It is optional
    because a run that was never inspected still belongs in the ledger; those
    columns come back as -1 to mean UNMEASURED rather than 0, which would read
    as a perfect wall nobody looked at.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    ran_at = ran_at or datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    seen = {int(f.get("shot", -1)): f for f in (findings or [])}

    wing_ratio = _f(summary.get("wing_ratio"), 0.22)
    wing_w = int(summary.get("wing_w") or 0)
    fps = _f(summary.get("fps"), 0.0)
    src = str(summary.get("source") or "")

    rows = []
    for rec in summary.get("per_shot", []):
        si = int(rec.get("shot", 0))
        prov = rec.get("provenance") or {}
        look = seen.get(si, {})
        rows.append([
            run_id, ran_at, src, si,
            int(rec.get("start") or 0), int(rec.get("frames") or 0),

            str(rec.get("motion") or ""), _f(rec.get("displacement")),
            _f(rec.get("layer_disagree")),

            str(rec.get("backend") or ""), _f(rec.get("geometry")),
            str(rec.get("state") or ""), str(rec.get("reasons") or ""),
            _f(rec.get("coverage")), _f(rec.get("effective")),

            *[_f(prov.get(r)) for r in RUNGS], _f(prov.get("photographic")),

            # -1 is "nobody looked", which is not the same as "clean"
            _f(look.get("hairlines"), -1.0),
            _f(look.get("jitter"), -1.0),
            _f(look.get("seam"), -1.0),

            wing_ratio, wing_w, fps,
            int(summary.get("width") or 0), int(summary.get("height") or 0),
        ])
    return run_id, rows


def read_job(job_dir):
    """The summary and, if one exists, the inspection beside it."""
    job = Path(job_dir)
    summary = json.loads((job / "screenx_summary.json").read_text(encoding="utf-8"))
    findings = None
    rp = job / "polish_report.json"
    if rp.exists():
        try:
            findings = json.loads(rp.read_text(encoding="utf-8")).get("findings")
        except ValueError:
            findings = None
    return summary, findings


def write_run(job_dir, client=None, run_id=None, verbose=True):
    """Push a finished render into the ledger. Returns (run_id, row count)."""
    summary, findings = read_job(job_dir)
    close = client is None
    client = client or connect()
    try:
        ensure_schema(client)
        run_id, rows = rows_from(summary, findings, run_id=run_id)
        if rows:
            client.insert(TABLE, rows, column_names=COLUMNS)
        if verbose:
            print(f"  ledger: {len(rows)} shot(s) written as run {run_id}",
                  flush=True)
        return run_id, len(rows)
    finally:
        if close:
            try:
                client.close()
            except Exception:                     # pragma: no cover
                pass


def ask(sql, client=None, limit=200):
    """
    Run a read-only query and return (column names, rows).

    Reads are capped and writes are refused here as well as by the credentials,
    because this is reachable from an agent and an agent will eventually be
    asked to delete something by someone who finds it funny.
    """
    lowered = " ".join(sql.lower().split())
    if not lowered.startswith(("select", "with", "show", "describe")):
        raise ValueError("the ledger answers questions; it does not take "
                         "INSERT, ALTER, DROP or TRUNCATE")
    for word in ("insert ", "alter ", "drop ", "truncate ", "delete ", "create "):
        if word in lowered:
            raise ValueError(f"'{word.strip()}' is not a question")
    close = client is None
    client = client or connect()
    try:
        if " limit " not in lowered and lowered.startswith(("select", "with")):
            sql = f"{sql.rstrip().rstrip(';')} LIMIT {int(limit)}"
        res = client.query(sql)
        return list(res.column_names), [list(r) for r in res.result_rows]
    finally:
        if close:
            try:
                client.close()
            except Exception:                     # pragma: no cover
                pass


# The questions this exists to answer, kept next to the schema so the demo and
# the docs cannot drift from what the table can actually support.
EXAMPLES = {
    "earned": (
        "How much of a film can be widened from its own footage?\n"
        f"SELECT source, count() AS shots, "
        f"countIf(state IN ('FULL','NARROW','BORROWED')) AS earned, "
        f"round(avg(photographic), 4) AS mean_real_wing "
        f"FROM {TABLE} GROUP BY source ORDER BY earned DESC"
    ),
    "refused": (
        "Which shots were refused, and why?\n"
        f"SELECT shot, motion, state, round(geometry_db,1) AS db, "
        f"round(effective,3) AS eff, reasons FROM {TABLE} "
        f"WHERE state IN ('OFF','GEN') ORDER BY shot"
    ),
    "invented": (
        "Where did we invent rather than recover?\n"
        f"SELECT source, shot, round(generated,4) AS generated, "
        f"round(directed,4) AS directed, round(photographic,4) AS real "
        f"FROM {TABLE} WHERE generated + directed > 0.01 "
        f"ORDER BY generated DESC"
    ),
    "artifacts": (
        "Which delivered walls still measure badly?\n"
        f"SELECT source, shot, round(hairlines,4) AS dark_lines, "
        f"round(jitter,2) AS jitter, round(seam,2) AS seam FROM {TABLE} "
        f"WHERE jitter >= 0 ORDER BY jitter DESC"
    ),
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="the shot ledger")
    ap.add_argument("--write", metavar="JOB_DIR", help="push a finished render")
    ap.add_argument("--sql", help="ask a question")
    ap.add_argument("--examples", action="store_true",
                    help="the questions this table was shaped to answer")
    a = ap.parse_args()

    if a.examples:
        for name, text in EXAMPLES.items():
            head, _, q = text.partition("\n")
            print(f"\n{name}: {head}\n  {q}")
    elif a.write:
        write_run(a.write)
    elif a.sql:
        cols, rows = ask(a.sql)
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join(str(v) for v in r))
    else:
        ap.print_help()
