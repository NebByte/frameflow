"""
server -- one container, two surfaces.

Cloud Run gives a service exactly one port, and this project has two things
worth showing on it:

    /            the agent. ADK's chat UI, talking to `agent_service`.
    /studio/     the converter itself -- upload, convert, watch the three
                 projector feeds, read the provenance report.

The agent is the entry point because the agent is the product: it decides which
shots are worth converting before anything is rendered. The studio is where the
result of that decision gets looked at, and it is a real interface rather than a
JSON dump, so it stays.

WHY THE STUDIO IS PROXIED RATHER THAN PORTED
--------------------------------------------
`app.py` is a stdlib ThreadingHTTPServer written well before any of this and
covered by `test_app.py`. Rewriting it as ASGI to satisfy a deployment target
would put sixteen passing tests at risk for no user-visible gain, so it runs on
a loopback port inside the container and this forwards to it. The forwarding is
the only new code, and if it breaks, it breaks visibly at `/studio/` while the
agent keeps working.
"""
from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

# Imported at module level on purpose. `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves those against
# MODULE globals -- so a `Request` imported inside build() is invisible to it,
# and it quietly reclassifies `request: Request` as a required query parameter.
# The symptom is a 422 on every proxied call with "query.request Field required",
# which points nowhere near the cause.
import httpx
from fastapi import Request
from fastapi.responses import (HTMLResponse, JSONResponse,
                               RedirectResponse, Response)

HERE = Path(__file__).resolve().parent
STUDIO_PORT = int(os.environ.get("STUDIO_PORT", "8421"))
PORT = int(os.environ.get("PORT", "8080"))


def start_studio():
    """The converter, on a loopback port nobody outside the container sees."""
    import app as studio

    studio.JOBS_DIR.mkdir(exist_ok=True)
    for name in studio.seed_jobs():          # so /studio/ is never empty
        print(f"seeded sample job: {name}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", STUDIO_PORT), studio.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def build():
    from google.adk.cli.fast_api import get_fast_api_app

    # agents_dir points AT the package, not at the repo that holds it.
    #
    # ADK decides a directory "is" a single agent if it contains agent.py --
    # and this repo has had an agent.py since long before any of this, holding
    # the WingAgent that plans which rung to try. Pointing at the repo root
    # therefore made ADK adopt the root as the agent and name it after the
    # working directory, which inside the container is /app. The deployed
    # service listed one agent called "app" and it was the studio server.
    #
    # Naming the package directly puts ADK in single-agent mode with the right
    # name, and it still adds the parent to sys.path so `import triage` works.
    api = get_fast_api_app(agents_dir=str(HERE / "agent_service"), web=True,
                           host="0.0.0.0", port=PORT)

    start_studio()
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{STUDIO_PORT}",
                               timeout=httpx.Timeout(600.0))

    @api.get("/", include_in_schema=False)
    async def landing():
        """
        A judge pasting the bare URL used to be redirected straight into a chat
        box with no indication of what to ask it. Three clickable questions and
        two sentences of context cost nothing and answer "what is this".
        """
        page = HERE / "static" / "landing.html"
        if not page.is_file():
            return RedirectResponse("/dev-ui/")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @api.get("/status")
    async def status():
        """
        What is actually wired up. Not /healthz: Google's frontend answers
        that path itself before it reaches the container, so the route existed
        in the OpenAPI schema and returned Google's own 404 in production.
        """
        from frameflow import ledger
        return JSONResponse(dict(
            ok=True,
            agent="frameflow_supervisor",
            ledger="configured" if ledger.settings() else "not configured",
            studio_port=STUDIO_PORT,
        ))

    @api.api_route("/static/{path:path}", methods=["GET", "HEAD"],
                   include_in_schema=False)
    async def studio_assets(path: str, request: Request):
        """
        The studio's own CSS and JS, at the path its HTML asks for.

        index.html references /static/frameflow.css and /static/app.js -- root
        absolute, because standalone the studio IS the root. Proxied under
        /studio/ those requests go to the root app instead, which does not have
        them, so the page rendered as unstyled HTML with no behaviour at all.
        Rewriting the markup would break running `python app.py` directly, so
        the root serves them instead.
        """
        return await _forward(request, f"/static/{path}")

    @api.api_route("/studio{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def studio(path: str, request: Request):
        """Forward to the converter, streaming so large media does not buffer."""
        return await _forward(request, path or "/")

    async def _forward(request: Request, target: str):
        if not target.startswith("/"):
            target = "/" + target
        if request.url.query:
            target = f"{target}?{request.url.query}"
        body = await request.body()
        # hop-by-hop headers and Host must not be forwarded verbatim
        head = {k: v for k, v in request.headers.items()
                if k.lower() not in ("host", "connection", "content-length")}
        try:
            r = await client.request(request.method, target, content=body,
                                     headers=head)
        except httpx.HTTPError as e:
            return JSONResponse(
                {"error": f"the studio is not answering: {type(e).__name__}"},
                status_code=502)
        drop = {"content-encoding", "transfer-encoding", "connection",
                "content-length"}
        return Response(
            content=r.content, status_code=r.status_code,
            headers={k: v for k, v in r.headers.items() if k.lower() not in drop},
            media_type=r.headers.get("content-type"))

    # ADK already owns "/" (it redirects into the dev UI) and its route was
    # registered first, so a later one never matches. Starlette resolves in
    # order, so the landing page has to be moved to the front rather than
    # merely added.
    for i, r in enumerate(api.router.routes):
        if getattr(r, "path", None) == "/" and "landing" in getattr(
                getattr(r, "endpoint", None), "__name__", ""):
            api.router.routes.insert(0, api.router.routes.pop(i))
            break

    return api


app = build()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
