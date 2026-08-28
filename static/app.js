/* Frameflow Studio -- operator shell.
 *
 * Vanilla ES modules on purpose: no build step, so the repo stays
 * `python app.py` and two dependencies.
 *
 * The flow is staged rather than one-shot. A clip uploads and waits, because
 * what else rides along with it -- another cut, another setup, context files --
 * changes which rungs the run can even reach.
 */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = { job: null, shots: [], summary: null, caps: null, sel: null, file: null };

const pct  = v => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const db   = v => (v == null ? "—" : Number(v).toFixed(1) + " dB");
const mb   = b => (b > 1048576 ? (b / 1048576).toFixed(1) + " MB"
                               : Math.max(1, Math.round(b / 1024)) + " KB");
const esc  = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------------ stages */

function go(name) {
  $$("[data-panel]").forEach(p => { p.hidden = p.dataset.panel !== name; });
  $$(".stage").forEach(b => b.setAttribute("aria-current", String(b.dataset.stage === name)));
}
$("#stages").addEventListener("click", e => {
  const b = e.target.closest(".stage");
  if (b && !b.disabled) go(b.dataset.stage);
});
const unlock = n => { const b = $(`.stage[data-stage="${n}"]`); if (b) b.disabled = false; };

/* ------------------------------------------------------------ capabilities */

const FLAGS = [
  { id: "sources",   label: "Index the film and try the real rungs first",
    note: "DONATED and RETRIEVED before anything is invented", needs: null },
  { id: "sfm",       label: "COLMAP reconstruction (--sfm)",
    note: "required for RETRIEVED", needs: "colmap" },
  { id: "prefer_3d", label: "Gaussian backend (--prefer-3d)",
    note: "3D recovery instead of layered warps", needs: "gpu" },
  { id: "reason",    label: "Reason about what left frame (--reason)",
    note: "measured evidence, slower", needs: null },
  { id: "vision",    label: "Add a vision model's claims (--vision)",
    note: "lands as asserted, never measured", needs: "gemini" },
  { id: "online",    label: "Query Openverse for licensed material (--online)",
    note: "external, needs network", needs: null },
];


/* The key box belongs to one paid, optional generator, and showing it to
   everyone reads as "this needs credentials" -- which is exactly backwards.
   The whole argument of this tool is that the walls come out of the footage
   for free. So it stays hidden until somebody actually picks a generator that
   needs one, and it says so when it appears. */
const HOSTED = new Set(["gemini-edit"]);

function syncKeyPanel() {
  const panel = $("#keypanel");
  if (!panel) return;
  const picked = [$("#dark"), $("#polishgen")]
    .filter(Boolean).map(el => el.value);
  const needs = picked.some(v => HOSTED.has(v));
  const already = state.caps && state.caps.gemini && state.caps.gemini.ok;
  panel.hidden = !needs || already;
}

async function loadCaps() {
  const caps = state.caps = await (await fetch("/api/capabilities")).json();
  $("#caps").innerHTML = Object.entries(caps).map(([, c]) => `
    <div class="check ${c.ok ? "" : "off"}">
      <span style="color:${c.ok ? "var(--real)" : "var(--ink-faint)"}">${c.ok ? "●" : "○"}</span>
      <span>${esc(c.label)}${c.ok ? "" : `<span class="why">${esc(c.reason)}</span>`}</span>
    </div>`).join("");

  syncKeyPanel();

  $("#flags").innerHTML = FLAGS.map(f => {
    const cap = f.needs ? caps[f.needs] : null;
    const off = cap && !cap.ok;
    return `<label class="check ${off ? "off" : ""}">
      <input type="checkbox" data-flag="${f.id}" ${off ? "disabled" : ""}>
      <span>${esc(f.label)}<span class="why">${esc(off ? cap.reason : f.note)}</span></span>
    </label>`;
  }).join("");
  $$("[data-flag]").forEach(cb => cb.addEventListener("change", showCommand));

  const cb = caps.colab || { ok: false, reason: "unavailable" };
  $("#remote").disabled = !cb.ok;
  $("#remoterow").classList.toggle("off", !cb.ok);
  if (!cb.ok) $("#remotewhy").textContent = cb.reason;
}

$("#savekey").addEventListener("click", async () => {
  const key = $("#wskey").value.trim();
  const res = await fetch("/api/keys", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ GEMINI_API_KEY: key }),
  });
  const out = await res.json();
  $("#wskey").value = "";                       // not kept in the page either
  $("#keystatus").textContent = out.error
    ? ` ${out.error}` : ` in use: ${out.set.join(", ") || "none"}`;
  loadCaps();
});

/* -------------------------------------------------------------------- jobs */

async function loadJobs() {
  const jobs = await (await fetch("/api/jobs")).json();
  $("#jobcount").textContent = jobs.length ? `${jobs.length} on this machine` : "";
  $("#jobs tbody").innerHTML = jobs.length ? jobs.map(j => `
    <tr data-job="${esc(j.id)}">
      <td class="n">${esc(j.id)}</td>
      <td>${esc(j.name)}</td>
      <td>${esc(j.state)}</td>
      <td class="n">${j.real == null ? "—" : pct(j.real)}</td>
      <td><button class="ghost" data-del="${esc(j.id)}">Delete</button></td>
    </tr>`).join("") : `<tr><td colspan="5" class="empty">No jobs yet.</td></tr>`;
}

$("#jobs").addEventListener("click", async e => {
  const del = e.target.closest("[data-del]");
  if (del) {
    e.stopPropagation();
    await fetch(`/api/jobs/${del.dataset.del}`, { method: "DELETE" });
    return loadJobs();
  }
  const id = e.target.closest("tr")?.dataset.job;
  if (!id) return;
  const job = await (await fetch(`/api/jobs/${id}`)).json();
  if (job.error) return;
  state.job = id; state.shots = job.shots || []; state.summary = job.summary;
  unlock("review"); unlock("report");
  renderReview(); renderReport(); loadFiles(); loadPolish(); go("review");
});

/* -------------------------------------------------------------------- load */

const drop = $("#drop"), fileInput = $("#file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("hot"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("hot");
  if (e.dataTransfer.files[0]) accept(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => fileInput.files[0] && accept(fileInput.files[0]));

/* Probe locally first: both fatal inputs are knowable from the file alone, and
 * saying so after a five minute render is too late to be useful. */
function accept(file) {
  state.file = file;
  const v = document.createElement("video");
  v.preload = "metadata";
  v.onloadedmetadata = async () => {
    const w = v.videoWidth, h = v.videoHeight, dur = v.duration, aspect = w / h;
    const warn = [];
    if (h > w) warn.push("Portrait. The vertical axis binds the wall geometry: a portrait clip measured 30.6% real wing against 91.5% for the same room shot landscape.");
    if (aspect > 2.2) warn.push("Scope aspect. Wall depth comes back 0.00 m at 2.39:1 — the walls have nowhere to land in the modeled auditorium.");
    if (dur < 8) warn.push("Under 8 seconds. Two-second clips gave COLMAP 58 points and two disconnected models.");
    $("#probe").innerHTML = `
      <div class="readouts" style="border:1px solid var(--line);margin-top:12px">
        <div class="readout"><div class="k">Source</div><div class="v small">${esc(file.name)}</div></div>
        <div class="readout"><div class="k">Resolution</div><div class="v small">${w}×${h}</div>
          <div class="n">${aspect.toFixed(2)}:1 · ${h > w ? "portrait" : "landscape"}</div></div>
        <div class="readout"><div class="k">Duration</div><div class="v small">${dur.toFixed(1)} s</div></div>
        <div class="readout"><div class="k">Size</div><div class="v small">${mb(file.size)}</div></div>
      </div>
      ${warn.map(t => `<p class="note warn" style="margin-top:10px">${esc(t)}</p>`).join("")}
      <p class="note" id="upstate" style="margin-top:10px">staging&hellip;</p>`;
    URL.revokeObjectURL(v.src);

    const res = await fetch(`/api/jobs?name=${encodeURIComponent(file.name)}`,
                            { method: "POST", body: file });
    const out = await res.json();
    if (out.error) { $("#upstate").textContent = out.error; return; }
    state.job = out.id;
    $("#jobid").textContent = out.id;
    $("#upstate").innerHTML = `staged as <strong>${esc(out.id)}</strong> — attach anything else below, or go on.`;
    $("#attachpanel").hidden = false;
    unlock("analyse"); unlock("render");
    $("#run").disabled = false;
    showCommand();
  };
  v.onerror = () => { $("#probe").innerHTML = `<p class="note warn">Could not read that file.</p>`; };
  v.src = URL.createObjectURL(file);
}

/* --------------------------------------------------------------- attaching */

$$("[data-attach]").forEach(input => {
  input.addEventListener("change", async () => {
    if (!state.job) return;
    for (const f of input.files) {
      const q = `kind=${input.dataset.attach}&name=${encodeURIComponent(f.name)}`;
      const res = await fetch(`/api/jobs/${state.job}/attach?${q}`,
                              { method: "POST", body: f });
      const out = await res.json();
      if (out.error) { $("#attached").textContent = out.error; return; }
      renderAttached(out.attachments);
    }
    input.value = "";
    showCommand();
  });
});

const RUNG = { other_cut: "DONATED", also: "RETRIEVED", context: "DIRECTED" };
function renderAttached(att) {
  state.attachments = att;
  const rows = Object.entries(att || {}).flatMap(([kind, files]) =>
    files.map(f => `<div>${esc(kind)} <span style="color:var(--ink-dim)">${esc(f.split(/[\\/]/).pop())}</span>
      <span style="color:var(--accent)">→ ${RUNG[kind]}</span></div>`));
  $("#attached").innerHTML = rows.join("") || "";
}

/* ----------------------------------------------------------------- analyse */

$("#analyse").addEventListener("click", async () => {
  if (!state.job) return;
  $("#analyse").disabled = true;
  $("#analyse").textContent = "Detecting…";
  const out = await (await fetch(`/api/jobs/${state.job}/analyse`, { method: "POST" })).json();
  $("#analyse").disabled = false;
  $("#analyse").textContent = "Detect shots";
  if (out.error) { $("#analysemeta").textContent = out.error; return; }

  $("#analysemeta").textContent =
    `${out.total} shots · ${out.fps} fps · crop ${out.crop.join(",")}`;
  $("#analysetable tbody").innerHTML = out.shots.map(s => {
    /* Displacement predicted effective coverage on every clip measured, so it
     * is worth stating as a prospect rather than leaving as a bare number. */
    const d = s.displacement || 0;
    const prospect = s.motion === "LOCKED" ? ["none", "var(--ink-faint)"]
      : d >= 12 ? ["strong", "var(--real)"]
      : d >= 5  ? ["some", "var(--accent)"]
      : ["weak", "var(--ink-faint)"];
    return `<tr>
      <td class="n">${s.shot}</td><td class="n">${s.start}</td>
      <td class="n">${s.frames}</td><td class="n">${s.seconds}</td>
      <td>${esc(s.motion)}</td><td class="n">${d.toFixed(2)}</td>
      <td style="color:${prospect[1]}">${prospect[0]}</td></tr>`;
  }).join("");
});

/* ------------------------------------------------------------------ render */

function options() {
  const o = {
    maxw: $("#maxw").value,
    frames_per_shot: $("#fps").value,
    max_shots: $("#maxshots").value,
    rotate: $("#rotate").value,
    wings_on_dark: $("#dark").value,
  };
  ["wing", "screen_width", "screen_height", "viewer_distance",
   "gate_geometry", "gate_full", "gate_narrow", "gate_detail", "gate_stale"]
    .forEach(k => { const v = $("#" + k).value.trim(); if (v) o[k] = v; });
  $$("[data-flag]").forEach(cb => { if (cb.checked) o[cb.dataset.flag] = "1"; });
  if ($("#remote").checked) o.remote = "1";
  return o;
}

/* Show the command this would run. The browser and the CLI are the same
 * pipeline, and printing the argv is how that stays checkable rather than
 * claimed. */
function showCommand() {
  if (!state.file) return;
  const o = options();
  const parts = ["python render.py", state.file.name,
                 "-o jobs/" + (state.job || "…"),
                 "--maxw " + o.maxw, "--frames-per-shot " + o.frames_per_shot];
  if (o.max_shots !== "0") parts.push("--max-shots " + o.max_shots);
  if (o.rotate !== "0") parts.push("--rotate " + o.rotate);
  if (o.wings_on_dark) parts.push("--wings-on-dark " + o.wings_on_dark);
  ["sources", "prefer_3d", "reason", "vision", "online"].forEach(f => {
    if (o[f]) parts.push("--" + f.replace(/_/g, "-"));
  });
  if (o.sfm) parts.push("--sfm jobs/" + (state.job || "…") + "/sfm");
  ["wing", "screen_width", "screen_height", "viewer_distance",
   "gate_geometry", "gate_full", "gate_narrow", "gate_detail", "gate_stale"]
    .forEach(k => { if (o[k]) parts.push("--" + k.replace(/_/g, "-") + " " + o[k]); });
  Object.entries(state.attachments || {}).forEach(([kind, files]) => {
    const flag = { other_cut: "--other-cut", also: "--also", context: "--context" }[kind];
    files.forEach(f => parts.push(flag + " " + f.split(/[\\/]/).pop()));
  });
  $("#cmd").textContent = parts.join(" ");
}
$$("#dark,#polishgen").forEach(el =>
  el && el.addEventListener("change", syncKeyPanel));
$$("#maxw,#fps,#maxshots,#rotate,#dark,#wing,#screen_width,#screen_height,#viewer_distance,#gate_geometry,#gate_full,#gate_narrow,#gate_detail,#gate_stale")
  .forEach(el => el.addEventListener("change", showCommand));

$("#run").addEventListener("click", async () => {
  if (!state.job) return;
  $("#run").disabled = true;
  $("#runstate").textContent = "starting";
  $("#live tbody").innerHTML = ""; $("#log").textContent = ""; state.shots = [];

  const res = await fetch(`/api/jobs/${state.job}/start`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options()),
  });
  const out = await res.json();
  if (out.error) {
    $("#runstate").textContent = out.error; $("#run").disabled = false; return;
  }
  $("#runstate").textContent = "running";
  watch(state.job);
});

function watch(id) {
  const src = new EventSource(`/api/jobs/${id}/events`);
  src.addEventListener("shot", e => {
    const rec = JSON.parse(e.data);
    state.shots.push(rec);
    $("#live tbody").insertAdjacentHTML("beforeend", `
      <tr><td class="n">${rec.shot}</td><td>${esc(rec.motion)}</td>
      <td>${esc(rec.backend)}</td><td class="n">${db(rec.geometry)}</td>
      <td><span class="verdict v-${esc(rec.state)}">${esc(rec.state)}</span></td>
      <td class="n">${pct(rec.effective)}</td></tr>`);
    unlock("review"); renderReview();
  });
  src.addEventListener("log", e => {
    const el = $("#log");
    el.textContent += JSON.parse(e.data) + "\n";
    el.scrollTop = el.scrollHeight;
  });
  src.addEventListener("end", e => {
    const d = JSON.parse(e.data);
    src.close();
    $("#run").disabled = false;
    $("#runstate").textContent = d.state === "done" ? "done" : `error — ${d.error || ""}`;
    if (d.summary) {
      state.summary = d.summary;
      state.shots = d.summary.per_shot || state.shots;
      unlock("report"); renderReview(); renderReport(); loadFiles();
      renderPolish({ state: "none", log: [], report: null, repaired: [] });
      go("report");
      loadJobs();
    }
  });
}

/* ------------------------------------------------------------------ review */

function renderReview() {
  const rows = state.shots;
  $("#shotcount").textContent = rows.length ? `${rows.length} shots` : "";
  $("#shots tbody").innerHTML = rows.length ? rows.map(r => `
    <tr data-shot="${r.shot}" class="${r.shot === state.sel ? "sel" : ""}">
      <td class="n">${r.shot}</td><td>${esc(r.motion)}</td><td>${esc(r.backend)}</td>
      <td class="n">${db(r.geometry)}</td>
      <td><span class="verdict v-${esc(r.state)}">${esc(r.state)}</span></td>
      <td class="n">${pct(r.effective)}</td>
      <td class="wrap">${esc(r.reasons || "")}</td></tr>`).join("")
    : `<tr><td colspan="7" class="empty">No shots yet.</td></tr>`;
}

$("#shots").addEventListener("click", e => {
  const id = e.target.closest("tr")?.dataset.shot;
  if (id == null) return;
  state.sel = Number(id);
  renderReview();
  renderDetail(state.shots.find(r => r.shot === state.sel));
});

const LABEL = `font:500 10px/1 var(--ui);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-faint);margin:16px 0 8px`;

/* The fields no interface has ever shown. action_gain is the important one: a
 * planner step is logged even at zero gain, so "the tool ran" and "the tool
 * landed pixels" read identically without it. */
function renderDetail(r) {
  if (!r) return;
  $("#detailid").textContent = `shot ${r.shot}`;
  const gains = r.action_gain || {};
  const gainRows = Object.keys(gains).length
    ? `<table><thead><tr><th>Planner action</th><th>Landed</th></tr></thead><tbody>${
        Object.entries(gains).map(([k, v]) => `<tr><td>${esc(k)}</td>
          <td class="n" style="color:${v > 0 ? "var(--real)" : "var(--ink-faint)"}">${v.toFixed(1)}%</td></tr>`).join("")
      }</tbody></table>`
    : `<p class="note">The planner did not run on this shot.</p>`;

  $("#detail").innerHTML = `
    <div class="readouts" style="border:1px solid var(--line)">
      <div class="readout"><div class="k">Verdict</div>
        <div class="v small"><span class="verdict v-${esc(r.state)}">${esc(r.state)}</span></div>
        <div class="n">${esc(r.reasons || "")}</div></div>
      <div class="readout"><div class="k">Geometry</div><div class="v">${db(r.geometry)}</div>
        <div class="n">gate is 20 dB</div></div>
      <div class="readout"><div class="k">Effective</div><div class="v">${pct(r.effective)}</div>
        <div class="n">NARROW at 25%, FULL at 55%</div></div>
    </div>
    <dl class="kv" style="margin-top:14px">
      <dt>motion</dt><dd>${esc(r.motion)}</dd>
      <dt>backend</dt><dd>${esc(r.backend)}</dd>
      <dt>coverage</dt><dd>${pct(r.coverage)}</dd>
      <dt>displacement</dt><dd>${r.displacement ?? "—"} px/frame</dd>
      <dt>second layer</dt><dd>${r.layer2 ?? "—"}</dd>
      <dt>layer disagreement</dt><dd>${r.layer_disagree ?? "—"} px</dd>
      <dt>frames</dt><dd>${r.frames ?? "—"} from ${r.start ?? "—"}</dd>
    </dl>
    ${r.provenance ? `<h4 style="${LABEL}">Where this shot's wall came from</h4>
      <dl class="kv">${["primary","recovered","donated","retrieved","referenced","directed","generated"]
        .filter(k => (r.provenance[k] || 0) > 0.0005)
        .map(k => `<dt>${k}</dt><dd>${((r.provenance[k]) * 100).toFixed(1)}%</dd>`).join("")}</dl>` : ""}
    <h4 style="${LABEL}">What the planner landed</h4>
    ${gainRows}
    ${r.reasoning ? `<h4 style="${LABEL}">Plan</h4>
      <pre class="log" style="max-height:180px">${esc(r.reasoning)}</pre>` : ""}
    <h4 style="${LABEL}">Direction</h4>
    <p class="note">Say what belongs off-frame here. It binds to this shot, and the
      pixels it drives are labelled <strong>DIRECTED</strong> — outside PHOTOGRAPHIC,
      because someone who knows the place is still not a camera.</p>
    <textarea id="notetext" placeholder="e.g. a fire escape, camera left"></textarea>
    <button class="ghost" id="notesave" style="margin-top:8px">Pin to shot ${r.shot}</button>
    <span class="note" id="notestatus"></span>`;

  $("#notesave").addEventListener("click", async () => {
    const text = $("#notetext").value.trim();
    if (!text || !state.job) return;
    const res = await fetch(`/api/jobs/${state.job}/notes`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shot: r.shot, text }),
    });
    const out = await res.json();
    $("#notestatus").textContent = out.ok ? " saved — read on the next run" : ` ${out.error}`;
  });
}

/* ------------------------------------------------------------------ report */

async function loadFiles() {
  if (!state.job) return;
  const files = await (await fetch(`/api/jobs/${state.job}/files`)).json();
  if (files.error) return;
  $("#files tbody").innerHTML = files.map(f => `
    <tr><td><a href="/api/jobs/${state.job}/file/${encodeURI(f.name)}"
      style="color:var(--ink-dim)">${esc(f.name)}</a></td>
    <td class="n">${mb(f.bytes)}</td></tr>`).join("");
}

function renderReport() {
  const s = state.summary;
  if (!s) return;
  $("#reportsource").textContent = s.source || "";

  const rows = s.per_shot || state.shots;
  const fps = 24;
  const lit = rows.filter(r => !["OFF", "GEN"].includes(r.state));
  const litFrames = lit.reduce((a, r) => a + (r.frames || 0), 0);
  const genFrames = rows.filter(r => r.state === "GEN").reduce((a, r) => a + (r.frames || 0), 0);
  const mins = f => f / fps / 60;

  /* Wing-on minutes is what this industry is judged on: early ScreenX titles
   * opened the side screens 20-30 minutes, current ones 60-100. The split into
   * filmed versus invented is the part nothing else reports. */
  $("#headline").innerHTML = `
    <div class="readout"><div class="k">Wings open</div>
      <div class="v">${mins(litFrames).toFixed(1)}<em> min</em></div>
      <div class="n">${lit.length} of ${rows.length} shots, from filmed periphery</div></div>
    <div class="readout"><div class="k">Wings invented</div>
      <div class="v">${mins(genFrames).toFixed(1)}<em> min</em></div>
      <div class="n">${s.wings_generated ?? 0} shots, generated</div></div>
    <div class="readout"><div class="k">Real wing pixels</div>
      <div class="v">${pct(s.mean_real_wing)}</div>
      <div class="n">photographed, both wings</div></div>
    <div class="readout"><div class="k">Mean effective</div>
      <div class="v">${pct(s.mean_effective)}</div>
      <div class="n">staleness and detail weighted</div></div>
    <div class="readout"><div class="k">Wall depth</div>
      <div class="v">${(s.extent?.depth ?? 0).toFixed(2)}<em> m</em></div>
      <div class="n">${pct(s.extent?.fraction_of_room)} of the room</div></div>`;

  const real = (s.mean_real_wing ?? 0) * 100;
  $("#provbar").innerHTML =
    `<i class="real" style="width:${real}%"></i><i class="invented" style="width:${100 - real}%"></i>`;
  /* Which rungs actually put pixels on a wall. "DONATED fired" was never a
   * checkable statement before: the label lived in the pixels and never in the
   * summary, so the interface could not show it either. */
  const RUNGS = ["primary", "recovered", "donated", "retrieved",
                 "referenced", "directed", "generated"];
  const PHOTO = new Set(["primary", "recovered", "donated", "retrieved"]);
  const prov = s.provenance || {};
  const rungs = RUNGS.filter(k => (prov[k] || 0) > 0.0005).map(k =>
    `<span><i class="swatch" style="background:${PHOTO.has(k) ? "var(--real)" : "var(--invented)"}"></i>${k} <b>${((prov[k] || 0) * 100).toFixed(1)}%</b></span>`);
  $("#provlegend").innerHTML = `
    <span><i class="swatch real"></i>filmed at this place <b>${real.toFixed(1)}%</b></span>
    <span><i class="swatch invented"></i>not filmed here <b>${(100 - real).toFixed(1)}%</b></span>`
    + (rungs.length ? `<span style="width:100%;height:1px"></span>` + rungs.join("") : "");

  const e = s.extent || {};
  $("#extent").innerHTML = Object.entries({
    "wall depth": `${(e.depth ?? 0).toFixed(2)} m`,
    "nearest wall point": `${(e.z_near ?? 0).toFixed(2)} m`,
    "fraction of room": pct(e.fraction_of_room),
    "binding axis": e.binding ?? "—",
    "wing ratio": s.wing_ratio ?? "—",
  }).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("#extentnote").textContent = e.binding === "vertical"
    ? "The vertical axis binds: the frame's height, not the wing width, decides how far the walls reach."
    : "The horizontal axis binds: wing width decides the reach.";

  /* A summary that does not say which bar it used cannot be compared with one
   * that used another. */
  $("#gateused").innerHTML = s.gate
    ? `<dt style="grid-column:1/-1;color:var(--ink-dim)">Judged against</dt>` +
      Object.entries(s.gate).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
    : "";

  $("#video").src = `/api/jobs/${state.job}/file/frameflow_demo.mp4`;
  $("#delivered").src = `/api/jobs/${state.job}/file/deliverable/master_widened.mp4`;

  $("#finaltable tbody").innerHTML = rows.map(r => `
    <tr><td class="n">${r.shot}</td><td>${esc(r.motion)}</td><td>${esc(r.backend)}</td>
    <td class="n">${db(r.geometry)}</td>
    <td><span class="verdict v-${esc(r.state)}">${esc(r.state)}</span></td>
    <td class="n">${pct(r.coverage)}</td><td class="n">${pct(r.effective)}</td>
    <td class="n">${r.displacement ?? "—"}</td>
    <td>${esc(r.actions || "—")}</td>
    <td class="wrap">${esc(r.reasons || "")}</td></tr>`).join("");
}

$("#dl").addEventListener("click", () => {
  if (state.job) location.href = `/api/jobs/${state.job}/file/frameflow_summary.json`;
});

/* ---------------------------------------------------------- finishing pass */

/* What a repaint costs in money. What it costs in truth is on the panel itself,
 * because that one applies to every generator including the free ones. */
const POLISH_COST = {
  "gemini-edit": "Billed per frame, and subject to your image quota.",
};

$("#polishgen").addEventListener("change", () => {
  const gen = $("#polishgen").value;
  $("#polishrun").textContent = gen ? "Settle, then repaint" : "Settle and inspect";
  const cost = POLISH_COST[gen];
  $("#polishcost").hidden = !cost;
  if (cost) {
    const faulted = ((state.polish || {}).report || {}).repairable || [];
    const n = faulted.length;
    $("#polishcost").textContent = cost + (n
      ? `  ${n} shot${n === 1 ? "" : "s"} are faulted, so leaving the box empty `
        + `pays for ${n}. Name the ones worth it.`
      : "  Name the shots worth paying for; empty means every faulted shot.");
  }
});

$("#polishrun").addEventListener("click", async () => {
  if (!state.job) return;
  $("#polishrun").disabled = true;
  $("#polishstate").textContent = "starting…";
  const res = await fetch(`/api/jobs/${state.job}/polish`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repair: $("#polishgen").value || null,
                           shots: $("#polishshots").value.trim() }),
  });
  const out = await res.json();
  if (out.error) {
    $("#polishrun").disabled = false;
    $("#polishstate").textContent = out.error;
    return;
  }
  pollPolish();
});

async function pollPolish() {
  if (!state.job) return;
  const p = await (await fetch(`/api/jobs/${state.job}/polish`)).json();
  if (p.error && !p.report) $("#polishstate").textContent = p.error;
  renderPolish(p);
  if (p.state === "running") { setTimeout(pollPolish, 900); return; }
  $("#polishrun").disabled = false;

  /* A repaint moves the headline down. Leaving the old figure on screen would
   * report pixels as filmed that a model drew a minute ago. */
  if ((p.repaired || []).length || (p.settled || []).length) {
    const job = await (await fetch(`/api/jobs/${state.job}`)).json();
    if (job.summary) {
      state.summary = job.summary;
      state.shots = job.summary.per_shot || state.shots;
      renderReport(); renderReview(); loadFiles();
      /* The repaint rewrote master_widened.mp4 at the same URL, so the player
       * would keep showing the cached, faulted cut and the polish would look
       * as though it had done nothing. */
      const v = `?v=${Date.now()}`;
      $("#delivered").src = `/api/jobs/${state.job}/file/deliverable/master_widened.mp4${v}`;
      $("#video").src = `/api/jobs/${state.job}/file/frameflow_demo.mp4${v}`;
      renderPolish(p);
    }
  }
}

function renderPolish(p) {
  state.polish = p;
  $("#polishstate").textContent =
    p.state === "running" ? (p.repairing ? "repainting…"
                             : p.settling ? "settling…" : "inspecting…")
      : p.state === "done" ? (p.error || "done") : $("#polishstate").textContent;

  const log = p.log || [];
  $("#polishlog").hidden = !log.length;
  $("#polishlog").textContent = log.join("\n");

  const found = (p.report && p.report.findings) || [];
  if (!found.length) {
    $("#polishout").innerHTML = `<div class="empty">${p.state === "running"
      ? "Sampling the wings and asking the model…" : "Not inspected yet."}</div>`;
    return;
  }
  const done = Object.fromEntries((p.repaired || []).map(r => [r.shot, r]));
  const eased = Object.fromEntries(((p.report || {}).settled || []).map(r => [r.shot, r]));

  /* The settle pass costs nothing and changes no number, so what it did has to
   * be shown as a movement rather than a state: "0.4% and it was 2.3%" is the
   * only way a reader can tell the pass worked. */
  const arrow = (was, now, fmt) =>
    was == null && now == null ? "—"
      : was == null || now == null ? fmt(now == null ? was : now)
        : `${fmt(was)} <span class="dim">→</span> <strong>${fmt(now)}</strong>`;
  const p2 = v => v == null ? "—" : pct(v);
  const x1 = v => v == null ? "—" : (+v).toFixed(2) + "×";

  $("#polishout").innerHTML = `<table>
    <thead><tr>
      <th>Shot</th><th>Verdict</th><th>Dark lines</th><th>Jitter</th><th>Seam</th>
      <th>Photographed</th><th>What the model sees</th><th>Outcome</th>
    </tr></thead>
    <tbody>${found.map(f => {
      const r = done[f.shot], e = eased[f.shot];
      const b = (e && e.before) || {}, a = (e && e.after) || {};
      const outcome = r
        ? `repainted &mdash; <strong>${pct(r.photographed_repainted)}</strong> of its
           photographed wing is now ${esc(r.label)}, driven by ${esc(r.driven_by)}`
        : e ? "settled — its own photography, nothing invented"
          : f.bad ? "faulted — repaint to fix"
            : f.note ? esc(f.note) : "acceptable";
      return `<tr>
        <td class="n">${f.shot}</td>
        <td><span class="verdict v-${esc(f.state)}">${esc(f.state)}</span></td>
        <td class="n">${e ? arrow(b.hairlines, a.hairlines, p2) : p2(f.hairlines)}</td>
        <td class="n">${e ? arrow(b.jitter, a.jitter, x1) : x1(f.jitter)}</td>
        <td class="n">${e ? arrow(b.seam, a.seam, x1) : x1(f.seam)}</td>
        <td class="n">${pct(f.photographed)}</td>
        <td class="wrap">${esc((f.claims || []).join("; ")) || "—"}</td>
        <td class="wrap">${outcome}</td></tr>`;
    }).join("")}</tbody></table>`;
}

/* Streak is measured, not asked -- so a reader can see whether the number and
 * the model agree before trusting either. */
async function loadPolish() {
  if (!state.job) return;
  const p = await (await fetch(`/api/jobs/${state.job}/polish`)).json();
  if (p && p.state && p.state !== "none") renderPolish(p);
}

loadCaps();
loadJobs();
