// Page bootstrap: build the tabs the backend advertises, render the selected
// feature's form, and drive the shared start / stop / log plumbing.

import { getJSON } from "./core/api.js";
import { LogConsole } from "./core/console.js";
import { $, clear, el } from "./core/dom.js";
import { renderForm } from "./core/form.js";
import { startJob, stopJob, watchJob } from "./core/jobs.js";
import { FEATURE_UI } from "./features/index.js";

const log = new LogConsole($("log"));
const state = {
  options: {},      // from /api/options
  features: [],     // from /api/features
  current: null,    // selected feature name
  form: null,       // renderForm handle for the selected feature
  values: {},       // per-feature field values, kept across tab switches
  running: false,
};

// --- rendering ------------------------------------------------------------ //
function buildTabs() {
  const tabs = clear($("tabs"));
  for (const feature of state.features) {
    const tab = el("div", { class: "tab", title: feature.description || "" }, feature.label || feature.name);
    tab.onclick = () => selectFeature(feature.name);
    tab.dataset.feature = feature.name;
    tabs.appendChild(tab);
  }
}

function selectFeature(name) {
  const ui = FEATURE_UI[name];
  if (!ui) {
    log.append([`[webui] no form defined for feature "${name}"`]);
    return;
  }
  if (state.form && state.current) state.values[state.current] = state.form.read();

  state.current = name;
  for (const tab of $("tabs").children) tab.classList.toggle("active", tab.dataset.feature === name);
  state.form = renderForm($("form"), ui.fields, state.options, state.values[name] || {});
}

function renderMeta(meta = {}) {
  const rows = [
    ["python", state.options.python],
    ["root", state.options.root],
    ["last run", meta.cmd],
    ["output", meta.outdir],
    ["started", meta.started],
  ].filter(([, value]) => value);

  const box = clear($("meta"));
  for (const [key, value] of rows) {
    box.appendChild(el("div", {}, [el("span", { class: "k" }, `${key}: `), el("code", {}, String(value))]));
  }
}

function setRunning(running, meta = {}) {
  state.running = running;
  $("run").disabled = running;
  $("stop").disabled = !running;
  $("dot").classList.toggle("on", running);
  if (running) {
    $("status").textContent = `running · ${meta.feature || state.current || ""}`;
  } else if (meta.exit_code !== null && meta.exit_code !== undefined) {
    $("status").textContent = `exited (${meta.exit_code})`;
  } else {
    $("status").textContent = "idle";
  }
  renderMeta(meta);
}

// --- actions -------------------------------------------------------------- //
async function run() {
  if (!state.current || !state.form) return;
  const params = state.form.read();
  state.values[state.current] = params;
  try {
    const result = await startJob(state.current, params);
    watcher.reset();
    log.clear();
    setRunning(true, result.meta || {});
  } catch (err) {
    log.append([`[webui] ${err.message}`]);
  }
}

async function stop() {
  try {
    await stopJob();
  } catch (err) {
    log.append([`[webui] ${err.message}`]);
  }
}

// --- boot ----------------------------------------------------------------- //
const watcher = watchJob({
  onLines: (lines) => log.append(lines),
  onState: (running, meta) => setRunning(running, meta),
  onReset: () => log.clear(),
});

$("run").onclick = run;
$("stop").onclick = stop;
$("clear").onclick = () => log.clear();

(async function boot() {
  const [features, options] = await Promise.all([getJSON("/api/features"), getJSON("/api/options")]);
  state.features = features.features || [];
  state.options = options;
  buildTabs();
  renderMeta();
  if (state.features.length) selectFeature(state.features[0].name);
})();
