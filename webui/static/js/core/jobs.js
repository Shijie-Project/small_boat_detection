// Talking to the shared job runner: start, stop, and follow the log.

import { getJSON, postJSON } from "./api.js";

export function startJob(feature, params) {
  return postJSON(`/api/${feature}/start`, params);
}

export function stopJob() {
  return postJSON("/api/job/stop");
}

// Polls /api/job/status and reports new lines / state changes. The server
// restarts its line counter on every run, so a cursor that runs ahead of it
// means a new job began (possibly from another tab) and the view is stale.
export function watchJob({ onLines, onState, onReset, interval = 1000 }) {
  let cursor = 0;
  let running = null;
  let started = null;

  async function tick() {
    try {
      const status = await getJSON(`/api/job/status?after=${cursor}`);
      if (typeof status.cursor === "number" && status.cursor < cursor) {
        cursor = 0;
        onReset?.();
      } else {
        if (status.lines?.length) onLines(status.lines);
        if (typeof status.cursor === "number") cursor = status.cursor;
        const stamp = status.meta?.started ?? null;
        if (status.running !== running || stamp !== started) {
          running = status.running;
          started = stamp;
          onState(status.running, status.meta || {});
        }
      }
    } catch {
      /* server busy or restarting; try again on the next tick */
    }
    setTimeout(tick, interval);
  }

  tick();
  return { reset() { cursor = 0; } };
}
