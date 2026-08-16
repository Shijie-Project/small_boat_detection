# webui

Gradio dashboard for launching train / test runs, splitting satellite imagery, and the
Label Studio annotation server from a browser.

```bash
gradio webui/app.py      # hot reload: edit any file under webui/ and the page rebuilds
python -m webui          # plain run, from the project root
python webui/webui.py    # same thing
# http://127.0.0.1:8000  (override with WEBUI_HOST / WEBUI_PORT)
```

Jobs run from the project root, one per **slot**. A slot is what a feature competes with:
train and test share `run` and are serialised, because they both want every GPU; Label
Studio wants none, so it sits in `service` and can stay up across any number of training
runs; image splitting is CPU work and gets `data`. All three can run at once. The console
on the right has one tab per slot. Adding a slot is adding a name to `SLOTS` in
`core/jobs.py` — the layout follows.

A job started before a hot reload keeps going, keeps streaming, and can still be stopped
(see `_shared_jobs()` in `core/jobs.py`).

## Layout

```
webui/
├── app.py                 the Blocks: one tab per feature, one console per slot
├── core/                  shared by every feature
│   ├── paths.py           project root, config/ckpt/data dirs, safe path resolution
│   ├── discovery.py       what fills the dropdowns
│   ├── jobs.py            the slots, one subprocess each + its log ring buffer
│   └── ui.py              field rendering, console, start/stop wiring
└── features/              one module per tab
    ├── base.py            Feature/Field/JobSpec + the argument handling train & test share
    ├── train.py
    ├── test.py
    ├── tile.py            cut satellite imagery into tiles (tools/dataset/tile_satellite.py)
    ├── regions.py         the same cut, but only the rectangles you draw on the preview
    └── label_studio.py    the annotation server, same as `tools/starter.sh label-studio`
```

**Manual split** (`features/regions.py`) is the tab that is more than a form. It shrinks the
whole scene to a preview, turns two clicks into a rectangle in source pixels, and runs
`tile_satellite.py --regions <file>` so only those rectangles are cut — a scene that grids
into 3000 mostly-empty tiles gives 20 useful ones. The rectangles are saved as
`split_images/<image>/regions.json` beside the tiles, so the same cut repeats from the
command line. The click maths lives in `on_click`; the counts under the preview come from
the tiler's own window functions, so what the page promises is what the script writes.

The console repaints on a 1 s `gr.Timer`, so the page always reflects the real job —
reload the browser, open a second tab, or hot reload the server and it picks up again.
Every handler also repaints on the spot, and no button is ever greyed out: clicking Start
while something runs says so, rather than silently doing nothing.

Two hot-reload notes: the browser page belongs to the app that served it, so refresh it
after "Changes detected"; and a job that is running when you reload keeps the `core/jobs.py`
it started with (edits there land once it finishes).

## Adding a feature

1. `features/<name>.py`: subclass `Feature`, declare `fields`, implement
   `build(params) -> JobSpec`, reusing the helpers in `features/base.py`. Raise
   `ValueError` for bad input — it shows up as an error toast.
1. Register it in `features/__init__.py` (`FEATURES`).
1. Set `slot` if it does not belong with the GPU work (the default is `run`). A feature
   outside `run` gets its own Stop button in its tab, since the shared one below the tabs
   stops the run slot.

That is the whole story for a form-shaped feature. Fields render as a text box, a dropdown
(`kind="choice"`, from a discovery `source` or a literal `choices` tuple), or a checkbox
(`kind="flag"`, read back with `flag(params, name)`). For a tab that needs more than fields
(file upload, a gallery of results), override `Feature.panel(options)` and build the
components yourself; return `{name: component}` and the start button wires itself up.
