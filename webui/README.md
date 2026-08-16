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

A form has one button, Start. Stop lives in the console tab instead, next to the log of
the thing it kills: a slot runs one job at a time, so a Stop per feature was several
buttons for one process, and the tab you started from is not necessarily the tab you are
looking at when you want it to end.

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
    ├── inference.py       detect on a folder of tiles (tools/inference/torch_inf_dir.py)
    ├── label_studio.py    the annotation server, same as `tools/starter.sh label-studio`
    └── ls_import.py       predictions.json -> annotations in the LS project
```

**Inference** runs on what **Split images** produced, so its dropdown lists tile folders
rather than images: a `split_images/<scene>/`, or `split_images/` itself for every scene at
once. That second case is the script's `--all`, which the tab always passes when the folder
holds scenes rather than tiles — a job started from a browser has no stdin, and the script
would otherwise stop to ask which scene it meant. It shares the `run` slot with train and
test, since it wants the same GPU.

**Import to LS** is the step after that: the `predictions.json` an inference run wrote
becomes annotations in the Label Studio project, one per task whose image the file
mentions. It talks to the running server rather than the database, so Label Studio has to
be up — it is the only feature that depends on another one. Dry run is ticked by default,
because this writes into live annotation work; `Undo instead` removes what an earlier
import created (every imported box carries an id starting with `pred`).

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
1. Set `slot` if it does not belong with the GPU work (the default is `run`). A slot it is
   the first to use gets its own console tab, Stop button included; nothing else to wire.

That is the whole story for a form-shaped feature. Fields render as a text box, a dropdown
(`kind="choice"`, from a discovery `source` or a literal `choices` tuple), or a checkbox
(`kind="flag"`, read back with `flag(params, name)`). For a tab that needs more than fields
(file upload, a gallery of results), override `Feature.panel(options)` and build the
components yourself; return `{name: component}` and the start button wires itself up.
