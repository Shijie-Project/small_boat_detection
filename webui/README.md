# webui

Gradio dashboard for launching train / test runs from a browser.

```bash
gradio webui/app.py      # hot reload: edit any file under webui/ and the page rebuilds
python -m webui          # plain run, from the project root
python webui/webui.py    # same thing
# http://127.0.0.1:8000  (override with WEBUI_HOST / WEBUI_PORT)
```

Jobs run from the project root, one at a time — training and testing both want the GPUs.
A run started before a hot reload keeps going, keeps streaming, and can still be stopped
(see `_shared_job()` in `core/jobs.py`).

## Layout

```
webui/
├── app.py                 the Blocks: one tab per feature, one shared console
├── core/                  shared by every feature
│   ├── paths.py           project root, config/ckpt/data dirs, safe path resolution
│   ├── discovery.py       what fills the dropdowns
│   ├── jobs.py            the single subprocess + its log ring buffer
│   └── ui.py              field rendering, console, start/stop wiring
└── features/              one module per tab
    ├── base.py            Feature/Field/JobSpec + the argument handling train & test share
    ├── train.py
    └── test.py
```

The console repaints on a 1 s `gr.Timer`, so the page always reflects the real job —
reload the browser, open a second tab, or hot reload the server and it picks up again.

## Adding a feature

1. `features/<name>.py`: subclass `Feature`, declare `fields`, implement
   `build(params) -> JobSpec`, reusing the helpers in `features/base.py`. Raise
   `ValueError` for bad input — it shows up as an error toast.
1. Register it in `features/__init__.py` (`FEATURES`).

That is the whole story for a form-shaped feature. For a tab that needs more than fields
(file upload, a gallery of results), override `Feature.panel(options)` and build the
components yourself; return `{name: component}` and the start button wires itself up.
