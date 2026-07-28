"""Gradio dashboard for launching train / test runs from a browser.

    gradio webui/app.py      # with hot reload, for editing
    python -m webui          # plain run, from the project root

Layout: ``core/`` is what every feature shares (paths, discovery, the job
runner, the Gradio widgets), ``features/`` is one module per tab.
"""
