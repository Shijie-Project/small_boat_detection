"""The dashboard itself: one tab per feature, one shared console.

Hot reload -- edit any file under ``webui/`` and the page rebuilds itself:

    gradio webui/app.py

Plain run (no reload):

    python -m webui
"""

import sys
from pathlib import Path


# gradio's reloader runs this file as a script, so the project has to be
# importable before anything below can be.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from webui.core.discovery import options
from webui.core.paths import HOST, PORT, ROOT
from webui.core.ui import CONSOLE_CSS, flatten, make_refresh, make_rescan, make_start, stop
from webui.features import all_features


TITLE = "Small Boat Detection"


def build_ui():
    opts = options()
    features = all_features()
    start_buttons = []
    choice_fields = []
    choice_components = []

    with gr.Blocks(title=TITLE, fill_width=True) as demo:
        gr.Markdown(f"### {TITLE} — train / test")

        with gr.Row():
            with gr.Column(scale=4, min_width=380):
                with gr.Tabs():
                    for feature in features:
                        with gr.Tab(feature.label):
                            if feature.description:
                                gr.Markdown(feature.description)
                            inputs = feature.panel(opts)
                            names = list(inputs)
                            button = gr.Button(f"Start {feature.label}", variant="primary")
                            button.click(
                                make_start(feature, names),
                                inputs=[inputs[name] for name in names],
                                outputs=[],
                            )
                            start_buttons.append(button)
                            for field in flatten(feature.fields):
                                if field.kind == "choice":
                                    choice_fields.append(field)
                                    choice_components.append(inputs[field.name])

                with gr.Row():
                    stop_button = gr.Button("Stop", variant="stop", interactive=False)
                    rescan_button = gr.Button("↻ Rescan", variant="secondary")
                gr.Markdown(f"<sub>root: `{ROOT}`</sub>")

            with gr.Column(scale=7):
                status = gr.Markdown("⚪ **idle**")
                log = gr.Textbox(
                    label="log",
                    lines=30,
                    max_lines=30,
                    autoscroll=True,
                    interactive=False,
                    show_label=False,
                    buttons=["copy"],
                    elem_classes="console",
                )

        stop_button.click(stop, outputs=[])
        rescan_button.click(make_rescan(choice_fields), outputs=choice_components)

        seen = gr.State(())
        gr.Timer(1.0).tick(
            make_refresh(len(start_buttons)),
            inputs=[seen],
            outputs=[status, log, *start_buttons, stop_button, seen],
            show_progress="hidden",
        )

    return demo


demo = build_ui()


def main():
    demo.launch(server_name=HOST, server_port=PORT, css=CONSOLE_CSS, show_error=True)


if __name__ == "__main__":
    main()
