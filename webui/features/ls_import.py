"""Import to LS: ``tools/annotation/predictions_to_annotations.py``.

Takes a ``predictions.json`` -- an inference run's, or a ``<split>_preds.json``
-- and writes its boxes into the Label Studio project as annotations, ready to
correct. The file decides the scope: a task whose image is in it gets the boxes,
a task that is not in it is left alone.

Its own tab rather than a second button on the Label Studio one: that tab starts
a server and this is a job that finishes, and a feature gets one Start. It runs
in the ``data`` slot, so it neither waits for the GPU nor for the annotation
server it talks to (which has to be up, in the ``service`` slot).

**Dry run is on by default.** It writes into the live annotation project, so the
first click reports what would land and touches nothing; clear the box to commit.
"""

from ..core.jobs import DATA_SLOT
from ..core.paths import LS_IMPORT_SCRIPT, rel, resolve
from .base import Feature, Field, JobSpec, existing_file, flag, python_executable, text


DEFAULT_LABEL = "ship"  # the rectanglelabel the project's GT boxes use


def score(params, key, default):
    """A 0-1 form value."""
    raw = text(params, key) or str(default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be between 0 and 1, got {value}")
    return f"{value:g}"


class LabelStudioImportFeature(Feature):
    name = "ls-import"
    label = "Import to LS"
    slot = DATA_SLOT
    description = (
        "Turn a `predictions.json` into Label Studio annotations. Every task in "
        "the project is checked: the ones whose image is in the file get the "
        "predicted boxes as an annotation to correct, the ones that are not are "
        "skipped. Tasks that already have an annotation are left alone unless "
        "you say otherwise. **Dry run is ticked** — untick it to actually write. "
        "Label Studio must be running (*Label Studio* tab), and the settings come "
        "from `../data/.env`."
    )
    fields = [
        Field(
            "predictions",
            "Predictions json (-p)",
            kind="choice",
            source="predictions",
            prefer="inf_det",
            info="An inference run's `inf_det/predictions.json`, or a `<split>_preds.json`.",
        ),
        [
            Field("label", "Label to give the boxes (--label)", value=DEFAULT_LABEL),
            Field("min_score", "Min score (--min-score)", value="0", info="0 keeps everything in the file."),
        ],
        [
            Field(
                "existing",
                "Task already annotated (--existing)",
                kind="choice",
                choices=("skip", "replace", "append"),
                value="skip",
            ),
            Field(
                "coco",
                "Annotations json (--coco, optional)",
                info="Only for a `<split>_preds.json`: the COCO file its image_ids belong to.",
            ),
        ],
        [
            Field("dry_run", "Dry run (write nothing)", kind="flag", value="1"),
            Field(
                "undo",
                "Undo instead",
                kind="flag",
                info="Remove the boxes an earlier import created; ignores the file above.",
            ),
        ],
        Field("project", "Project id (optional)", info="Blank: LABEL_STUDIO_PROJECT_ID from the .env."),
    ]

    def build(self, params):
        cmd = [python_executable(params), LS_IMPORT_SCRIPT]

        if flag(params, "undo"):
            cmd.append("--undo")
        else:
            predictions = existing_file(params, "predictions", "predictions json")
            label = text(params, "label") or DEFAULT_LABEL
            cmd += [
                "-p",
                predictions,
                "--label",
                label,
                "--min-score",
                score(params, "min_score", 0),
                "--existing",
                text(params, "existing") or "skip",
            ]
            coco = existing_file(params, "coco", "annotations json", required=False)
            if coco:
                cmd += ["--coco", coco]

        if flag(params, "dry_run"):
            cmd.append("--dry-run")
        project = text(params, "project")
        if project:
            if not project.isdigit():
                raise ValueError(f"project id must be a number, got {project!r}")
            cmd += ["--project", project]

        source = "" if flag(params, "undo") else rel(resolve(text(params, "predictions")))
        meta = {"feature": self.name, "outdir": source, "cmd": " ".join(cmd)}
        return JobSpec(cmd, meta=meta)
