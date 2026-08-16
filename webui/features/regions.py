"""Manual split: mark the parts of a scene worth cutting, then cut only those.

The automatic grid cuts everything, and a satellite scene is mostly empty
water -- thousands of tiles nobody wants to annotate. This tab shows the whole
scene shrunk to fit the page, turns the rectangles you click on it back into
full-resolution pixel coordinates, and hands ``tile_satellite.py`` just those
windows through its ``--regions`` file.

Two clicks make a rectangle: the first corner, then the opposite one. The table
underneath is the same regions in source pixels and is editable, so a rectangle
clicked roughly can be typed exactly.
"""

import importlib.util
import json

import cv2
import gradio as gr

from ..core.jobs import DATA_SLOT
from ..core.paths import ROOT, TILE_SCRIPT, rel, resolve
from ..core.ui import render_fields
from .base import Feature, Field, JobSpec, flag, positive_int, python_executable, text, whole_int
from .tile import BLACK_PCT_DEFAULT, black_fraction, output_root, source_path


PREVIEW_MAX = 1600  # longest side of the preview; a scene is far bigger
MIN_SIDE = 8  # preview px: anything smaller is a double click, not a rectangle
BOX_COLOUR = (0, 220, 255)  # RGB -- the preview is handed to gradio as RGB
PENDING_COLOUR = (60, 255, 60)

_TILER = None


def tiler():
    """``tools/dataset/tile_satellite.py`` as a module.

    The preview has to be read exactly the way the cut will read it -- the SAR
    percentile stretch above all -- and the tile counts shown here have to be
    the counts the script will produce, so both come from the script itself
    rather than from a copy that can drift. It is a script, not a package, so
    it is loaded by path.
    """
    global _TILER
    if _TILER is None:
        path = ROOT / TILE_SCRIPT
        spec = importlib.util.spec_from_file_location("tile_satellite", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TILER = module
    return _TILER


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
def preview_of(path):
    """``(RGB preview, width, height, scale)`` -- scale is source px per preview px."""
    image = tiler().load_image(str(path))
    height, width = image.shape[:2]
    scale = max(1.0, max(width, height) / PREVIEW_MAX)
    size = (max(1, round(width / scale)), max(1, round(height / scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB), width, height, scale


def draw(base, view, regions, pending):
    """The preview with every region outlined and the pending corner marked."""
    if base is None:
        return None
    canvas = base.copy()
    scale = view["scale"]
    for index, (x, y, w, h) in enumerate(regions, 1):
        corner = (int(x / scale), int(y / scale))
        far = (int((x + w) / scale), int((y + h) / scale))
        cv2.rectangle(canvas, corner, far, BOX_COLOUR, 2)
        cv2.putText(
            canvas,
            f"R{index}",
            (corner[0] + 4, corner[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOUR,
            2,
            cv2.LINE_AA,
        )
    if pending:
        spot = (int(pending[0] / scale), int(pending[1] / scale))
        cv2.drawMarker(canvas, spot, PENDING_COLOUR, cv2.MARKER_CROSS, 24, 2)
    return canvas


def counts(view, regions, tile, overlap, whole):
    """``(tiles the full grid would make, tiles these regions make)``."""
    module = tiler()
    try:
        full = len(module.grid_windows(view["width"], view["height"], tile, overlap))
        mine = len(module.region_windows(regions, tile, overlap, whole)) if regions else 0
    except ValueError:  # overlap >= tile, which the form reports on Start
        return 0, 0
    return full, mine


def status(view, regions, pending, tile, overlap, whole):
    """The line under the preview: what is loaded, what is marked, what it costs."""
    if not view:
        return "_Pick an image and press **Load preview**._"
    head = f"`{view['source']}` — {view['width']}×{view['height']} px, preview 1:{view['scale']:.0f}"
    if pending:
        return f"{head} · **first corner at ({pending[0]}, {pending[1]})** — click the opposite one"
    if not regions:
        return f"{head} · _click two opposite corners to add a region_"
    full, mine = counts(view, regions, tile, overlap, whole)
    saved = f" instead of {full}" if full else ""
    return f"{head} · **{len(regions)} region(s) → {mine} tiles**{saved}"


# --------------------------------------------------------------------------- #
# Region bookkeeping
# --------------------------------------------------------------------------- #
def as_int(value, default, minimum=0):
    """A form number for the tile counts on screen -- half-typed is not an error.

    Start reports a bad tile size properly; a preview redraw only needs
    something sane to count with.
    """
    try:
        return max(minimum, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


def sizes(tile, overlap):
    """``(tile, overlap)`` as the counts on screen should read them."""
    return as_int(tile, 1024, minimum=1), as_int(overlap, 0)


def snapped(box, tile, image_size):
    """A region grown out to whole tiles, so its cut lines up with the auto grid."""
    x, y, w, h = box
    width, height = image_size
    x0 = (x // tile) * tile
    y0 = (y // tile) * tile
    x1 = min(width, -(-(x + w) // tile) * tile)
    y1 = min(height, -(-(y + h) // tile) * tile)
    return [x0, y0, x1 - x0, y1 - y0]


def clipped(box, view):
    """One region held inside the image; ``None`` when nothing is left of it."""
    x, y, w, h = (int(round(float(value))) for value in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(view["width"], x + w), min(view["height"], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def rows_of(regions):
    """Regions as the table shows them."""
    return [list(region) for region in regions]


def parse_rows(rows, view):
    """The table back into regions, dropping the blank and the impossible."""
    regions = []
    for row in rows if rows is not None else []:
        values = list(row)[:4]
        if len(values) < 4 or any(value is None or str(value).strip() == "" for value in values):
            continue
        try:
            box = clipped(values, view)
        except ValueError:
            continue
        if box:
            regions.append(box)
    return regions


def regions_path(outdir, stem):
    """Where this image's ``regions.json`` lives: beside the tiles it cuts."""
    root = resolve(outdir)
    if root is None:
        raise ValueError(f"output root is outside the project: {outdir}")
    return root / stem / "regions.json"


def write_regions(target, source, view, regions):
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "width": view["width"],
        "height": view["height"],
        "regions": [{"x": x, "y": y, "w": w, "h": h} for x, y, w, h in regions],
    }
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=1)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def load_preview(source, tile, overlap, whole):
    """Read the picked image and start with a clean set of regions."""
    if not source:
        raise gr.Error("pick an image first")
    path = resolve(source)
    if path is None or not path.exists():
        raise gr.Error(f"image not found: {source}")
    if path.is_dir():
        raise gr.Error("pick a single image, not the folder — regions are per image")

    base, width, height, scale = preview_of(path)
    view = {"source": source, "width": width, "height": height, "scale": scale}
    tile, overlap = sizes(tile, overlap)
    gr.Info(f"{width}×{height} loaded")
    return base, base, view, [], None, rows_of([]), status(view, [], None, tile, overlap, whole)


def on_click(base, view, regions, pending, tile, overlap, snap, whole, evt: gr.SelectData):
    """First click sets a corner, second one turns the pair into a region."""
    if not view:
        gr.Warning("load a preview first")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    tile, overlap = sizes(tile, overlap)
    scale = view["scale"]
    click = (
        min(view["width"] - 1, max(0, int(round(evt.index[0] * scale)))),
        min(view["height"] - 1, max(0, int(round(evt.index[1] * scale)))),
    )

    if pending is None:
        return (
            draw(base, view, regions, click),
            regions,
            click,
            gr.skip(),
            status(view, regions, click, tile, overlap, whole),
        )

    x0, x1 = sorted((pending[0], click[0]))
    y0, y1 = sorted((pending[1], click[1]))
    if (x1 - x0) < MIN_SIDE * scale or (y1 - y0) < MIN_SIDE * scale:
        gr.Warning("those two corners are on top of each other — starting again from this one")
        return (
            draw(base, view, regions, click),
            regions,
            click,
            gr.skip(),
            status(view, regions, click, tile, overlap, whole),
        )

    box = [x0, y0, x1 - x0, y1 - y0]
    if snap:
        box = snapped(box, tile, (view["width"], view["height"]))
    regions = list(regions) + [box]
    return (
        draw(base, view, regions, None),
        regions,
        None,
        rows_of(regions),
        status(view, regions, None, tile, overlap, whole),
    )


def on_table(base, view, rows, tile, overlap, whole):
    """A hand-typed table wins over what was clicked -- half-drawn rectangle included."""
    if not view:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    tile, overlap = sizes(tile, overlap)
    regions = parse_rows(rows, view)
    return (draw(base, view, regions, None), regions, None, status(view, regions, None, tile, overlap, whole))


def make_edit(change):
    """Undo / Clear share everything except what they do to the list."""

    def edit(base, view, regions, tile, overlap, whole):
        if not view:
            gr.Warning("load a preview first")
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        tile, overlap = sizes(tile, overlap)
        regions = change(list(regions))
        return (
            draw(base, view, regions, None),
            regions,
            None,
            rows_of(regions),
            status(view, regions, None, tile, overlap, whole),
        )

    return edit


def save_regions(view, regions, output):
    """Write ``regions.json`` now, for a run from the command line."""
    if not view:
        raise gr.Error("load a preview first")
    if not regions:
        raise gr.Error("no regions to save")
    source = resolve(view["source"])
    _, outdir = output_root({"output": output}, source)
    target = regions_path(outdir, source.stem)
    write_regions(target, view["source"], view, regions)
    gr.Info(f"saved {len(regions)} region(s) → {rel(target)}")


# --------------------------------------------------------------------------- #
class RegionFeature(Feature):
    name = "regions"
    label = "Manual split"
    slot = DATA_SLOT
    description = (
        "Cut only the parts of a scene worth annotating. Load an image, click "
        "two opposite corners for each region you care about, then Start: the "
        "tiles come out of those rectangles only, at original resolution, into "
        "the same `split_images/<image>/` as the automatic split. The regions "
        "are saved next to them as `regions.json`, so the same cut can be "
        "repeated from the command line."
    )
    fields = [
        Field(
            "source",
            "Image (-i)",
            kind="choice",
            source="satellite",
            info="One image — regions belong to the scene they were drawn on.",
        ),
        Field("output", "Output root (-o, optional)", info="Blank: <image folder>/split_images/"),
        [
            Field("tile", "Tile size (--tile)", value="1024"),
            Field("overlap", "Overlap (--overlap)", value="0", info="stride = tile - overlap"),
        ],
        [
            Field("format", "Tile format", kind="choice", choices=("png", "jpg"), value="png"),
            Field(
                "snap",
                "Snap regions to the tile grid",
                kind="flag",
                value="1",
                info="Grow each rectangle out to whole tiles, so nothing is padded inside it.",
            ),
        ],
        [
            Field(
                "whole",
                "Cut each region whole",
                kind="flag",
                info="One image per region at its own size, instead of a grid inside it.",
            ),
            Field(
                "drop_black",
                "Delete black background",
                kind="flag",
                value="1",
                info="Throw away the all-black tiles a region may reach into.",
            ),
        ],
        Field(
            "black_pct",
            "Black threshold (%)",
            value=str(BLACK_PCT_DEFAULT),
            info="A tile at least this black goes; 100 keeps anything with a lit pixel.",
        ),
    ]

    def panel(self, options):
        inputs = {}
        with gr.Row():
            inputs.update(render_fields([self.fields[0]], options))
            load = gr.Button("Load preview", variant="secondary", scale=0, min_width=150)

        preview = gr.Image(
            label="click two opposite corners",
            type="numpy",
            interactive=False,
            height=560,
        )
        note = gr.Markdown("_Pick an image and press **Load preview**._")
        with gr.Row():
            undo = gr.Button("↶ Undo region", size="sm")
            clear = gr.Button("Clear regions", size="sm")
            save = gr.Button("Save regions.json", size="sm")
        with gr.Accordion("Regions (source pixels — editable)", open=False):
            table = gr.Dataframe(
                headers=["x", "y", "w", "h"],
                datatype="number",
                column_count=(4, "fixed"),
                type="array",
                value=[],
                interactive=True,
            )

        inputs.update(render_fields(self.fields[1:], options))

        base = gr.State(None)  # the untouched preview, redrawn on every change
        view = gr.State({})  # source path and the scale back to full resolution
        pending = gr.State(None)  # the first corner, waiting for its opposite
        inputs["view"] = view
        inputs["regions"] = gr.State([])

        source, output = inputs["source"], inputs["output"]
        tile, overlap = inputs["tile"], inputs["overlap"]
        snap, whole = inputs["snap"], inputs["whole"]
        regions = inputs["regions"]

        load.click(
            load_preview,
            inputs=[source, tile, overlap, whole],
            outputs=[preview, base, view, regions, pending, table, note],
        )
        preview.select(
            on_click,
            inputs=[base, view, regions, pending, tile, overlap, snap, whole],
            outputs=[preview, regions, pending, table, note],
        )
        table.input(
            on_table,
            inputs=[base, view, table, tile, overlap, whole],
            outputs=[preview, regions, pending, note],
        )
        edit_io = {
            "inputs": [base, view, regions, tile, overlap, whole],
            "outputs": [preview, regions, pending, table, note],
        }
        undo.click(make_edit(lambda items: items[:-1]), **edit_io)
        clear.click(make_edit(lambda items: []), **edit_io)
        save.click(save_regions, inputs=[view, regions, output])
        return inputs

    def build(self, params):
        source, path = source_path(params)
        if path.is_dir():
            raise ValueError("pick a single image, not the folder — regions are per image")
        regions = list(params.get("regions") or [])
        if not regions:
            raise ValueError("no regions marked — click two opposite corners on the preview")

        view = params.get("view") or {}
        if view.get("source") != source:
            raise ValueError(
                f"the regions belong to {view.get('source', 'another image')} — "
                "press Load preview for the image you picked"
            )

        output, outdir = output_root(params, path)
        tile = positive_int(params, "tile", 1024)
        overlap = whole_int(params, "overlap", 0, minimum=0)
        if overlap >= tile:
            raise ValueError(f"overlap ({overlap}) must be smaller than tile ({tile})")
        image_format = text(params, "format") or "png"

        target = regions_path(outdir, path.stem)
        write_regions(target, source, view, regions)

        cmd = [
            python_executable(params),
            TILE_SCRIPT,
            "-i",
            source,
            "--regions",
            rel(target),
            "--tile",
            str(tile),
            "--overlap",
            str(overlap),
            "--format",
            image_format,
        ]
        if output:
            cmd += ["-o", output]
        if flag(params, "whole"):
            cmd.append("--whole-regions")
        if flag(params, "drop_black"):
            cmd += ["--drop-black", "--black-frac", black_fraction(params)]

        meta = {
            "feature": self.name,
            "source": source,
            "outdir": rel(target.parent),
            "cmd": " ".join(cmd),
        }
        notes = [f"[webui] {len(regions)} region(s) -> {rel(target)}"]
        return JobSpec(cmd, meta=meta, notes=notes)
