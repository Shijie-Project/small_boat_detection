"""Split images: ``tools/dataset/tile_satellite.py``, the annotation prep step.

Cuts the purchased satellite imagery into 1024 tiles. It is CPU work, so it
runs in the ``data`` slot and neither waits for the GPUs nor holds them up.
"""

from ..core.jobs import DATA_SLOT
from ..core.paths import SATELLITE_DIR, TILE_SCRIPT, rel, resolve
from .base import Feature, Field, JobSpec, flag, positive_int, python_executable, text, whole_int


BLACK_PCT_DEFAULT = 99  # a tile this black is background, not water


SPLIT_DIRNAME = "split_images"  # must match tile_satellite.py


def source_path(params):
    """The image or folder to cut, as listed in the dropdown."""
    value = text(params, "source")
    if not value:
        raise ValueError("source image or folder is required")
    path = resolve(value)
    if path is None:
        raise ValueError(f"source is outside the project: {value}")
    if not path.exists():
        raise ValueError(f"source not found: {value}")
    return value, path


def output_root(params, source):
    """Where the tiles go: the form's answer, else what the script would pick."""
    value = text(params, "output")
    if not value:
        parent = source if source.is_dir() else source.parent
        return "", rel(parent / SPLIT_DIRNAME)
    if resolve(value) is None:
        raise ValueError(f"output root is outside the project: {value}")
    return value, rel(value)


def black_fraction(params):
    """The "how black is background" percentage, as the fraction the script takes."""
    percent = whole_int(params, "black_pct", BLACK_PCT_DEFAULT, minimum=1)
    if percent > 100:
        raise ValueError(f"black threshold must be <= 100, got {percent}")
    return f"{percent / 100:.4f}"


class TileFeature(Feature):
    name = "tile"
    label = "Split images"
    slot = DATA_SLOT
    description = (
        "Pad each satellite image out to a whole number of tiles and cut it into "
        "a grid, at original resolution. Tiles land in `<folder>/split_images/"
        "<image>/`, next to a `tiles.json` recording where each one came from. "
        "Tiles that are nothing but the black background are thrown away as they "
        "are cut, so nothing empty reaches the annotation step."
    )
    fields = [
        Field(
            "source",
            "Image or folder (-i)",
            kind="choice",
            source="satellite",
            value=rel(SATELLITE_DIR),
            info="The folder cuts every image in it.",
        ),
        Field(
            "output",
            "Output root (-o, optional)",
            info=f"Blank: <source folder>/{SPLIT_DIRNAME}/",
        ),
        [
            Field("tile", "Tile size (--tile)", value="1024"),
            Field("overlap", "Overlap (--overlap)", value="0", info="stride = tile - overlap"),
        ],
        [
            Field("format", "Tile format", kind="choice", choices=("png", "jpg"), value="png"),
            Field(
                "skip_blank",
                "Skip blank tiles",
                kind="flag",
                info="Drop tiles that are one flat colour (padding, no-data).",
            ),
        ],
        [
            Field(
                "drop_black",
                "Delete black background",
                kind="flag",
                value="1",
                info="Throw away the all-black tiles around the imagery.",
            ),
            Field(
                "black_pct",
                "Black threshold (%)",
                value=str(BLACK_PCT_DEFAULT),
                info="A tile at least this black goes; 100 keeps anything with a lit pixel.",
            ),
        ],
    ]

    def build(self, params):
        source, path = source_path(params)
        output, outdir = output_root(params, path)
        tile = positive_int(params, "tile", 1024)
        overlap = whole_int(params, "overlap", 0, minimum=0)
        if overlap >= tile:
            raise ValueError(f"overlap ({overlap}) must be smaller than tile ({tile})")
        image_format = text(params, "format") or "png"

        cmd = [
            python_executable(params),
            TILE_SCRIPT,
            "-i",
            source,
            "--tile",
            str(tile),
            "--overlap",
            str(overlap),
            "--format",
            image_format,
        ]
        if output:
            cmd += ["-o", output]
        if flag(params, "skip_blank"):
            cmd.append("--skip-blank")
        if flag(params, "drop_black"):
            cmd += ["--drop-black", "--black-frac", black_fraction(params)]

        meta = {"feature": self.name, "source": source, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, meta=meta)
