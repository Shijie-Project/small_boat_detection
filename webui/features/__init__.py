"""Feature registry -- one module per tab.

To add a feature: drop a module here with a :class:`~webui.features.base.Feature`
subclass (fields + ``build``), then list it in ``FEATURES`` below.
"""

from .base import Feature, Field, JobSpec
from .label_studio import LabelStudioFeature
from .regions import RegionFeature
from .test import TestFeature
from .tile import TileFeature
from .train import TrainFeature


FEATURES = [
    TrainFeature(),
    TestFeature(),
    TileFeature(),
    RegionFeature(),
    LabelStudioFeature(),
]


def all_features():
    return list(FEATURES)


def get(name):
    for feature in FEATURES:
        if feature.name == name:
            return feature
    return None
