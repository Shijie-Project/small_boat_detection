"""Feature registry -- one module per tab.

To add a feature: drop a module here with a :class:`~webui.features.base.Feature`
subclass, list it in ``FEATURES`` below, and add the matching form spec in
``static/js/features/``.
"""

from .base import Feature, JobSpec
from .test import TestFeature
from .train import TrainFeature


FEATURES = [TrainFeature(), TestFeature()]


def all_features():
    return list(FEATURES)


def get(name):
    for feature in FEATURES:
        if feature.name == name:
            return feature
    return None
