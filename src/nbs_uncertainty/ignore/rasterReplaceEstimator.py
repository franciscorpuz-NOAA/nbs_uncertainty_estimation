from copy import deepcopy

import numpy as np

from .surfaceEstimators import RasterEstimator
from scipy.stats import genextreme
from ..readers.bathymetryDataset import RasterBathymetry
from dataclasses import dataclass


class RasterReplaceEstimator(RasterEstimator):
    """
    Estimator for replacing-methods.
    """

    def estimate_surface(self):
        self.data_strip = super().pre_process()
