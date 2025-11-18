from .uncertaintyEstimators import UncertaintyEstimator
import numpy as np
from ..utils.helper import matrix2strip, strip2matrix
from typing import Callable
from scipy.stats import genextreme
from functools import partial

class SpatialEstimator(UncertaintyEstimator):
    """
    Estimator for spectrally-based uncertainty models.
    """

    def __init__(self,
                 data: np.ndarray,
                 interpolation_cell_distance: int,
                 multiple: int) -> None:
        data = matrix2strip(data)
        self.data = data
        self.interpolation_cell_distance = interpolation_cell_distance
        self.multiple = multiple
        self.min_window = 2

    @property
    def compute_uncertainty(self) -> np.ndarray:

        """
        Calculate the variance of the provided data array in parts.
        """
        data = self.data
        interpolation_cell_distance = self.interpolation_cell_distance
        multiple = self.multiple

        num_lines, num_samples = data.shape
        interpolation_cell_distance = ((interpolation_cell_distance - 2) // multiple) + 2
        # interpolation_cell_distance = num_samples
        difference_max = np.full((num_lines, interpolation_cell_distance), 0.0)
        # difference_mean = np.full((num_lines, interpolation_cell_distance), 0.0)
        # difference_std = np.full((num_lines, interpolation_cell_distance), 0.0)
        for win_len in range(self.min_window, interpolation_cell_distance // 2 + 1):
            num_convolutions = num_samples - win_len + 1
            differences = np.full((num_lines, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins = np.min(data[:, step:step + win_len], axis=-1)
                maxs = np.max(data[:, step:step + win_len], axis=-1)
                differences[:, step] = maxs - mins
            difference_stat[:, win_len - 1] = self.selectMethod().max(differences, axis=-1)
            # difference_mean[:, win_len-1] = np.mean(differences, axis =-1)
            # difference_std[:, win_len-1] = np.std(differences, axis =-1)
        # print(f"final convolutions for window length {win_len} is {num_convolutions}")
        difference_max[:, -win_len:] = np.fliplr(difference_max[:, :win_len])
        # difference_mean[:,-win_len:] = np.fliplr(difference_mean[:, :win_len])
        # difference_std[:,-win_len:] = np.fliplr(difference_std[:,:win_len])

        output_data = strip2matrix(difference_max)
        return output_data

    def selectMethod(self, method: str) -> Callable:
        method_dict = {
            'average': np.mean,
            'median': np.median,
            'max': np.max,
            'std': np.std,
            'gev': genextreme,
            'p95': partial(np.percentile, q=95, interpolation='nearest'),
            'p99': partial(np.percentile, q=99, interpolation='nearest'),
        }






