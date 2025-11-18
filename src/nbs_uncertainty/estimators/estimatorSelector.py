from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .uncertaintyEstimators import UncertaintyEstimator
from ..readers.bathymetry import (Bathymetry,
                                  RasterBathymetry,
                                  CSVBathymetry,
                                  BPSBathymetry)
from ..utils.helper import (get_column_indices,
                            matrix2strip,
                            strip2matrix,
                            subsample)
from .spectralEstimators import amplitude_v2,psd_v2


class EstimatorSelector:
    """
    Factory for creating surface estimators based on bathymetry data type.
    """
    estimators_dict = {
        'raster': 'RasterEstimator',
        'csv': 'CSVEstimator',
        'bps': 'BPSEstimator'
    }

    @staticmethod
    def create_estimator(bathy_file: Bathymetry) -> 'SurfaceEstimator':
        """
        Selects and returns an appropriate surface estimator based on the data type.
        """
        data_type = bathy_file.data_type
        estimator_class_name = EstimatorSelector.estimators_dict.get(data_type)
        if estimator_class_name:
            estimator_class = globals()[estimator_class_name]
            return estimator_class(bathy_file)
        else:
            raise RuntimeError(f"Unrecognized Bathymetry type: {data_type}")


class SurfaceEstimator(ABC):
    """
    Abstract class for estimating surface given sampling method
    """

    bathy_file: Bathymetry
    interpolation: np.ndarray
    residual: np.ndarray


    def __init__(self, bathy_file: Bathymetry) -> None:
        self.bathy_file = bathy_file

    @abstractmethod
    def estimate_surface(self) -> np.ndarray:
        """
        Estimates the surface from the bathymetry data.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_residual(self) -> None:
        """
        Computes the residual between the bathymetry data and its interpolation
        :return:
        """
        raise NotImplementedError



class RasterEstimator(SurfaceEstimator):
    """
    Adapter class for estimating raster surfaces
    """

    method: str = None
    resolution: int = 1
    linespacing_m: int = 1
    current_multiple: int = 1
    max_multiple: int = 1
    data_array: np.ndarray
    column_indices: np.ndarray
    subsampling: str = None # possible values: "along, across, x_tiled"

    def __init__(self, bathy_file: RasterBathymetry) -> None:
        super().__init__(bathy_file)
        self.data_array = bathy_file.data
        self.resolution = bathy_file.resolution

    def __repr__(self):
        return f"RasterEstimator({self.bathy_file})"

    def set_method(self, method: str) -> None:
        self.method = method

    def set_subsampling_method(self, method: str):
        self.subsampling = method

    def set_max_multiple(self, max_multiple: Union[int, list[int]]) -> None:
        if type(max_multiple) == int:
            self.max_multiple = max_multiple
        else:
            self.max_multiple = np.max(max_multiple)

    def set_linespacing_m(self, linespacing_m: int) -> None:
        self.linespacing_m = linespacing_m

        # trigger the computation of column indices based on linespacing
        self.column_indices = get_column_indices(self.data_array.shape[1],
                                                 self.resolution,
                                                 self.linespacing_m,
                                                 self.max_multiple)

        # compute residual surface to serve as truth value
        self.residual, self.interpolation = self.compute_residual(self.data_array)

    def compute_residual(self,
                         input_array: np.ndarray,
                         input_multiple: int = 1) -> tuple[np.ndarray, np.ndarray]:

        # convert arrays into a strip to optimize vectorization
        data_in_strip = matrix2strip(input_array,
                                    column_indices=self.column_indices,
                                    multiple=input_multiple)

        # perform calculations for interpolation and residual
        interpolation_in_strip = np.linspace(start=data_in_strip[:, 0],
                                         stop=data_in_strip[:, -1],
                                         num=data_in_strip.shape[1])
        interpolated_strip = interpolation_in_strip.T
        residual_in_strip = data_in_strip - interpolated_strip

        # convert "strips" back to original data dimensions
        output_interpolation = strip2matrix(interpolation_in_strip,
                                          original_shape=input_array.shape,
                                          column_indices=self.column_indices)

        output_residual = strip2matrix(data_strip=residual_in_strip,
                                     original_shape=input_array.shape,
                                     column_indices=self.column_indices)

        return output_residual, output_interpolation

    def set_current_multiple(self, input_multiple: int) -> None:
        self.current_multiple = input_multiple

    def estimate_surface(self) -> np.ndarray:

        # compute new indices based on current linespacing and multiple
        current_column_indices = get_column_indices(self.data_array.shape[1],
                                                 self.resolution,
                                                 self.linespacing_m,
                                                 self.current_multiple)
        # subsample depth data
        subsampled_data_array = subsample(data = self.data_array,
                                          column_indices = current_column_indices,
                                          method = self.subsampling)
        subsampled_residual, _ = self.compute_residual(subsampled_data_array,
                                                       self.current_multiple)

        uncertainty_function = UncertaintyMethodSelector.select_estimator(self.method)

        uncertainty_output = uncertainty_function.compute_uncertainty(subsampled_residual)

        # data_strip_of_multiple = matrix2strip(subsampled_residual,
        #                                     self.column_indices,
        #                                     multiple=self.current_multiple)




        return uncertainty_output



# For Future Implementation
class CSVEstimator(SurfaceEstimator):
    """"
    Adapter class for estimating csv data
    """
    # For future implementation
    def __init__(self, bathy_file: CSVBathymetry) -> None:
        super().__init__(bathy_file)

    def estimate_surface(self) -> np.ndarray:
        return super().estimate_surface()


# For Future Implementation
class BPSEstimator(SurfaceEstimator):
    """"
    Adapter class for estimating bps data
    """
    # For future implementation
    def __init__(self, bathy_file: BPSBathymetry) -> None:
        super().__init__(bathy_file)

    def estimate_surface(self) -> np.ndarray:
        return super().estimate_surface()



class UncertaintyMethodSelector:
    """
    Class template for computing uncertainty estimation
    """
    uncertainty_methods = {
        # 'amplitude_v1': 'amplitude_v1',
        # 'psd_v1': 'psd_v1',
        # 'spectrum_v1': 'spectrum_v1',
        'amplitude_v2': amplitude_v2,
        'psd_v2': psd_v2,
        'spectrum_v2': 'spectrum_v2',
        'diff_max': 'diff_max',
        'diff_ave': 'diff_ave',
        'diff_std': 'diff_std',

    }

    @staticmethod
    def select_estimator(self, method: str) -> UncertaintyEstimator:
        if method in UncertaintyMethodSelector.uncertainty_methods:
            return UncertaintyMethodSelector.uncertainty_methods[method]
        else:
            raise NotImplementedError(f"Method {method} not implemented")


