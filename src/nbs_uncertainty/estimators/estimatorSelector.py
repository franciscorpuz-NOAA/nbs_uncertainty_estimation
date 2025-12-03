from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union
from .spectralEstimators import SpectralEstimator
from .spatialEstimators import SpatialEstimator

import numpy as np

from .uncertaintyEstimators import UncertaintyEstimator
from ..readers.bathymetry import (Bathymetry,
                                  RasterBathymetry,
                                  CSVBathymetry,
                                  BPSBathymetry)
from ..utils.helper import (get_column_indices,
                            matrix2strip,
                            strip2matrix,
                            subsample, upsample)



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
        output_interpolation = strip2matrix(interpolated_strip,
                                          original_shape=input_array.shape,
                                          column_indices=self.column_indices)

        output_residual = strip2matrix(data_strip=residual_in_strip,
                                     original_shape=input_array.shape,
                                     column_indices=self.column_indices)

        return output_residual, output_interpolation

    def set_current_multiple(self, input_multiple: int) -> None:
        self.current_multiple = input_multiple

    def estimate_surface(self) -> np.ndarray:

        subsampling_method = self.subsampling
        # compute new indices based on current linespacing and multiple
        current_column_indices = get_column_indices(self.data_array.shape[1],
                                                 self.resolution,
                                                 self.linespacing_m,
                                                 self.current_multiple)
        # subsample depth data
        subsampled_data_array = subsample(data = self.data_array,
                                          column_indices = current_column_indices,
                                          method = subsampling_method)

        subsampled_residual, _ = self.compute_residual(subsampled_data_array,
                                                       self.current_multiple)

        uncertainty_function = UncertaintyMethodSelector.select_estimator(self.method)

        data_strip_of_multiple = matrix2strip(subsampled_residual,
                                            self.column_indices,
                                            multiple=self.current_multiple)

        uncertainty_in_strip = uncertainty_function.compute_uncertainty(data_strip_of_multiple,
                                                                      multiple=self.current_multiple,
                                                                      method=self.method)



        uncertainty_output = upsample(subsampled_data=uncertainty_in_strip,
                                      column_indices=current_column_indices,
                                      method=subsampling_method)


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
        pass

    def compute_residual(self) -> None:
        pass



# For Future Implementation
class BPSEstimator(SurfaceEstimator):
    """"
    Adapter class for estimating bps data
    """
    # For future implementation
    def __init__(self, bathy_file: BPSBathymetry) -> None:
        super().__init__(bathy_file)

    def estimate_surface(self) -> np.ndarray:
        pass

    def compute_residual(self) -> None:
        pass



class EstimatorSelector:
    """
    Factory for creating surface estimators based on bathymetry data type.
    """

    @staticmethod
    def create_estimator(bathy_file: Bathymetry) -> 'SurfaceEstimator':
        """
        Selects and returns an appropriate surface estimator based on the data type.
        """
        estimator_types = {
            'raster': RasterEstimator(bathy_file),
            'csv': CSVEstimator(bathy_file),
            'bps': BPSEstimator(bathy_file)
        }
        estimator = estimator_types.get(bathy_file.data_type)
        if estimator:
            return estimator
        else:
            raise RuntimeError(f"Unrecognized Bathymetry type: {bathy_file.data_type}")


class UncertaintyMethodSelector:
    """
    Class template for computing uncertainty estimation
    """


    @staticmethod
    def select_estimator(method: str) -> UncertaintyEstimator:
        uncertainty_methods = {
            # 'amplitude_v1': 'amplitude_v1',
            # 'psd_v1': 'psd_v1',
            # 'spectrum_v1': 'spectrum_v1',
            'amplitude_v2': SpectralEstimator,
            'psd_v2': SpectralEstimator,
            # 'std_envelope_1' :
            # 'spectrum_v2': spectrum_v2,
            'diff_max': SpatialEstimator,
            'diff_ave': SpatialEstimator,
            'diff_std': SpatialEstimator,
        }
        if method in uncertainty_methods:
            return uncertainty_methods.get(method)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
