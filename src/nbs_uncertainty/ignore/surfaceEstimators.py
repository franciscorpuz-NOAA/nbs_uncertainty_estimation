# from abc import ABC, abstractmethod
# from collections.abc import Callable
# from typing import Union
# from .spectralEstimators import SpectralEstimator
# from .spatialEstimators import SpatialEstimator

# import numpy as np
#
# # from .uncertaintyEstimators import UncertaintyEstimator
# from ..readers.bathymetryDataset import (BathymetryFromFile,
#                                       RasterBathymetry,
#                                       CSVBathymetry,
#                                       BPSBathymetry)
# from ..utils.utils import (get_column_indices,
#                            matrix2strip,
#                            strip2matrix,
#                            upsample,
#                            subsample)
#
# class SurfaceEstimator:
#     """
#     Abstract class for estimating surface given sampling method
#     """
#
#     bathy_file: BathymetryFromFile
#     # interpolation: np.ndarray
#     # residual: np.ndarray
#
#
#     def __init__(self, bathy_file: BathymetryFromFile) -> None:
#         self.bathy_file = bathy_file
#
#     def compute_residual(self) -> None:
#         """
#         Computes the residual between the bathymetry depth and its interpolation
#         :return:
#         """
#         raise NotImplementedError
#
#     def estimate_surface(self) -> np.ndarray:
#         """
#         Estimates the surface from the bathymetry depth.
#         """
#         raise NotImplementedError
#
#
# class RasterEstimator(SurfaceEstimator):
#     """
#     Adapter class for estimating raster surfaces
#     """
#
#     method: str = None
#     resolution: int = 1
#     linespacing_m: int = 1
#     current_multiple: int = 1
#     max_multiple: int = 1
#     data_array: np.ndarray
#     column_indices: np.ndarray
#     # bathy_file : RasterBathymetry
#     subsampling: str = None # possible values: "along, across, x_tiled"
#     subsampled_shape = tuple[int, int]
#     depth_data_strip_shape = tuple[int, int]
#
#
#     def __init__(self, bathy_file: RasterBathymetry) -> None:
#         super().__init__(bathy_file)
#         self.depth_array = self.bathy_file.depth_data.copy()
#         self.resolution = self.bathy_file.metadata['resolution']
#         self.linespacing = self.bathy_file.metadata['linespacing']
#         self.max_multiple = self.bathy_file.metadata['max_multiple']
#         self.current_multiple = self.bathy_file.metadata['current_multiple']
#         self.windowing = self.bathy_file.metadata['windowing']
#         self.sampling = self.bathy_file.metadata['sampling']
#         self.column_indices = get_column_indices(array_len=self.depth_array.shape[1],
#                                                  resolution=self.resolution,
#                                                  linespacing_meters=self.linespacing,
#                                                  max_multiple=self.max_multiple)
#
#
#
#     def __repr__(self):
#         return f"RasterEstimator({self.bathy_file})"
#
#     def pre_process(self) -> np.ndarray:
#         subsampled_depth = subsample(self.depth_array, self.column_indices, method=self.sampling)
#         self.subsampled_shape = subsampled_depth.shape
#
#         depth_data_strip = matrix2strip(subsampled_depth,
#                                         column_indices=self.column_indices,
#                                         multiple=self.current_multiple)
#
#         if depth_data_strip.ndim < 2:
#             depth_data_strip = depth_data_strip.reshape(1, -1)
#
#         self.depth_data_strip_shape = depth_data_strip.shape
#         return depth_data_strip
#
#     def post_process(self, uncertainty_strip:np.ndarray) -> np.ndarray:
#         # Remove edges when computing the original linespacing
#         current_multiple = self.current_multiple
#         linespacing_width = int((self.depth_data_strip_shape[1] - 2) / current_multiple)
#         # Include edges again for the output strip
#         output = np.zeros(shape=(self.depth_data_strip_shape[0], linespacing_width + 2))
#         num_cols = output.shape[1]
#
#         selected_data = uncertainty_strip[:, :int(num_cols / 2)]
#         output[:, :int(num_cols / 2)] = selected_data
#         output[:, int(num_cols / 2):] = np.fliplr(selected_data)
#
#         output = strip2matrix(data_strip=output,
#                      original_shape=self.depth_array.shape,
#                      column_indices=self.column_indices)
#
#         sampling = self.bathy_file.metadata['sampling']
#         output = upsample(output, list(self.column_indices), method=sampling)
#
#         return output
#
#     def estimate_surface(self) -> None:
#         raise NotImplementedError
#
#     def compute_residual(self) -> None:
#         raise NotImplementedError

    # def compute_residual(self,
    #                      input_array: np.ndarray,
    #                      input_multiple: int = 1) -> tuple[np.ndarray, np.ndarray]:
    #
    #     # convert arrays into a strip to optimize vectorization
    #     data_in_strip = matrix2strip(input_array,
    #                                 column_indices=self.column_indices,
    #                                 multiple=input_multiple)
    #
    #     # perform calculations for interpolation and residual
    #     interpolation_in_strip = np.linspace(start=data_in_strip[:, 0],
    #                                      stop=data_in_strip[:, -1],
    #                                      num=data_in_strip.shape[1])
    #
    #     interpolated_strip = interpolation_in_strip.T
    #     residual_in_strip = data_in_strip - interpolated_strip
    #
    #     # convert "strips" back to original depth dimensions
    #     output_interpolation = strip2matrix(interpolated_strip,
    #                                       original_shape=input_array.shape,
    #                                       column_indices=self.column_indices)
    #
    #     output_residual = strip2matrix(data_strip=residual_in_strip,
    #                                  original_shape=input_array.shape,
    #                                  column_indices=self.column_indices)
    #
    #     return output_residual, output_interpolation

    # def estimate_surface(self) -> np.ndarray:
    #
    #     subsampling_method = self.subsampling
    #     # compute new indices based on current linespacing and multiple
    #     current_column_indices = get_column_indices(self.data_array.shape[1],
    #                                              self.resolution,
    #                                              self.linespacing_m,
    #                                              self.current_multiple)
    #     # subsample depth depth
    #     subsampled_data_array = subsample(data = self.data_array,
    #                                       column_indices = current_column_indices,
    #                                       method = subsampling_method)
    #
    #     subsampled_residual, _ = self.compute_residual(subsampled_data_array,
    #                                                    self.current_multiple)
    #
    #     uncertainty_function = UncertaintyMethodSelector.select_estimator(self.method)
    #
    #     data_strip_of_multiple = matrix2strip(subsampled_residual,
    #                                         self.column_indices,
    #                                         multiple=self.current_multiple)
    #
    #     uncertainty_in_strip = uncertainty_function.compute_uncertainty(data_strip_of_multiple,
    #                                                                   multiple=self.current_multiple,
    #                                                                   method=self.method)
    #
    #
    #
    #     uncertainty_output = upsample(subsampled_data=uncertainty_in_strip,
    #                                   column_indices=current_column_indices,
    #                                   method=subsampling_method)
    #
    #
    #     return uncertainty_output



# # For Future Implementation
# class CSVEstimator(SurfaceEstimator):
#     """"
#     Adapter class for estimating csv depth
#     """
#     # For future implementation
#     def __init__(self, bathy_file: CSVBathymetry) -> None:
#         super().__init__(bathy_file)
#
#
#     def estimate_surface(self) -> np.ndarray:
#         pass
#
#     def compute_residual(self) -> None:
#         pass
#
#
#
# # For Future Implementation
# class BPSEstimator(SurfaceEstimator):
#     """"
#     Adapter class for estimating bps depth
#     """
#     # For future implementation
#     def __init__(self, bathy_file: BPSBathymetry) -> None:
#         super().__init__(bathy_file)
#
#     def estimate_surface(self) -> np.ndarray:
#         pass
#
#     def compute_residual(self) -> None:
#         pass



# class EstimatorSelector:
#     """
#     Factory for creating surface ignore based on bathymetry depth type.
#     """
#
#     @staticmethod
#     def create_estimator(bathy_file: Bathymetry) -> 'SurfaceEstimator':
#         """
#         Selects and returns an appropriate surface estimator based on the depth type.
#         """
#         estimator_types = {
#             'raster': RasterEstimator(bathy_file),
#             'csv': CSVEstimator(bathy_file),
#             'bps': BPSEstimator(bathy_file)
#         }
#         estimator = estimator_types.get(bathy_file.data_type)
#         if estimator:
#             return estimator
#         else:
#             raise RuntimeError(f"Unrecognized Bathymetry type: {bathy_file.data_type}")


# class UncertaintyMethodSelector:
#     """
#     Class template for computing uncertainty estimation
#     """
#
#
#     @staticmethod
#     def select_estimator(method: str) -> UncertaintyEstimator:
#         uncertainty_methods = {
#             # 'amplitude_v1': 'amplitude_v1',
#             # 'psd_v1': 'psd_v1',
#             # 'spectrum_v1': 'spectrum_v1',
#             'amplitude_v2': SpectralEstimator,
#             'psd_v2': SpectralEstimator,
#             # 'std_envelope_1' :
#             # 'spectrum_v2': spectrum_v2,
#             'diff_max': SpatialEstimator,
#             'diff_ave': SpatialEstimator,
#             'diff_std': SpatialEstimator,
#         }
#         if method in uncertainty_methods:
#             return uncertainty_methods.get(method)
#         else:
#             raise NotImplementedError(f"Method {method} not implemented")
