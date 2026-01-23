import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from nbs_uncertainty.readers.bathymetryDataset import RasterDataset
from nbs_uncertainty.methods.rasterSpectralMethods import (GlenPSD,
                                                           GlenAmplitude,
                                                           EliasUncertainty)
from nbs_uncertainty.methods.rasterSpatialMethods import (RasterSpatialStd,
                                                          RasterSpatialDiff,
                                                          RasterSpatialGaussian)

from functools import partial

raster_methods = {'amp_v1': GlenAmplitude,
                  'psd_v1': GlenPSD,
                  'amp_v2': partial(EliasUncertainty, method='amplitude'),
                  'psd_n': partial(EliasUncertainty, method='psd_n'),
                  'psd_lf': partial(EliasUncertainty, method='psd_lf'),
                  'psd_df': partial(EliasUncertainty, method='psd_df'),
                  'spectrum': partial(EliasUncertainty, method='spectrum'),
                  'spatial_std': partial(RasterSpatialStd, method='spatial_std'),
                  'spatial_diff': partial(RasterSpatialDiff, method='spatial_diff'),
                  'spatial_gaussian': partial(RasterSpatialGaussian, method='spatial_gaussian')
                  }

spectral_methods = ['amp_v1', 'psd_v1', 'amp_v2', 'psd_n', 'psd_lf', 'psd_df', 'spectrum']
spatial_methods = ['spatial_std', 'spatial_diff', 'spatial_gaussian']


class RasterProcessor:
    """
    Base class for methods that uses BathymetryDataset as input
    """
    def __init__(self, linespacing_meters: int,
                        multiple: int,
                        max_multiple: int):

        # define parameters needed for most of the methods
        self.linespacing_meters = linespacing_meters
        self.multiple = multiple
        self.max_multiple = max_multiple


    def get_column_indices(self, rasterDataset: RasterDataset) -> np.ndarray:

        """
        Determine indices of the columns to be used as simulated sampling lines

        Parameters:
        -------------
        array_len : int
                    Length of the array containing bathymetry depth
        resolution : float
                    Spatial Resolution of the bathymetry depth
        linespacing_meters : int
                            distance between 2 vertical sample lines (m)
        max_multiple : int
                        maximum multiple of the linespacing to be used
                        as window size for FFT


        Returns:
        -----------
        col_indxs : np.array
                    array indices corresponding to sample points/lines
                    :rtype: np.ndarray

        """
        array_len: int = rasterDataset.shape[1]
        resolution: int = rasterDataset.metadata["resolution"]
        linespacing_meters: int = self.linespacing_meters
        max_multiple: int = self.max_multiple

        # Round the desired linespacing to the nearest even integer
        # for computational convenience
        linespacing_in_pixels  = np.round(linespacing_meters / resolution)
        if (linespacing_in_pixels % 2) != 0:
            linespacing_in_pixels = linespacing_in_pixels - 1

        if array_len < linespacing_in_pixels:
            raise ValueError(
                f"""
                Desired linespacing should be less than the Spatial coverage.
                Entered Linespacing: {linespacing_meters}m
                Bathymetric coverage: {array_len * resolution}m"""
            )

        # Valid sampling columns is determined by the window length
        # Window lengths are multiples of the linespacing
        window_size_pixels = linespacing_in_pixels * max_multiple
        start_col = int(
            (window_size_pixels // 2) -
            (linespacing_in_pixels // 2)
        )

        last_col = int(
            array_len
            - (window_size_pixels // 2)
            + (linespacing_in_pixels // 2)
            - 1
        )

        # actual sampling indices will be determined by the desired linespacing
        col_indxs = np.arange(start_col,
                              last_col,
                              (linespacing_in_pixels + 1)).astype(int)

        return col_indxs

    def matrix2strip(self, depth: RasterDataset) -> RasterDataset:
        """
        Transform depth matrix into a single vertical strip with
        width equal to the linespacing

        Parameters
        ----------
        depth : np.ndarray
                1d vector or 2d array of bathymetric values
        column_indices : np.array
                        column indices/location of the sampling lines
        multiple : int
                Multiple of the linespacing used to define the window size

        Returns
        --------
        strip : np.ndarray
                a segment of the bathymetric depth for further FFT processing


        """
        print(type(depth))
        column_indices = self.get_column_indices(depth)
        # if depth is a vector, convert to matrix
        if len(depth.shape) < 1:
            depth = np.expand_dims(depth, axis=0)
            depth = depth.view(RasterDataset)
            depth.__dict__.update(depth.__dict__)
        current_multiple = self.multiple
        start, end = column_indices[0], column_indices[1]
        linespacing = end - start - 1
        window_size = linespacing * current_multiple
        midpoint = start + (linespacing // 2) + 1

        # Determine column boundaries for window segment
        # -1 / +1 will include sampling columns at the edges
        start_col = int(midpoint - (window_size // 2)) - 1
        end_col = int(column_indices[-1] + (window_size // 2) + 1)

        # Get sliding window view of depth using window_size
        # +2 will compensate for the additional pixels on the edges
        window_views = sliding_window_view(depth[:, start_col:end_col],
                                           window_shape=(depth.shape[0],
                                                         window_size + 2))

        # remove extra dimension and only retain views of the window size
        stride = linespacing + 1
        window_views = window_views.squeeze()[::stride]

        # reshape to 2d matrix
        strips = window_views.reshape(-1, window_size + 2)

        # cast to rasterDataset
        output = strips.view(RasterDataset)
        output.__dict__.update(depth.__dict__)

        return output

    @staticmethod
    def strip2matrix(data_strip: RasterDataset,
                    column_indices: np.ndarray) -> RasterDataset:

        """
        Reverses the matrix2strip function, reverts the strip back
        to the original dimensions of the bathymetric depth

        Parameters
        ----------
        data_strip : np.ndarray
                    processed depth in strip form
        column_indices : np.array
                         column indices/location of the sampling lines

        Returns
        --------
        unstripped : np.array
                    values in their original spatial location


        """

        # placeholder for reconstructed matrix
        output = np.full((data_strip.orig_shape), np.nan)


        # "Cut" the long strip into vertical segments with length (rows)
        # equal to the original depth matrix
        num_rows, num_cols = data_strip.orig_shape[0], data_strip.shape[1]
        window_views = sliding_window_view(data_strip,
                                           window_shape=(num_rows,
                                                         num_cols))

        # remove extra dimension and only retain views of the window size
        stride = num_rows
        segment_strips = window_views.squeeze()[::stride]

        # Start with the first slice of the segment_strips
        strip_0 = segment_strips[0, :, :]

        # Remove the first column of the succeeding slides as they overlap
        # with the last column of the previous slice
        strip_rest = segment_strips[1:, :, 1:]

        # concatenate succeeding slices in the 2nd dimension
        strip_rest = np.transpose(strip_rest, (0, 2, 1))
        strip_rest = strip_rest.reshape(-1, strip_rest.shape[2]).T

        # concatenate first slice with the rest of the segments
        unstripped = np.concatenate((strip_0, strip_rest), axis=1)

        # Place reconstructed array into proper columns
        output[:, column_indices[0]:column_indices[-1] + 1] = unstripped

        # Crop out np.nan pixels
        cols_with_nan = np.isnan(output).any(axis=0)
        output = output[:, ~cols_with_nan]

        # cast to rasterDataset
        output = output.view(RasterDataset)
        output.__dict__.update(data_strip.__dict__)

        return output

    def compute_interpolation (self, rasterDataset: RasterDataset) -> RasterDataset:
        interpolated_strip = np.linspace(start=rasterDataset[:, 0],
                                         stop=rasterDataset[:, -1],
                                         num=rasterDataset.shape[1]).T

        # cast to RasterDataset
        output = interpolated_strip.view(RasterDataset)
        output.__dict__.update(rasterDataset.__dict__)

        return output

    def compute_interpolated_surface(self, rasterDataset: RasterDataset):
        column_indices = self.get_column_indices(rasterDataset)
        depthdata_as_strip = self.matrix2strip(rasterDataset)
        interpolated_strip = self.compute_interpolation(depthdata_as_strip)
        return self.strip2matrix(data_strip=interpolated_strip,
                                 column_indices=column_indices)

    def compute_residual_surface(self, rasterDataset: RasterDataset) -> RasterDataset:
        interpolated_surface = self.compute_interpolated_surface(rasterDataset)

        # number of columns may change due to subsampling
        new_cols = interpolated_surface.shape[1]
        residual_surface = rasterDataset[:, :new_cols] - interpolated_surface
        return np.abs(residual_surface)

    def compute_residual_strip(self, rasterDataset: RasterDataset) -> RasterDataset:
        column_indices = self.get_column_indices(rasterDataset)
        return np.abs(self.compute_interpolation(self.matrix2strip(depth= rasterDataset)))

    def post_process(self, uncertainty_strip:RasterDataset) -> RasterDataset:
        # Remove edges when computing the original linespacing
        linespacing_width = int((uncertainty_strip.shape[1] - 2) / self.multiple)

        # Include edges again for the output strip
        output = np.zeros(shape=(uncertainty_strip.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        # Create mirror image
        selected_data = uncertainty_strip[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        output = output.view(RasterDataset)
        output.__dict__.update(uncertainty_strip.__dict__)

        return output


    def estimate_surface(self, method: str, bathy_data: RasterDataset, **kwargs):
        estimation_method = raster_methods[method]
        bathy_strip = self.compute_residual_strip(bathy_data)
        output_strip = estimation_method(data_strip=bathy_strip,
                                 linespacing_meters=self.linespacing_meters,
                                 current_multiple=self.multiple,
                                 max_multiple=self.max_multiple,
                                 **kwargs).estimate_uncertainty()
        column_indices = self.get_column_indices(bathy_data)
        if method in spectral_methods:
            # Cast to RasterDataset
            output_strip = output_strip.view(RasterDataset)
            output_strip.__dict__.update(bathy_data.__dict__)
            output = self.post_process(output_strip)
            return self.strip2matrix(data_strip=output, column_indices=column_indices)
        elif method in spatial_methods:
            for key in output_strip.keys():
                current_output = output_strip[key]

                # Cast to RasterDataset
                current_output = current_output.view(RasterDataset)
                current_output.__dict__.update(bathy_data.__dict__)
                post_processed = self.post_process(current_output)
                output_strip[key] = self.strip2matrix(data_strip=post_processed, column_indices=column_indices)
            return output_strip
        else:
            raise ValueError(f"Unexpected method: {method}")












