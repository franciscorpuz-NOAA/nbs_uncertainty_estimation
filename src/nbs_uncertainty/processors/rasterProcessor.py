# from nbs_uncertainty.processors.bathyProcessors import BaseProcessorClass
from nbs_uncertainty.readers.bathymetryDataset import RasterDataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class RasterProcessorClass:
    """
    Base class for Processors that work with Raster datasets
    """

    def __init__(self, bathydataset: RasterDataset,
                 linespacing_meters: int,
                 current_multiple: int,
                 max_multiple: int):

        self.bathydataset = bathydataset
        self.linespacing_meters = linespacing_meters
        self.current_multiple = current_multiple
        self.max_multiple = max_multiple

    @property
    def column_indices(self):

        """
        Determine indices of the columns to be used as sampling lines

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
        array_len: int = self.bathydataset.depth_data.shape[1]
        resolution: int = self.bathydataset.metadata["resolution"]
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

    @staticmethod
    def matrix2strip(depth: np.ndarray,
                    column_indices: np.ndarray,
                    multiple: int) -> np.ndarray:
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
        strip : np.array
                a segment of the bathymetric depth for further FFT processing


        """

        # if depth is a vector, convert to matrix
        if len(depth.shape) < 1:
            depth = np.expand_dims(depth, axis=0)
        if len(column_indices) < 2:
            # already a strip
            return depth
        current_multiple = multiple
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

        return strips

    @staticmethod
    def strip2matrix(data_strip: np.ndarray,
                    original_shape: tuple,
                    column_indices: np.ndarray) -> np.ndarray:

        """
        Reverses the matrix2strip function, reverts the strip back
        to the original dimensions of the bathymetric depth

        Parameters
        ----------
        data_strip : np.ndarray
                    processed depth in strip form
        original_shape : np.array
                        original dimensions of the bathymetric depth
        column_indices : np.array
                         column indices/location of the sampling lines

        Returns
        --------
        unstripped : np.array
                    values in their original spatial location


        """

        # placeholder for reconstructed matrix
        output = np.zeros(shape=original_shape)
        output[:] = np.nan

        # "Cut" the long strip into vertical segments with length (rows)
        # equal to the original depth matrix
        num_rows, num_cols = original_shape[0], data_strip.shape[1]
        # print(f"original shape: {original_shape}")
        # print(f"data_strip shape:{data_strip.shape}")
        # print(f"window shape: {num_rows, num_cols}")
        window_views = sliding_window_view(data_strip,
                                           window_shape=(num_rows,
                                                         num_cols))

        # remove extra dimension and only retain views of the window size
        stride = num_rows
        segment_strips = window_views.squeeze()[::stride]

        if len(segment_strips.shape) < 3:
            return data_strip

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

        return output

    @property
    def as_strip(self):
        return RasterProcessorClass.matrix2strip(depth=self.bathydataset.depth_data,
                                                 column_indices=self.column_indices,
                                                 multiple=self.current_multiple)
    @property
    def interpolated_surface(self):
        depthdata_as_strip = self.as_strip
        interpolated_strip = np.linspace(start=depthdata_as_strip[:, 0],
                                         stop=depthdata_as_strip[:, -1],
                                         num=depthdata_as_strip.shape[1]).T
        return RasterProcessorClass.strip2matrix(data_strip=interpolated_strip,
                                                 original_shape=self.bathydataset.depth_data.shape,
                                                 column_indices=self.column_indices)

    @property
    def residual_surface(self):
        return self.bathydataset.depth_data - self.interpolated_surface

    @property
    def residual_strip(self):
        return self.matrix2strip(depth=self.residual_surface,
                                 column_indices=self.column_indices,
                                 multiple=self.current_multiple)


    def post_process(self, uncertainty_strip:np.ndarray) -> np.ndarray:
        # Remove edges when computing the original linespacing
        linespacing_width = int((self.as_strip.shape[1] - 2) / self.current_multiple)

        # Include edges again for the output strip
        output = np.zeros(shape=(self.as_strip.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        # Create mirror image
        selected_data = uncertainty_strip[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        return output