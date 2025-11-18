import numpy as np
from ..readers.bathymetry import RasterBathymetry

def remove_edge_ndv(raster_data: np.ndarray,
                    ndv_value: np.number,
                    max_iterations: int = None) -> RasterBathymetry:
    """
    Iteratively removes edge rows and columns containing no-data values
    in the raster data

    Parameters
    ----------
    raster_data : RasterBathymetry
        A 2D NumPy array representing surface elevation or bathymetry data.
        Expected to be numeric (e.g., float, int).

    max_iterations : int, optional
        Maximum number of iterations to perform. Defaults to half the data width

    Returns
    -------
    RasterBathymetry
        A 2D NumPy array representing the cropped data

    """

    depth = raster_data
    # Type check for depth
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"Input 'depth' must be a NumPy array (np.ndarray). type:{type(depth)}")

    # Handle initial empty array or None input
    if depth is None or depth.size == 0:
        raise ValueError("Input 'depth' array cannot be None or empty.")

    # Create a working copy to avoid modifying the original array passed in
    elev = depth.copy()
    original_shape = depth.shape

    # Set up value for max_iteration if none declared
    if max_iterations is None:
        max_dimensions = np.max(original_shape)
        max_iterations = int(np.max(max_dimensions) / 2)

    # Extract no_data_value
    ndv = ndv_value

    if ndv == np.nan:
        def is_ndv(data_array):
            return np.any(np.isnan(data_array))
    else:
        def is_ndv(data_array):
            return np.any(data_array == ndv)

    shrink_idx = 0
    have_ndv = True
    # remove edges that have ndv elements
    # continue until all edges are ndv-free or exceeded 100 iterations
    # assumes that all inner elements are non NaN
    while have_ndv:
        tmp = elev[0, :]
        if is_ndv(tmp):
            elev = elev[1:, :]
        tmp = elev[:, 0]
        if is_ndv(tmp):
            elev = elev[:, 1:]
        tmp = elev[-1, :]
        if is_ndv(tmp):
            elev = elev[:-1, :]
        tmp = elev[:, -1]
        if is_ndv(tmp):
            elev = elev[:, :-1]
        shrink_idx += 1
        if not np.any(is_ndv(elev)):
            have_ndv = False
        if shrink_idx > max_iterations:
            break

    if is_ndv(elev):
        print("Warning: Depth data still contains NDV values.")

    return elev

def make_square(raster_data: np.ndarray) -> np.ndarray:
    new_dim = np.min(raster_data.shape)
    new_raster = raster_data[:new_dim, :new_dim]
    return new_raster

