import configparser
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from ..readers.bathymetry import Bathymetry
from typing import Dict

def get_config(config_path: str = '../config/config.ini') -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_column_indices(
    array_len: int,
    resolution: int,
    linespacing_meters: int,
    max_multiple: int,
) -> np.ndarray:
    """
    Determine indices of the columns to be used as sampling lines

    Parameters:
    -------------
    array_len : int
                Length of the array containing bathymetry data
    resolution : float
                    Spatial Resolution of the bathymetry data
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

    # Round the desired linespacing to the nearest even integer
    # for computational convenience
    linespacing_in_pixels = np.round(linespacing_meters / resolution)
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

def matrix2strip(
    depth: np.ndarray,
    column_indices: np.ndarray,
    multiple: int,
) -> np.ndarray:
    """
    Transform depth matrix into a continuous strip with
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
            a segment of the bathymetric data for further FFT processing


    """

    # if depth is a vector, convert to matrix
    if len(depth.shape) < 1:
        depth = np.expand_dims(depth, axis=0)

    current_multiple = multiple
    start, end = column_indices[0], column_indices[1]
    linespacing = end - start - 1
    window_size = linespacing * current_multiple
    midpoint = start + (linespacing // 2) + 1

    # Determine column boundaries for window segment
    # -1 / +1 will include sampling columns at the edges
    start_col = int(midpoint - (window_size // 2)) - 1
    end_col = int(column_indices[-1] + (window_size // 2) + 1)

    # Get sliding window view of data using window_size
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


def strip2matrix(
    data_strip: np.ndarray,
    original_shape: tuple,
    column_indices: np.ndarray,
) -> np.ndarray:
    """
    Reverses the matrix2strip function, reverts the strip back
    to the original dimensions of the bathymetric data

    Parameters
    ----------
    data_strip : np.ndarray
                processed data in strip form
    original_shape : np.array
                    original dimensions of the bathymetric data
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
    # equal to the original data matrix
    num_rows, num_cols = original_shape[0], data_strip.shape[1]
    window_views = sliding_window_view(data_strip,
                                       window_shape=(num_rows,
                                                     num_cols))

    # remove extra dimension and only retain views of the window size
    stride = num_rows
    segment_strips = window_views.squeeze()[::stride]

    # Start with the first slice of the segment_strips
    strip_0 = segment_strips[0, :, :]

    # Remove the first column of the succeeding slides as as they overlap
    # with the last column of the previous slice
    strip_rest = segment_strips[1:, :, 1:]

    # concatenate succeeding slices in the 2nd dimension
    strip_rest = np.transpose(strip_rest, (0, 2, 1))
    strip_rest = strip_rest.reshape(-1, strip_rest.shape[2]).T

    # concatenate first slice with the rest of the segments
    unstripped = np.concatenate((strip_0, strip_rest), axis=1)
    # print(f"strip_0 shape: {strip_0.shape}")
    # print(f"strip_rest shape: {strip_rest.shape}")
    # print(f"target shape: {output[:, column_indices[0]:column_indices[-1]+1].shape}")

    # Place reconstructed array into proper columns
    output[:, column_indices[0]:column_indices[-1]+1] = unstripped

    return output


def transform_matrix(matrix_data, row_indices):
    matrix_data_rows = matrix_data[row_indices[:-1], :]
    n_times = int(row_indices[1] - row_indices[0])
    matrix_data_tiles = np.repeat(matrix_data_rows, n_times, axis=0)
    matrix_data[:row_indices[0], :] = np.nan
    matrix_data[row_indices[-1]:, :] = np.nan
    matrix_data[row_indices[0]:row_indices[-1], :] = matrix_data_tiles
    return matrix_data


def subsample(data:np.ndarray, column_indices: np.ndarray, method: str):
    # possible values: "along", "across", "along-tiled", "across-tiled"
    if method == "across":
        return data
    elif method == "across-tiled":
        # subsample rows every linespacing (defined by the column indices)
        subsampled_data = data[column_indices[:-1], :]
        return subsampled_data
    elif method == "along":
        transposed_data = data.transpose()
        return transposed_data
    elif method == "along-tiled":
        transposed_data = data.transpose()
        subsampled_transposed_data = transposed_data[column_indices[:-1], :]
        return subsampled_transposed_data
    else:
        raise ValueError(f"Unknown method: {method}")


def upsample(subsampled_data:np.ndarray, column_indices: list[int], method: str):
    # possible values: "along", "across", "along-tiled", "across-tiled"
    if method == "across":
        return subsampled_data
    elif method == "across-tiled":
        # repeat rows every linespacing (defined by the column indices)
        n_times = int(column_indices[1] - column_indices[0])
        upsampled_data = np.repeat(subsampled_data, n_times, axis=0)
        return upsampled_data
    elif method == "along":
        upsampled_data = subsampled_data.transpose()
        return upsampled_data
    elif method == "along-tiled":
        n_times = int(column_indices[1] - column_indices[0])
        repeated_subsampled_data = np.repeat(subsampled_data, n_times, axis=0)
        upsampled_data = repeated_subsampled_data.transpose()
        return upsampled_data
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_residual(bathy_data: Bathymetry, params: Dict | None) -> np.ndarray:
    """
    Compute the residual error from estimating the data using
    linear interpolation

    This function computes the estimate for the data strip
    using the edge values and returns the residual error

    Parameters
    ----------

    bathy_data : Bathymetry
                 Bathymetry object

    params : Dict
            custom parameters based on data type

    Returns
    -------
    residual : Bathymetry
               Bathymetry object containing residual in the "data"

    """

    depth_data = bathy_data.data.copy()

    # compute column indices based on pre-defined params
    column_indices = get_column_indices(array_len=depth_data.shape[1],
                                        resolution=bathy_data.metadata['resolution'],
                                        linespacing_meters=params['linespacing'],
                                        max_multiple=params['max_multiple'])

    depth_data_strip = matrix2strip(depth_data,
                                         column_indices=column_indices,
                                         multiple=1)


    interpolated_strip = np.linspace(start=depth_data_strip[:, 0],
                                     stop=depth_data_strip[:, -1],
                                     num=depth_data_strip.shape[1])

    interpolated_strip = interpolated_strip.T
    residual_strip = depth_data_strip - interpolated_strip

    residual = strip2matrix(data_strip=residual_strip,
                             original_shape=depth_data.shape,
                             column_indices=column_indices)

    return residual

def uncertainty_comparison(residuals, uncertainties):
    nonzero_idx = np.nonzero(
        (residuals != 0) & (~np.isnan(residuals)) & (uncertainties != 0)
    )
    uncertainty_ratio = np.full(residuals.shape, np.nan)
    uncertainty_ratio[nonzero_idx] = uncertainties[nonzero_idx] / np.abs(residuals[nonzero_idx])
    fail_points = np.nonzero(uncertainty_ratio < 1)
    ur_flat = uncertainty_ratio[nonzero_idx].flatten()
    total_count = len(ur_flat)
    fail_count = len(fail_points[0])
    pass_percentage = 100 - fail_count / total_count * 100
    current_rmse = np.sqrt(np.mean((residuals[nonzero_idx] - uncertainties[nonzero_idx]) ** 2))
    mean_error = np.mean(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
    std_dev = np.std(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
    sharp = np.mean(uncertainties[nonzero_idx])
    corr = np.corrcoef(uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx]))[0, 1] if len(np.abs(residuals[nonzero_idx])) > 1 else np.nan

    return {
        "total_cts": total_count,
        "fail_cts": fail_count,
        "percentage": pass_percentage,
        "rmse": current_rmse,
        "mean": mean_error,
        "std_dev": std_dev,
        "sharp": sharp,
        "corr": corr}, uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx])
    
    
def multi_uncertainty_comparison(
    residuals: np.ndarray,
    uncertainties_dict: dict[str, np.ndarray],
    resolution,
    desired_linespacing_meters=None,
    fn=None,
    plot_grid=(4, 3),
    path=None,
    plot_boxplots=True
):
    """
    Compare multiple uncertainty surfaces against residuals in one figure.

    Parameters
    ----------
    residuals : np.ndarray
        2D array of residual surface.
    uncertainties_dict : dict
        Dictionary of uncertainty name -> uncertainty array.
    resolution : float
        Grid resolution in meters.
    desired_linespacing_meters : float, optional
        Used for labeling titles.
    fn : str, optional
        Surface name for the first title.
    plot_grid : tuple
        (nrows, ncols) for subplot grid.
    """
    
    import matplotlib.pyplot as plt

    def uncertainty_comparison(residuals:np.ndarray, uncertainties:np.ndarray):
        nonzero_idx = np.nonzero(
            (residuals != 0) & (~np.isnan(residuals)) & (uncertainties != 0)
        )
        print(nonzero_idx)
        uncertainty_ratio = np.full(residuals.shape, np.nan)
        uncertainty_ratio[nonzero_idx] = uncertainties[nonzero_idx] / np.abs(residuals[nonzero_idx])
        fail_points = np.nonzero(uncertainty_ratio < 1)
        ur_flat = uncertainty_ratio[nonzero_idx].flatten()
        total_count = len(ur_flat)
        fail_count = len(fail_points[0])
        pass_percentage = 100 - fail_count / total_count * 100
        current_rmse = np.sqrt(np.mean((residuals[nonzero_idx] - uncertainties[nonzero_idx]) ** 2))
        mean_error = np.mean(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
        std_dev = np.std(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
        sharp = np.mean(uncertainties[nonzero_idx])
        corr = np.corrcoef(uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx]))[0, 1] if len(np.abs(residuals[nonzero_idx])) > 1 else np.nan

        return pass_percentage, current_rmse, mean_error, std_dev, sharp, corr

    # ---- Create figure ----
    nrows, ncols = plot_grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12), layout="constrained")

    axes = axes.flatten()
    names = list(uncertainties_dict.keys())
    results = []

    for i, (name, uncertainty) in enumerate(uncertainties_dict.items()):
        ax = axes[i]

        # Compute stats
        pass_percentage, rmse, mean_error, std_dev, sharp, corr = uncertainty_comparison(residuals, uncertainty)

        # Append stats for CSV
        results.append({
            "Seabed": fn,
            "Uncertainty Method": name,
            "Line spacing ": desired_linespacing_meters,
            "Pass %": pass_percentage,
            "RMSE": rmse,
            "Bias (Mean Error)": mean_error,
            "Std Dev": std_dev,
            "Sharpness": sharp,
            "Correlation": corr
        })

        # Scatter comparison plot
        nonzero_idx = np.nonzero(
            (residuals != 0) & (~np.isnan(residuals)) & (uncertainty != 0)
        )
        max_unc = np.max(uncertainty[nonzero_idx])
        ax.plot(np.abs(residuals[nonzero_idx]), uncertainty[nonzero_idx], ".", alpha=0.3)
        ax.plot([0, max_unc], [0, max_unc], "r", lw=1)

        ax.set_xlabel("Abs. Residual (m)")
        ax.set_ylabel("Uncertainty (m)")
        ax.set_xlim(0, max_unc)
        ax.set_ylim(0, max_unc)
        ax.grid(True, alpha=0.3)

        # Title with stats
        ax.set_title(
            f"{name}\nPass: {pass_percentage:.1f}%  RMSE: {rmse:.2f}  Bias: {mean_error:.2f}\n  "
            f"Corr: {corr:.2f}, Sharp: {sharp:.2f}",  fontsize=12,
        )

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if fn and desired_linespacing_meters:
        fig.suptitle(
            f"Uncertainty Comparisons for {fn} ({resolution}m grid, {desired_linespacing_meters}m spacing)",
            fontsize=14,
        )
    else:
        fig.suptitle("Uncertainty Comparisons", fontsize=14)
    # outpath = f'{path}_uncertainty_comparisons.png'
    # plt.savefig(outpath, bbox_inches='tight')
    plt.show()


    # # ---- Export to CSV ----
    # if path:
    #     df = pd.DataFrame(results)
    #     outpath = f'{path}_stats.csv'
    #     df.to_csv(f'{outpath}', index=False)
    #     print(f"Statistics exported to {outpath}")

    # # ---- Optional: Combined Boxplot of Residuals vs Uncertainties ----

    # if plot_boxplots:
    #     # Collect data
    #     data = []
    #     labels = []

    #     # Residuals (absolute values)
    #     res_vals = np.abs(residuals[(residuals != 0) & (~np.isnan(residuals))]).flatten()
    #     data.append(res_vals)
    #     labels.append("Abs.  Residuals")

    #     # Each uncertainty
    #     for name, uncertainty in uncertainties_dict.items():
    #         unc_vals = uncertainty[(uncertainty != 0) & (~np.isnan(uncertainty))].flatten()
    #         data.append(unc_vals)
    #         labels.append(name)

    #     # Combined boxplot
    #     plt.figure(figsize=(10, 5))
    #     plt.boxplot(data, patch_artist=True, labels=labels,
    #                 boxprops=dict(facecolor='lightgray', alpha=0.7),
    #                 medianprops=dict(color='red', linewidth=1.5))
    #     plt.title(f"Uncertainty Boxplots for {fn} ({resolution}m grid, {desired_linespacing_meters}m spacing)")
    #     plt.ylabel("Uncertainty (m)")
    #     plt.grid(alpha=0.3)
    #     plt.xticks(rotation=30)
    #     outpath = f'{path}_uncertainty_boxplots.png'
    #     plt.savefig(outpath, bbox_inches='tight')
    #     plt.show()