import numpy as np
from ..utils.helper import matrix2strip, strip2matrix
from typing import Callable
from scipy.stats import genextreme
from functools import partial
from abc import ABC, abstractmethod



class SpatialEstimator:
    """
    Estimator for spectrally-based uncertainty models.
    """


    @staticmethod
    def compute_uncertainty(data: np.ndarray,
                            multiple: int,
                            method: str,
                            min_window: int = 2) -> np.ndarray:

        """
        Calculate the variance of the provided data array in parts.
        """
        # data = matrix2strip(data)
        interpolation_cell_distance = data.shape[1]
        num_lines, num_samples = data.shape
        interpolation_cell_distance = ((interpolation_cell_distance - 2) // multiple) + 2
        # interpolation_cell_distance = num_samples
        difference_stat = np.full((num_lines, interpolation_cell_distance), 0.0)
        # difference_mean = np.full((num_lines, interpolation_cell_distance), 0.0)
        # difference_std = np.full((num_lines, interpolation_cell_distance), 0.0)
        for win_len in range(min_window, interpolation_cell_distance // 2 + 1):
            num_convolutions = num_samples - win_len + 1
            differences = np.full((num_lines, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins = np.min(data[:, step:step + win_len], axis=-1)
                maxs = np.max(data[:, step:step + win_len], axis=-1)
                differences[:, step] = maxs - mins
            current_method = selectMethod(method)
            difference_stat[:, win_len - 1] = current_method(differences, axis=-1)
            # difference_mean[:, win_len-1] = np.mean(differences, axis =-1)
            # difference_std[:, win_len-1] = np.std(differences, axis =-1)
        # print(f"final convolutions for window length {win_len} is {num_convolutions}")
        difference_stat[:, -win_len:] = np.fliplr(difference_stat[:, :win_len])
        # difference_mean[:,-win_len:] = np.fliplr(difference_mean[:, :win_len])
        # difference_std[:,-win_len:] = np.fliplr(difference_std[:,:win_len])

        output_data = strip2matrix(difference_stat)
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
        # function for computing variance
    }

    return method_dict.get(method)


def get_interp_residual(data):
    """
    Calculate the residual between the interpolated data and the original data.
    Interpolated data is a line between the two end values.
    """

    slopes = (data[:, -1] - data[:,0]) / data.shape[1]
    start_vals = data[:, 0]
    # interpolation = np.arange(len(data))
    # interpolation = interpolation * slopes + start_vals
    interpolation = np.tile(np.arange(data.shape[1]), (data.shape[0],1))
    interpolation = interpolation * slopes + start_vals
    residual = data - interpolation

    return residual, interpolation


def preprocess_strip(data):
    """
    Remove the mean and apply a Hann window to the rows of the provided data array.
    """
    shape = data.shape[1]
    window = np.hanning(shape)
    data = data * window
    return data


def psd_strip_modified(data, resolution):
    """
    Calculate the power spectral density of the provided data array.

    For background on the inline comments, see https://docs.scipy.org/doc/scipy/tutorial/signal.html#sampled-sine-signal
    """
    shape = data.shape[1]
    hann_win = np.hanning(shape)
    hann_scale = 1 / np.sqrt(np.sum(np.square(hann_win)))
    fft_data_windowed = np.fft.rfft(data, axis=1) * hann_scale  # this is $X_l^w$
    psd_1sided = 2 * resolution * np.square(np.abs(fft_data_windowed))
    # drop zero frequency
    frequencies = np.fft.rfftfreq(shape, resolution)[1:]
    # frequencies = np.fft.rfftfreq(shape, resolution)
    # psd_result = np.zeros_like(psd_1sided)
    # psd_result[:, 1:shape] = psd_1sided[:, 1:shape]
    psd_result = psd_1sided[:, 1:shape]
    return psd_result, frequencies


def spectrum_strip(data, resolution):
    """
    Modified psd_strip_modified function to compute spectrum instead of psd
    Need to check if units are similar

    """
    shape = data.shape[1]
    hann_win = np.hanning(shape)
    hann_scale = 1 / np.sqrt(np.sum(hann_win)**2)
    fft_data_windowed = np.fft.rfft(data, axis=1) * hann_scale  # this is $X_l^w$
    spectrum_1sided = 2 * resolution * np.square(np.abs(fft_data_windowed))
    # drop zero frequency
    frequencies = np.fft.rfftfreq(shape, resolution)[1:]
    # frequencies = np.fft.rfftfreq(shape, resolution)
    # psd_result = spectrum_1sided[:, 1:shape]
    # psd_result = np.zeros_like(spectrum_1sided)
    # psd_result[:, 1:shape] = spectrum_1sided[:, 1:shape]
    psd_result = spectrum_1sided[:, 1:shape]
    return psd_result, frequencies


def amp_strip(data, resolution):
    """
    Calculate the power spectral density of the provided data array.

    """
    shape = data.shape[1]
    hann_win = np.hanning(shape)
    hann_scale = 1 / np.sum(hann_win)
    # fft_data_windowed = np.fft.rfft(data, axis=1) * hann_scale
    fft_data_windowed = np.fft.rfft(data, axis=1)
    fft_data_1sided = 2 * np.abs(fft_data_windowed)
    # drop zero frequency
    frequencies = np.fft.rfftfreq(data.shape[1], resolution)[1:]
    # frequencies = np.fft.rfftfreq(data.shape[1], resolution)
    # amp_result = fft_data_1sided[:, 1:]
    # amp_result = np.zeros_like(fft_data_1sided)
    # amp_result[:, 1:] = fft_data_1sided[:, 1:]
    amp_result = fft_data_1sided[:, 1:]
    return amp_result, frequencies


class spatial_contributions:
    def __init__(self, resolution, max_cell_number):
        """
        Create the distance and frequency dependent scaling factors.
        """
        # self.frequencies = frequencies inside the window
        self.frequencies = np.fft.rfftfreq(max_cell_number, resolution)[1:]
        # self.frequencies = np.fft.rfftfreq(max_cell_number, resolution)
        self.distances = np.arange(max_cell_number) * resolution
        distances_2d, freq_2d = np.meshgrid(self.distances, self.frequencies)
        self.spatial_scale = distances_2d * freq_2d
        self.spatial_scale = np.where(
            self.spatial_scale < 0.25, self.spatial_scale, 0.25
        )
        self.spatial_signal = np.sin(self.spatial_scale * 2 * np.pi)

    def get_uncertainties(self, pxx, fft_freq,
                          interpolation_cell_distance, multiple):
        """
        Multiply the scaling factors by the amplitude to get the uncertainties.
        """
        freq_idx = np.where(np.isin(fft_freq, self.frequencies))[0]
        # freq_contributions = psd * self.spatial_signal[freq_idx,:]
        # should add a check to make sure self.frequencies are matched appropriately to psd_freq
        # print(f"fft_freq: {fft_freq}, fft_freq length: {len(fft_freq)}")
        # print(f"frequencies: {self.frequencies}, frequencies length: {len(self.frequencies)}")
        # print(f"pxx shape: {pxx.shape}, freq_idx length: {len(freq_idx)}")
        # print(f"spatial signal shape: {self.spatial_signal.shape}")
        window_uncertainties = (pxx @ self.spatial_signal) / len(freq_idx)
        # window_uncertainties = (pxx @ self.spatial_signal)
        # zero_column = np.zeros((pxx.shape[0], 1))
        # window_uncertainties = np.column_stack((zero_column, window_uncertainties))
        # print(f"window_uncertainties shape: {window_uncertainties.shape}")
        interpolation_cell_distance = int(interpolation_cell_distance)
        interpolation_uncertainties = np.zeros((pxx.shape[0], ((interpolation_cell_distance-2)//multiple)+2))

        half_window = (interpolation_uncertainties.shape[1] // 2)
        interpolation_uncertainties[:, :half_window] = (
            window_uncertainties[:, :half_window])
        interpolation_uncertainties[:, half_window:] = np.fliplr(
            window_uncertainties[:, :half_window])

        return interpolation_uncertainties

def glen_get_uncertainties(data, resolution, multiple, method="amplitude"):

    data = preprocess_strip(data)
    if method == "amplitude":
        energy, freq = amp_strip(data, resolution)
        scaler = spatial_contributions(resolution, data.shape[1])
        uncertainty = scaler.get_uncertainties(energy,
                                               freq,
                                               data.shape[1],
                                               multiple)

    elif method == "psd":
        energy, freq = psd_strip_modified(data, resolution)
        scaler = spatial_contributions(resolution, data.shape[1])
        uncertainty = scaler.get_uncertainties(energy,
                                               freq,
                                               data.shape[1],
                                               multiple)
        uncertainty = np.sqrt(uncertainty)

    elif method == "spectrum":
        energy, freq = spectrum_strip(data, resolution)
        scaler = spatial_contributions(resolution, data.shape[1])
        uncertainty = scaler.get_uncertainties(energy,
                                               freq,
                                               data.shape[1],
                                               multiple)
        uncertainty = np.sqrt(uncertainty) * resolution

    else:
        raise ValueError(f"Unrecognized Method: {method}")

    return uncertainty



def get_difference_uncertainties(data:np.ndarray, 
                                      interpolation_cell_distance:int, 
                                      min_window:int = 2, 
                                      multiple:int = 1,
                                      method:str = 'max') -> np.ndarray:
    """
    Calculate the variance of the provided data array in parts.
    """
    num_lines, num_samples = data.shape
    interpolation_cell_distance = ((interpolation_cell_distance-2)//multiple)+2
    # interpolation_cell_distance = num_samples
    differences = np.array([])
    output = np.full((num_lines, interpolation_cell_distance), 0.0)
    # difference_max = np.full((num_lines, interpolation_cell_distance), 0.0)
    # difference_mean = np.full((num_lines, interpolation_cell_distance), 0.0)
    # difference_std = np.full((num_lines, interpolation_cell_distance), 0.0)
    for win_len in range(min_window, interpolation_cell_distance//2+1):
        # num_convolutions = num_samples - win_len + 1
        # differences = np.full((num_lines, num_convolutions), 0.0)
        # for step in range(num_convolutions):
        #     mins = np.min(data[:,step:step+win_len], axis = -1)
        #     maxs = np.max(data[:,step:step+win_len], axis = -1)
        #     stds = np.std(data[:,step:step+win_len], axis = -1)
        #     differences[:,step] = maxs - mins
        # difference_max[:, win_len-1] = np.max(differences, axis=-1)
        # difference_mean[:, win_len-1] = np.mean(differences, axis =-1)
        # difference_std[:, win_len-1] = np.std(differences, axis =-1)
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=win_len, axis=-1)
        mins = np.min(windows, axis = -1)
        maxs = np.max(windows, axis = -1)
        stds = np.std(windows, axis = -1)
        differences = maxs - mins
    
        if method == 'max':
            output[:, win_len-1] = np.max(differences, axis=-1)
        elif method == 'mean':
            output[:, win_len-1] = np.mean(differences, axis =-1)
        elif method == 'std':
            output[:, win_len-1] = np.std(differences, axis =-1)
        elif method == 'gaussian':
            output[:, win_len-1] = 
        elif method == 'gev':
            return difference_std   

    # print(f"final convolutions for window length {win_len} is {num_convolutions}")
    # difference_max[:,-win_len:] = np.fliplr(difference_max[:, :win_len])
    # difference_mean[:,-win_len:] = np.fliplr(difference_mean[:, :win_len])
    # difference_std[:,-win_len:] = np.fliplr(difference_std[:,:win_len])
    
    else:
        raise ValueError(f"Unknown method: {method}. \
                           Choose from 'max', 'mean', or 'std'.")


