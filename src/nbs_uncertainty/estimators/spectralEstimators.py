from .uncertaintyEstimators import UncertaintyEstimator
from ..utils.helper import matrix2strip, strip2matrix

import numpy as np
from scipy import signal


class SpectralEstimator(UncertaintyEstimator):
    """
    Estimator for spectrally-based uncertainty models.
    """

    def __init__(self,
                 data: np.ndarray,
                 resolution: int,
                 multiple: int) -> None:
        self.data = matrix2strip(data)
        self.resolution = resolution
        self.multiple = multiple
        self.windowing = 'hann'

    def compute_uncertainty(self) -> np.ndarray:
        # Implement spectral estimation logic here
        raise NotImplementedError


    def compute_energy(data: np.ndarray,
                       resolution: int,
                       method: str,
                       window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT energy using 'method' process

        Parameters
        ----------
        data : np.array
               Input data
        resolution : int
                     Spatial resoluton of the array
        method : str
                 FFT Method used to estimate signal energy

        Returns
        -------
        np.array
                Spectral energy in the signal
        """

        rfft_values = np.abs(np.fft.rfft(data, axis=1))
        _, num_cols = rfft_values.shape
        r_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)

        if method == "amplitude":
            scale_factor = np.sum(window_values)
            if num_cols % 2 == 0:
                rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
                rfft_values = rfft_values[:, :-1]
                r_frequencies = r_frequencies[:-1]
            else:
                rfft_values[:, 1:] = rfft_values[:, 1:] * 2

        elif method == "psd":
            scale_factor = np.sum(window_values ** 2) / resolution
            rfft_values = rfft_values ** 2
            if num_cols % 2 == 0:
                rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
            else:
                rfft_values[:, 1:] = rfft_values[:, 1:] * 2
            rfft_values = rfft_values[:, :-1]
            r_frequencies = r_frequencies[:-1]

        else:
            raise ValueError(
                f"""Unknown FFT Method: {method}
                    FFT options: {'amplitude', 'psd'}
                    """)

        energy = rfft_values / scale_factor

        return energy, r_frequencies


class amplitude_v2(UncertaintyEstimator):
    def compute_uncertainty(self) -> np.ndarray:
        data = self.data
        resolution = self.resolution

        if data.ndim < 2:
            data = data.reshape(1, -1)

        # create window
        segment_window = signal.windows.get_window(
            window=self.windowing, Nx=data.shape[1], fftbins=False)

        # preprocess_signal, could be modified later
        # data_mean = np.mean(data, axis=1)
        preprocessed_signal = data * segment_window

        energy, energy_freqs = self.compute_energy(
            preprocessed_signal,
            resolution,
            "amplitude",
            segment_window
        )

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(resolution, data.shape[1])
        variance = energy @ spatial_signal
        window_uncertainty = variance

        # Remove edges when computing the original linespacing
        linespacing_width = int((data.shape[1] - 2) / self.multiple)
        # Include edges again for the output strip
        output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        selected_data = window_uncertainty[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        return output

class psd_v2(UncertaintyEstimator):
    def compute_uncertainty(self) -> np.ndarray:
        data = self.data
        resolution = self.resolution

        if data.ndim < 2:
            data = data.reshape(1, -1)

        # create window
        segment_window = signal.windows.get_window(
            window=self.windowing, Nx=data.shape[1], fftbins=False)

        # preprocess_signal, could be modified later
        # data_mean = np.mean(data, axis=1)
        preprocessed_signal = data * segment_window

        energy, energy_freqs = self.compute_energy(
            preprocessed_signal,
            resolution,
            "psd",
            segment_window
        )

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(resolution, data.shape[1])
        variance = energy @ spatial_signal
        window_uncertainty = variance

        # Remove edges when computing the original linespacing
        linespacing_width = int((data.shape[1] - 2) / self.multiple)
        # Include edges again for the output strip
        output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        selected_data = window_uncertainty[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        return output


def create_spatial_signal(resolution: int, max_cell_number: int):
    """
    Create the distance and frequency dependent scaling factors.
    """
    frequencies = np.fft.rfftfreq(max_cell_number, resolution)
    if len(frequencies) % 2 == 0:
        frequencies = frequencies[:-1]
    distances = np.arange(max_cell_number) * resolution
    distances_2d, freq_2d = np.meshgrid(distances, frequencies)
    spatial_scale = distances_2d * freq_2d
    spatial_scale = np.where(spatial_scale < 0.25, spatial_scale, 0.25)
    spatial_signal = np.sin(spatial_scale * 2 * np.pi)

    return spatial_signal