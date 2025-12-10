# from ..utils.helper import matrix2strip, strip2matrix
from copy import deepcopy

import numpy as np
from scipy import signal
from dataclasses import dataclass
from ..utils.helper import get_column_indices
from ..readers.bathymetry import RasterBathymetry
from ..utils.helper import matrix2strip, strip2matrix

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


@dataclass
class SpectralV1:
    bathy_data: RasterBathymetry
    method: str

    column_indices: np.ndarray = None
    segment_window: np.ndarray = None
    depth_data_strip: np.ndarray = None

    def pre_process(self) -> np.ndarray:
        depth_data = self.bathy_data.data
        resolution = self.bathy_data.metadata['resolution']
        linespacing = self.bathy_data.metadata['linespacing']
        max_multiple = self.bathy_data.metadata['max_multiple']
        current_multiple = self.bathy_data.metadata['current_multiple']
        windowing = self.bathy_data.metadata['windowing']

        self.column_indices = get_column_indices(array_len=depth_data.shape[1],
                                            resolution=resolution,
                                            linespacing_meters=linespacing,
                                            max_multiple=max_multiple)

        depth_data_strip = matrix2strip(depth_data,
                                        column_indices=self.column_indices,
                                        multiple=current_multiple)

        if depth_data_strip.ndim < 2:
            depth_data_strip = depth_data_strip.reshape(1, -1)

        # create window
        self.segment_window = signal.windows.get_window(
            window=windowing, Nx=depth_data_strip.shape[1], fftbins=False)

        # preprocess_signal, could be modified later
        # data_mean = np.mean(data, axis=1)
        self.depth_data_strip = depth_data_strip
        preprocessed_signal = depth_data_strip * self.segment_window
        return preprocessed_signal

    def post_process(self, uncertainty_strip:np.ndarray) -> np.ndarray:
        # Remove edges when computing the original linespacing
        current_multiple = self.bathy_data.metadata['current_multiple']
        linespacing_width = int((self.depth_data_strip.shape[1] - 2) / current_multiple)
        # Include edges again for the output strip
        output = np.zeros(shape=(self.depth_data_strip.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        selected_data = uncertainty_strip[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        output = strip2matrix(data_strip=output,
                     original_shape=self.bathy_data.data.shape,
                     column_indices=self.column_indices)

        return output

    def compute_uncertainty(self) -> np.ndarray:
        raise NotImplementedError




class AmpV1(SpectralV1):

    def compute_uncertainty(self) -> RasterBathymetry:
        preprocessed_signal = self.pre_process()
        resolution = self.bathy_data.metadata['resolution']
        rfft_values = np.abs(np.fft.rfft(preprocessed_signal, axis=1))
        _, num_cols = rfft_values.shape
        r_frequencies = np.fft.rfftfreq(preprocessed_signal.shape[1], d=resolution)

        scale_factor = np.sum(self.segment_window)
        if num_cols % 2 == 0:
            rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
            rfft_values = rfft_values[:, :-1]
            r_frequencies = r_frequencies[:-1]
        else:
            rfft_values[:, 1:] = rfft_values[:, 1:] * 2

        energy = rfft_values
        energy_freqs = r_frequencies

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(resolution, preprocessed_signal.shape[1])
        variance = energy @ spatial_signal

        window_uncertainty = variance
        output = self.post_process(window_uncertainty)

        new_bathy = deepcopy(self.bathy_data)
        new_bathy.data = output
        return new_bathy


class PSDV1(SpectralV1):

    def compute_uncertainty(self) -> RasterBathymetry:
        preprocessed_signal = self.pre_process()
        resolution = self.bathy_data.metadata['resolution']
        rfft_values = np.abs(np.fft.rfft(preprocessed_signal, axis=1))
        _, num_cols = rfft_values.shape
        r_frequencies = np.fft.rfftfreq(preprocessed_signal.shape[1], d=resolution)

        scale_factor = np.sum(self.segment_window ** 2) / resolution
        rfft_values = rfft_values ** 2
        if num_cols % 2 == 0:
            rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
        else:
            rfft_values[:, 1:] = rfft_values[:, 1:] * 2
        rfft_values = rfft_values[:, :-1]
        r_frequencies = r_frequencies[:-1]

        energy = rfft_values
        energy_freqs = r_frequencies

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(resolution, preprocessed_signal.shape[1])
        variance = energy @ spatial_signal

        # normalize energy (convert m^2 to meters)
        variance = variance / len(energy_freqs)
        window_uncertainty = np.sqrt(variance)
        output = self.post_process(window_uncertainty)

        new_bathy = deepcopy(self.bathy_data)
        new_bathy.data = output
        return new_bathy


class EliasUncertainty(SpectralV1):

    def compute_energy_elias(self, data: np.ndarray,
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

        if method == "amplitude":  # ASD in Scipy doc

            cden = np.sqrt(np.sum(window_values ** 2))
            energy = np.sqrt(resolution) * rfft_values / cden

            if num_cols % 2 == 0:
                energy[:, 1:-1] = energy[:, 1:-1] * 2
            else:
                energy[:, 1:] = energy[:, 1:] * 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        elif method == "psd":
            cden = np.sqrt(np.sum(window_values ** 2))
            energy = resolution * (np.abs(rfft_values / cden) ** 2)

            if num_cols % 2 == 0:  # even length → Nyquist bin exists
                energy[:, 1:-1] *= 2
            else:  # odd length → no Nyquist bin
                energy[:, 1:] *= 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        elif method == "psd_n":
            cden = np.sqrt(np.sum(window_values ** 2))
            energy = resolution * (np.abs(rfft_values / cden) ** 2)

            if num_cols % 2 == 0:  # even length → Nyquist bin exists
                energy[:, 1:-1] *= 2
            else:  # odd length → no Nyquist bin
                energy[:, 1:] *= 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        elif method == "psd_lf":
            cden = np.sqrt(np.sum(window_values ** 2))
            energy = resolution * (np.abs(rfft_values / cden) ** 2)

            if num_cols % 2 == 0:  # even length → Nyquist bin exists
                energy[:, 1:-1] *= 2
            else:  # odd length → no Nyquist bin
                energy[:, 1:] *= 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        elif method == "psd_df":
            cden = np.sqrt(np.sum(window_values ** 2))
            energy = resolution * (np.abs(rfft_values / cden) ** 2)

            if num_cols % 2 == 0:  # even length → Nyquist bin exists
                energy[:, 1:-1] *= 2
            else:  # odd length → no Nyquist bin
                energy[:, 1:] *= 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        elif method == "spectrum":  # amplitude spectrum
            # camp = np.abs(np.sum(window_values))
            #
            # energy = rfft_values / camp
            #
            # if num_cols % 2 == 0:  # even length → Nyquist bin exists
            #     energy[:, 1:-1] *= 2
            # else:  # odd length → no Nyquist bin
            #     energy[:, 1:] *= 2

            cden = np.sqrt(np.sum(window_values ** 2))
            energy = resolution * (np.abs(rfft_values / cden) ** 2)

            if num_cols % 2 == 0:  # even length → Nyquist bin exists
                energy[:, 1:-1] *= 2
            else:  # odd length → no Nyquist bin
                energy[:, 1:] *= 2

            energy = energy[:, :-1]
            r_frequencies = r_frequencies[:-1]

        else:
            raise ValueError(
                f"""Unknown FFT Method: {method}
                    FFT options: {'amplitude', 'psd_n', 'psd_lf', 'spectrum'}
                    """)

        # energy = rfft_values / scale_factor

        return energy, r_frequencies


    def compute_uncertainty(self) -> RasterBathymetry:
        preprocessed_signal = self.pre_process()
        energy, energy_freqs = self.compute_energy_elias(preprocessed_signal,
                                                    self.bathy_data.metadata['resolution'],
                                                    self.method,
                                                    self.segment_window)

        df = 1.0 / (preprocessed_signal.shape[1] * self.bathy_data.metadata['resolution'])

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(self.bathy_data.metadata['resolution'], preprocessed_signal.shape[1])

        variance = None
        if self.method == "amplitude":
            variance = ((energy ** 2) / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd":
            variance = (energy / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd_n":
            variance = (energy / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd_lf":
            variance = (energy / len(energy_freqs)) @ spatial_signal
        elif self.method == "psd_df":
            variance = (energy * df) @ spatial_signal
        elif self.method == "spectrum":
            variance = (energy / len(energy_freqs)) @ spatial_signal
        else:
            raise ValueError(f"Method not found: {self.method}")

        # if method == "amplitude":
        #     variance = (((energy**2)*df)) @ spatial_signal
        # elif method == "psd":
        #     variance = ((energy*df)) @ spatial_signal
        # elif method == "spectrum":
        #     variance = ((energy*df)) @ spatial_signal

        window_uncertainty = np.sqrt(variance)
        output = self.post_process(window_uncertainty)

        new_bathy = deepcopy(self.bathy_data)
        new_bathy.data = output
        return new_bathy

#
# def compute_fft_uncertainty_elias(
#     data: np.ndarray,
#     multiple: int,
#     resolution: int,
#     windowing: str = 'hann',
#     method: str = "amplitude",
#     selection: str = "half",
# ) -> np.ndarray:
#     """
#     Estimate the uncertainty using FFT
#
#     Parameters
#     ----------
#     data : np.array
#            Input data for FFT estimation
#     multiple : int
#                Window length as multiple of the linespacing
#     resolution : int
#                  Input data resolution for frequency calculation
#     windowing : str
#                 Type of window to taper input
#                 options: scipy.signal.windows type
#     method : str
#              Type of FFT to estimate energy, defaults to 'amplitude'
#              options: ['amplitude', 'psd', 'spectrum']
#
#
#     Returns
#     -------
#     output : np.ndarray
#         Uncertainty estimate from the FFT method
#         To be compared with the residual error
#
#     """
#
#     if data.ndim < 2:
#         data = data.reshape(1, -1)
#
#     # create window
#     segment_window = signal.windows.get_window(
#         window=windowing, Nx=data.shape[1], fftbins=True)
#
#     # preprocess_signal, could be modified later
#     # data_mean = np.mean(data, axis=1)
#     preprocessed_signal = data * segment_window
#
#     energy, energy_freqs = compute_energy_elias(preprocessed_signal,
#                                             resolution,
#                                             method,
#                                             segment_window)
#
#     df = 1.0 / (data.shape[1] * resolution)
#
#     # compute contribution per frequency
#     spatial_signal = create_spatial_signal(resolution, data.shape[1])
#
#     if method == "amplitude":
#         variance = (((energy**2)/data.shape[1])) @ spatial_signal
#     elif method == "psd_n":
#         variance = ((energy/data.shape[1])) @ spatial_signal
#     elif method == "psd_lf":
#         variance = ((energy/len(energy_freqs))) @ spatial_signal
#     elif method == "spectrum":
#         variance = ((energy/len(energy_freqs))) @ spatial_signal
#
#     # if method == "amplitude":
#     #     variance = (((energy**2)*df)) @ spatial_signal
#     # elif method == "psd":
#     #     variance = ((energy*df)) @ spatial_signal
#     # elif method == "spectrum":
#     #     variance = ((energy*df)) @ spatial_signal
#
#     window_uncertainty = np.sqrt(variance)
#
#
#     # Remove edges when computing the original linespacing
#     linespacing_width = int((data.shape[1]-2) / multiple)
#     # Include edges again for the output strip
#     output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
#     num_cols = output.shape[1]
#
#     if selection == "half":
#         selected_data = window_uncertainty[:, :int(num_cols/2)]
#         output[:, :int(num_cols/2)] = selected_data
#         output[:, int(num_cols/2):] = np.fliplr(selected_data)
#     else:
#         # pick energy from frequencies only present in original data
#         freqs_window = np.fft.rfftfreq(int(data.shape[1] / multiple),
#                                        resolution)
#         freq_idxs = np.where(np.isin(freqs_window, energy_freqs))[0]
#         selected_data = window_uncertainty[:, freq_idxs]
#         output[:, :int(num_cols/2)] = selected_data
#         output[:, int(num_cols/2):] = np.fliplr(selected_data)
#
#     return output
#
# def compute_energy_elias(data: np.ndarray,
#                    resolution: int,
#                    method: str,
#                    window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Compute FFT energy using 'method' process
#
#     Parameters
#     ----------
#     data : np.array
#            Input data
#     resolution : int
#                  Spatial resoluton of the array
#     method : str
#              FFT Method used to estimate signal energy
#
#     Returns
#     -------
#     np.array
#             Spectral energy in the signal
#     """
#
#     rfft_values = np.abs(np.fft.rfft(data, axis=1))
#     _, num_cols = rfft_values.shape
#     r_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)
#
#     if method == "amplitude": #ASD in Scipy doc
#
#         cden = np.sqrt(np.sum(window_values** 2))
#         energy = np.sqrt(resolution) * rfft_values / cden
#
#         if num_cols % 2 == 0:
#             energy[:, 1:-1] = energy[:, 1:-1] * 2
#         else:
#             energy[:, 1:] = energy[:, 1:] * 2
#
#         energy = energy[:, :-1]
#         r_frequencies = r_frequencies[:-1]
#
#     elif method == "psd_n":
#         cden = np.sqrt(np.sum(window_values** 2))
#         energy =  resolution * (np.abs(rfft_values / cden)**2)
#
#         if num_cols % 2 == 0:  # even length → Nyquist bin exists
#             energy[:, 1:-1] *= 2
#         else:  # odd length → no Nyquist bin
#             energy[:, 1:] *= 2
#
#         energy = energy[:, :-1]
#         r_frequencies = r_frequencies[:-1]
#
#     elif method == "psd_lf":
#         cden = np.sqrt(np.sum(window_values** 2))
#         energy =  resolution * (np.abs(rfft_values / cden)**2)
#
#         if num_cols % 2 == 0:  # even length → Nyquist bin exists
#             energy[:, 1:-1] *= 2
#         else:  # odd length → no Nyquist bin
#             energy[:, 1:] *= 2
#
#         energy = energy[:, :-1]
#         r_frequencies = r_frequencies[:-1]
#
#     elif method == "spectrum": #amplitude spectrum
#         # camp = np.abs(np.sum(window_values))
#         #
#         # energy = rfft_values / camp
#         #
#         # if num_cols % 2 == 0:  # even length → Nyquist bin exists
#         #     energy[:, 1:-1] *= 2
#         # else:  # odd length → no Nyquist bin
#         #     energy[:, 1:] *= 2
#
#         cden = np.sqrt(np.sum(window_values** 2))
#         energy =  resolution * (np.abs(rfft_values / cden)**2)
#
#         if num_cols % 2 == 0:  # even length → Nyquist bin exists
#             energy[:, 1:-1] *= 2
#         else:  # odd length → no Nyquist bin
#             energy[:, 1:] *= 2
#
#         energy = energy[:, :-1]
#         r_frequencies = r_frequencies[:-1]
#
#     else:
#         raise ValueError(
#             f"""Unknown FFT Method: {method}
#                 FFT options: {'amplitude', 'psd', 'spectrum'}
#                 """)
#
#     # energy = rfft_values / scale_factor
#
#     return energy, r_frequencies



# class SpectralEstimator:
#     """
#     Estimator for spectrally-based uncertainty models.
#     """

#     def __init__(self,
#                  data: np.ndarray,
#                  resolution: int,
#                  multiple: int) -> None:
#         self.data = matrix2strip(data)
#         self.resolution = resolution
#         self.multiple = multiple
#         self.windowing = 'hann'

#     def compute_uncertainty(self) -> np.ndarray:
#         # Implement spectral estimation logic here
#         raise NotImplementedError


#     def compute_energy(data: np.ndarray,
#                        resolution: int,
#                        method: str,
#                        window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Compute FFT energy using 'method' process

#         Parameters
#         ----------
#         data : np.array
#                Input data
#         resolution : int
#                      Spatial resoluton of the array
#         method : str
#                  FFT Method used to estimate signal energy

#         Returns
#         -------
#         np.array
#                 Spectral energy in the signal
#         """

#         rfft_values = np.abs(np.fft.rfft(data, axis=1))
#         _, num_cols = rfft_values.shape
#         r_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)

#         if method == "amplitude":
#             scale_factor = np.sum(window_values)
#             if num_cols % 2 == 0:
#                 rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
#                 rfft_values = rfft_values[:, :-1]
#                 r_frequencies = r_frequencies[:-1]
#             else:
#                 rfft_values[:, 1:] = rfft_values[:, 1:] * 2

#         elif method == "psd":
#             scale_factor = np.sum(window_values ** 2) / resolution
#             rfft_values = rfft_values ** 2
#             if num_cols % 2 == 0:
#                 rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
#             else:
#                 rfft_values[:, 1:] = rfft_values[:, 1:] * 2
#             rfft_values = rfft_values[:, :-1]
#             r_frequencies = r_frequencies[:-1]

#         else:
#             raise ValueError(
#                 f"""Unknown FFT Method: {method}
#                     FFT options: {'amplitude', 'psd'}
#                     """)

#         energy = rfft_values / scale_factor

#         return energy, r_frequencies


# class amplitude_v2(SpectralEstimator):
#     def compute_uncertainty(self) -> np.ndarray:
#         data = self.data
#         resolution = self.resolution

#         if data.ndim < 2:
#             data = data.reshape(1, -1)

#         # create window
#         segment_window = signal.windows.get_window(
#             window=self.windowing, Nx=data.shape[1], fftbins=False)

#         # preprocess_signal, could be modified later
#         # data_mean = np.mean(data, axis=1)
#         preprocessed_signal = data * segment_window

#         energy, energy_freqs = self.compute_energy(
#             preprocessed_signal,
#             resolution,
#             "amplitude",
#             segment_window
#         )

#         # compute contribution per frequency
#         spatial_signal = create_spatial_signal(resolution, data.shape[1])
#         variance = energy @ spatial_signal
#         window_uncertainty = variance

#         # Remove edges when computing the original linespacing
#         linespacing_width = int((data.shape[1] - 2) / self.multiple)
#         # Include edges again for the output strip
#         output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
#         num_cols = output.shape[1]

#         selected_data = window_uncertainty[:, :int(num_cols / 2)]
#         output[:, :int(num_cols / 2)] = selected_data
#         output[:, int(num_cols / 2):] = np.fliplr(selected_data)

#         return output

# class psd_v2(SpectralEstimator):
#     def compute_uncertainty(self) -> np.ndarray:
#         data = self.data
#         resolution = self.resolution

#         if data.ndim < 2:
#             data = data.reshape(1, -1)

#         # create window
#         segment_window = signal.windows.get_window(
#             window=self.windowing, Nx=data.shape[1], fftbins=False)

#         # preprocess_signal, could be modified later
#         # data_mean = np.mean(data, axis=1)
#         preprocessed_signal = data * segment_window

#         energy, energy_freqs = self.compute_energy(
#             preprocessed_signal,
#             resolution,
#             "psd",
#             segment_window
#         )

#         # compute contribution per frequency
#         spatial_signal = create_spatial_signal(resolution, data.shape[1])
#         variance = energy @ spatial_signal
#         window_uncertainty = variance

#         # Remove edges when computing the original linespacing
#         linespacing_width = int((data.shape[1] - 2) / self.multiple)
#         # Include edges again for the output strip
#         output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
#         num_cols = output.shape[1]

#         selected_data = window_uncertainty[:, :int(num_cols / 2)]
#         output[:, :int(num_cols / 2)] = selected_data
#         output[:, int(num_cols / 2):] = np.fliplr(selected_data)

#         return output




#
# def compute_fft_uncertainty(
#     data: np.ndarray,
#     multiple: int,
#     resolution: int,
#     windowing: str = 'hann',
#     method: str = "amplitude",
# ) -> np.ndarray:
#     """
#     Estimate the uncertainty using FFT
#
#     Parameters
#     ----------
#     data : np.array
#            Input data for FFT estimation
#     multiple : int
#                Window length as multiple of the linespacing
#     resolution : int
#                  Input data resolution for frequency calculation
#     windowing : str
#                 Type of window to taper input
#                 options: scipy.signal.windows type
#     method : str
#              Type of FFT to estimate energy, defaults to 'amplitude'
#              options: ['amplitude', 'psd', 'spectrum']
#
#
#     Returns
#     -------
#     output : np.ndarray
#         Uncertainty estimate from the FFT method
#         To be compared with the residual error
#
#     """
#
#     if data.ndim < 2:
#         data = data.reshape(1, -1)
#
#     # create window
#     segment_window = signal.windows.get_window(
#         window=windowing, Nx=data.shape[1], fftbins=False)
#
#     # preprocess_signal, could be modified later
#     # data_mean = np.mean(data, axis=1)
#     preprocessed_signal = data * segment_window
#
#     energy, energy_freqs = compute_energy(
#         preprocessed_signal,
#         resolution,
#         method,
#         segment_window
#     )
#
#     # compute contribution per frequency
#     spatial_signal = create_spatial_signal(resolution, data.shape[1])
#     variance = energy @ spatial_signal
#
#     # normalize energy (convert m^2 to meters)
#     if method == "amplitude":
#         window_uncertainty = variance
#     else:
#         variance = variance / len(energy_freqs)
#         window_uncertainty = np.sqrt(variance)
#
#     # Remove edges when computing the original linespacing
#     linespacing_width = int((data.shape[1]-2) / multiple)
#     # Include edges again for the output strip
#     output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
#     num_cols = output.shape[1]
#
#     selected_data = window_uncertainty[:, :int(num_cols/2)]
#     output[:, :int(num_cols/2)] = selected_data
#     output[:, int(num_cols/2):] = np.fliplr(selected_data)
#
#
#     return output
#
# def compute_energy(data: np.ndarray,
#                    resolution: int,
#                    method: str,
#                    window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Compute FFT energy using 'method' process
#
#     Parameters
#     ----------
#     data : np.array
#            Input data
#     resolution : int
#                  Spatial resoluton of the array
#     method : str
#              FFT Method used to estimate signal energy
#
#     Returns
#     -------
#     np.array
#             Spectral energy in the signal
#     """
#
#     rfft_values = np.abs(np.fft.rfft(data, axis=1))
#     _, num_cols = rfft_values.shape
#     r_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)
#
#     if method == "amplitude":
#         scale_factor = np.sum(window_values)
#         if num_cols % 2 == 0:
#             rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
#             rfft_values = rfft_values[:, :-1]
#             r_frequencies = r_frequencies[:-1]
#         else:
#             rfft_values[:, 1:] = rfft_values[:, 1:] * 2
#
#     elif method == "psd":
#         scale_factor = np.sum(window_values**2) / resolution
#         rfft_values = rfft_values**2
#         if num_cols % 2 == 0:
#             rfft_values[:, 1:-1] = rfft_values[:, 1:-1] * 2
#         else:
#             rfft_values[:, 1:] = rfft_values[:, 1:] * 2
#         rfft_values = rfft_values[:, :-1]
#         r_frequencies = r_frequencies[:-1]
#
#     else:
#         raise ValueError(
#             f"""Unknown FFT Method: {method}
#                 FFT options: {'amplitude', 'psd'}
#                 """)
#
#     energy = rfft_values / scale_factor
#
#     return energy, r_frequencies



# # create specialized functions for each FFT Method
# compute_amp_uncertainty = partial(compute_fft_uncertainty,
#                                   selection='half',
#                                   windowing='hann',
#                                   method='amplitude')
# compute_psd_uncertainty = partial(compute_fft_uncertainty,
#                                   selection='half',
#                                   windowing='hann',
#                                   method='psd')

# include updated computation from Elias