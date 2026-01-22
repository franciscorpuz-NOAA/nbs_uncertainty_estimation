# from copy import deepcopy
#
# import numpy as np
#
# from .surfaceEstimators import RasterEstimator
# from scipy.stats import genextreme
# from ..readers.bathymetryDataset import RasterBathymetry
# from dataclasses import dataclass
#
# @dataclass
# class RasterSpatialEstimator(RasterEstimator):
#     """
#     Estimator for spectrally-based uncertainty models.
#     """
#
#     # bathy_data: RasterBathymetry
#     # multiple: int
#     # resolution: int
#     # windowing: str
#     # method: str
#     num_lines: int
#     num_samples: int
#     interpolation_cell_distance: int
#     shape: tuple[int, int]
#     data_strip: np.ndarray
#     min_window: int = 2
#
#
#
#     # column_indices: np.ndarray = None
#     # segment_window: np.ndarray = None
#     # depth_data_strip: np.ndarray = None
#     # depth_data_shape: tuple = None
#
#     def __init__(self, bathy_file: RasterBathymetry, *args, **kwargs) -> None:
#         super().__init__(bathy_file)
#         if 'min_window' in kwargs:
#             self.min_window = int(kwargs['min_window'])
#
#     # def pre_process(self) -> np.ndarray:
#     #     return super().pre_process()
#
#     # def post_process(self, uncertainty_strip:np.ndarray) -> np.ndarray:
#     #     return super().post_process(uncertainty_strip)
#
#     def estimate_surface(self):
#         self.data_strip = super().pre_process()
#         interpolation_cell_distance = self.data_strip.shape[1]
#         multiple = self.bathy_file.metadata['current_multiple']
#
#         self.num_lines, self.num_samples = self.data_strip.shape
#         self.interpolation_cell_distance = ((interpolation_cell_distance - 2) // multiple) + 2
#         self.shape = (self.num_lines, interpolation_cell_distance)


# class RasterResidual(RasterSpatialEstimator):
#
#     # def __init__(self, bathy_file: RasterBathymetry) -> None:
#     #     super().__init__(bathy_file)
#
#     def estimate_surface(self):
#         """
#         Calculate the variance of the provided depth array in parts.
#         """
#         super().estimate_surface()
#         depth_data_strip = self.data_strip
#         interpolated_strip = np.linspace(start=depth_data_strip[:, 0],
#                                          stop=depth_data_strip[:, -1],
#                                          num=depth_data_strip.shape[1])
#
#         interpolated_strip = interpolated_strip.T
#         residual_strip = depth_data_strip - interpolated_strip
#
#         residual = self.post_process(residual_strip)
#         new_bathy = deepcopy(self.bathy_file)
#         new_bathy.depth_data = residual
#         return new_bathy
#
#     def show_depth(self):
#         res = self.bathy_file.metadata['resolution']
#         super().__show_depth__(title=f"Interpolated Surface @{res}")


# class RasterSpatialStd(RasterSpatialEstimator):
#
#     # def __init__(self, bathy_file: RasterBathymetry) -> None:
#     #     super().__init__(bathy_file)
#
#     def estimate_surface(self):
#         """
#         Calculate the variance of the provided depth array in parts.
#         """
#         super().estimate_surface()
#
#         # containers
#         std_mean = np.zeros(self.shape)
#         std_max = np.zeros(self.shape)
#         std_envelope1 = np.zeros(self.shape)
#         std_envelope2 = np.zeros(self.shape)
#         std_envelope3 = np.zeros(self.shape)
#         win_len = 0
#
#
#         for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
#             num_convolutions = self.num_samples - win_len + 1
#             differences = np.full((self.num_lines, num_convolutions), 0.0)
#             for step in range(num_convolutions):
#                 mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
#                 maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
#                 stds = np.std(self.data_strip[:, step:step + win_len], axis=-1)
#                 differences[:, step] = maxs - mins
#
#                 std_mean[:, win_len - 1] = np.mean(stds, axis=-1)
#                 std_max[:, win_len - 1] = np.max(stds, axis=-1)
#                 std_std = np.std(stds, axis=-1)
#                 std_envelope1[:, win_len - 1] = std_mean[:, win_len - 1] + std_std
#                 std_envelope2[:, win_len - 1] = std_mean[:, win_len - 1] + 2 * std_std
#                 std_envelope3[:, win_len - 1] = std_mean[:, win_len - 1] + 3 * std_std
#
#         std_mean[:, -win_len:] = np.fliplr(std_mean[:, :win_len])
#         std_max[:, -win_len:] = np.fliplr(std_max[:, :win_len])
#         std_envelope1[:, -win_len:] = np.fliplr(std_envelope1[:, :win_len])
#         std_envelope2[:, -win_len:] = np.fliplr(std_envelope2[:, :win_len])
#         std_envelope3[:, -win_len:] = np.fliplr(std_envelope3[:, :win_len])
#         results = {'std_mean': std_mean,
#                    'std_max': std_max,
#                    'std_envelope1': std_envelope1,
#                    'std_envelope2': std_envelope2,
#                    'std_envelope3': std_envelope3}
#
#         for key in results.keys():
#             results[key] = self.post_process(results[key])
#
#         new_bathy = deepcopy(self.bathy_file)
#         new_bathy.metadata['results'] = results
#
#         return new_bathy


# class RasterSpatialDiff(RasterSpatialEstimator):
#
#     # def __init__(self, bathy_file: RasterBathymetry) -> None:
#     #     super().__init__(bathy_file)
#
#     def estimate_surface(self):
#
#         """
#         Calculate the variance of the provided depth array in parts.
#         """
#
#         super().estimate_surface()
#
#         # containers
#         difference_mean = np.zeros(self.shape)
#         difference_max = np.zeros(self.shape)
#         diff_envelope1 = np.zeros(self.shape)
#         diff_envelope2 = np.zeros(self.shape)
#         diff_envelope3 = np.zeros(self.shape)
#         win_len = 0
#
#         for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
#             num_convolutions = self.num_samples - win_len + 1
#             differences = np.full((self.num_lines, num_convolutions), 0.0)
#             for step in range(num_convolutions):
#                 mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
#                 maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
#                 differences[:, step] = maxs - mins
#
#                 diff_mean = np.mean(differences, axis=-1)
#                 diff_std = np.std(differences, axis=-1)
#                 diff_max = np.max(differences, axis=-1)
#                 difference_mean[:, win_len - 1] = diff_mean
#                 difference_max[:, win_len - 1] = diff_max
#                 diff_envelope1[:, win_len - 1] = diff_mean + diff_std
#                 diff_envelope2[:, win_len - 1] = diff_mean + 2 * diff_std
#                 diff_envelope3[:, win_len - 1] = diff_mean + 3 * diff_std
#
#         difference_mean[:, -win_len:] = np.fliplr(difference_mean[:, :win_len])
#         difference_max[:, -win_len:] = np.fliplr(difference_max[:, :win_len])
#         diff_envelope1[:, -win_len:] = np.fliplr(diff_envelope1[:, :win_len])
#         diff_envelope2[:, -win_len:] = np.fliplr(diff_envelope2[:, :win_len])
#         diff_envelope3[:, -win_len:] = np.fliplr(diff_envelope3[:, :win_len])
#         results = {'difference_mean': difference_mean,
#                    'difference_max': difference_max,
#                    'difference_envelope1': diff_envelope1,
#                    'difference_envelope2': diff_envelope2,
#                    'diff_envelope3': diff_envelope3,
#                    }
#
#         for key in results.keys():
#             results[key] = self.post_process(results[key])
#
#         new_bathy = deepcopy(self.bathy_file)
#         new_bathy.metadata['results'] = results
#
#         return new_bathy
#
# class RasterSpatialGEV(RasterSpatialEstimator):
#
#     # def __init__(self, bathy_file: RasterBathymetry) -> None:
#     #     super().__init__(bathy_file)
#
#     def estimate_surface(self):
#
#         """
#         Calculate the variance of the provided depth array in parts.
#         """
#
#         super().estimate_surface()
#
#         # containers
#         gev_mean = np.zeros(self.shape)
#         gev_p95_stats = np.zeros(self.shape)
#         gev_p99_stats = np.zeros(self.shape)
#         win_len = 0
#
#
#         for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
#             num_convolutions = self.num_samples - win_len + 1
#             differences = np.full((self.num_lines, num_convolutions), 0.0)
#             for step in range(num_convolutions):
#                 mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
#                 maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
#                 differences[:, step] = maxs - mins
#
#                 for i in range(self.num_lines):
#                     shape_, loc_, scale_ = genextreme.fit(differences[i])
#                     gev_mean[i, win_len - 1] = loc_
#                     gev_p95_stats[i, win_len - 1] = genextreme.ppf(0.95, shape_, loc_, scale_)
#                     gev_p99_stats[i, win_len - 1] = genextreme.ppf(0.99, shape_, loc_, scale_)
#
#         gev_mean[:, -win_len:] = np.fliplr(gev_mean[:, :win_len])
#         gev_p95_stats[:, -win_len:] = np.fliplr(gev_p95_stats[:, :win_len])
#         gev_p99_stats[:, -win_len:] = np.fliplr(gev_p99_stats[:, :win_len])
#
#         results = {'gev_mean': gev_mean,
#                    'gev_p95_stats': gev_p95_stats,
#                    'gev_p99_stats': gev_p99_stats
#                    }
#
#         for key in results.keys():
#             results[key] = self.post_process(results[key])
#
#         new_bathy = deepcopy(self.bathy_file)
#         new_bathy.metadata['results'] = results
#
#         return new_bathy
#
# class RasterSpatialGaussian(RasterSpatialEstimator):
#
#     # def __init__(self, bathy_file: RasterBathymetry) -> None:
#     #     super().__init__(bathy_file)
#
#     def estimate_surface(self):
#
#         """
#         Calculate the variance of the provided depth array in parts.
#         """
#
#         super().estimate_surface()
#
#         # containers
#         gaussian_mean = np.zeros(self.shape)
#         gaussian_p95_stats = np.zeros(self.shape)
#         gaussian_p99_stats = np.zeros(self.shape)
#         win_len = 0
#
#
#         for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
#             num_convolutions = self.num_samples - win_len + 1
#             differences = np.full((self.num_lines, num_convolutions), 0.0)
#             for step in range(num_convolutions):
#                 mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
#                 maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
#                 stds = np.std(self.data_strip[:, step:step + win_len], axis=-1)
#                 differences[:, step] = maxs - mins
#
#                 gaussian_mean[:, win_len - 1] = np.mean(differences, axis=-1)
#                 gaussian_p95_stats[:, win_len - 1] = np.percentile(differences, 95, axis=-1)
#                 gaussian_p99_stats[:, win_len - 1] = np.percentile(differences, 99, axis=-1)
#
#         gaussian_mean[:, -win_len:] = np.fliplr(gaussian_mean[:, :win_len])
#         gaussian_p95_stats[:, -win_len:] = np.fliplr(gaussian_p95_stats[:, :win_len])
#         gaussian_p99_stats[:, -win_len:] = np.fliplr(gaussian_p99_stats[:, :win_len])
#         results = {'gaussian_mean': gaussian_mean,
#                    'gaussian_p95_stats': gaussian_p95_stats,
#                    'gaussian_p99_stats': gaussian_p99_stats
#                    }
#         for key in results.keys():
#             results[key] = self.post_process(results[key])
#
#         new_bathy = deepcopy(self.bathy_file)
#         new_bathy.metadata['results'] = results
#
#         return new_bathy


















#
# def get_difference_uncertainties(depth:np.ndarray,
#                                       interpolation_cell_distance:int,
#                                       min_window:int = 2,
#                                       multiple:int = 1,
#                                       method:str = 'max') -> np.ndarray:
#     """
#     Calculate the variance of the provided depth array in parts.
#     """
#     num_lines, num_samples = depth.shape
#     interpolation_cell_distance = ((interpolation_cell_distance-2)//multiple)+2
#     # interpolation_cell_distance = num_samples
#     differences = np.array([])
#     output = np.full((num_lines, interpolation_cell_distance), 0.0)
#     # difference_max = np.full((num_lines, interpolation_cell_distance), 0.0)
#     # difference_mean = np.full((num_lines, interpolation_cell_distance), 0.0)
#     # difference_std = np.full((num_lines, interpolation_cell_distance), 0.0)
#     for win_len in range(min_window, interpolation_cell_distance//2+1):
#         # num_convolutions = num_samples - win_len + 1
#         # differences = np.full((num_lines, num_convolutions), 0.0)
#         # for step in range(num_convolutions):
#         #     mins = np.min(depth[:,step:step+win_len], axis = -1)
#         #     maxs = np.max(depth[:,step:step+win_len], axis = -1)
#         #     stds = np.std(depth[:,step:step+win_len], axis = -1)
#         #     differences[:,step] = maxs - mins
#         # difference_max[:, win_len-1] = np.max(differences, axis=-1)
#         # difference_mean[:, win_len-1] = np.mean(differences, axis =-1)
#         # difference_std[:, win_len-1] = np.std(differences, axis =-1)
#         windows = np.lib.stride_tricks.sliding_window_view(depth, window_shape=win_len, axis=-1)
#         mins = np.min(windows, axis = -1)
#         maxs = np.max(windows, axis = -1)
#         stds = np.std(windows, axis = -1)
#         differences = maxs - mins
#
#         if method == 'max':
#             output[:, win_len-1] = np.max(differences, axis=-1)
#         elif method == 'mean':
#             output[:, win_len-1] = np.mean(differences, axis =-1)
#         elif method == 'std':
#             output[:, win_len-1] = np.std(differences, axis =-1)
#         else:
#             raise ValueError(f"Unknown method: {method}. \
#                            Choose from 'max', 'mean', or 'std'.")
#
#         # elif method == 'gaussian':
#         #     output[:, win_len-1] =
#         # elif method == 'gev':
#         #     return difference_std
#
#     # print(f"final convolutions for window length {win_len} is {num_convolutions}")
#     # difference_max[:,-win_len:] = np.fliplr(difference_max[:, :win_len])
#     # difference_mean[:,-win_len:] = np.fliplr(difference_mean[:, :win_len])
#     # difference_std[:,-win_len:] = np.fliplr(difference_std[:,:win_len])
#
#     else:
#         raise ValueError(f"Unknown method: {method}. \
#                            Choose from 'max', 'mean', or 'std'.")
#
#
#


#
# def get_interp_residual(depth):
#     """
#     Calculate the residual between the interpolated depth and the original depth.
#     Interpolated depth is a line between the two end values.
#     """
#
#     slopes = (depth[:, -1] - depth[:,0]) / depth.shape[1]
#     start_vals = depth[:, 0]
#     # interpolation = np.arange(len(depth))
#     # interpolation = interpolation * slopes + start_vals
#     interpolation = np.tile(np.arange(depth.shape[1]), (depth.shape[0],1))
#     interpolation = interpolation * slopes + start_vals
#     residual = depth - interpolation
#
#     return residual, interpolation
#
#
# def preprocess_strip(depth):
#     """
#     Remove the mean and apply a Hann window to the rows of the provided depth array.
#     """
#     shape = depth.shape[1]
#     window = np.hanning(shape)
#     depth = depth * window
#     return depth
#
#
# def psd_strip_modified(depth, resolution):
#     """
#     Calculate the power spectral density of the provided depth array.
#
#     For background on the inline comments, see https://docs.scipy.org/doc/scipy/tutorial/signal.html#sampled-sine-signal
#     """
#     shape = depth.shape[1]
#     hann_win = np.hanning(shape)
#     hann_scale = 1 / np.sqrt(np.sum(np.square(hann_win)))
#     fft_data_windowed = np.fft.rfft(depth, axis=1) * hann_scale  # this is $X_l^w$
#     psd_1sided = 2 * resolution * np.square(np.abs(fft_data_windowed))
#     # drop zero frequency
#     frequencies = np.fft.rfftfreq(shape, resolution)[1:]
#     # frequencies = np.fft.rfftfreq(shape, resolution)
#     # psd_result = np.zeros_like(psd_1sided)
#     # psd_result[:, 1:shape] = psd_1sided[:, 1:shape]
#     psd_result = psd_1sided[:, 1:shape]
#     return psd_result, frequencies
#
#
# def spectrum_strip(depth, resolution):
#     """
#     Modified psd_strip_modified function to compute spectrum instead of psd
#     Need to check if units are similar
#
#     """
#     shape = depth.shape[1]
#     hann_win = np.hanning(shape)
#     hann_scale = 1 / np.sqrt(np.sum(hann_win)**2)
#     fft_data_windowed = np.fft.rfft(depth, axis=1) * hann_scale  # this is $X_l^w$
#     spectrum_1sided = 2 * resolution * np.square(np.abs(fft_data_windowed))
#     # drop zero frequency
#     frequencies = np.fft.rfftfreq(shape, resolution)[1:]
#     # frequencies = np.fft.rfftfreq(shape, resolution)
#     # psd_result = spectrum_1sided[:, 1:shape]
#     # psd_result = np.zeros_like(spectrum_1sided)
#     # psd_result[:, 1:shape] = spectrum_1sided[:, 1:shape]
#     psd_result = spectrum_1sided[:, 1:shape]
#     return psd_result, frequencies
#
#
# def amp_strip(depth, resolution):
#     """
#     Calculate the power spectral density of the provided depth array.
#
#     """
#     shape = depth.shape[1]
#     hann_win = np.hanning(shape)
#     hann_scale = 1 / np.sum(hann_win)
#     # fft_data_windowed = np.fft.rfft(depth, axis=1) * hann_scale
#     fft_data_windowed = np.fft.rfft(depth, axis=1)
#     fft_data_1sided = 2 * np.abs(fft_data_windowed)
#     # drop zero frequency
#     frequencies = np.fft.rfftfreq(depth.shape[1], resolution)[1:]
#     # frequencies = np.fft.rfftfreq(depth.shape[1], resolution)
#     # amp_result = fft_data_1sided[:, 1:]
#     # amp_result = np.zeros_like(fft_data_1sided)
#     # amp_result[:, 1:] = fft_data_1sided[:, 1:]
#     amp_result = fft_data_1sided[:, 1:]
#     return amp_result, frequencies
#
#
# class spatial_contributions:
#     def __init__(self, resolution, max_cell_number):
#         """
#         Create the distance and frequency dependent scaling factors.
#         """
#         # self.frequencies = frequencies inside the window
#         self.frequencies = np.fft.rfftfreq(max_cell_number, resolution)[1:]
#         # self.frequencies = np.fft.rfftfreq(max_cell_number, resolution)
#         self.distances = np.arange(max_cell_number) * resolution
#         distances_2d, freq_2d = np.meshgrid(self.distances, self.frequencies)
#         self.spatial_scale = distances_2d * freq_2d
#         self.spatial_scale = np.where(
#             self.spatial_scale < 0.25, self.spatial_scale, 0.25
#         )
#         self.spatial_signal = np.sin(self.spatial_scale * 2 * np.pi)
#
#     def get_uncertainties(self, pxx, fft_freq,
#                           interpolation_cell_distance, multiple):
#         """
#         Multiply the scaling factors by the amplitude to get the uncertainties.
#         """
#         freq_idx = np.where(np.isin(fft_freq, self.frequencies))[0]
#         # freq_contributions = psd * self.spatial_signal[freq_idx,:]
#         # should add a check to make sure self.frequencies are matched appropriately to psd_freq
#         # print(f"fft_freq: {fft_freq}, fft_freq length: {len(fft_freq)}")
#         # print(f"frequencies: {self.frequencies}, frequencies length: {len(self.frequencies)}")
#         # print(f"pxx shape: {pxx.shape}, freq_idx length: {len(freq_idx)}")
#         # print(f"spatial signal shape: {self.spatial_signal.shape}")
#         window_uncertainties = (pxx @ self.spatial_signal) / len(freq_idx)
#         # window_uncertainties = (pxx @ self.spatial_signal)
#         # zero_column = np.zeros((pxx.shape[0], 1))
#         # window_uncertainties = np.column_stack((zero_column, window_uncertainties))
#         # print(f"window_uncertainties shape: {window_uncertainties.shape}")
#         interpolation_cell_distance = int(interpolation_cell_distance)
#         interpolation_uncertainties = np.zeros((pxx.shape[0], ((interpolation_cell_distance-2)//multiple)+2))
#
#         half_window = (interpolation_uncertainties.shape[1] // 2)
#         interpolation_uncertainties[:, :half_window] = (
#             window_uncertainties[:, :half_window])
#         interpolation_uncertainties[:, half_window:] = np.fliplr(
#             window_uncertainties[:, :half_window])
#
#         return interpolation_uncertainties
#
# def glen_get_uncertainties(depth, resolution, multiple, method="amplitude"):
#
#     depth = preprocess_strip(depth)
#     if method == "amplitude":
#         energy, freq = amp_strip(depth, resolution)
#         scaler = spatial_contributions(resolution, depth.shape[1])
#         uncertainty = scaler.get_uncertainties(energy,
#                                                freq,
#                                                depth.shape[1],
#                                                multiple)
#
#     elif method == "psd":
#         energy, freq = psd_strip_modified(depth, resolution)
#         scaler = spatial_contributions(resolution, depth.shape[1])
#         uncertainty = scaler.get_uncertainties(energy,
#                                                freq,
#                                                depth.shape[1],
#                                                multiple)
#         uncertainty = np.sqrt(uncertainty)
#
#     elif method == "spectrum":
#         energy, freq = spectrum_strip(depth, resolution)
#         scaler = spatial_contributions(resolution, depth.shape[1])
#         uncertainty = scaler.get_uncertainties(energy,
#                                                freq,
#                                                depth.shape[1],
#                                                multiple)
#         uncertainty = np.sqrt(uncertainty) * resolution
#
#     else:
#         raise ValueError(f"Unrecognized Method: {method}")
#
#     return uncertainty
#
#
