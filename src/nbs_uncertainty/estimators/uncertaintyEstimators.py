from ..processors.bathy_processors import BathyProcessor
from ..readers.bathymetry import Bathymetry, RasterBathymetry
from ..utils import helper
from typing import Type
from ..estimators.spectralEstimators import compute_fft_uncertainty, compute_fft_uncertainty_elias
from ..estimators.spatialEstimators import get_difference_uncertainties


# Method lists
@BathyProcessor.register("residual", RasterBathymetry)
def compute_raster_residual(bathy_data: RasterBathymetry, param: dict | None):
    return helper.compute_residual(bathy_data, param)


@BathyProcessor.register("psd_v1", RasterBathymetry)
def glen_psd(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty(
        data=bathy_data.data,
        multiple=param.get("multiple", 1),
        resolution=param.get("resolution", 1),
        windowing=param.get("windowing", "hann"),
        method="psd",
        selection="half")


@BathyProcessor.register("amp_v1", RasterBathymetry)
def glen_amplitude(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty(
        data=bathy_data.data,
        multiple=param.get("multiple", 1),
        resolution=param.get("resolution", 1),
        windowing=param.get("windowing", "hann"),
        method="amplitude",
        selection="half")
    
    
@BathyProcessor.register("amp_v2", RasterBathymetry)
def elias_amplitude(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty_elias(
            data=bathy_data.data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "amplitude"),
            selection=param.get("selection", "half"),
        )
    
    
@BathyProcessor.register("psd_n", RasterBathymetry)
def elias_psd_n(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty_elias(
            data=bathy_data.data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd_n"),
            selection=param.get("selection", "half"),
        )
    
    
@BathyProcessor.register("psd_lf", RasterBathymetry)
def elias_psd_lf(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty_elias(
            data=bathy_data.data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd_lf"),
            selection=param.get("selection", "half"),
        )
    
    
@BathyProcessor.register("spectrum", RasterBathymetry)
def elias_spectrum(bathy_data: RasterBathymetry, param: dict):
    return compute_fft_uncertainty_elias(
            data=bathy_data.data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "spectrum"),
            selection=param.get("selection", "half"),
        )
    
    
@BathyProcessor.register("diff_max", RasterBathymetry)
def diff_max(bathy_data: RasterBathymetry, param: dict):
    return get_difference_uncertainties(data=bathy_data.data,
                                        interpolation_cell_distance=bathy_data.data.shape[1],
                                        min_window=2, 
                                        multiple=param.get("multiple", 1),
                                        method='max')
    
@BathyProcessor.register("diff_max", CSVBathymetry)
def diff_max(bathy_data: RasterBathymetry, param: dict):
    return get_difference_uncertainties_csv(data=bathy_data.data,
                                        interpolation_cell_distance=bathy_data.data.shape[1],
                                        min_window=2, 
                                        multiple=param.get("multiple", 1),
                                        method='max')

@BathyProcessor.register("diff_mean", RasterBathymetry)
def diff_mean(bathy_data: RasterBathymetry, param: dict):
    return get_difference_uncertainties(data=bathy_data.data,
                                        interpolation_cell_distance=bathy_data.data.shape[1],
                                        min_window=2, 
                                        multiple=param.get("multiple", 1),
                                        method='min')


