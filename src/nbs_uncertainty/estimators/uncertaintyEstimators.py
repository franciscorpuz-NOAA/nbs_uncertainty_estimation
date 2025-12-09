from ..processors.bathy_processors import BathyProcessor
from ..readers.bathymetry import RasterBathymetry
from ..utils import helper
from ..estimators.spectralEstimators import AmpV1, PSDV1, EliasUncertainty
from ..estimators.spatialEstimators import SpatialStd, SpatialDiff, SpatialGEV, SpatialGaussian
# from ..estimators.spatialEstimators import get_difference_uncertainties
# from ..estimators.spectralEstimators import compute_fft_uncertainty, compute_fft_uncertainty_elias

# Method lists
@BathyProcessor.register("residual", RasterBathymetry)
def compute_raster_residual(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return helper.compute_residual(bathy_data, param)

@BathyProcessor.register("amp_v1", RasterBathymetry)
def glen_amplitude(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return AmpV1(
        data=bathy_data,
        multiple=param.get("multiple", 1),
        resolution=param.get("resolution", 1),
        windowing=param.get("windowing", "hann"),
        method="amplitude").compute_uncertainty()

@BathyProcessor.register("psd_v1", RasterBathymetry)
def glen_psd(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return PSDV1(
        data=bathy_data,
        multiple=param.get("multiple", 1),
        resolution=param.get("resolution", 1),
        windowing=param.get("windowing", "hann"),
        method="psd").compute_uncertainty()
    
    
@BathyProcessor.register("amp_v2", RasterBathymetry)
def elias_amplitude(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "amplitude"),
        ).compute_uncertainty()
    
@BathyProcessor.register("psd_v2", RasterBathymetry)
def elias_psd(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd"),
        ).compute_uncertainty()

@BathyProcessor.register("psd_n", RasterBathymetry)
def elias_psd_n(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd_n"),
        ).compute_uncertainty()
    
    
@BathyProcessor.register("psd_lf", RasterBathymetry)
def elias_psd_lf(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd_lf"),
        ).compute_uncertainty()

@BathyProcessor.register("psd_df", RasterBathymetry)
def elias_psd_lf(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "psd_df"),
        ).compute_uncertainty()
    
@BathyProcessor.register("spectrum", RasterBathymetry)
def elias_spectrum(bathy_data: RasterBathymetry, *args, **kwargs):
    param = kwargs.get("param", {})
    return EliasUncertainty(
            data=bathy_data,
            multiple=param.get("multiple", 1),
            resolution=bathy_data.metadata.get('resolution', 1),
            windowing=param.get("windowing", "hann"),
            method=param.get("method", "spectrum"),
        ).compute_uncertainty()
    

## Spatial Estimators
@BathyProcessor.register("spatial_std", RasterBathymetry)
def get_spatial_std(bathy_data: RasterBathymetry, *args, **kwargs):
    return SpatialStd(
        bathy_data=bathy_data,
        # interpolation_cell_distance=bathy_data.data.shape[1],
        min_window=2,
        # multiple=param.get("multiple", 1),
        # method='max'
    ).compute_uncertainty()


@BathyProcessor.register("spatial_diff", RasterBathymetry)
def get_spatial_diff(bathy_data: RasterBathymetry, *args, **kwargs):
    return SpatialDiff(
        bathy_data=bathy_data,
        # interpolation_cell_distance=bathy_data.data.shape[1],
        min_window=2,
        # multiple=param.get("multiple", 1),
        # method='ave'
    ).compute_uncertainty()

@BathyProcessor.register("spatial_gev", RasterBathymetry)
def get_spatial_gev(bathy_data: RasterBathymetry, *args, **kwargs):
    return SpatialGEV(
        bathy_data=bathy_data,
        # interpolation_cell_distance=bathy_data.data.shape[1],
        min_window=2,
        # multiple=param.get("multiple", 1),
        # method='std'
    ).compute_uncertainty()


@BathyProcessor.register("spatial_gaussian", RasterBathymetry)
def get_spatial_gaussian(bathy_data: RasterBathymetry, *args, **kwargs):
    return SpatialGaussian(
        bathy_data=bathy_data,
        # interpolation_cell_distance=bathy_data.data.shape[1],
        min_window=2,
        # multiple=param.get("multiple", 1),
        # method='p95'
    ).compute_uncertainty()
