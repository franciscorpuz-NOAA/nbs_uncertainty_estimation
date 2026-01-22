from nbs_uncertainty.ignore.estimatorRegister import BathyProcessor
from nbs_uncertainty.processors.rasterSpatialProcessors import (RasterSpatialDiff,
                                                                RasterSpatialGaussian,
                                                                RasterSpatialGEV,
                                                                RasterSpatialStd)
from nbs_uncertainty.processors.rasterSpectralProcessors import (EliasUncertainty)
# from ..ignore.spatialEstimators import get_difference_uncertainties
# from ..ignore.spectralEstimators import compute_fft_uncertainty, compute_fft_uncertainty_elias

# Spectral Estimators
@BathyProcessor.register("amp_v1", RasterBathymetry)
def glen_amplitude(bathy_file: RasterBathymetry, *args, **kwargs):
    return AmpEstimatorRaster(
        bathy_file=bathy_file,
        method = '').estimate_surface()

@BathyProcessor.register("psd_v1", RasterBathymetry)
def glen_psd(bathy_file: RasterBathymetry, *args, **kwargs):
    return PSDV1Raster(
        bathy_file=bathy_file,
        method = '').estimate_surface()

@BathyProcessor.register("amp_v2", RasterBathymetry)
def elias_amplitude(bathy_file: RasterBathymetry, *args, **kwargs):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="amplitude").estimate_surface()

@BathyProcessor.register("psd_v2", RasterBathymetry)
def elias_psd_n(bathy_file: RasterBathymetry):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="psd").estimate_surface()

@BathyProcessor.register("psd_n", RasterBathymetry)
def elias_psd_n(bathy_file: RasterBathymetry):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="psd_n").estimate_surface()
    
    
@BathyProcessor.register("psd_lf", RasterBathymetry)
def elias_psd_lf(bathy_file: RasterBathymetry):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="psd_lf").estimate_surface()

@BathyProcessor.register("psd_df", RasterBathymetry)
def elias_psd_lf(bathy_file: RasterBathymetry):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="psd_df").estimate_surface()

@BathyProcessor.register("spectrum", RasterBathymetry)
def elias_spectrum(bathy_file: RasterBathymetry):
    return EliasUncertainty(
            bathy_file=bathy_file,
            method="spectrum").estimate_surface()
    

## Spatial Estimators
@BathyProcessor.register("residual", RasterBathymetry)
def compute_raster_residual(bathy_file: RasterBathymetry, *args, **kwargs):
    return RasterResidual(bathy_file).estimate_surface()

@BathyProcessor.register("spatial_std", RasterBathymetry)
def get_spatial_std(bathy_file: RasterBathymetry, *args, **kwargs):
    return RasterSpatialStd(
        bathy_file=bathy_file,
        # interpolation_cell_distance=bathy_data.depth.shape[1],
        min_window=2
        # multiple=param.get("multiple", 1),
        # method='max'
    ).estimate_surface()


@BathyProcessor.register("spatial_diff", RasterBathymetry)
def get_spatial_diff(bathy_file: RasterBathymetry, *args, **kwargs):
    return RasterSpatialDiff(
        bathy_file=bathy_file,
        # interpolation_cell_distance=bathy_data.depth.shape[1],
        min_window=2
        # multiple=param.get("multiple", 1),
        # method='ave'
    ).estimate_surface()

@BathyProcessor.register("spatial_gev", RasterBathymetry)
def get_spatial_gev(bathy_file: RasterBathymetry, *args, **kwargs):
    return RasterSpatialGEV(
        bathy_file=bathy_file,
        # interpolation_cell_distance=bathy_data.depth.shape[1],
        min_window=2
        # multiple=param.get("multiple", 1),
        # method='std'
    ).estimate_surface()


@BathyProcessor.register("spatial_gaussian", RasterBathymetry)
def get_spatial_gaussian(bathy_file: RasterBathymetry, *args, **kwargs):
    return RasterSpatialGaussian(
        bathy_file=bathy_file,
        # interpolation_cell_distance=bathy_data.depth.shape[1],
        min_window=2
        # multiple=param.get("multiple", 1),
        # method='p95'
    ).estimate_surface()


# Replacement Estimators
