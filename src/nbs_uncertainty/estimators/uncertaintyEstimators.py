from ..processors.bathy_processors import BathyProcessor
from ..readers.bathymetry import RasterBathymetry
from ..utils import helper


#Method lists
@BathyProcessor.register("residual", RasterBathymetry)
def compute_raster_residual(bathy_data: RasterBathymetry, param: dict | None):
    return helper.compute_residual(bathy_data, param)

# @BathyProcessor.register("psd_v1", RasterBathymetry)
# def glen_psd()



