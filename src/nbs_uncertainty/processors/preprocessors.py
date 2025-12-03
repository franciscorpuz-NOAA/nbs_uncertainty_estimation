import numpy as np
from ..readers.bathymetry import RasterBathymetry


def make_square(raster_data: RasterBathymetry) -> RasterBathymetry:
    new_dim = np.min(raster_data.data.shape)
    new_raster = raster_data.data[:new_dim, :new_dim]
    raster_data.set_data(new_raster)
    return raster_data

