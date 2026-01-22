import numpy as np

from nbs_uncertainty.readers.bathymetryDataset import RasterDataset
from nbs_uncertainty.processors.rasterProcessor import RasterProcessorClass
from nbs_uncertainty.processors.rasterSpectralProcessors import (GlenPSD,
                                                                 GlenAmplitude,
                                                                 EliasUncertainty)
from nbs_uncertainty.processors.rasterSpatialProcessors import (RasterSpatialStd,
                                                                RasterSpatialDiff,
                                                                RasterSpatialGEV,
                                                                RasterSpatialGaussian)

from functools import partial

raster_methods = {'amp_v1': GlenAmplitude,
                  'psd_v1': GlenPSD,
                  'amp_v2': partial(EliasUncertainty, method='amplitude'),
                  'psd_n': partial(EliasUncertainty, method='psd_n'),
                  'psd_lf': partial(EliasUncertainty, method='psd_lf'),
                  'psd_df': partial(EliasUncertainty, method='psd_df'),
                  'spectrum': partial(EliasUncertainty, method='spectrum'),
                  'spatial_std': partial(RasterSpatialStd, method='spatial_std'),
                  'spatial_diff': partial(RasterSpatialDiff, method='spatial_diff'),
                  'spatial_gaussian': partial(RasterSpatialGaussian, method='spatial_gaussian')
                  }

class RasterProcessorFactory:
    """
    Base class for methods that uses BathymetryDataset as input
    """
    def __init__(self, linespacing_meters: int,
                        multiple: int,
                        max_multiple: int):

        self.linespacing_meters = linespacing_meters
        self.multiple = multiple
        self.max_multiple = max_multiple

    def to_strip(self, bathy_data: RasterDataset):
        rasterProcessor = RasterProcessorClass(bathydataset=bathy_data,
                                               linespacing_meters=self.linespacing_meters,
                                               current_multiple=self.multiple,
                                               max_multiple=self.max_multiple)
        return rasterProcessor.as_strip


    def compute_residual(self, bathy_data: RasterDataset) -> np.ndarray:
        rasterProcessor = RasterProcessorClass(bathydataset=bathy_data,
                                               linespacing_meters=self.linespacing_meters,
                                               current_multiple=self.multiple,
                                               max_multiple=self.max_multiple)
        return rasterProcessor.residual_surface

    def get_column_indices(self, bathy_data: RasterDataset) -> np.ndarray:
        rasterProcessor = RasterProcessorClass(bathydataset=bathy_data,
                                               linespacing_meters=self.linespacing_meters,
                                               current_multiple=self.multiple,
                                               max_multiple=self.max_multiple)
        return rasterProcessor.column_indices

    def estimate_surface(self, method: str, bathy_data: RasterDataset):
        estimation_method = raster_methods[method]
        return estimation_method(bathydataset=bathy_data,
                                 linespacing_meters=self.linespacing_meters,
                                 current_multiple=self.multiple,
                                 max_multiple=self.max_multiple).estimate_uncertainty()






