from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class Bathymetry:
    """
    Class for storing generic bathymetry data
    """
    name: str
    data_type: str

    def __repr__(self) -> str:
        return f"Bathymetry File: ({self.name}, {self.data_type})"



class RasterBathymetry(Bathymetry):
    """
    Class for storing bathymetry data from rasters

    Includes variables for the resolution, bathy_data and nan_value
    """
    resolution: int
    data: np.ndarray[Tuple[int, int], np.dtype[float]]
    ndv_value: np.dtype[np.number]

    def __init__(self, name: str):
        self.name = name
        self.data_type = "raster"

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_data(self, data):
        self.data = data

    def set_nodata_value(self, ndv_value):
        self.ndv_value = ndv_value

    def __repr__(self) -> str:
        return (f"RasterBathymetry File: \n \
        filename: {self.name} \n \
        data_type: {self.data_type} \n \
        resolution: {self.resolution} \n \
        ndv_value: {self.ndv_value} \n \
        min\max value: [{np.nanmin(self.data.flatten()),np.nanmax(self.data.flatten())}] \n \
        data_shape: {self.data.shape}")


class BPSBathymetry(Bathymetry):
    """
    Class for storing Bathy Point Store bathymetry data
    [Placeholder]
    """

    def __init__(self, name: str):
        self.name = name
        self.data_type = "bps"


class CSVBathymetry(Bathymetry):
    """
    Class for storing CSV bathymetry data
    [Placeholder]
    """

    def __init__(self, name: str):
        self.name = name
        self.data_type = "csv"