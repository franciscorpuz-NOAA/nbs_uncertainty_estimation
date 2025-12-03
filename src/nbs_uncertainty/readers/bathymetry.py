from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

from .read_helpers import (is_valid_filename,
                          read_config_file,
                          remove_edge_ndv)

CONFIG_FILE = "../config/config.ini"

@dataclass
class Bathymetry:
    """
    Class for storing generic bathymetry data

    :param
    filename - bathymetry source filename
    """
    data: np.ndarray
    metadata: Dict

    def __init__(self,
                 filename: str | Path,
                 config_file: str = CONFIG_FILE):

        # extract directory location of bathymetry data from config file
        if not is_valid_filename(config_file):
            raise FileNotFoundError(f"Config file not found in {config_file}")
        config = read_config_file(config_file)

        # check if filename exists
        data_dir = str(config['DEFAULT']['data_directory'])
        filename = str(Path(data_dir) / filename)
        if not is_valid_filename(filename):
            raise RuntimeError(f"Error reading bathymetry file at {self.metadata['filename']}")

        self.metadata = {'filename': filename, 'config': config}

    @property
    def filename(self):
        full_path = self.metadata['filename']
        return (f"Filename: {Path(full_path).name}"
                f"\n Location: {Path(full_path).parent}")

    def __repr__(self) -> str:
        return (f"data shape: {self.data.shape}"
                f"\n metadata: {self.metadata}")


@dataclass
class RasterBathymetry(Bathymetry):
    """
    Bathymetry subclass representing depth data from rasters

    Data Structure: [data, metadata]
    data : depth data from raster
    metadata : dictionary of raster-specific data (resolution, no_data_value, etc)
    """

    def __init__(self,
                 filename: str | Path,
                 config_file: str = CONFIG_FILE):
        """

        Parameters
        ----------
        filename: TIFF filename
        config_file: Optional config file

        Implements the following operations:
        1. Read input TIFF file
        2. Saves other TFF information in the "metadata" dictionary (ndv_value, resolution)
        3. Removes NDV values from the data array (can be controlled in the config file under RASTER)

        """
        super().__init__(filename, config_file)
        self.metadata['data_type'] = "raster"

        # read raster file using GDAL
        filename = self.metadata['filename']
        with gdal.Open(str(filename)) as ds:
            if not ds:
                raise RuntimeError(
                    f"GDAL failed to open TIFF file: '{filename}'")

            # Retrieve Bathymetric data
            depth_band = ds.GetRasterBand(1)
            if not depth_band:
                raise RuntimeError(
                    f"Error retrieving depth data from {filename}.")

            self.metadata['ndv_value'] = depth_band.GetNoDataValue()
            self.data = depth_band.ReadAsArray()
            depth_gt = ds.GetGeoTransform()
            self.metadata['resolution'] = depth_gt[1]
            if self.metadata['resolution'] < 1:
                print(f"WARNING: detected resolution value is <= 1"
                      f"\n Setting resolution value to 1")
                self.metadata['resolution'] = 1

        if bool(self.metadata['config']['RASTER']['remove_ndv']):
            self.data = remove_edge_ndv(self.data, self.metadata['ndv_value'])

    @property
    def resolution(self):
        return self.metadata['resolution']

    @property
    def min_val(self):
        return np.nanmin(self.data.flatten())

    @property
    def max_val(self):
        return np.nanmax(self.data.flatten())

    @property
    def ndv_value(self):
        return self.metadata['ndv_value']

    def __repr__(self) -> str:
        return (f'Type: RasterBathymetry'
                f'\n {self.filename}'
                f'\n Resolution: {self.resolution}'
                f'\n No Data Value: {self.ndv_value}'
                f'\n Min\Max value: [{self.min_val, self.max_val}]'
                f'\n Data Shape: {self.data.shape}'
                )


class BPSBathymetry(Bathymetry):
    """
    Bathymetry subclass representing depth data from BPS

    Data Structure: [data, metadata]
    data : depth data from BPS
    metadata : dictionary of BPS-specific data (to be defined...)
    """

    def __post_init__(self):
        self.metadata['data_type'] = "bps"



class CSVBathymetry(Bathymetry):
    """
    Data Structure: [data, metadata]

    data : depth data from CSV
    metadata : dictionary of CSV-specific data (to be defined...)
    """

    def __post_init__(self):
        self.metadata['data_type'] = "csv"





# Helper Functions

def load_file(filename: str | Path):
    """
    Function to read bathymetric filenames
    :param filename:
    :return: Bathymetry class object
    """

    filename = str(filename)
    if filename.endswith(".csv"):
        return CSVBathymetry(filename)
    elif filename.endswith((".tif", ".tiff")):
        return RasterBathymetry(filename)
    elif filename.endswith(".bps"):
        return BPSBathymetry(filename)
    else:
        raise RuntimeError(f"Unrecognized file type: {filename}")



