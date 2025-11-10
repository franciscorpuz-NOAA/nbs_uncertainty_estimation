from abc import ABC, abstractmethod
from .bathymetry import RasterBathymetry, BPSBathymetry, CSVBathymetry
from osgeo import gdal
from pathlib import Path


class BathyReader(ABC):
    """
    Abstract class for reading bathymetry files
    """

    @abstractmethod
    def read_file(self, filename: str):
        # To be implemented by child classes
        raise NotImplementedError

    @staticmethod
    def is_valid_filename(filename: str) -> bool:
        # check validity of filename
        if not isinstance(filename, (str, Path)):
            raise TypeError("Filename must be a String or Path.")
        elif not Path(filename).exists():
            raise FileNotFoundError(f"TIFF file not found at path: {filename}")
        else:
            return True


class RasterBathyReader(BathyReader):
    """
    Class implementation for reading RasterBathymetry files
    Uses the GDal Library to read the bathymetry TIFF files
    """

    def read_file(self, filename: str) -> RasterBathymetry:

        if not self.is_valid_filename(filename):
            raise RuntimeError("Error reading bathymetry file")

        # Create RasterBathymetry object
        tiff_data = RasterBathymetry(filename)

        # Read TIFF file
        with gdal.Open(str(filename)) as ds:
            if not ds:
                raise RuntimeError(
                    f"GDAL failed to open TIFF file: '{filename}'")

            # Retrieve Bathymetric data
            depth_band = ds.GetRasterBand(1)
            if not depth_band:
                raise RuntimeError(
                    f"Error retrieving depth data from {filename}.")

            ndv = depth_band.GetNoDataValue()
            depth = depth_band.ReadAsArray()
            depth_gt = ds.GetGeoTransform()
            resolution = depth_gt[1]
            if resolution < 1:
                print("WARNING: detected resolution value is <= 1. \n \
                      Setting resolution value to 1")
                resolution = 1

            tiff_data.set_resolution(resolution)
            tiff_data.set_data(depth)
            tiff_data.set_nodata_value(ndv)

            return tiff_data


class BPSBathyReader(BathyReader):
    """
    Class implementation for reading BPS Bathymetry files
    """
    def read_file(self, filename: str) -> BPSBathymetry:

        if not self.is_valid_filename(filename):
            raise RuntimeError("Error reading bathymetry file")
        bps_data = BPSBathymetry(filename)
        return bps_data

class CSVBathyReader(BathyReader):
    """
    Class implementation for reading CSV Bathymetry files
    """
    def read_file(self, filename: str) -> CSVBathymetry:
        if not self.is_valid_filename(filename):
            raise RuntimeError("Error reading bathymetry file")
        csv_data = CSVBathymetry(filename)
        return csv_data


class FileReaderSelector:
    """
    Class implementation for selecting appropriate file reader by type
    """
    @staticmethod
    def select_reader(filename: str):
        if filename.endswith(".csv"):
            return CSVBathyReader
        elif filename.endswith(".tif"):
            return RasterBathyReader
        elif filename.endswith(".bps"):
            return BPSBathyReader
        else:
            raise RuntimeError(f"Unrecognized file type: {filename}")