# FOR REMOVAL

# from abc import ABC, abstractmethod
# from pathlib import Path
# import numpy as np
# import configparser
#
# from osgeo import gdal
#
# from .bathymetry import RasterBathymetry, BPSBathymetry, CSVBathymetry
# from ..processors.preprocessors import remove_edge_ndv, make_square
#
#
# class BathyReader(ABC):
#     """
#     Abstract class for reading bathymetry files
#     """
#
#     def __init__(self,
#                  config_path: str = "../config/config.ini",
#                  filename: str | Path = None) -> None:
#         self.config = configparser.ConfigParser()
#         self.config.read(Path(config_path))
#         self.filename = Path(filename)
#
#
#     @abstractmethod
#     def read_file(self):
#         # To be implemented by child classes
#         raise NotImplementedError
#
#     @staticmethod
#     def is_valid_filename(filename: str | Path) -> bool:
#         # check validity of filename
#         if not isinstance(filename, (str, Path)):
#             raise TypeError("Filename must be a String or Path.")
#         elif not Path(filename).exists():
#             raise FileNotFoundError(f"TIFF file not found at path: {filename}")
#         else:
#             return True
#
#
# class RasterBathyReader(BathyReader):
#     """
#     Class implementation for reading RasterBathymetry files
#     Uses the GDal Library to read the bathymetry TIFF files
#     """
#     gdal.UseExceptions()
#     def __init__(self, filename: str | Path) -> None:
#         super().__init__(filename=filename)
#
#     def read_file(self) -> RasterBathymetry:
#         filename = self.filename
#
#         if not self.is_valid_filename(filename):
#             raise RuntimeError("Error reading bathymetry file")
#
#         # Create RasterBathymetry object
#         tiff_data = RasterBathymetry(filename)
#
#         # Read TIFF file
#         with gdal.Open(str(filename)) as ds:
#             if not ds:
#                 raise RuntimeError(
#                     f"GDAL failed to open TIFF file: '{filename}'")
#
#             # Retrieve Bathymetric data
#             depth_band = ds.GetRasterBand(1)
#             if not depth_band:
#                 raise RuntimeError(
#                     f"Error retrieving depth data from {filename}.")
#
#             ndv = depth_band.GetNoDataValue()
#             depth = depth_band.ReadAsArray()
#             depth_gt = ds.GetGeoTransform()
#             resolution = depth_gt[1]
#             if resolution < 1:
#                 print("WARNING: detected resolution value is <= 1. \n \
#                       Setting resolution value to 1")
#                 resolution = 1
#
#             tiff_data.set_data(depth)
#             tiff_data.set_resolution(resolution)
#             tiff_data = remove_edge_ndv(tiff_data, ndv)
#             # depth = make_square(depth)
#             tiff_data.set_resolution(resolution)
#             tiff_data.set_nodata_value(ndv)
#
#             return tiff_data
#
#
# class BPSBathyReader(BathyReader):
#     """
#     Class implementation for reading BPS Bathymetry files
#     """
#     def __init__(self, filename: str | Path) -> None:
#         super().__init__(filename=filename)
#
#     def read_file(self) -> BPSBathymetry:
#         filename = self.filename
#         if not self.is_valid_filename(filename):
#             raise RuntimeError("Error reading bathymetry file")
#         bps_data = BPSBathymetry(filename)
#         return bps_data
#
# class CSVBathyReader(BathyReader):
#     """
#     Class implementation for reading CSV Bathymetry files
#     """
#
#     def __init__(self, filename: str | Path) -> None:
#         super().__init__(filename=filename)
#
#     def read_file(self) -> CSVBathymetry:
#         filename = self.filename
#         if not self.is_valid_filename(filename):
#             raise RuntimeError("Error reading bathymetry file")
#         csv_data = CSVBathymetry(filename)
#         return csv_data
#
#
# class FileReaderSelector:
#     """
#     Class implementation for selecting appropriate file reader by type
#     """
#     @staticmethod
#     def select_reader(filename: str | Path):
#         filename = str(filename)
#         if filename.endswith(".csv"):
#             return CSVBathyReader(filename)
#         elif filename.endswith((".tif", ".tiff")):
#             return RasterBathyReader(filename)
#         elif filename.endswith(".bps"):
#             return BPSBathyReader(filename)
#         else:
#             raise RuntimeError(f"Unrecognized file type: {filename}")