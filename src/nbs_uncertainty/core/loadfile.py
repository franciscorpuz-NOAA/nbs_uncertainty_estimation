from nbs_uncertainty.readers.bathymetryFileReaders import (RasterReader,
                                                           CSVReader,
                                                           BPSReader)
from pathlib import Path

def load_file(filename: str | Path, **kwargs):
    """
    Selects proper reader based on filetype

    Parameters
    ----------
    filename

    Returns
    -------
    BathymetryDataset
    """

    filename = str(filename)
    if filename.endswith(".csv"):
        reader = CSVReader()
    elif filename.endswith((".tif", ".tiff")):
        reader = RasterReader()
    elif filename.endswith(".bps"):
        reader =  BPSReader()
    else:
        raise RuntimeError(f"Unrecognized file type: {filename}")

    return reader.read_file(filename, **kwargs)