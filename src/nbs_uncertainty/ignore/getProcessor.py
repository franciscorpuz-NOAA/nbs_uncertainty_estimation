from nbs_uncertainty.readers.bathymetryDataset import BathymetryDataset
from nbs_uncertainty.processors.rasterProcessor import RasterProcessor

def getProcessor(dataset: BathymetryDataset,
                 linespacing_meters: int,
                 current_multiple: int,
                 max_multiple: int):

    dataset_type = dataset.type
    if dataset_type == 'raster':
        return RasterProcessor(linespacing_meters=linespacing_meters,
                               multiple=current_multiple,
                               max_multiple=max_multiple)
    elif dataset_type in ['csv', 'bps']:
        print("For future implementation")
        return None
    else:
        raise ValueError("Unrecognized dataset type")
