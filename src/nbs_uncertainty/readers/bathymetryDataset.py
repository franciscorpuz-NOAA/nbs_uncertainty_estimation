from dataclasses import dataclass
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class BathymetryDataset:
    """
    Class for storing generic bathymetry depth data.

    Attributes
    ----------
    filename : str
        Name of the source file.
    type : str
        Type of the dataset (e.g. 'raster', 'bag').
    depth_data : np.ndarray
        Numpy array containing depth values.
    metadata : Dict
        Dictionary containing metadata such as resolution, bounds, etc.
    """

    filename: str
    type: str
    depth_data: np.ndarray
    metadata: Dict


class RasterDataset(BathymetryDataset):
    """
    Subclass for Raster-type bathymetry with additional helper functions
    """

    def __init__(self, depth_data, filename='none', metadata=None, data_type='raster'):
        self.filename = filename
        self.metadata = metadata
        self.type = data_type
        self.depth_data = depth_data

    @property
    def resolution(self):
        """Returns data resolution extracted from the raster metadata"""
        return self.metadata['resolution']

    @property
    def min_val(self):
        """Returns minimum value in the depth array, excluding NaNs"""
        return np.nanmin(self.depth_data.flatten())

    @property
    def max_val(self):
        """Returns maximum value in the depth array, excluding NaNs"""
        return np.nanmax(self.depth_data.flatten())

    @property
    def ndv_value(self):
        """Returns no-data-value extracted from the raster metadata"""
        return self.metadata['ndv_value']

    def show_depth(self, title: str = None):
        """
        Plots the depth for visualization
        Parameters
        ----------
        title: str
            Custom plot title
            Default is filename with linespacing and resolution information

        Returns
        -------
        none

        """
        fig, ax1 = plt.subplots(1)
        im = ax1.imshow(self.depth_data, cmap='terrain', aspect='equal')
        res = self.metadata['resolution']
        shape_0 = self.depth_data.shape[0]
        shape_1 = self.depth_data.shape[1]
        fig.colorbar(im, label='Depth (m)')
        locs = ax1.get_xticks()
        ax1.set_xticks(locs)
        ax1.set_xticklabels([str(int(x * res)) for x in locs])
        locs = ax1.get_yticks()
        ax1.set_yticks(locs)
        ax1.set_yticklabels([str(int(y * res)) for y in locs])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_xlabel("West-East (m)")
        ax1.set_ylabel("North-South (m)")
        if title is None:
            fn = Path(self.filename).name
            title = f"{fn} at {res}m resolution"
        ax1.set_title(f"""
                    Surface:{title} at {res}m resolution
                    Dimensions: {shape_0 * res / 1000}km by {shape_1 * res / 1000}km
                        """)
        ax1.set_xlim(left=0, right=shape_1)
        ax1.set_ylim(top=0, bottom=shape_0)

    def __repr__(self) -> str:
        """
        Convenience function for printing raster-specific information

        Returns
        -------
        none
        """
        return (f'Type: {self.type}'
                f'\n Filename: {self.filename}'
                f'\n Full path: {self.metadata["full_path"]}'
                f'\n Resolution: {self.resolution}'
                f'\n No Data Value: {self.ndv_value}'
                f'\n Min\Max value: [{self.min_val, self.max_val}]'
                f'\n Data Shape: {self.depth_data.shape}'
                )
