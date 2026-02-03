# nbs_uncertainty_estimation

## What nbs_uncertainty does

Compute Uncertainty in subsampled Bathymetric Data using various methods

## How to install nbs_uncertainty

[//]: # (:::{todo})

[//]: # (- )

[//]: # (:::)

To install this package run:

`python -m pip install git+https://github.com/franciscorpuz-NOAA/nbs_uncertainty_estimation.git`

## Get started using nbs_uncertainty

Check 'exploratory_analysis' Jupyter Notebook for basic usage

```python
from nbs_uncertainty.core.loadfile import load_file
from nbs_uncertainty.processors.rasterProcessor import RasterProcessor
from nbs_uncertainty.utils import utils
import numpy as np

# Read bathymetry data from file
filename = "../data/raster/BlueTopo.tiff"
bathy_data = load_file(filename)

# Optional: Visualize the dataset
bathy_data.show_depth()

# Create processor with specified subsampling parameters
linespacing = 256
max_multiple = 1
current_multiple = 1
processor = RasterProcessor(linespacing_meters=linespacing,
                            max_multiple=max_multiple,
                            multiple=current_multiple)

# Use processor object to compute various surface calculations
# E.g., Compute the residual surface
residual = processor.compute_residual_surface(bathy_data)

# Compute Uncertainty using PSD spectrum, etc
uncertainty_psd_v1 = processor.estimate_uncertainty(method='psd_v1', residual_data=residual)




```

## How to cite nbs_uncertainty_estimation

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html