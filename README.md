# nbs_uncertainty_estimation

## What nbs_uncertainty does

Compute Uncertainty in subsampled Bathymetric Data

## How to install nbs_uncertainty

:::{todo}
- 
:::

To install this package run:

`python -m pip install git+https://github.com/franciscorpuz-NOAA/nbs_uncertainty_estimation.git`

## Get started using nbs_uncertainty

Check 'exploratory_analysis' Jupyter Notebook for basic usage

```python
>> > from nbs_uncertainty.readers.bathymetryFileReaders import FileReaderSelector
>> > from nbs_uncertainty.ignore.surfaceEstimators import EstimatorSelector

>> > file_reader = FileReaderSelector.select_reader(full_path)
>> > bathy_data = file_reader.read_file(full_path)
>> > estimator = EstimatorSelector.create_estimator(bathy_data)
>> > uncertainty_estimate = estimate_surface(bathy_data)



```

## How to cite nbs_uncertainty_estimation

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html