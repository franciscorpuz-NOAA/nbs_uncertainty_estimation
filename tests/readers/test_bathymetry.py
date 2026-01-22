import pytest
from nbs_uncertainty.readers.bathymetryDataset import BathymetryDataset
import numpy as np
from numpy.testing import assert_almost_equal

@pytest.fixture
def sample_bathymetrydataset():
    # Sample bathydataset for testing
    return BathymetryDataset(
        filename="Rocky.tiff",
        type="raster",
        depth_data=np.array([[2.0, 1.5], [2.0, 1.0]]),
        metadata={"resolution": 8, "ndv_value": 10000}
    )

def test_bathymetrydataset(sample_bathymetrydataset):
    assert sample_bathymetrydataset.metadata["resolution"] == 8
    assert sample_bathymetrydataset.metadata["ndv_value"] == 10000
    assert sample_bathymetrydataset.filename == "Rocky.tiff"
    assert_almost_equal(sample_bathymetrydataset.depth_data, np.array([[2.0, 1.5], [2.0, 1.0]]))

def test_metadata(sample_bathymetrydataset):
    assert isinstance(sample_bathymetrydataset.metadata, dict)
    assert sample_bathymetrydataset.metadata == {"resolution": 8, "ndv_value": 10000}
    assert "ndv_value" in sample_bathymetrydataset.metadata
