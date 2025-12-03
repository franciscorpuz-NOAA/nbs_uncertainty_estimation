from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class UncertaintyEstimator(ABC):
    """
    Abstract class for uncertainty estimation methods
    """

    @abstractmethod
    def compute_uncertainty(self) -> np.ndarray:
        raise NotImplementedError



