# from ..readers.bathymetryDataset import (RasterBathymetry,
#                                          BPSBathymetry,
#                                          CSVBathymetry)

# from typing import Dict, Type, Callable, Tuple, Any, Union
# from copy import deepcopy
# import numpy as np


class BathyProcessor:
    """
    Selects appropriate residual and ignore based on depth type
    """

    _method_dicts: Dict[Tuple[str, Type[Union[RasterBathymetry,
                                        BPSBathymetry,
                                        CSVBathymetry]]],
                        Callable[..., Any]] = {}

    @classmethod
    def register(cls, method_name: str,
                 bathy_type: Type[Union[RasterBathymetry,
                                        BPSBathymetry,
                                        CSVBathymetry]]):
        def decorator(func: Callable[..., Any]):
            cls._method_dicts[(method_name, bathy_type)] = func
            return func
        return decorator

    @classmethod
    def estimate_surface(cls, method_name: str,
                         bathy_data: Union[RasterBathymetry,
                                           BPSBathymetry,
                                           CSVBathymetry],
                         *args, **kwargs):
        
        key = (method_name, type(bathy_data))

        handler = cls._method_dicts.get(key)
        if handler is None:
            print(f"key: {key}")
            print(f"keys:{cls._method_dicts.keys()}")
            raise ValueError(f"{method_name} Processor not found for object {bathy_data}")
        if 'param' in kwargs.keys():
            settings = kwargs['param']
            bathy_data.metadata.update(settings)
        output = handler(bathy_data, *args, **kwargs)

        return output

    @classmethod
    def compute_residual(cls, bathy_data: RasterBathymetry, *args, **kwargs):
        return cls.estimate_surface('residual',
                                        bathy_data,
                                        *args, **kwargs)

    @classmethod
    def estimate_uncertainty(cls, method_name: str,
                             bathy_data: RasterBathymetry, *args, **kwargs):
        return cls.estimate_surface(method_name=method_name,
                                        bathy_data=bathy_data,
                                        *args, **kwargs)

