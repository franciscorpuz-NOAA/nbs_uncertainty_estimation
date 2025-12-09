from ..readers.bathymetry import (RasterBathymetry,
                                  BPSBathymetry,
                                  CSVBathymetry)

from typing import Dict, Type, Callable, Tuple, Any, Union
from copy import deepcopy


class BathyProcessor:
    """
    Selects appropriate residual and estimators based on data type
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
            print( cls._method_dicts.keys())
            raise ValueError(f"{method_name} Processor not found for type {type(bathy_data)}")
        output = handler(bathy_data, *args, **kwargs)
        new_bathy = deepcopy(bathy_data)
        new_bathy.data = output
        if 'param' in kwargs.keys():
            settings = kwargs['param']
            new_bathy.metadata.update(settings)
        return new_bathy

    @classmethod
    def compute_residual(cls, bathy_data: RasterBathymetry, *args, **kwargs):
        residual = cls.estimate_surface('residual',
                                        bathy_data,
                                        *args, **kwargs)
        return residual

    @classmethod
    def estimate_uncertainty(cls, method_name: str,
                             bathy_data: RasterBathymetry, *args, **kwargs):
        uncertainty = cls.estimate_surface(method_name=method_name,
                                        bathy_data=bathy_data,
                                        *args, **kwargs)
        return uncertainty

