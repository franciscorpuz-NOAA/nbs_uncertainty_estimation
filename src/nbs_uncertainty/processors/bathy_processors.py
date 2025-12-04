from ..readers.bathymetry import Bathymetry
from typing import Dict, Type, Callable, Tuple, Any

class BathyProcessor:
    """
    Selects appropriate residual and estimators based on data type
    """

    _method_dicts: Dict[Tuple[str, Type[Bathymetry]], Callable[..., Any]] = {}

    @classmethod
    def register(cls, method_name: str, bathy_data: Type[Bathymetry]):
        def decorator(func: Callable[..., Any]):
            cls._method_dicts[(method_name, bathy_data)] = func
            return func
        return decorator

    @classmethod
    def estimate_surface(cls, method_name: str, bathy_data: Bathymetry, *args, **kwargs):
        key = (method_name, type(bathy_data))

        handler = cls._method_dicts.get(key)
        if handler is None:
            raise ValueError(f"{method_name} Processor not found for type {type(bathy_data)}")
        return handler(bathy_data, *args, **kwargs)

    @classmethod
    def compute_residual(cls, bathy_data: Bathymetry, param: Dict | None):
        return cls.estimate_surface(method_name='residual', bathy_data=bathy_data, param = param)



