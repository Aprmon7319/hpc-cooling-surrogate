"""
Models module for fmu2ml package.
"""

from .base_model import BaseModel
from .fno import FNOCooling
from .hybrid_fno import HybridFNOCooling
from .deeponet import DeepONetCooling
from .model_registry import ModelRegistry, create_model

__all__ = [
    'BaseModel',
    'FNOCooling',
    'HybridFNOCooling',
    'DeepONetCooling',
    'ModelRegistry',
    'create_model',
]
