"""
Inference module for fmu2ml package.
"""

from .predictor import CoolingModelPredictor
from .batch_processor import BatchProcessor

__all__ = [
    'CoolingModelPredictor',
    'BatchProcessor',
]
