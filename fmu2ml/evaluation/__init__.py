"""
Evaluation module for fmu2ml package.
"""

from .metrics import calculate_metrics, MetricsCalculator
from .physics_validator import PhysicsValidator
from .comparator import ModelComparator

__all__ = [
    'calculate_metrics',
    'MetricsCalculator',
    'PhysicsValidator',
    'ModelComparator',
]
